import {
  ResponseError,
  type AmgixApi,
  type ClusterView,
  type NodeMetricSeries,
  type NodeView,
  type ReadyResponse,
  type WindowSample,
} from '@amgix/amgix-client'
import {
  CategoryScale,
  Chart,
  Legend,
  LineController,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
  type ChartConfiguration,
  type ChartOptions,
  type Scale,
  type Tick,
} from 'chart.js'
import $ from 'jquery'

import { hideDashboardError, showDashboardError } from '../error-bar'
import { DashboardPanel } from './panel-base'

Chart.register(LineController, LineElement, PointElement, CategoryScale, LinearScale, Tooltip, Legend)

const HOME_READY_POLL_MS = 10_000

const HOME_METRICS_TAB_API_ID = 'dashboard-home-metrics-tab-api'
const HOME_METRICS_TAB_ENCODER_ID = 'dashboard-home-metrics-tab-encoder'
const HOME_METRICS_PANEL_API_ID = 'dashboard-home-metrics-panel-api'
const HOME_METRICS_PANEL_ENCODER_ID = 'dashboard-home-metrics-panel-encoder'

/** Cluster table embedding columns use this rolling window (seconds). */
const CLUSTER_METRICS_WINDOW_SEC = 30

/** Cluster chart: client-side history span (API is snapshot-only). */
const CLUSTER_CHART_HISTORY_MS = 10 * 60 * 1000

/** Evenly spaced time ticks from min..max (avoids Chart.js “nice” gaps at the edges). */
const CLUSTER_CHART_X_TICK_COUNT = 7

function buildClusterChartEvenXTicks(min: number, max: number, count: number): Tick[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [{ value: min }, { value: max }]
  }
  if (max <= min) {
    return [{ value: min }]
  }
  const n = Math.max(2, Math.floor(count))
  const span = max - min
  const step = span / (n - 1)
  const ticks: Tick[] = []
  for (let i = 0; i < n; i += 1) {
    ticks.push({ value: i === n - 1 ? max : min + i * step })
  }
  return ticks
}

/** Survives full page reload; same origin only. */
const CLUSTER_CHART_STORAGE_KEY = 'amgix.dashboard.clusterThroughputHistory.v1'
const CLUSTER_API_CHART_STORAGE_KEY = 'amgix.dashboard.apiMetricsHistory.v1'

function clusterHelpIcon(tip: string): JQuery<HTMLElement> {
  return $('<button>', {
    type: 'button',
    class: 'dashboard-home-cluster-help',
    text: 'i',
    attr: {
      title: tip,
      'aria-label': tip,
    },
  })
}

function clusterThWithHelp(label: string, tip: string): JQuery<HTMLElement> {
  return $('<th>', { class: 'dashboard-home-cluster-th-with-help' }).append(
    $('<span>', { class: 'dashboard-home-cluster-th-inner' }).append(
      $('<span>', { class: 'dashboard-home-cluster-th-label', text: label }),
      clusterHelpIcon(tip),
    ),
  )
}

function clusterBatchesColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Number of embed requests that originated on this node in the last ${w}s.`
}

function clusterRateColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Documents embedded per second, originating on this node, over the last ${w}s.`
}

function clusterInferMsColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Local model inference time per document, in ms, over the last ${w}s.`
}

function clusterPipeMsColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Routed inference time per document, in ms, as seen by the originating encoder (includes RPC if the request was forwarded to another node), over the last ${w}s.`
}

function clusterErrPerSecColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Failed embed requests per second, originating on this node, over the last ${w}s.`
}

function clusterChartHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Inference and routed latency per document over time, ${w}s rolling window.`
}

function clusterApiChartHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `API request rate and mean latency over time. All traffic, search, and ingestion (async + sync + bulk), ${w}s rolling window per poll.`
}

function clusterApiAllReqColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `All HTTP requests per second handled by this API process, over the last ${w}s.`
}

function clusterApiAllMsColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Mean latency in ms for all HTTP requests on this API process, over the last ${w}s.`
}

function clusterApiAsyncReqColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Async single-document uploads per second (POST …/documents), over the last ${w}s.`
}

function clusterApiAsyncMsColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Mean latency in ms for async single-document uploads, over the last ${w}s.`
}

function clusterApiSyncReqColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Sync single-document uploads per second (POST …/documents/sync), over the last ${w}s.`
}

function clusterApiSyncMsColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Mean latency in ms for sync single-document uploads, over the last ${w}s.`
}

function clusterApiBulkReqColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Bulk upload requests per second (POST …/documents/bulk), over the last ${w}s.`
}

function clusterApiBulkMsColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Mean latency in ms for bulk upload requests, over the last ${w}s.`
}

function clusterApiSearchReqColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Search requests per second (POST …/search), over the last ${w}s.`
}

function clusterApiSearchMsColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Mean latency in ms for search requests, over the last ${w}s.`
}

function clusterApiErrPerSecColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `HTTP 4xx and 5xx responses per second from this API process, combined, over the last ${w}s.`
}

type HomeReadinessKey = 'database' | 'rabbitmq' | 'index' | 'query'

function formatRequestError(context: string, err: unknown): string {
  if (err instanceof ResponseError) {
    return `${context} (HTTP ${err.response.status})`
  }
  if (err instanceof Error) {
    return err.message
  }
  return context
}

async function fetchReadiness(): Promise<ReadyResponse> {
  const res = await fetch('/v1/health/ready')
  return (await res.json()) as ReadyResponse
}

function formatVersionLabel(raw: string): string {
  const t = raw.trim()
  return t.startsWith('v') ? t : `v${t}`
}

/** Backend-reported DB version for display: optional leading `v`; strip legacy `db ` prefix. */
function formatDatabaseVersionLabel(raw: string): string {
  let t = raw.trim().replace(/^db\s+/i, '')
  if (t.toLowerCase() === 'unknown') {
    return t
  }
  return formatVersionLabel(t)
}

function shouldShowInfrastructureVersion(raw: string | undefined | null): boolean {
  const t = (raw ?? '').trim()
  if (t === '') return false
  if (t.toLowerCase() === 'unknown') return false
  return true
}

function formatDatabaseSummary(info: { database_kind: string; database_version: string }): string {
  const type = info.database_kind.trim() || 'Database'
  const mid = shouldShowInfrastructureVersion(info.database_version)
    ? ` (${formatDatabaseVersionLabel(info.database_version)})`
    : ''
  return `${type}${mid}`
}

function formatRabbitmqSummary(info: { rabbitmq_version: string }): string {
  const mid = shouldShowInfrastructureVersion(info.rabbitmq_version)
    ? ` (${formatVersionLabel(info.rabbitmq_version.trim())})`
    : ''
  return `RabbitMQ${mid}`
}

function readinessLabel(ok: boolean): string {
  return ok ? 'Ready' : 'Not ready'
}

function formatLoadModelsCell(node: NodeView): string {
  if (!node.load_models) {
    return 'No'
  }
  const n = (node.loaded_models ?? []).length
  return `Yes (${n})`
}

function formatGpuStatus(node: NodeView): string {
  if (!node.gpu_support) {
    return 'n/a'
  }
  if (node.gpu_available) {
    return 'Yes'
  }
  return 'Undetected'
}

const clusterMetricsWindowKey = String(CLUSTER_METRICS_WINDOW_SEC)

function getNodeMetricWindowSample(s: NodeMetricSeries): WindowSample | undefined {
  const w = s.windows
  if (w == null) {
    return undefined
  }
  return w[clusterMetricsWindowKey] ?? w[CLUSTER_METRICS_WINDOW_SEC]
}

function getMetricWindowSampleByFirstKey(list: NodeMetricSeries[], name: string): WindowSample | undefined {
  for (const s of list) {
    if (s.key[0] === name) {
      return getNodeMetricWindowSample(s)
    }
  }
  return undefined
}

function formatApiRateCell(list: NodeMetricSeries[], rateKey: string): string {
  const ws = getMetricWindowSampleByFirstKey(list, rateKey)
  if (ws == null) {
    return ''
  }
  const v = Number(ws.value)
  if (!Number.isFinite(v)) {
    return ''
  }
  return formatClusterRpsCell(v)
}

function formatApiAvgMsCell(list: NodeMetricSeries[], msKey: string): string {
  const ws = getMetricWindowSampleByFirstKey(list, msKey)
  if (ws == null) {
    return ''
  }
  const n = Number(ws.n)
  const v = Number(ws.value)
  if (!Number.isFinite(n) || n <= 0 || !Number.isFinite(v)) {
    return ''
  }
  return formatClusterAvgMsCell(v)
}

function formatApiHttpErrorsRateCell(list: NodeMetricSeries[]): string {
  let sum = 0
  let saw = false
  for (const name of ['api_error_4xx', 'api_error_5xx'] as const) {
    const ws = getMetricWindowSampleByFirstKey(list, name)
    if (ws != null) {
      const v = Number(ws.value)
      if (Number.isFinite(v)) {
        sum += v
        saw = true
      }
    }
  }
  if (!saw) {
    return ''
  }
  return formatClusterRpsCell(sum)
}

function formatApiMetricsCells(node: NodeView): {
  allReq: string
  allMs: string
  asyncReq: string
  asyncMs: string
  syncReq: string
  syncMs: string
  bulkReq: string
  bulkMs: string
  searchReq: string
  searchMs: string
  errPerSec: string
} {
  const list = node.metrics ?? []
  return {
    allReq: formatApiRateCell(list, 'api_requests'),
    allMs: formatApiAvgMsCell(list, 'api_request_ms'),
    asyncReq: formatApiRateCell(list, 'api_async_upload'),
    asyncMs: formatApiAvgMsCell(list, 'api_async_upload_ms'),
    syncReq: formatApiRateCell(list, 'api_sync_upload'),
    syncMs: formatApiAvgMsCell(list, 'api_sync_upload_ms'),
    bulkReq: formatApiRateCell(list, 'api_bulk_upload'),
    bulkMs: formatApiAvgMsCell(list, 'api_bulk_upload_ms'),
    searchReq: formatApiRateCell(list, 'api_search'),
    searchMs: formatApiAvgMsCell(list, 'api_search_ms'),
    errPerSec: formatApiHttpErrorsRateCell(list),
  }
}

/** Stable merge key for dimensions key[1..3] (vector type, model, revision). */
function seriesDimKey(s: NodeMetricSeries): string {
  const k = s.key
  if (k.length < 4) {
    return k.join('\0')
  }
  return `${k[1]}\0${k[2] ?? ''}\0${k[3] ?? ''}`
}

function seriesDisplayLabel(s: NodeMetricSeries): string {
  const k = s.key
  const model = (k[2] ?? '').trim()
  const revision = (k[3] ?? '').trim()
  if (model !== '') {
    return revision !== '' ? `${model} (${revision})` : model
  }
  return k[1] ?? ''
}

/** Cluster grid: fixed decimal places for rates (docs/s, err/s). */
function formatClusterRpsCell(rps: number): string {
  if (!Number.isFinite(rps)) {
    return ''
  }
  return new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(rps)
}

/** Cluster grid and chart: fixed decimal places for latency (ms). */
function formatClusterAvgMsCell(ms: number): string {
  if (!Number.isFinite(ms)) {
    return ''
  }
  return new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(ms)
}

function latestFiniteSeriesY(data: ReadonlyArray<{ y?: number | null }>): number | null {
  for (let i = data.length - 1; i >= 0; i -= 1) {
    const y = data[i]?.y
    if (typeof y === 'number' && Number.isFinite(y)) {
      return y
    }
  }
  return null
}

function clusterChartLegendParts(
  baseLabel: string,
  data: ReadonlyArray<{ y?: number | null }>,
): { labelStem: string; valuePart: string | null } {
  const y = latestFiniteSeriesY(data)
  if (y == null) {
    return { labelStem: baseLabel, valuePart: null }
  }
  if (y === 0) {
    return { labelStem: `${baseLabel}: `, valuePart: '-' }
  }
  return { labelStem: `${baseLabel}: `, valuePart: `${formatClusterAvgMsCell(y)} ms` }
}

function clusterApiChartLegendPartsRps(
  baseLabel: string,
  data: ReadonlyArray<{ y?: number | null }>,
): { labelStem: string; valuePart: string | null } {
  const y = latestFiniteSeriesY(data)
  if (y == null) {
    return { labelStem: baseLabel, valuePart: null }
  }
  if (y === 0) {
    return { labelStem: `${baseLabel}: `, valuePart: '-' }
  }
  return { labelStem: `${baseLabel}: `, valuePart: `${formatClusterRpsCell(y)} /s` }
}

/**
 * Table cells: batches_origin (count), docs_origin (rate),
 * inference_ms_per_doc and inference_origin_ms_per_doc (latency),
 * inference_origin_errors (sum in window → per-second rate).
 */
function formatEmbeddingMetricsCells(node: NodeView): {
  rps: string
  avgMs: string
  e2eAvgMs: string
  requests: string
  errPerSec: string
} {
  const list = node.metrics ?? []
  let sawAny = false
  let totalBatches = 0
  let sumDocsRps = 0
  let weightedMs = 0
  let msN = 0
  let weightedOriginMs = 0
  let originN = 0
  let sumErrorsInWindow = 0
  for (const s of list) {
    const name = s.key[0]
    const ws = getNodeMetricWindowSample(s)
    if (ws == null) {
      continue
    }
    const n = Number(ws.n)
    const v = Number(ws.value)
    if (name === 'batches_origin') {
      sawAny = true
      if (Number.isFinite(n) && n > 0) {
        totalBatches += Math.trunc(n)
      }
    } else if (name === 'docs_origin') {
      sawAny = true
      if (Number.isFinite(v)) {
        sumDocsRps += v
      }
    } else if (name === 'inference_ms_per_doc') {
      if (Number.isFinite(n) && n > 0) {
        sawAny = true
        const ni = Math.trunc(n)
        if (Number.isFinite(v)) {
          weightedMs += v * ni
          msN += ni
        }
      }
    } else if (name === 'inference_origin_ms_per_doc') {
      if (Number.isFinite(n) && n > 0) {
        sawAny = true
        const ni = Math.trunc(n)
        if (Number.isFinite(v)) {
          weightedOriginMs += v * ni
          originN += ni
        }
      }
    } else if (name === 'inference_origin_errors') {
      sawAny = true
      if (Number.isFinite(v)) {
        sumErrorsInWindow += v
      }
    }
  }
  if (!sawAny) {
    return { rps: '', avgMs: '', e2eAvgMs: '', requests: '', errPerSec: '' }
  }
  const errPerSec =
    CLUSTER_METRICS_WINDOW_SEC > 0 ? sumErrorsInWindow / CLUSTER_METRICS_WINDOW_SEC : 0
  return {
    rps: formatClusterRpsCell(sumDocsRps),
    avgMs: msN > 0 ? formatClusterAvgMsCell(weightedMs / msN) : '',
    e2eAvgMs: originN > 0 ? formatClusterAvgMsCell(weightedOriginMs / originN) : '',
    requests: String(totalBatches),
    errPerSec: formatClusterRpsCell(errPerSec),
  }
}

const LEGACY_CHART_TYPE_MODEL_PREFIX = /^(dense_model|sparse_model):\s*(.+)$/i

/** Human-readable series name from merge key `type\\0model\\0revision`. */
function labelFromVectorMetricsMergeKey(key: string): string {
  const parts = key.split('\0')
  const typ = parts[0] ?? ''
  const model = (parts[1] ?? '').trim()
  const revision = (parts[2] ?? '').trim()
  if (model !== '') {
    return revision !== '' ? `${model} (${revision})` : model
  }
  return typ
}

/** Chart legend/tooltip caption; strips legacy `dense_model: …` from storage and fixes type-only stubs. */
function normalizeVectorSeriesLabelForKey(key: string, stored: string | undefined): string {
  if (stored != null && stored !== '') {
    const legacy = LEGACY_CHART_TYPE_MODEL_PREFIX.exec(stored)
    if (legacy) {
      return legacy[2]!.trim()
    }
    if (stored === 'dense_model' || stored === 'sparse_model') {
      return labelFromVectorMetricsMergeKey(key)
    }
    return stored
  }
  return labelFromVectorMetricsMergeKey(key)
}

type ClusterThroughputAgg = {
  chartRows: Array<{ key: string; label: string; inferenceMeanMs: number }>
  globalAvgMs: number | null
  globalE2eMs: number | null
}

function aggregateClusterThroughput(view: ClusterView | null): ClusterThroughputAgg {
  const empty: ClusterThroughputAgg = {
    chartRows: [],
    globalAvgMs: null,
    globalE2eMs: null,
  }
  if (view == null || view.nodes == null) {
    return empty
  }

  const byKey = new Map<string, { label: string; weightedMsSum: number; nSum: number }>()
  let weightedAvgSum = 0
  let weightedAvgN = 0
  let weightedE2eSum = 0
  let weightedE2eN = 0

  for (const node of Object.values(view.nodes)) {
    for (const s of node.metrics ?? []) {
      if (s.key[0] !== 'inference_ms_per_doc') {
        continue
      }
      const wm = getNodeMetricWindowSample(s)
      if (wm == null) {
        continue
      }
      const n = Number(wm.n)
      if (!Number.isFinite(n) || n <= 0) {
        continue
      }
      const ni = Math.trunc(n)
      const key = seriesDimKey(s)
      const label = seriesDisplayLabel(s)
      let cur = byKey.get(key)
      if (!cur) {
        cur = { label, weightedMsSum: 0, nSum: 0 }
        byKey.set(key, cur)
      }

      const avgMs = Number(wm.value)
      if (Number.isFinite(avgMs)) {
        cur.weightedMsSum += avgMs * ni
        cur.nSum += ni
        weightedAvgSum += avgMs * ni
        weightedAvgN += ni
      }
    }
    for (const s of node.metrics ?? []) {
      if (s.key[0] !== 'inference_origin_ms_per_doc') {
        continue
      }
      const wm = getNodeMetricWindowSample(s)
      if (wm == null) {
        continue
      }
      const n = Number(wm.n)
      if (!Number.isFinite(n) || n <= 0) {
        continue
      }
      const ni = Math.trunc(n)
      const e2e = Number(wm.value)
      if (Number.isFinite(e2e)) {
        weightedE2eSum += e2e * ni
        weightedE2eN += ni
      }
    }
  }

  const chartRows = Array.from(byKey.entries())
    .filter(([, v]) => v.nSum > 0)
    .map(([key, v]) => ({
      key,
      label: v.label,
      inferenceMeanMs: v.weightedMsSum / v.nSum,
    }))
    .sort((a, b) => b.inferenceMeanMs - a.inferenceMeanMs)
  return {
    chartRows,
    globalAvgMs: weightedAvgN > 0 ? weightedAvgSum / weightedAvgN : null,
    globalE2eMs: weightedE2eN > 0 ? weightedE2eSum / weightedE2eN : null,
  }
}

function sortApiNodeEntries(entries: Array<[string, NodeView]>): Array<[string, NodeView]> {
  const copy = [...entries]
  copy.sort((a, b) => a[0].localeCompare(b[0], undefined, { sensitivity: 'base' }))
  return copy
}

function sortEncoderNodeEntries(entries: Array<[string, NodeView]>): Array<[string, NodeView]> {
  const copy = [...entries]
  copy.sort((a, b) => {
    const byRole = a[1].role.localeCompare(b[1].role, undefined, { sensitivity: 'base' })
    if (byRole !== 0) {
      return byRole
    }
    return a[0].localeCompare(b[0], undefined, { sensitivity: 'base' })
  })
  return copy
}

function partitionApiAndEncoderNodes(
  nodes: { [key: string]: NodeView },
): { api: Array<[string, NodeView]>; encoders: Array<[string, NodeView]> } {
  const api: Array<[string, NodeView]> = []
  const encoders: Array<[string, NodeView]> = []
  for (const entry of Object.entries(nodes)) {
    if (entry[1].role === 'api') {
      api.push(entry)
    } else {
      encoders.push(entry)
    }
  }
  return { api: sortApiNodeEntries(api), encoders: sortEncoderNodeEntries(encoders) }
}

type ApiMetricsHistoryPoint = {
  t: number
  allRps: number | null
  allMs: number | null
  searchRps: number | null
  searchMs: number | null
  ingestRps: number | null
  ingestMs: number | null
  errRps: number | null
}

/** Cluster-wide API aggregates; err4xx/err5xx are for Cluster Info, errRps is their sum (chart). */
type AggregateApiChartMetrics = Omit<ApiMetricsHistoryPoint, 't'> & {
  err4xxRps: number | null
  err5xxRps: number | null
}

function aggregateApiChartMetrics(view: ClusterView | null): AggregateApiChartMetrics {
  const empty: AggregateApiChartMetrics = {
    allRps: null,
    allMs: null,
    searchRps: null,
    searchMs: null,
    ingestRps: null,
    ingestMs: null,
    errRps: null,
    err4xxRps: null,
    err5xxRps: null,
  }
  if (view == null || view.nodes == null) {
    return empty
  }
  let sumAllRps = 0
  let sawAllRps = false
  let allMsSumVN = 0
  let allMsSumN = 0
  let sumSearchRps = 0
  let sawSearchRps = false
  let searchMsSumVN = 0
  let searchMsSumN = 0
  let sumIngestRps = 0
  let sawIngestRps = false
  let ingestMsSumVN = 0
  let ingestMsSumN = 0
  let sumErr4xx = 0
  let sawErr4xx = false
  let sumErr5xx = 0
  let sawErr5xx = false

  for (const node of Object.values(view.nodes)) {
    if (node.role !== 'api') {
      continue
    }
    const list = node.metrics ?? []

    const wsAllR = getMetricWindowSampleByFirstKey(list, 'api_requests')
    if (wsAllR != null) {
      const v = Number(wsAllR.value)
      if (Number.isFinite(v)) {
        sumAllRps += v
        sawAllRps = true
      }
    }
    const wsAllM = getMetricWindowSampleByFirstKey(list, 'api_request_ms')
    if (wsAllM != null) {
      const v = Number(wsAllM.value)
      const n = Number(wsAllM.n)
      if (Number.isFinite(v) && Number.isFinite(n) && n > 0) {
        const ni = Math.trunc(n)
        allMsSumVN += v * ni
        allMsSumN += ni
      }
    }

    const wsSr = getMetricWindowSampleByFirstKey(list, 'api_search')
    if (wsSr != null) {
      const v = Number(wsSr.value)
      if (Number.isFinite(v)) {
        sumSearchRps += v
        sawSearchRps = true
      }
    }
    const wsSm = getMetricWindowSampleByFirstKey(list, 'api_search_ms')
    if (wsSm != null) {
      const v = Number(wsSm.value)
      const n = Number(wsSm.n)
      if (Number.isFinite(v) && Number.isFinite(n) && n > 0) {
        const ni = Math.trunc(n)
        searchMsSumVN += v * ni
        searchMsSumN += ni
      }
    }

    for (const rk of ['api_async_upload', 'api_sync_upload', 'api_bulk_upload'] as const) {
      const ws = getMetricWindowSampleByFirstKey(list, rk)
      if (ws != null) {
        const v = Number(ws.value)
        if (Number.isFinite(v)) {
          sumIngestRps += v
          sawIngestRps = true
        }
      }
    }
    for (const mk of ['api_async_upload_ms', 'api_sync_upload_ms', 'api_bulk_upload_ms'] as const) {
      const ws = getMetricWindowSampleByFirstKey(list, mk)
      if (ws != null) {
        const v = Number(ws.value)
        const n = Number(ws.n)
        if (Number.isFinite(v) && Number.isFinite(n) && n > 0) {
          const ni = Math.trunc(n)
          ingestMsSumVN += v * ni
          ingestMsSumN += ni
        }
      }
    }

    const ws4 = getMetricWindowSampleByFirstKey(list, 'api_error_4xx')
    if (ws4 != null) {
      const v = Number(ws4.value)
      if (Number.isFinite(v)) {
        sumErr4xx += v
        sawErr4xx = true
      }
    }
    const ws5 = getMetricWindowSampleByFirstKey(list, 'api_error_5xx')
    if (ws5 != null) {
      const v = Number(ws5.value)
      if (Number.isFinite(v)) {
        sumErr5xx += v
        sawErr5xx = true
      }
    }
  }

  const sawAnyErr = sawErr4xx || sawErr5xx
  return {
    allRps: sawAllRps ? sumAllRps : null,
    allMs: allMsSumN > 0 ? allMsSumVN / allMsSumN : null,
    searchRps: sawSearchRps ? sumSearchRps : null,
    searchMs: searchMsSumN > 0 ? searchMsSumVN / searchMsSumN : null,
    ingestRps: sawIngestRps ? sumIngestRps : null,
    ingestMs: ingestMsSumN > 0 ? ingestMsSumVN / ingestMsSumN : null,
    err4xxRps: sawErr4xx ? sumErr4xx : null,
    err5xxRps: sawErr5xx ? sumErr5xx : null,
    errRps: sawAnyErr ? sumErr4xx + sumErr5xx : null,
  }
}

type ApiChartStoredPayload = {
  v: 1
  points: ApiMetricsHistoryPoint[]
}

function loadApiMetricsHistoryFromStorage(cutoff: number): ApiMetricsHistoryPoint[] | null {
  try {
    const s = localStorage.getItem(CLUSTER_API_CHART_STORAGE_KEY)
    if (s == null || s === '') {
      return null
    }
    const parsed = JSON.parse(s) as unknown
    if (!parsed || typeof parsed !== 'object') {
      return null
    }
    const p = parsed as Partial<ApiChartStoredPayload>
    if (p.v !== 1 || !Array.isArray(p.points)) {
      return null
    }
    const points: ApiMetricsHistoryPoint[] = []
    for (const item of p.points) {
      if (
        item != null &&
        typeof item === 'object' &&
        typeof (item as ApiMetricsHistoryPoint).t === 'number' &&
        Number.isFinite((item as ApiMetricsHistoryPoint).t)
      ) {
        const raw = item as ApiMetricsHistoryPoint
        points.push({
          t: raw.t,
          allRps: raw.allRps,
          allMs: raw.allMs,
          searchRps: raw.searchRps,
          searchMs: raw.searchMs,
          ingestRps: raw.ingestRps,
          ingestMs: raw.ingestMs,
          errRps: raw.errRps ?? null,
        })
      }
    }
    const filtered = points.filter((pt) => pt.t >= cutoff)
    filtered.sort((a, b) => a.t - b.t)
    return filtered
  } catch {
    return null
  }
}

function saveApiMetricsHistoryToStorage(points: ApiMetricsHistoryPoint[]): void {
  try {
    const payload: ApiChartStoredPayload = { v: 1, points }
    localStorage.setItem(CLUSTER_API_CHART_STORAGE_KEY, JSON.stringify(payload))
  } catch {
    // ignore
  }
}

function readClusterChartThemeColors(): { grid: string; tick: string } {
  const s = getComputedStyle(document.documentElement)
  const grid = s.getPropertyValue('--chart-cluster-grid').trim()
  const tick = s.getPropertyValue('--chart-cluster-tick').trim()
  return {
    grid: grid || 'rgba(120, 120, 120, 0.18)',
    tick: tick || '#555555',
  }
}

/** Match :root font family; axis sizes kept smaller than body text so they fit the chart. */
function readClusterChartFont(): { family: string; tickPx: number; titlePx: number; tooltipPx: number } {
  const s = getComputedStyle(document.documentElement)
  const family = (s.fontFamily || '').trim() || 'system-ui, sans-serif'
  const rootPx = parseFloat(s.fontSize)
  const base = Number.isFinite(rootPx) && rootPx > 0 ? rootPx : 16
  return {
    family,
    tickPx: Math.max(10, Math.round(base * 0.67)),
    titlePx: Math.max(9, Math.round(base * 0.61)),
    tooltipPx: Math.max(11, Math.round(base * 0.77)),
  }
}

function clusterChartDevicePixelRatio(): number {
  if (typeof window === 'undefined') {
    return 1
  }
  const d = window.devicePixelRatio
  return d != null && d > 0 ? d : 1
}

function buildApiMetricsChartOptions(
  cutoff: number,
  now: number,
  gridColor: string,
  tickColor: string,
  chartFont: ReturnType<typeof readClusterChartFont>,
  yAxisTitle: string,
  latencyTooltip: boolean,
): ChartOptions<'line'> {
  return {
    color: tickColor,
    devicePixelRatio: clusterChartDevicePixelRatio(),
    font: {
      family: chartFont.family,
      size: chartFont.tickPx,
    },
    parsing: false,
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'nearest', axis: 'x', intersect: false },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        titleFont: { family: chartFont.family, size: chartFont.tooltipPx },
        bodyFont: { family: chartFont.family, size: chartFont.tooltipPx },
        callbacks: {
          title(items) {
            const raw = items[0]?.parsed.x
            if (typeof raw !== 'number') {
              return ''
            }
            return new Date(raw).toLocaleString()
          },
          label(ctx) {
            const v = ctx.parsed.y
            const lab = ctx.dataset.label ?? ''
            if (typeof v !== 'number') {
              return lab
            }
            if (latencyTooltip) {
              return `${lab}: ${formatClusterAvgMsCell(v)} ms`
            }
            return `${lab}: ${formatClusterRpsCell(v)} /s`
          },
        },
      },
    },
    scales: {
      x: {
        type: 'linear',
        min: cutoff,
        max: now,
        afterBuildTicks: (axis: Scale) => {
          axis.ticks = buildClusterChartEvenXTicks(
            Number(axis.min),
            Number(axis.max),
            CLUSTER_CHART_X_TICK_COUNT,
          )
        },
        grid: { color: gridColor },
        ticks: {
          autoSkip: false,
          color: tickColor,
          font: { family: chartFont.family, size: chartFont.tickPx },
          callback: (tickValue) => {
            const n = typeof tickValue === 'number' ? tickValue : Number(tickValue)
            if (!Number.isFinite(n)) {
              return ''
            }
            return new Date(n).toLocaleTimeString(undefined, {
              hour: '2-digit',
              minute: '2-digit',
            })
          },
        },
      },
      y: {
        type: 'linear',
        position: 'left',
        beginAtZero: true,
        title: {
          display: true,
          text: yAxisTitle,
          color: tickColor,
          font: { family: chartFont.family, size: chartFont.titlePx, weight: 600 },
        },
        ticks: {
          color: tickColor,
          font: { family: chartFont.family, size: chartFont.tickPx },
        },
        grid: { color: gridColor },
      },
    },
  }
}

function clusterLineColors(count: number): string[] {
  const base = ['#16a34a', '#2563eb', '#ca8a04', '#9333ea', '#db2777', '#0891b2', '#ea580c']
  const out: string[] = []
  for (let i = 0; i < count; i += 1) {
    out.push(base[i % base.length]!)
  }
  return out
}

function renderClusterChartHtmlLegend(
  $ul: JQuery<HTMLElement>,
  items: Array<{ color: string; labelStem: string; valuePart: string | null }>,
  labelColor: string,
): void {
  $ul.empty()
  for (const { color, labelStem, valuePart } of items) {
    const $swatch = $('<span>', { class: 'dashboard-home-cluster-chart-legend-swatch' }).css(
      'background-color',
      color,
    )
    const $text = $('<span>', {
      class: 'dashboard-home-cluster-chart-legend-label',
      css: { color: labelColor },
    })
    $text.append(document.createTextNode(labelStem))
    if (valuePart != null) {
      $text.append(
        $('<strong>', {
          class: 'dashboard-home-cluster-chart-legend-value',
          text: valuePart,
        }),
      )
    }
    $ul.append($('<li>', { class: 'dashboard-home-cluster-chart-legend-item' }).append($swatch, $text))
  }
}

function $readinessStatus(ok: boolean, readinessKey: HomeReadinessKey): JQuery<HTMLElement> {
  return $('<span>', {
    class: ok ? 'dashboard-home-status dashboard-home-status--ready' : 'dashboard-home-status dashboard-home-status--not-ready',
    text: readinessLabel(ok),
    attr: { 'data-home-ready': readinessKey },
  })
}

/** byKey: cluster-wide weighted mean inference latency (ms) per vector merge key */
type ClusterThroughputHistoryPoint = {
  t: number
  byKey: Map<string, number>
  avgMs: number | null
  e2eMs: number | null
}

type ClusterChartStoredPoint = {
  t: number
  avgMs: number | null
  e2eMs: number | null
  byKey: [string, number][]
}

type ClusterChartStoredPayload = {
  v: 1
  points: ClusterChartStoredPoint[]
  labels?: [string, string][]
}

function clusterPointToStored(pt: ClusterThroughputHistoryPoint): ClusterChartStoredPoint {
  return {
    t: pt.t,
    avgMs: pt.avgMs,
    e2eMs: pt.e2eMs,
    byKey: Array.from(pt.byKey.entries()),
  }
}

function storedToClusterPoint(raw: ClusterChartStoredPoint): ClusterThroughputHistoryPoint | null {
  if (typeof raw.t !== 'number' || !Number.isFinite(raw.t)) {
    return null
  }
  if (!Array.isArray(raw.byKey)) {
    return null
  }
  const byKey = new Map<string, number>()
  for (const pair of raw.byKey) {
    if (!Array.isArray(pair) || pair.length !== 2) {
      continue
    }
    const [k, v] = pair
    if (typeof k !== 'string' || typeof v !== 'number' || !Number.isFinite(v)) {
      continue
    }
    byKey.set(k, v)
  }
  const avgRaw = raw.avgMs == null ? null : Number(raw.avgMs)
  const e2eRaw = raw.e2eMs == null ? null : Number(raw.e2eMs)
  return {
    t: raw.t,
    byKey,
    avgMs: avgRaw != null && Number.isFinite(avgRaw) ? avgRaw : null,
    e2eMs: e2eRaw != null && Number.isFinite(e2eRaw) ? e2eRaw : null,
  }
}

function loadClusterThroughputHistoryFromStorage(
  cutoff: number,
): { points: ClusterThroughputHistoryPoint[]; labels: [string, string][] } | null {
  try {
    const s = localStorage.getItem(CLUSTER_CHART_STORAGE_KEY)
    if (s == null || s === '') {
      return null
    }
    const parsed = JSON.parse(s) as unknown
    if (!parsed || typeof parsed !== 'object') {
      return null
    }
    const p = parsed as Partial<ClusterChartStoredPayload>
    if (p.v !== 1 || !Array.isArray(p.points)) {
      return null
    }
    const points: ClusterThroughputHistoryPoint[] = []
    for (const item of p.points) {
      const pt = storedToClusterPoint(item as ClusterChartStoredPoint)
      if (pt != null && pt.t >= cutoff) {
        points.push(pt)
      }
    }
    points.sort((a, b) => a.t - b.t)
    const labels = Array.isArray(p.labels) ? (p.labels as [string, string][]) : []
    return { points, labels }
  } catch {
    return null
  }
}

function saveClusterThroughputHistoryToStorage(
  points: ClusterThroughputHistoryPoint[],
  labels: Map<string, string>,
): void {
  try {
    const normalizedLabels: [string, string][] = []
    for (const [k, v] of labels) {
      normalizedLabels.push([k, normalizeVectorSeriesLabelForKey(k, v)])
    }
    const payload: ClusterChartStoredPayload = {
      v: 1,
      points: points.map(clusterPointToStored),
      labels: normalizedLabels,
    }
    localStorage.setItem(CLUSTER_CHART_STORAGE_KEY, JSON.stringify(payload))
  } catch {
    // QuotaExceededError, private mode, disabled storage, etc.
  }
}

export class HomePanel extends DashboardPanel {
  private readyPollTimer: number | null = null
  private readyPollGeneration = 0
  private clusterThroughputChart: Chart<'line'> | null = null
  private clusterThroughputHistory: ClusterThroughputHistoryPoint[] = []
  private clusterSeriesLabels = new Map<string, string>()
  private apiMetricsRequestsChart: Chart<'line'> | null = null
  private apiMetricsLatenciesChart: Chart<'line'> | null = null
  private apiMetricsHistory: ApiMetricsHistoryPoint[] = []

  init(api: AmgixApi): void {
    const $root = $('#panel-home [data-home-root]')
    if (!$root.length) {
      return
    }
    void this.loadHome(api, $root)
  }

  private clearReadyPoll(): void {
    if (this.readyPollTimer != null) {
      window.clearInterval(this.readyPollTimer)
      this.readyPollTimer = null
    }
    this.destroyApiMetricsChart()
    this.destroyClusterThroughputChart()
    this.clusterThroughputHistory = []
    this.clusterSeriesLabels.clear()
    this.apiMetricsHistory = []
  }

  private destroyApiMetricsChart(): void {
    this.apiMetricsRequestsChart?.destroy()
    this.apiMetricsRequestsChart = null
    this.apiMetricsLatenciesChart?.destroy()
    this.apiMetricsLatenciesChart = null
  }

  private syncSingleYAxisApiMetricsChartTheme(
    ch: Chart<'line'>,
    cutoff: number,
    now: number,
    gridColor: string,
    tickColor: string,
    chartFont: ReturnType<typeof readClusterChartFont>,
    yAxisTitle: string,
  ): void {
    const xScale = ch.options.scales?.x
    if (xScale && typeof xScale === 'object' && 'min' in xScale && 'max' in xScale) {
      ;(xScale as { min?: number; max?: number }).min = cutoff
      ;(xScale as { min?: number; max?: number }).max = now
    }
    ch.options.color = tickColor
    if (ch.options.font && typeof ch.options.font === 'object') {
      const f = ch.options.font as { family?: string; size?: number }
      f.family = chartFont.family
      f.size = chartFont.tickPx
    }
    const tip = ch.options.plugins?.tooltip
    if (tip && typeof tip === 'object') {
      const t = tip as {
        titleFont?: { family?: string; size?: number }
        bodyFont?: { family?: string; size?: number }
      }
      if (t.titleFont) {
        t.titleFont.family = chartFont.family
        t.titleFont.size = chartFont.tooltipPx
      }
      if (t.bodyFont) {
        t.bodyFont.family = chartFont.family
        t.bodyFont.size = chartFont.tooltipPx
      }
    }
    const xs = ch.options.scales?.x
    if (xs && typeof xs === 'object') {
      const gx = (xs as { grid?: { color?: string }; ticks?: { color?: string; font?: { family?: string; size?: number } } }).grid
      if (gx) {
        gx.color = gridColor
      }
      const tx = (xs as { ticks?: { color?: string; font?: { family?: string; size?: number } } }).ticks
      if (tx) {
        tx.color = tickColor
        if (tx.font) {
          tx.font.family = chartFont.family
          tx.font.size = chartFont.tickPx
        }
      }
    }
    const ys = ch.options.scales?.y
    if (ys && typeof ys === 'object') {
      const gy = (ys as { grid?: { color?: string }; ticks?: { color?: string; font?: { family?: string; size?: number } }; title?: { color?: string; text?: string; font?: { family?: string; size?: number; weight?: number } } }).grid
      if (gy) {
        gy.color = gridColor
      }
      const ty = (ys as { ticks?: { color?: string; font?: { family?: string; size?: number } } }).ticks
      if (ty) {
        ty.color = tickColor
        if (ty.font) {
          ty.font.family = chartFont.family
          ty.font.size = chartFont.tickPx
        }
      }
      const ysWithTitle = ys as {
        title?: { display?: boolean; text?: string; color?: string; font?: { family?: string; size?: number; weight?: number } }
      }
      if (!ysWithTitle.title) {
        ysWithTitle.title = {}
      }
      ysWithTitle.title.display = true
      ysWithTitle.title.text = yAxisTitle
      ysWithTitle.title.color = tickColor
      if (ysWithTitle.title.font) {
        ysWithTitle.title.font.family = chartFont.family
        ysWithTitle.title.font.size = chartFont.titlePx
        ysWithTitle.title.font.weight = 600
      } else {
        ysWithTitle.title.font = { family: chartFont.family, size: chartFont.titlePx, weight: 600 }
      }
    }
    const sc = ch.options.scales as Record<string, unknown> | undefined
    if (sc?.y1 !== undefined) {
      delete sc.y1
    }
    ch.update('none')
  }

  private destroyClusterThroughputChart(): void {
    this.clusterThroughputChart?.destroy()
    this.clusterThroughputChart = null
  }

  private refreshClusterThroughputChart($root: JQuery<HTMLElement>, view: ClusterView | null): void {
    const $canvas = $root.find('[data-home-cluster-throughput-chart]')
    const canvas = $canvas.get(0) as HTMLCanvasElement | undefined
    const $wrap = $root.find('[data-home-cluster-chart-wrap]')
    const $canvasWrap = $root.find('[data-home-cluster-chart-canvas-wrap]')
    const $legendUl = $root.find('[data-home-cluster-chart-legend]')
    if (!canvas || !$wrap.length || !$canvasWrap.length || !$legendUl.length) {
      return
    }

    const agg = aggregateClusterThroughput(view)
    const now = Date.now()
    const cutoff = now - CLUSTER_CHART_HISTORY_MS

    if (this.clusterThroughputHistory.length === 0) {
      const restored = loadClusterThroughputHistoryFromStorage(cutoff)
      if (restored != null) {
        this.clusterThroughputHistory = restored.points
        for (const [k, lab] of restored.labels) {
          if (typeof k === 'string' && typeof lab === 'string' && k !== '') {
            this.clusterSeriesLabels.set(k, normalizeVectorSeriesLabelForKey(k, lab))
          }
        }
      }
    }

    for (const row of agg.chartRows) {
      this.clusterSeriesLabels.set(row.key, row.label)
    }

    const pointMap = new Map<string, number>()
    for (const row of agg.chartRows) {
      pointMap.set(row.key, row.inferenceMeanMs)
    }
    this.clusterThroughputHistory.push({
      t: now,
      byKey: pointMap,
      avgMs: agg.globalAvgMs,
      e2eMs: agg.globalE2eMs,
    })
    while (this.clusterThroughputHistory.length > 0 && this.clusterThroughputHistory[0]!.t < cutoff) {
      this.clusterThroughputHistory.shift()
    }

    saveClusterThroughputHistoryToStorage(this.clusterThroughputHistory, this.clusterSeriesLabels)

    const keySet = new Set<string>()
    for (const pt of this.clusterThroughputHistory) {
      for (const k of pt.byKey.keys()) {
        keySet.add(k)
      }
    }
    const seriesKeys = Array.from(keySet).sort((a, b) =>
      a.localeCompare(b, undefined, { sensitivity: 'base' }),
    )

    const batchColors = clusterLineColors(Math.max(1, seriesKeys.length))
    const avgLineColor = '#64748b'
    const e2eLineColor = '#ea580c'

    const avgMsDataset = {
      label: 'Inference Latency',
      data: this.clusterThroughputHistory.map((pt) => ({
        x: pt.t,
        y: typeof pt.avgMs === 'number' && Number.isFinite(pt.avgMs) ? pt.avgMs : 0,
      })),
      borderColor: avgLineColor,
      backgroundColor: avgLineColor,
      borderWidth: 2,
      fill: false,
      tension: 0.4,
      pointRadius: 0,
      pointHoverRadius: 5,
      pointHitRadius: 12,
      spanGaps: false,
    }
    const e2eMsDataset = {
      label: 'Routed Latency',
      data: this.clusterThroughputHistory.map((pt) => ({
        x: pt.t,
        y: typeof pt.e2eMs === 'number' && Number.isFinite(pt.e2eMs) ? pt.e2eMs : 0,
      })),
      borderColor: e2eLineColor,
      backgroundColor: e2eLineColor,
      borderWidth: 2,
      fill: false,
      tension: 0.4,
      pointRadius: 0,
      pointHoverRadius: 5,
      pointHitRadius: 12,
      spanGaps: false,
    }

    const perVectorDatasets = seriesKeys.map((key, i) => {
      const typeModel = normalizeVectorSeriesLabelForKey(key, this.clusterSeriesLabels.get(key))
      return {
        label: `Inference Latency (${typeModel})`,
        data: this.clusterThroughputHistory.map((pt) => {
          const y = pt.byKey.get(key)
          return {
            x: pt.t,
            y: typeof y === 'number' && Number.isFinite(y) ? y : 0,
          }
        }),
        borderColor: batchColors[i]!,
        backgroundColor: batchColors[i]!,
        borderWidth: 2,
        fill: false,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 5,
        pointHitRadius: 12,
        spanGaps: false,
      }
    })

    const datasets = [avgMsDataset, e2eMsDataset, ...perVectorDatasets]

    $canvasWrap.css('height', '150px')

    const { grid: gridColor, tick: tickColor } = readClusterChartThemeColors()
    const chartFont = readClusterChartFont()

    const legendItems = datasets.map((ds) => ({
      color: String(ds.borderColor ?? ds.backgroundColor ?? '#888'),
      ...clusterChartLegendParts(String(ds.label ?? ''), ds.data as Array<{ y?: number | null }>),
    }))
    renderClusterChartHtmlLegend($legendUl, legendItems, tickColor)

    if (this.clusterThroughputChart == null) {
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        return
      }
      const config: ChartConfiguration<'line'> = {
        type: 'line',
        data: { datasets },
        options: {
          color: tickColor,
          devicePixelRatio: clusterChartDevicePixelRatio(),
          font: {
            family: chartFont.family,
            size: chartFont.tickPx,
          },
          parsing: false,
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'nearest', axis: 'x', intersect: false },
          plugins: {
            legend: {
              display: false,
            },
            tooltip: {
              titleFont: { family: chartFont.family, size: chartFont.tooltipPx },
              bodyFont: { family: chartFont.family, size: chartFont.tooltipPx },
              callbacks: {
                title(items) {
                  const raw = items[0]?.parsed.x
                  if (typeof raw !== 'number') {
                    return ''
                  }
                  return new Date(raw).toLocaleString()
                },
                label(ctx) {
                  const v = ctx.parsed.y
                  const lab = ctx.dataset.label ?? ''
                  if (typeof v !== 'number') {
                    return lab
                  }
                  return `${lab}: ${formatClusterAvgMsCell(v)} ms`
                },
              },
            },
          },
          scales: {
            x: {
              type: 'linear',
              min: cutoff,
              max: now,
              afterBuildTicks: (axis: Scale) => {
                axis.ticks = buildClusterChartEvenXTicks(
                  Number(axis.min),
                  Number(axis.max),
                  CLUSTER_CHART_X_TICK_COUNT,
                )
              },
              grid: { color: gridColor },
              ticks: {
                autoSkip: false,
                color: tickColor,
                font: { family: chartFont.family, size: chartFont.tickPx },
                callback: (tickValue) => {
                  const n = typeof tickValue === 'number' ? tickValue : Number(tickValue)
                  if (!Number.isFinite(n)) {
                    return ''
                  }
                  return new Date(n).toLocaleTimeString(undefined, {
                    hour: '2-digit',
                    minute: '2-digit',
                  })
                },
              },
            },
            y: {
              type: 'linear',
              position: 'left',
              beginAtZero: true,
              title: {
                display: true,
                text: 'Milliseconds',
                color: tickColor,
                font: { family: chartFont.family, size: chartFont.titlePx, weight: 600 },
              },
              ticks: {
                color: tickColor,
                font: { family: chartFont.family, size: chartFont.tickPx },
              },
              grid: { color: gridColor },
            },
          },
        },
      }
      this.clusterThroughputChart = new Chart(ctx, config)
    } else {
      const ch = this.clusterThroughputChart
      ch.data.datasets = datasets as typeof ch.data.datasets
      const xScale = ch.options.scales?.x
      if (xScale && typeof xScale === 'object' && 'min' in xScale && 'max' in xScale) {
        ;(xScale as { min?: number; max?: number }).min = cutoff
        ;(xScale as { min?: number; max?: number }).max = now
      }
      const leg = ch.options.plugins?.legend
      if (leg && typeof leg === 'object' && 'display' in leg) {
        ;(leg as { display?: boolean }).display = false
      }
      ch.options.color = tickColor
      const xs = ch.options.scales?.x
      if (xs && typeof xs === 'object') {
        const gx = (xs as { grid?: { color?: string }; ticks?: { color?: string } }).grid
        if (gx) {
          gx.color = gridColor
        }
        const tx = (xs as { ticks?: { color?: string } }).ticks
        if (tx) {
          tx.color = tickColor
        }
      }
      const ys = ch.options.scales?.y
      if (ys && typeof ys === 'object') {
        const gy = (ys as { grid?: { color?: string }; ticks?: { color?: string }; title?: { color?: string } }).grid
        if (gy) {
          gy.color = gridColor
        }
        const ty = (ys as { ticks?: { color?: string } }).ticks
        if (ty) {
          ty.color = tickColor
        }
        const ysWithTitle = ys as {
          title?: { display?: boolean; text?: string; color?: string }
        }
        if (!ysWithTitle.title) {
          ysWithTitle.title = {}
        }
        ysWithTitle.title.display = true
        ysWithTitle.title.text = 'Milliseconds'
        ysWithTitle.title.color = tickColor
      }
      const sc = ch.options.scales as Record<string, unknown> | undefined
      if (sc?.y1 !== undefined) {
        delete sc.y1
      }
      ch.update('none')
    }
  }

  private refreshApiMetricsChart($root: JQuery<HTMLElement>, view: ClusterView | null): void {
    const $reqCanvas = $root.find('[data-home-api-requests-metrics-chart]')
    const reqCanvas = $reqCanvas.get(0) as HTMLCanvasElement | undefined
    const $reqWrap = $root.find('[data-home-api-requests-chart-wrap]')
    const $reqCanvasWrap = $root.find('[data-home-api-requests-chart-canvas-wrap]')
    const $reqLegendUl = $root.find('[data-home-api-requests-chart-legend]')

    const $latCanvas = $root.find('[data-home-api-latencies-metrics-chart]')
    const latCanvas = $latCanvas.get(0) as HTMLCanvasElement | undefined
    const $latWrap = $root.find('[data-home-api-latencies-chart-wrap]')
    const $latCanvasWrap = $root.find('[data-home-api-latencies-chart-canvas-wrap]')
    const $latLegendUl = $root.find('[data-home-api-latencies-chart-legend]')

    if (
      !reqCanvas ||
      !latCanvas ||
      !$reqWrap.length ||
      !$reqCanvasWrap.length ||
      !$reqLegendUl.length ||
      !$latWrap.length ||
      !$latCanvasWrap.length ||
      !$latLegendUl.length
    ) {
      return
    }

    const agg = aggregateApiChartMetrics(view)
    const now = Date.now()
    const cutoff = now - CLUSTER_CHART_HISTORY_MS

    if (this.apiMetricsHistory.length === 0) {
      const restored = loadApiMetricsHistoryFromStorage(cutoff)
      if (restored != null && restored.length > 0) {
        this.apiMetricsHistory = restored
      }
    }

    this.apiMetricsHistory.push({
      t: now,
      allRps: agg.allRps,
      allMs: agg.allMs,
      searchRps: agg.searchRps,
      searchMs: agg.searchMs,
      ingestRps: agg.ingestRps,
      ingestMs: agg.ingestMs,
      errRps: agg.errRps,
    })
    while (this.apiMetricsHistory.length > 0 && this.apiMetricsHistory[0]!.t < cutoff) {
      this.apiMetricsHistory.shift()
    }
    saveApiMetricsHistoryToStorage(this.apiMetricsHistory)

    const yNum = (v: number | null) => (typeof v === 'number' && Number.isFinite(v) ? v : 0)

    /** Same hues for All / Search / Ingest on both requests and latency charts. */
    const apiGroupColors = ['#8b5cf6', '#38bdf8', '#f59e0b']
    const apiErrorsReqColor = '#dc2626'

    const line = (label: string, color: string, pick: (pt: ApiMetricsHistoryPoint) => number) => ({
      label,
      data: this.apiMetricsHistory.map((pt) => ({ x: pt.t, y: pick(pt) })),
      borderColor: color,
      backgroundColor: color,
      borderWidth: 2,
      fill: false,
      tension: 0.4,
      pointRadius: 0,
      pointHoverRadius: 5,
      pointHitRadius: 12,
      spanGaps: false,
    })

    const datasetsRps = [
      line('All', apiGroupColors[0]!, (pt) => yNum(pt.allRps)),
      line('Search', apiGroupColors[1]!, (pt) => yNum(pt.searchRps)),
      line('Ingest', apiGroupColors[2]!, (pt) => yNum(pt.ingestRps)),
      line('Err/s', apiErrorsReqColor, (pt) => yNum(pt.errRps)),
    ]

    const datasetsMs = [
      line('All', apiGroupColors[0]!, (pt) => yNum(pt.allMs)),
      line('Search', apiGroupColors[1]!, (pt) => yNum(pt.searchMs)),
      line('Ingest', apiGroupColors[2]!, (pt) => yNum(pt.ingestMs)),
    ]

    $reqCanvasWrap.css('height', '150px')
    $latCanvasWrap.css('height', '150px')

    const { grid: gridColor, tick: tickColor } = readClusterChartThemeColors()
    const chartFont = readClusterChartFont()

    const legendItemsRps = datasetsRps.map((ds) => ({
      color: String(ds.borderColor ?? '#888'),
      ...clusterApiChartLegendPartsRps(String(ds.label ?? ''), ds.data as Array<{ y?: number | null }>),
    }))
    const legendItemsMs = datasetsMs.map((ds) => ({
      color: String(ds.borderColor ?? '#888'),
      ...clusterChartLegendParts(String(ds.label ?? ''), ds.data as Array<{ y?: number | null }>),
    }))
    renderClusterChartHtmlLegend($reqLegendUl, legendItemsRps, tickColor)
    renderClusterChartHtmlLegend($latLegendUl, legendItemsMs, tickColor)

    if (this.apiMetricsRequestsChart == null) {
      const ctx = reqCanvas.getContext('2d')
      if (!ctx) {
        return
      }
      this.apiMetricsRequestsChart = new Chart(ctx, {
        type: 'line',
        data: { datasets: datasetsRps },
        options: buildApiMetricsChartOptions(cutoff, now, gridColor, tickColor, chartFont, 'Req/s', false),
      })
    } else {
      const ch = this.apiMetricsRequestsChart
      ch.data.datasets = datasetsRps as typeof ch.data.datasets
      this.syncSingleYAxisApiMetricsChartTheme(ch, cutoff, now, gridColor, tickColor, chartFont, 'Req/s')
    }

    if (this.apiMetricsLatenciesChart == null) {
      const ctx = latCanvas.getContext('2d')
      if (!ctx) {
        return
      }
      this.apiMetricsLatenciesChart = new Chart(ctx, {
        type: 'line',
        data: { datasets: datasetsMs },
        options: buildApiMetricsChartOptions(
          cutoff,
          now,
          gridColor,
          tickColor,
          chartFont,
          'Milliseconds',
          true,
        ),
      })
    } else {
      const ch = this.apiMetricsLatenciesChart
      ch.data.datasets = datasetsMs as typeof ch.data.datasets
      this.syncSingleYAxisApiMetricsChartTheme(ch, cutoff, now, gridColor, tickColor, chartFont, 'Milliseconds')
    }
  }

  private activateHomeMetricsTab($root: JQuery<HTMLElement>, panel: 'api' | 'encoder'): void {
    const $apiPanel = $root.find('[data-home-metrics-tab-panel="api"]')
    const $encPanel = $root.find('[data-home-metrics-tab-panel="encoder"]')
    const $btnApi = $root.find('[data-home-metrics-tab="api"]')
    const $btnEnc = $root.find('[data-home-metrics-tab="encoder"]')
    if (!$apiPanel.length || !$encPanel.length || !$btnApi.length || !$btnEnc.length) {
      return
    }
    const showApi = panel === 'api'
    $btnApi.attr('aria-selected', showApi ? 'true' : 'false')
    $btnEnc.attr('aria-selected', showApi ? 'false' : 'true')
    $btnApi.attr('tabindex', showApi ? '0' : '-1')
    $btnEnc.attr('tabindex', showApi ? '-1' : '0')
    $apiPanel.prop('hidden', !showApi)
    $encPanel.prop('hidden', showApi)
    window.requestAnimationFrame(() => {
      if (showApi) {
        this.apiMetricsRequestsChart?.resize()
        this.apiMetricsLatenciesChart?.resize()
      } else {
        this.clusterThroughputChart?.resize()
      }
    })
  }

  private applyClusterInfoToDom($root: JQuery<HTMLElement>, view: ClusterView | null): void {
    if (!$root.find('[data-home-cluster-info]').length) {
      return
    }
    const dash = '\u2014'
    const nodes = view?.nodes
    if (view == null || nodes === undefined) {
      $root.find('[data-home-cluster-info="api-nodes"]').text(dash)
      $root.find('[data-home-cluster-info="encoder-nodes"]').text(dash)
      $root.find('[data-home-cluster-info="api-rps"]').text(dash)
      $root.find('[data-home-cluster-info="api-ms"]').text(dash)
      $root.find('[data-home-cluster-info="infer-ms"]').text(dash)
      $root.find('[data-home-cluster-info="api-err-4xx"]').text(dash)
      $root.find('[data-home-cluster-info="api-err-5xx"]').text(dash)
      return
    }
    const { api: apiNodeEntries, encoders: encoderNodeEntries } = partitionApiAndEncoderNodes(nodes)
    $root.find('[data-home-cluster-info="api-nodes"]').text(String(apiNodeEntries.length))
    $root.find('[data-home-cluster-info="encoder-nodes"]').text(String(encoderNodeEntries.length))
    const apiAgg = aggregateApiChartMetrics(view)
    const $apiRps = $root.find('[data-home-cluster-info="api-rps"]')
    if (apiAgg.allRps != null && Number.isFinite(apiAgg.allRps)) {
      $apiRps.text(formatClusterRpsCell(apiAgg.allRps))
    } else {
      $apiRps.text(dash)
    }
    const $apiMs = $root.find('[data-home-cluster-info="api-ms"]')
    if (apiAgg.allMs != null && Number.isFinite(apiAgg.allMs)) {
      $apiMs.text(`${formatClusterAvgMsCell(apiAgg.allMs)} ms`)
    } else {
      $apiMs.text(dash)
    }
    const thr = aggregateClusterThroughput(view)
    const $inferMs = $root.find('[data-home-cluster-info="infer-ms"]')
    if (thr.globalAvgMs != null && Number.isFinite(thr.globalAvgMs)) {
      $inferMs.text(`${formatClusterAvgMsCell(thr.globalAvgMs)} ms/doc`)
    } else {
      $inferMs.text(dash)
    }
    const $err4 = $root.find('[data-home-cluster-info="api-err-4xx"]')
    if (apiAgg.err4xxRps != null && Number.isFinite(apiAgg.err4xxRps)) {
      $err4.text(`${formatClusterRpsCell(apiAgg.err4xxRps)} /s`)
    } else {
      $err4.text(dash)
    }
    const $err5 = $root.find('[data-home-cluster-info="api-err-5xx"]')
    if (apiAgg.err5xxRps != null && Number.isFinite(apiAgg.err5xxRps)) {
      $err5.text(`${formatClusterRpsCell(apiAgg.err5xxRps)} /s`)
    } else {
      $err5.text(dash)
    }
  }

  private applyClusterViewToDom($root: JQuery<HTMLElement>, view: ClusterView | null): void {
    try {
      this.applyClusterTablesAndChartsToDom($root, view)
    } finally {
      this.applyClusterInfoToDom($root, view)
    }
  }

  private applyClusterTablesAndChartsToDom($root: JQuery<HTMLElement>, view: ClusterView | null): void {
    const CLUSTER_API_COLSPAN = 13
    const CLUSTER_ENCODER_COLSPAN = 10
    const $apiTbody = $root.find('[data-home-api-cluster-tbody]')
    const $encTbody = $root.find('[data-home-cluster-tbody]')
    if (!$apiTbody.length && !$encTbody.length) {
      this.refreshApiMetricsChart($root, view)
      this.refreshClusterThroughputChart($root, view)
      return
    }
    $apiTbody.empty()
    $encTbody.empty()
    const nodes = view?.nodes
    if (view == null || nodes === undefined) {
      if ($apiTbody.length) {
        $apiTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_API_COLSPAN,
              text: 'Cluster view unavailable.',
            }),
          ),
        )
      }
      if ($encTbody.length) {
        $encTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_ENCODER_COLSPAN,
              text: 'Cluster view unavailable.',
            }),
          ),
        )
      }
      this.refreshApiMetricsChart($root, view)
      this.refreshClusterThroughputChart($root, view)
      return
    }
    if (Object.keys(nodes).length === 0) {
      if ($apiTbody.length) {
        $apiTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_API_COLSPAN,
              text: 'No API nodes in cluster view.',
            }),
          ),
        )
      }
      if ($encTbody.length) {
        $encTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_ENCODER_COLSPAN,
              text: 'No encoder nodes in cluster view.',
            }),
          ),
        )
      }
      this.refreshApiMetricsChart($root, view)
      this.refreshClusterThroughputChart($root, view)
      return
    }
    const { api: apiEntries, encoders: encoderEntries } = partitionApiAndEncoderNodes(nodes)
    if ($apiTbody.length) {
      if (apiEntries.length === 0) {
        $apiTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_API_COLSPAN,
              text: 'No API nodes in cluster view.',
            }),
          ),
        )
      } else {
        for (const [hostname, node] of apiEntries) {
          const apiM = formatApiMetricsCells(node)
          const $nodeTd = $('<td>', { class: 'dashboard-home-v' })
          if (node.is_leader) {
            $nodeTd.append($('<strong>', { text: `${hostname}*` }))
          } else {
            $nodeTd.text(hostname)
          }
          $apiTbody.append(
            $('<tr>').append(
              $('<td>', { class: 'dashboard-home-v', text: node.role }),
              $nodeTd,
              $('<td>', { class: 'dashboard-home-v', text: apiM.allReq }),
              $('<td>', { class: 'dashboard-home-v', text: apiM.allMs }),
              $('<td>', { class: 'dashboard-home-v', text: apiM.asyncReq }),
              $('<td>', { class: 'dashboard-home-v', text: apiM.asyncMs }),
              $('<td>', { class: 'dashboard-home-v', text: apiM.syncReq }),
              $('<td>', { class: 'dashboard-home-v', text: apiM.syncMs }),
              $('<td>', { class: 'dashboard-home-v', text: apiM.bulkReq }),
              $('<td>', { class: 'dashboard-home-v', text: apiM.bulkMs }),
              $('<td>', { class: 'dashboard-home-v', text: apiM.searchReq }),
              $('<td>', { class: 'dashboard-home-v', text: apiM.searchMs }),
              $('<td>', { class: 'dashboard-home-v', text: apiM.errPerSec }),
            ),
          )
        }
      }
    }
    if ($encTbody.length) {
      if (encoderEntries.length === 0) {
        $encTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_ENCODER_COLSPAN,
              text: 'No encoder nodes in cluster view.',
            }),
          ),
        )
      } else {
        for (const [hostname, node] of encoderEntries) {
          const m = formatEmbeddingMetricsCells(node)
          const $nodeTd = $('<td>', { class: 'dashboard-home-v' })
          if (node.is_leader) {
            $nodeTd.append($('<strong>', { text: `${hostname}*` }))
          } else {
            $nodeTd.text(hostname)
          }
          $encTbody.append(
            $('<tr>').append(
              $('<td>', { class: 'dashboard-home-v', text: node.role }),
              $nodeTd,
              $('<td>', { class: 'dashboard-home-v', text: formatLoadModelsCell(node) }),
              $('<td>', { class: 'dashboard-home-v', text: node.at_capacity ? 'Yes' : 'No' }),
              $('<td>', { class: 'dashboard-home-v', text: formatGpuStatus(node) }),
              $('<td>', { class: 'dashboard-home-v', text: m.requests }),
              $('<td>', { class: 'dashboard-home-v', text: m.rps }),
              $('<td>', { class: 'dashboard-home-v', text: m.avgMs }),
              $('<td>', { class: 'dashboard-home-v', text: m.e2eAvgMs }),
              $('<td>', { class: 'dashboard-home-v', text: m.errPerSec }),
            ),
          )
        }
      }
    }
    this.refreshApiMetricsChart($root, view)
    this.refreshClusterThroughputChart($root, view)
  }

  private applyReadinessToDom($root: JQuery<HTMLElement>, ready: ReadyResponse): void {
    const keys: HomeReadinessKey[] = ['database', 'rabbitmq', 'index', 'query']
    for (const key of keys) {
      const ok = ready[key]
      const $el = $root.find(`[data-home-ready="${key}"]`)
      if (!$el.length) {
        continue
      }
      $el.text(readinessLabel(ok))
      $el.toggleClass('dashboard-home-status--ready', ok)
      $el.toggleClass('dashboard-home-status--not-ready', !ok)
    }
  }

  private async pollHomeRefresh(api: AmgixApi, $root: JQuery<HTMLElement>, generation: number): Promise<void> {
    if (generation !== this.readyPollGeneration) {
      return
    }
    if ($('#panel-home').prop('hidden')) {
      return
    }
    let clusterView: ClusterView | null = null
    try {
      clusterView = await api.clusterView()
    } catch {
      clusterView = null
    }
    if (generation !== this.readyPollGeneration) {
      return
    }
    this.applyClusterViewToDom($root, clusterView)
    try {
      const ready = await fetchReadiness()
      if (generation !== this.readyPollGeneration) {
        return
      }
      this.applyReadinessToDom($root, ready)
    } catch {
      // Transient failures: next interval retries (same as collections stats poll).
    }
  }

  private async loadHome(api: AmgixApi, $root: JQuery<HTMLElement>): Promise<void> {
    this.clearReadyPoll()
    this.readyPollGeneration += 1
    const generation = this.readyPollGeneration

    $root.empty().append($('<p>', { class: 'dashboard-home-loading', text: 'Loading…' }))

    try {
      const [info, ready, clusterView] = await Promise.all([
        api.systemInfo(),
        fetchReadiness(),
        api.clusterView().catch((): null => null),
      ])
      if (generation !== this.readyPollGeneration) {
        return
      }
      hideDashboardError()

      const dbSummary = formatDatabaseSummary(info)
      const rmqSummary = formatRabbitmqSummary(info)

      type HomeRow = {
        key: string
        value: string | JQuery<HTMLElement>
        readiness: { ok: boolean; dataKey: HomeReadinessKey } | null
      }
      const rows: HomeRow[] = [
        { key: 'Amgix version', value: formatVersionLabel(info.amgix_version), readiness: null },
        { key: 'Database', value: dbSummary, readiness: { ok: ready.database, dataKey: 'database' } },
        { key: 'Broker', value: rmqSummary, readiness: { ok: ready.rabbitmq, dataKey: 'rabbitmq' } },
        { key: 'Index workers', value: '', readiness: { ok: ready.index, dataKey: 'index' } },
        { key: 'Query workers', value: '', readiness: { ok: ready.query, dataKey: 'query' } },
        { key: 'Collection count', value: String(info.collection_count), readiness: null },
      ]

      const $tbody = $('<tbody>')
      for (const { key, value, readiness } of rows) {
        const $tdV = $('<td>', { class: 'dashboard-home-v' })
        if (typeof value === 'string') {
          $tdV.text(value)
        } else {
          $tdV.append(value)
        }
        const $tdR = $('<td>', { class: 'dashboard-home-r' })
        if (readiness !== null) {
          $tdR.append($readinessStatus(readiness.ok, readiness.dataKey))
        }
        $tbody.append($('<tr>').append($('<td>', { class: 'dashboard-home-k', text: `${key}:` }), $tdV, $tdR))
      }

      const $thead = $('<thead>').append(
        $('<tr>').append(
          $('<th>', {
            class: 'dashboard-home-table-heading',
            colspan: 3,
            text: 'System Info',
          }),
        ),
      )

      const $systemTable = $('<table>', { class: 'dashboard-home-table' }).append($thead, $tbody)
      const $systemBlock = $('<div>', { class: 'dashboard-home-system-block' }).append($systemTable)

      const clusterInfoRow = (label: string, key: string) =>
        $('<tr>').append(
          $('<td>', { class: 'dashboard-home-k', text: `${label}:` }),
          $('<td>', {
            class: 'dashboard-home-v',
            attr: { 'data-home-cluster-info': key },
          }),
        )
      const $clusterInfoThead = $('<thead>').append(
        $('<tr>').append(
          $('<th>', {
            class: 'dashboard-home-table-heading',
            colspan: 2,
            text: 'Cluster Info',
          }),
        ),
      )
      const $clusterInfoTbody = $('<tbody>').append(
        clusterInfoRow('API Nodes', 'api-nodes'),
        clusterInfoRow('Encoder Nodes', 'encoder-nodes'),
        clusterInfoRow('Global RPS', 'api-rps'),
        clusterInfoRow('Avg Latency', 'api-ms'),
        clusterInfoRow('Avg Inference Latency', 'infer-ms'),
        clusterInfoRow('4xx Errors', 'api-err-4xx'),
        clusterInfoRow('5xx Errors', 'api-err-5xx'),
      )
      const $clusterInfoTable = $('<table>', { class: 'dashboard-home-table' }).append(
        $clusterInfoThead,
        $clusterInfoTbody,
      )
      const $clusterInfoBlock = $('<div>', { class: 'dashboard-home-cluster-info' }).append($clusterInfoTable)

      const $topBand = $('<div>', { class: 'dashboard-home-top-band' }).append($systemBlock, $clusterInfoBlock)

      const $clusterChartSection = $('<section>', { class: 'dashboard-home-cluster-metrics' }).append(
        $('<div>', {
          class: 'dashboard-home-cluster-chart-inner',
          attr: { 'data-home-cluster-chart-wrap': '' },
        }).append(
          $('<div>', {
            class:
              'dashboard-home-cluster-chart-hint-row dashboard-home-cluster-chart-hint-row--with-title',
          }).append(
            $('<span>', {
              class: 'dashboard-home-cluster-chart-hint-spacer',
              attr: { 'aria-hidden': 'true' },
            }),
            $('<span>', { class: 'dashboard-home-cluster-chart-title', text: 'Inference Latencies' }),
            clusterHelpIcon(clusterChartHelpText()),
          ),
          $('<div>', {
            class: 'dashboard-home-cluster-chart-canvas-wrap',
            attr: { 'data-home-cluster-chart-canvas-wrap': '' },
          }).append(
            $('<canvas>', {
              attr: { 'data-home-cluster-throughput-chart': '' },
              'aria-label': `Cluster inference and routed latency over time, per document, ${CLUSTER_METRICS_WINDOW_SEC}s rolling window`,
            }),
          ),
          $('<ul>', {
            class: 'dashboard-home-cluster-chart-legend',
            attr: { 'data-home-cluster-chart-legend': '' },
            'aria-label': 'Chart series',
          }),
        ),
      )

      const $apiClusterThead = $('<thead>')
        .append(
          $('<tr>').append(
            $('<th>', {
              class: 'dashboard-home-table-heading',
              colspan: 13,
              text: 'API nodes',
            }),
          ),
        )
        .append(
          $('<tr>', { class: 'dashboard-home-cluster-colhead' }).append(
            $('<th>', { text: 'Role' }),
            $('<th>', { text: 'Node' }),
            clusterThWithHelp('All/s', clusterApiAllReqColumnHelpText()),
            clusterThWithHelp('All ms', clusterApiAllMsColumnHelpText()),
            clusterThWithHelp('Async/s', clusterApiAsyncReqColumnHelpText()),
            clusterThWithHelp('Async ms', clusterApiAsyncMsColumnHelpText()),
            clusterThWithHelp('Sync/s', clusterApiSyncReqColumnHelpText()),
            clusterThWithHelp('Sync ms', clusterApiSyncMsColumnHelpText()),
            clusterThWithHelp('Bulk/s', clusterApiBulkReqColumnHelpText()),
            clusterThWithHelp('Bulk ms', clusterApiBulkMsColumnHelpText()),
            clusterThWithHelp('Srch/s', clusterApiSearchReqColumnHelpText()),
            clusterThWithHelp('Srch ms', clusterApiSearchMsColumnHelpText()),
            clusterThWithHelp('Err/s', clusterApiErrPerSecColumnHelpText()),
          ),
        )

      const $apiClusterTable = $('<table>', {
        class: 'dashboard-home-table dashboard-home-cluster-table dashboard-home-cluster-table--api',
      }).append($apiClusterThead, $('<tbody>', { attr: { 'data-home-api-cluster-tbody': '' } }))

      const $apiChartsRow = $('<div>', { class: 'dashboard-home-api-charts-row' }).append(
        $('<section>', { class: 'dashboard-home-cluster-metrics' }).append(
          $('<div>', {
            class: 'dashboard-home-cluster-chart-inner',
            attr: { 'data-home-api-requests-chart-wrap': '' },
          }).append(
            $('<div>', {
              class:
                'dashboard-home-cluster-chart-hint-row dashboard-home-cluster-chart-hint-row--with-title',
            }).append(
              $('<span>', {
                class: 'dashboard-home-cluster-chart-hint-spacer',
                attr: { 'aria-hidden': 'true' },
              }),
              $('<span>', {
                class: 'dashboard-home-cluster-chart-title',
                text: 'Cluster API Requests',
              }),
              clusterHelpIcon(clusterApiChartHelpText()),
            ),
            $('<div>', {
              class: 'dashboard-home-cluster-chart-canvas-wrap',
              attr: { 'data-home-api-requests-chart-canvas-wrap': '' },
            }).append(
              $('<canvas>', {
                attr: { 'data-home-api-requests-metrics-chart': '' },
                'aria-label': `Cluster API request rate, ${CLUSTER_METRICS_WINDOW_SEC}s rolling window`,
              }),
            ),
            $('<ul>', {
              class: 'dashboard-home-cluster-chart-legend dashboard-home-cluster-chart-legend--inline',
              attr: { 'data-home-api-requests-chart-legend': '' },
              'aria-label': 'API requests chart series',
            }),
          ),
        ),
        $('<section>', { class: 'dashboard-home-cluster-metrics' }).append(
          $('<div>', {
            class: 'dashboard-home-cluster-chart-inner',
            attr: { 'data-home-api-latencies-chart-wrap': '' },
          }).append(
            $('<div>', {
              class:
                'dashboard-home-cluster-chart-hint-row dashboard-home-cluster-chart-hint-row--with-title',
            }).append(
              $('<span>', {
                class: 'dashboard-home-cluster-chart-hint-spacer',
                attr: { 'aria-hidden': 'true' },
              }),
              $('<span>', {
                class: 'dashboard-home-cluster-chart-title',
                text: 'Cluster API Latencies',
              }),
              clusterHelpIcon(clusterApiChartHelpText()),
            ),
            $('<div>', {
              class: 'dashboard-home-cluster-chart-canvas-wrap',
              attr: { 'data-home-api-latencies-chart-canvas-wrap': '' },
            }).append(
              $('<canvas>', {
                attr: { 'data-home-api-latencies-metrics-chart': '' },
                'aria-label': `Cluster API mean latency, ${CLUSTER_METRICS_WINDOW_SEC}s rolling window`,
              }),
            ),
            $('<ul>', {
              class: 'dashboard-home-cluster-chart-legend dashboard-home-cluster-chart-legend--inline',
              attr: { 'data-home-api-latencies-chart-legend': '' },
              'aria-label': 'API latencies chart series',
            }),
          ),
        ),
      )

      const $encoderClusterThead = $('<thead>')
        .append(
          $('<tr>').append(
            $('<th>', {
              class: 'dashboard-home-table-heading',
              colspan: 10,
              text: 'Encoder nodes',
            }),
          ),
        )
        .append(
          $('<tr>', { class: 'dashboard-home-cluster-colhead' }).append(
            $('<th>', { text: 'Role' }),
            $('<th>', { text: 'Node' }),
            $('<th>', { text: 'Models' }),
            $('<th>', { text: 'Capacity' }),
            $('<th>', { text: 'GPU' }),
            clusterThWithHelp('Batches', clusterBatchesColumnHelpText()),
            clusterThWithHelp('Docs/s', clusterRateColumnHelpText()),
            clusterThWithHelp('Infer ms', clusterInferMsColumnHelpText()),
            clusterThWithHelp('Routed ms', clusterPipeMsColumnHelpText()),
            clusterThWithHelp('Err/s', clusterErrPerSecColumnHelpText()),
          ),
        )

      const $encoderClusterTable = $('<table>', {
        class: 'dashboard-home-table dashboard-home-cluster-table dashboard-home-cluster-table--encoders',
      }).append($encoderClusterThead, $('<tbody>', { attr: { 'data-home-cluster-tbody': '' } }))

      const $btnApiMetrics = $('<button>', {
        type: 'button',
        role: 'tab',
        class: 'dashboard-home-metrics-tab',
        id: HOME_METRICS_TAB_API_ID,
        text: 'API Metrics',
        attr: {
          'aria-selected': 'true',
          'aria-controls': HOME_METRICS_PANEL_API_ID,
          'data-home-metrics-tab': 'api',
          tabindex: '0',
        },
      })
      const $btnEncoderMetrics = $('<button>', {
        type: 'button',
        role: 'tab',
        class: 'dashboard-home-metrics-tab',
        id: HOME_METRICS_TAB_ENCODER_ID,
        text: 'Encoder Metrics',
        attr: {
          'aria-selected': 'false',
          'aria-controls': HOME_METRICS_PANEL_ENCODER_ID,
          'data-home-metrics-tab': 'encoder',
          tabindex: '-1',
        },
      })
      const $metricsTabList = $('<div>', {
        role: 'tablist',
        class: 'dashboard-home-metrics-tablist',
        'aria-label': 'Cluster metrics',
      }).append($btnApiMetrics, $btnEncoderMetrics)

      const $apiMetricsPanel = $('<div>', {
        class: 'dashboard-home-cluster-tables dashboard-home-metrics-tab-panel',
        id: HOME_METRICS_PANEL_API_ID,
        role: 'tabpanel',
        attr: {
          'data-home-metrics-tab-panel': 'api',
          'aria-labelledby': HOME_METRICS_TAB_API_ID,
        },
      }).append($apiChartsRow, $apiClusterTable)

      const $encoderMetricsPanel = $('<div>', {
        class: 'dashboard-home-cluster-tables dashboard-home-metrics-tab-panel',
        id: HOME_METRICS_PANEL_ENCODER_ID,
        role: 'tabpanel',
        attr: {
          'data-home-metrics-tab-panel': 'encoder',
          'aria-labelledby': HOME_METRICS_TAB_ENCODER_ID,
        },
      }).append($clusterChartSection, $encoderClusterTable)

      $encoderMetricsPanel.prop('hidden', true)

      const $metricsShell = $('<div>', { class: 'dashboard-home-metrics-shell' }).append(
        $metricsTabList,
        $apiMetricsPanel,
        $encoderMetricsPanel,
      )

      $btnApiMetrics.on('click', () => {
        this.activateHomeMetricsTab($root, 'api')
      })
      $btnEncoderMetrics.on('click', () => {
        this.activateHomeMetricsTab($root, 'encoder')
      })
      $metricsTabList.on('keydown', (e) => {
        const key = e.key
        if (key !== 'ArrowRight' && key !== 'ArrowLeft' && key !== 'Home' && key !== 'End') {
          return
        }
        e.preventDefault()
        const current = $btnApiMetrics.attr('aria-selected') === 'true' ? 'api' : 'encoder'
        let next: 'api' | 'encoder'
        if (key === 'Home') {
          next = 'api'
        } else if (key === 'End') {
          next = 'encoder'
        } else if (key === 'ArrowRight') {
          next = current === 'api' ? 'encoder' : 'api'
        } else {
          next = current === 'encoder' ? 'api' : 'encoder'
        }
        this.activateHomeMetricsTab($root, next)
        $root.find(`[data-home-metrics-tab="${next}"]`).get(0)?.focus()
      })

      $root.empty().append($topBand, $metricsShell)
      this.applyClusterViewToDom($root, clusterView)

      this.readyPollTimer = window.setInterval(() => {
        void this.pollHomeRefresh(api, $root, generation)
      }, HOME_READY_POLL_MS)
    } catch (err) {
      if (generation !== this.readyPollGeneration) {
        return
      }
      $root.empty()
      showDashboardError(formatRequestError('Could not load home.', err))
    }
  }
}

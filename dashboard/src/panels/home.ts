import {
  type AmgixApi,
  type MetricTrend,
  type Metrics,
  type MetricsBucket,
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
import zoomPlugin from 'chartjs-plugin-zoom'
import type { ZoomPluginOptions } from 'chartjs-plugin-zoom/types/options'
import $ from 'jquery'

import { hideDashboardError, showDashboardError } from '../error-bar'
import { formatDashboardRouteHash, parseDashboardRouteHash, type HomeMetricsTabId } from '../route-hash'
import { formatRequestError, stripModelNamespaceForDisplay } from './common'
import { DashboardPanel } from './panel-base'

Chart.register(LineController, LineElement, PointElement, CategoryScale, LinearScale, Tooltip, Legend, zoomPlugin)

const HOME_READY_POLL_MS = 10_000

const HOME_METRICS_TAB_API_ID = 'dashboard-home-metrics-tab-api'
const HOME_METRICS_TAB_INDEXING_ID = 'dashboard-home-metrics-tab-indexing'
const HOME_METRICS_TAB_ENCODER_ID = 'dashboard-home-metrics-tab-encoder'
const HOME_METRICS_PANEL_API_ID = 'dashboard-home-metrics-panel-api'
const HOME_METRICS_PANEL_INDEXING_ID = 'dashboard-home-metrics-panel-indexing'
const HOME_METRICS_PANEL_ENCODER_ID = 'dashboard-home-metrics-panel-encoder'

const CLUSTER_METRICS_WINDOW_SEC = 60

const DEFAULT_HOME_CHART_HISTORY_MS = 10 * 60 * 1000

const HOME_CHART_HISTORY_ONE_DAY_MS = 24 * 60 * 60 * 1000
const HOME_CHART_HISTORY_SEVEN_DAYS_MS = 7 * HOME_CHART_HISTORY_ONE_DAY_MS

const HOME_CHART_HISTORY_OPTIONS: ReadonlyArray<{ label: string; valueMs: number }> = [
  { label: '10 m', valueMs: 10 * 60 * 1000 },
  { label: '30 m', valueMs: 30 * 60 * 1000 },
  { label: '1 h', valueMs: 60 * 60 * 1000 },
  { label: '6 h', valueMs: 6 * 60 * 60 * 1000 },
  { label: '12 h', valueMs: 12 * 60 * 60 * 1000 },
  { label: '1 d', valueMs: HOME_CHART_HISTORY_ONE_DAY_MS },
  { label: '2 d', valueMs: 2 * HOME_CHART_HISTORY_ONE_DAY_MS },
  { label: '3 d', valueMs: 3 * HOME_CHART_HISTORY_ONE_DAY_MS },
  { label: '7 d', valueMs: HOME_CHART_HISTORY_SEVEN_DAYS_MS },
]

const HOME_CHART_METRICS_LIVE_WINDOW_SEC = 60 as const

function homeChartTrendResolutionSec(historyMs: number): 60 | 300 {
  return historyMs <= HOME_CHART_HISTORY_ONE_DAY_MS ? 60 : 300
}

function homeChartTrendPatchMs(historyMs: number, resolutionSec: 60 | 300): number {
  if (resolutionSec >= 300) {
    return Math.min(historyMs, 20 * 60 * 1000)
  }
  return Math.min(historyMs, 2 * 60 * 1000)
}

function formatClusterChartXAxisTickLabel(valueMs: number, spanMs: number): string {
  if (!Number.isFinite(valueMs) || !Number.isFinite(spanMs) || spanMs <= 0) {
    return ''
  }
  const d = new Date(valueMs)
  if (spanMs <= HOME_CHART_HISTORY_ONE_DAY_MS) {
    return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
  }
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

const API_CHART_TREND_KEYS = [
  'api_requests',
  'api_request_ms',
  'api_async_upload',
  'api_async_upload_ms',
  'api_sync_upload',
  'api_sync_upload_ms',
  'api_bulk_upload',
  'api_bulk_upload_ms',
  'api_search',
  'api_search_ms',
  'api_error_4xx',
  'api_error_5xx',
] as const

const ENCODER_CHART_TREND_KEYS = [
  'embed_inference_ms',
  'embed_passages',
  'embed_inference_origin_ms',
  'embed_passages_origin',
] as const

const INDEXING_CHART_TREND_KEYS = ['index_queue_job_ms', 'index_bulk_job_ms'] as const

const HOME_INDEXING_TABLE_EXTRA_KEYS = [
  'index_queue_docs_skipped_stale',
  'index_queue_docs_new',
  'index_queue_docs_updated',
  'index_queue_failed',
  'index_queue_requeued',
  'index_bulk_batches',
  'index_bulk_batch_size',
  'index_bulk_failed',
  'index_bulk_requeued',
] as const

const HOME_CLUSTER_INFO_API_KEYS = [
  'api_requests',
  'api_request_ms',
  'api_error_4xx',
  'api_error_5xx',
] as const

const HOME_ENCODER_TABLE_EXTRA_KEYS = [
  'embed_batches_origin',
  'embed_passages_origin',
  'embed_inference_origin_errors',
] as const

function uniqStrings(xs: readonly string[]): string[] {
  return [...new Set(xs)]
}

function homeEncoderTableMetricKeys(): string[] {
  return uniqStrings([...HOME_ENCODER_TABLE_EXTRA_KEYS, ...ENCODER_CHART_TREND_KEYS])
}

function homeIndexingTableMetricKeys(): string[] {
  return uniqStrings([...HOME_INDEXING_TABLE_EXTRA_KEYS, ...INDEXING_CHART_TREND_KEYS])
}

function homeMetricsCurrentKeys(tab: HomeMetricsTabId): string[] {
  const apiOverview = [...HOME_CLUSTER_INFO_API_KEYS]
  const encOverview = [...ENCODER_CHART_TREND_KEYS]
  if (tab === 'api') {
    return uniqStrings([...API_CHART_TREND_KEYS, ...encOverview, ...INDEXING_CHART_TREND_KEYS])
  }
  if (tab === 'encoder') {
    return uniqStrings([...homeEncoderTableMetricKeys(), ...apiOverview, ...INDEXING_CHART_TREND_KEYS])
  }
  if (tab === 'indexing') {
    return uniqStrings([...homeIndexingTableMetricKeys(), ...apiOverview, ...encOverview])
  }
  return uniqStrings([...apiOverview, ...encOverview])
}

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

const HOME_CHART_ZOOM_RESET_TARGETS = ['encoder', 'indexing', 'api-requests', 'api-latencies'] as const
type HomeChartZoomResetTarget = (typeof HOME_CHART_ZOOM_RESET_TARGETS)[number]

function isHomeChartZoomResetTarget(v: string): v is HomeChartZoomResetTarget {
  return (HOME_CHART_ZOOM_RESET_TARGETS as readonly string[]).includes(v)
}

function homeChartZoomResetButton(target: HomeChartZoomResetTarget): JQuery<HTMLElement> {
  return $('<button>', {
    type: 'button',
    class: 'dashboard-home-cluster-chart-zoom-reset',
    attr: { 'data-home-chart-zoom-reset': target },
    title: 'Reset chart zoom',
    'aria-label': 'Reset chart zoom',
  }).append(
    $('<span>', {
      class: 'material-symbols-outlined',
      text: 'restart_alt',
      'aria-hidden': 'true',
    }),
  )
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
  return `Number of embed requests that originated in the last ${w}s.`
}

function clusterRateColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Passages embedded per second from originating requests over the last ${w}s.`
}

function clusterInferMsColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Mean local model inference time per passage, in ms, over the last ${w}s.`
}

function clusterPipeMsColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Mean routed inference time per passage, in ms, as seen by the originating encoder (includes RPC if the request was forwarded to another node), over the last ${w}s.`
}

function clusterErrPerSecColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Failed embed requests per second from originating requests over the last ${w}s.`
}

function clusterByModelBatchesHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Embed batches that originated on any encoder node in the last ${w}s, summed by vector type and model.`
}

function clusterByModelRateHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Passages per second from originating embed traffic in the last ${w}s, summed by type and model across encoder nodes.`
}

function clusterByModelInferMsHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Mean local inference ms per passage in the last ${w}s, weighted across encoder nodes for each type and model.`
}

function clusterByModelRoutedMsHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Mean routed (origin) ms per passage in the last ${w}s, weighted across encoder nodes for each type and model.`
}

function clusterByModelErrHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Failed embed requests per second in the last ${w}s, summed by type and model across encoder nodes.`
}

function clusterChartHelpText(): string {
  return 'Inference latency is model time per passage on the node that ran the model. Routed latency is end-to-end time per passage on the node that received the request (includes RPC when work is forwarded to another encoder).'
}

function indexingClusterChartHelpText(): string {
  return 'Single-doc: mean time per completed indexing item. Bulk: mean time per completed bulk batch. Values are cluster-wide means (milliseconds).'
}

function clusterIndexingDocsPerSecHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `New and updated documents indexed per second (single-doc path and bulk batches), over the last ${w}s.`
}

function clusterIndexingStalePerSecHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Documents per second skipped because they were older than what is already indexed, over the last ${w}s.`
}

function clusterIndexingQueueFailHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Failures per second for single-document indexing, over the last ${w}s.`
}

function clusterIndexingQueueRequeueHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Requeues per second for single-document indexing, over the last ${w}s.`
}

function clusterIndexingQueueJobMsHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Mean time in ms per completed single-document indexing item, over the last ${w}s.`
}

function clusterIndexingBulkBatchesHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Bulk indexing batches completed per second, over the last ${w}s.`
}

function clusterIndexingBulkBatchSizeHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Mean documents per completed bulk batch, over the last ${w}s.`
}

function clusterIndexingBulkJobMsHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Mean time in ms per completed bulk indexing batch, over the last ${w}s.`
}

function clusterIndexingBulkFailHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Bulk indexing batches marked failed, per second, over the last ${w}s.`
}

function clusterIndexingBulkRequeueHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Bulk indexing batches requeued for retry, per second, over the last ${w}s.`
}

function clusterApiRequestsChartHelpText(): string {
  return 'Requests per second: total traffic, search, document uploads (async, sync, and bulk combined), and HTTP errors (4xx and 5xx).'
}

function clusterApiLatenciesChartHelpText(): string {
  return 'Mean HTTP request latency in milliseconds per series: all requests, search, and document uploads (async, sync, and bulk combined).'
}

function clusterApiAllReqColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `All HTTP requests per second, over the last ${w}s.`
}

function clusterApiAllMsColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Mean latency in ms for all HTTP requests, over the last ${w}s.`
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
  return `HTTP 4xx and 5xx responses per second, combined, over the last ${w}s.`
}

type HomeReadinessKey = 'database' | 'rabbitmq' | 'index' | 'query'

async function fetchReadiness(): Promise<ReadyResponse> {
  const res = await fetch('/v1/health/ready')
  if (!res.ok) {
    throw new Error(`Readiness request failed (HTTP ${res.status})`)
  }
  return (await res.json()) as ReadyResponse
}

function formatVersionLabel(raw: string): string {
  const t = raw.trim()
  return t.startsWith('v') ? t : `v${t}`
}

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

function nodeMeta(node: NodeView): Record<string, unknown> {
  const meta = (node as NodeView & { meta?: Record<string, unknown> }).meta
  return meta != null && typeof meta === 'object' ? (meta as Record<string, unknown>) : {}
}

function nodeMetaBool(node: NodeView, key: string, fallback: boolean = false): boolean {
  const value = nodeMeta(node)[key]
  return typeof value === 'boolean' ? value : fallback
}

function nodeMetaLoadedModelDisplayStrings(node: NodeView): string[] {
  const value = nodeMeta(node)['loaded_models']
  if (!Array.isArray(value)) {
    return []
  }
  const out: string[] = []
  for (const raw of value) {
    if (raw == null || typeof raw !== 'object') {
      continue
    }
    const o = raw as { label?: unknown; model_key?: unknown }
    if (typeof o.label === 'string') {
      const t = o.label.trim()
      if (t !== '') {
        out.push(t)
        continue
      }
    }
    if (Array.isArray(o.model_key)) {
      const parts = o.model_key.map((x) => String(x).trim()).filter((s) => s !== '')
      const t = parts.join(' · ')
      if (t !== '') {
        out.push(t)
      }
    }
  }
  return out
}

function openHomeLoadedModelsDialog(hostname: string, models: string[]): void {
  const dialogId = 'home-loaded-models-dialog'
  $(`#${dialogId}`).remove()

  const titleId = 'home-loaded-models-dialog-title'
  const $dialog = $('<dialog>', {
    id: dialogId,
    class: 'dashboard-collections-config-dialog',
    'aria-labelledby': titleId,
  })

  const $close = $('<button>', {
    type: 'button',
    class: 'dashboard-collections-config-dialog-close',
    text: 'Close',
  })
  $close.on('click', () => {
    ;($dialog.get(0) as HTMLDialogElement | undefined)?.close()
  })
  $dialog.on('close', () => {
    $dialog.remove()
  })

  const $list = $('<ul>', { class: 'dashboard-home-loaded-models-dialog-list' })
  for (const m of models) {
    $list.append($('<li>', { text: m }))
  }

  $dialog.append(
    $('<h4>', {
      id: titleId,
      class: 'dashboard-collections-config-dialog-title',
      text: `Loaded models — ${hostname}`,
    }),
    $('<div>', { class: 'dashboard-home-loaded-models-dialog-body' }).append($list),
    $('<div>', { class: 'dashboard-collections-config-dialog-actions' }).append($close),
  )

  $('body').append($dialog)
  ;($dialog.get(0) as HTMLDialogElement).showModal()
}

function buildEncoderModelsCell(hostname: string, node: NodeView): JQuery<HTMLElement> {
  const $td = $('<td>', { class: 'dashboard-home-v' })
  if (!nodeMetaBool(node, 'load_models')) {
    $td.text('No')
    return $td
  }
  const labels = nodeMetaLoadedModelDisplayStrings(node)
  const n = labels.length
  if (n === 0) {
    $td.text('Yes (0)')
    return $td
  }
  const label =
    n === 1 ? `Show 1 loaded model on ${hostname}` : `Show ${n} loaded models on ${hostname}`
  $td.append(
    document.createTextNode('Yes ('),
    $('<a>', {
      href: '#',
      class: 'dashboard-home-loaded-models-count',
      text: String(n),
      attr: { 'aria-label': label },
    }).on('click', (e) => {
      e.preventDefault()
      openHomeLoadedModelsDialog(hostname, labels)
    }),
    document.createTextNode(')'),
  )
  return $td
}

function formatAtCapacityCell(node: NodeView): string {
  if (!nodeMetaBool(node, 'load_models')) {
    return ''
  }
  return nodeMetaBool(node, 'at_capacity') ? 'Yes' : 'No'
}

const LEGACY_CHART_TYPE_MODEL_PREFIX = /^(dense_model|sparse_model):\s*(.+)$/i

const EMBED_KEYS_FOR_BY_MODEL_TABLE = new Set<string>([
  'embed_batches_origin',
  'embed_passages_origin',
  'embed_inference_ms',
  'embed_passages',
  'embed_inference_origin_ms',
  'embed_inference_origin_errors',
])

function displayEmbeddingModelForTable(raw: string): string {
  let t = raw.trim()
  const legacy = LEGACY_CHART_TYPE_MODEL_PREFIX.exec(t)
  if (legacy) {
    t = legacy[2]!.trim()
  }
  return stripModelNamespaceForDisplay(t)
}

function groupingKeyForEmbeddingByModelSeries(s: NodeMetricSeries): string | null {
  if (!EMBED_KEYS_FOR_BY_MODEL_TABLE.has(s.key)) {
    return null
  }
  const d = s.dims ?? []
  const rawType = (d[0] ?? '').trim()
  if (rawType === '') {
    return null
  }
  const modelDisp = displayEmbeddingModelForTable(d[1] ?? '')
  return `${rawType}\0${modelDisp}`
}

function rowTypeModelFromEmbeddingGroupKey(key: string): { type: string; model: string } {
  const i = key.indexOf('\0')
  if (i < 0) {
    return { type: key, model: '' }
  }
  return { type: key.slice(0, i), model: key.slice(i + 1) }
}

function formatVectorMetricsTypeDisplay(rawType: string): string {
  if (rawType === 'dense_model') {
    return 'dense'
  }
  if (rawType === 'sparse_model') {
    return 'sparse'
  }
  if (rawType === 'dense_custom') {
    return 'dense (custom)'
  }
  if (rawType === 'sparse_custom') {
    return 'sparse (custom)'
  }
  return rawType
}

type EmbeddingByModelAgg = {
  totalBatches: number
  sumPassagesOrigin: number
  totalInferenceMs: number
  totalPassages: number
  totalOriginMs: number
  totalOriginPassages: number
  sumErrors: number
  sawAny: boolean
}

function aggregateEmbeddingMetricsByTypeModel(view: Metrics | null): Map<string, EmbeddingByModelAgg> {
  const out = new Map<string, EmbeddingByModelAgg>()
  if (view?.nodes == null) {
    return out
  }
  const bump = (gk: string): EmbeddingByModelAgg => {
    let a = out.get(gk)
    if (a == null) {
      a = {
        totalBatches: 0,
        sumPassagesOrigin: 0,
        totalInferenceMs: 0,
        totalPassages: 0,
        totalOriginMs: 0,
        totalOriginPassages: 0,
        sumErrors: 0,
        sawAny: false,
      }
      out.set(gk, a)
    }
    return a
  }
  for (const node of Object.values(view.nodes)) {
    if (node.role === 'api') {
      continue
    }
    for (const s of node.metrics ?? []) {
      const gk = groupingKeyForEmbeddingByModelSeries(s)
      if (gk == null) {
        continue
      }
      const ws = getNodeWindowSample(s, CLUSTER_METRICS_WINDOW_SEC)
      if (ws == null) {
        continue
      }
      const v = Number(ws.value)
      const name = s.key
      const a = bump(gk)
      if (name === 'embed_batches_origin') {
        a.sawAny = true
        if (Number.isFinite(v)) {
          a.totalBatches += Math.trunc(v)
        }
      } else if (name === 'embed_passages_origin') {
        a.sawAny = true
        if (Number.isFinite(v)) {
          a.sumPassagesOrigin += v
          a.totalOriginPassages += v
        }
      } else if (name === 'embed_inference_ms') {
        if (Number.isFinite(v)) {
          a.sawAny = true
          a.totalInferenceMs += v
        }
      } else if (name === 'embed_passages') {
        if (Number.isFinite(v)) {
          a.totalPassages += v
        }
      } else if (name === 'embed_inference_origin_ms') {
        if (Number.isFinite(v)) {
          a.sawAny = true
          a.totalOriginMs += v
        }
      } else if (name === 'embed_inference_origin_errors') {
        a.sawAny = true
        if (Number.isFinite(v)) {
          a.sumErrors += v
        }
      }
    }
  }
  return out
}

function isDenseOrSparseVectorType(t: string): boolean {
  const s = t.toLowerCase()
  return s.includes('dense') || s.includes('sparse')
}

function compareVectorMetricsRows(
  a: { typeRaw: string; model: string },
  b: { typeRaw: string; model: string },
): number {
  const aNo = a.model.trim() === ''
  const bNo = b.model.trim() === ''
  if (aNo !== bNo) {
    return aNo ? -1 : 1
  }
  if (aNo) {
    return a.typeRaw.localeCompare(b.typeRaw, undefined, { sensitivity: 'base' })
  }
  const aDs = isDenseOrSparseVectorType(a.typeRaw)
  const bDs = isDenseOrSparseVectorType(b.typeRaw)
  if (aDs && bDs) {
    const byModel = a.model.localeCompare(b.model, undefined, { sensitivity: 'base' })
    if (byModel !== 0) {
      return byModel
    }
    return a.typeRaw.localeCompare(b.typeRaw, undefined, { sensitivity: 'base' })
  }
  if (aDs !== bDs) {
    return aDs ? -1 : 1
  }
  const byType = a.typeRaw.localeCompare(b.typeRaw, undefined, { sensitivity: 'base' })
  if (byType !== 0) {
    return byType
  }
  return a.model.localeCompare(b.model, undefined, { sensitivity: 'base' })
}

function formatEmbeddingByModelTableRows(
  view: Metrics | null,
): Array<{
  type: string
  typeRaw: string
  model: string
  requests: string
  rps: string
  avgMs: string
  e2eAvgMs: string
  errPerSec: string
}> {
  const m = aggregateEmbeddingMetricsByTypeModel(view)
  const w = CLUSTER_METRICS_WINDOW_SEC
  const rows = [...m.entries()]
    .filter(([, a]) => a.sawAny)
    .map(([gk, a]) => {
      const { type: typeRaw, model } = rowTypeModelFromEmbeddingGroupKey(gk)
      const errPerSec = a.sumErrors / w
      return {
        type: formatVectorMetricsTypeDisplay(typeRaw),
        typeRaw,
        model,
        requests: String(a.totalBatches),
        rps: formatClusterRpsCell(a.sumPassagesOrigin / w),
        avgMs: a.totalPassages > 0 ? formatClusterAvgMsCell(a.totalInferenceMs / a.totalPassages) : '',
        e2eAvgMs: a.totalOriginPassages > 0 ? formatClusterAvgMsCell(a.totalOriginMs / a.totalOriginPassages) : '',
        errPerSec: formatClusterRpsCell(errPerSec),
      }
    })
  rows.sort(compareVectorMetricsRows)
  return rows
}

function formatGpuStatus(node: NodeView): string {
  if (!nodeMetaBool(node, 'gpu_support')) {
    return 'n/a'
  }
  if (nodeMetaBool(node, 'gpu_available')) {
    return 'Yes'
  }
  return 'Undetected'
}

function getNodeWindowSample(s: NodeMetricSeries, windowSec: number): WindowSample | undefined {
  const w = s.windows
  if (w == null) {
    return undefined
  }
  return w[String(windowSec)]
}

function getNodeMetricWindowSample(s: NodeMetricSeries): WindowSample | undefined {
  return getNodeWindowSample(s, CLUSTER_METRICS_WINDOW_SEC)
}

function getMetricWindowSampleByFirstKey(
  list: NodeMetricSeries[],
  name: string,
  windowSec: number = CLUSTER_METRICS_WINDOW_SEC,
): WindowSample | undefined {
  for (const s of list) {
    if (s.key === name) {
      return getNodeWindowSample(s, windowSec)
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
  return formatClusterRpsCell(v / CLUSTER_METRICS_WINDOW_SEC)
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
  return formatClusterAvgMsCell(v / n)
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
  return formatClusterRpsCell(sum / CLUSTER_METRICS_WINDOW_SEC)
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

function seriesDimKey(s: NodeMetricSeries): string {
  const d = s.dims ?? []
  return `${d[0] ?? ''}\0${d[1] ?? ''}\0${d[2] ?? ''}`
}

function seriesDisplayLabel(s: NodeMetricSeries): string {
  const d = s.dims ?? []
  const model = (d[1] ?? '').trim()
  const revision = (d[2] ?? '').trim()
  if (model !== '') {
    return revision !== '' ? `${model} (${revision})` : model
  }
  return d[0] ?? ''
}

function formatClusterRpsCell(rps: number): string {
  if (!Number.isFinite(rps)) {
    return ''
  }
  return new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(rps)
}

function formatClusterAvgMsCell(ms: number): string {
  if (!Number.isFinite(ms)) {
    return ''
  }
  return new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(ms)
}

function clusterChartLegendMsValue(baseLabel: string, y: number | null): {
  labelStem: string
  valuePart: string | null
} {
  return {
    labelStem: `${baseLabel}: `,
    valuePart:
      typeof y === 'number' && Number.isFinite(y) ? `${formatClusterAvgMsCell(y)} ms` : '- ms',
  }
}

function clusterChartLegendRpsValue(baseLabel: string, y: number | null): {
  labelStem: string
  valuePart: string | null
} {
  return {
    labelStem: `${baseLabel}: `,
    valuePart:
      typeof y === 'number' && Number.isFinite(y) ? `${formatClusterRpsCell(y)} /s` : '- /s',
  }
}

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
  let sumPassagesRps = 0
  let totalInferenceMs = 0
  let totalPassages = 0
  let totalOriginMs = 0
  let totalOriginPassages = 0
  let sumErrorsInWindow = 0
  for (const s of list) {
    const name = s.key
    const ws = getNodeMetricWindowSample(s)
    if (ws == null) {
      continue
    }
    const v = Number(ws.value)
    if (name === 'embed_batches_origin') {
      sawAny = true
      if (Number.isFinite(v)) {
        totalBatches += Math.trunc(v)
      }
    } else if (name === 'embed_passages_origin') {
      sawAny = true
      if (Number.isFinite(v)) {
        sumPassagesRps += v
        totalOriginPassages += v
      }
    } else if (name === 'embed_inference_ms') {
      if (Number.isFinite(v)) {
        sawAny = true
        totalInferenceMs += v
      }
    } else if (name === 'embed_passages') {
      if (Number.isFinite(v)) {
        totalPassages += v
      }
    } else if (name === 'embed_inference_origin_ms') {
      if (Number.isFinite(v)) {
        sawAny = true
        totalOriginMs += v
      }
    } else if (name === 'embed_inference_origin_errors') {
      sawAny = true
      if (Number.isFinite(v)) {
        sumErrorsInWindow += v
      }
    }
  }
  if (!sawAny) {
    return { rps: '', avgMs: '', e2eAvgMs: '', requests: '', errPerSec: '' }
  }
  const errPerSec = sumErrorsInWindow / CLUSTER_METRICS_WINDOW_SEC
  return {
    rps: formatClusterRpsCell(sumPassagesRps / CLUSTER_METRICS_WINDOW_SEC),
    avgMs: totalPassages > 0 ? formatClusterAvgMsCell(totalInferenceMs / totalPassages) : '',
    e2eAvgMs: totalOriginPassages > 0 ? formatClusterAvgMsCell(totalOriginMs / totalOriginPassages) : '',
    requests: String(totalBatches),
    errPerSec: formatClusterRpsCell(errPerSec),
  }
}

function formatIndexingSumRateCell(list: NodeMetricSeries[], keys: readonly string[]): string {
  let sum = 0
  let saw = false
  for (const name of keys) {
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
  return formatClusterRpsCell(sum / CLUSTER_METRICS_WINDOW_SEC)
}

function formatIndexingSingleSumRateCell(list: NodeMetricSeries[], key: string): string {
  return formatIndexingSumRateCell(list, [key])
}

function formatIndexingMetricsCells(node: NodeView): {
  docsPerSec: string
  stalePerSec: string
  queueFailPerSec: string
  queueRequeuePerSec: string
  queueJobMs: string
  bulkBatchesPerSec: string
  bulkBatchSize: string
  bulkJobMs: string
  bulkFailPerSec: string
  bulkRequeuePerSec: string
} {
  const list = node.metrics ?? []
  const docsPerSec = formatIndexingSumRateCell(list, ['index_queue_docs_new', 'index_queue_docs_updated'])
  const stalePerSec = formatIndexingSingleSumRateCell(list, 'index_queue_docs_skipped_stale')
  const queueFailPerSec = formatIndexingSingleSumRateCell(list, 'index_queue_failed')
  const queueRequeuePerSec = formatIndexingSingleSumRateCell(list, 'index_queue_requeued')
  const queueJobMs = formatApiAvgMsCell(list, 'index_queue_job_ms')
  const bulkBatchesPerSec = formatIndexingSingleSumRateCell(list, 'index_bulk_batches')
  const bulkBatchSize = formatApiAvgMsCell(list, 'index_bulk_batch_size')
  const bulkJobMs = formatApiAvgMsCell(list, 'index_bulk_job_ms')
  const bulkFailPerSec = formatIndexingSingleSumRateCell(list, 'index_bulk_failed')
  const bulkRequeuePerSec = formatIndexingSingleSumRateCell(list, 'index_bulk_requeued')
  const empty =
    docsPerSec === '' &&
    stalePerSec === '' &&
    queueFailPerSec === '' &&
    queueRequeuePerSec === '' &&
    queueJobMs === '' &&
    bulkBatchesPerSec === '' &&
    bulkBatchSize === '' &&
    bulkJobMs === '' &&
    bulkFailPerSec === '' &&
    bulkRequeuePerSec === ''
  if (empty) {
    return {
      docsPerSec: '',
      stalePerSec: '',
      queueFailPerSec: '',
      queueRequeuePerSec: '',
      queueJobMs: '',
      bulkBatchesPerSec: '',
      bulkBatchSize: '',
      bulkJobMs: '',
      bulkFailPerSec: '',
      bulkRequeuePerSec: '',
    }
  }
  return {
    docsPerSec,
    stalePerSec,
    queueFailPerSec,
    queueRequeuePerSec,
    queueJobMs,
    bulkBatchesPerSec,
    bulkBatchSize,
    bulkJobMs,
    bulkFailPerSec,
    bulkRequeuePerSec,
  }
}

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

function aggregateClusterThroughput(
  view: Metrics | null,
  windowSec: number = CLUSTER_METRICS_WINDOW_SEC,
): ClusterThroughputAgg {
  const empty: ClusterThroughputAgg = {
    chartRows: [],
    globalAvgMs: null,
    globalE2eMs: null,
  }
  if (view == null || view.nodes == null) {
    return empty
  }

  const byKey = new Map<string, { label: string; totalMs: number; totalPassages: number }>()
  let globalInferenceMs = 0
  let globalPassages = 0
  let globalOriginMs = 0
  let globalOriginPassages = 0

  for (const node of Object.values(view.nodes)) {
    const msByDim = new Map<string, { label: string; ms: number }>()
    const passagesByDim = new Map<string, number>()
    for (const s of node.metrics ?? []) {
      const wm = getNodeWindowSample(s, windowSec)
      if (wm == null) {
        continue
      }
      const v = Number(wm.value)
      if (!Number.isFinite(v)) {
        continue
      }
      const dimKey = seriesDimKey(s)
      if (s.key === 'embed_inference_ms') {
        const existing = msByDim.get(dimKey)
        msByDim.set(dimKey, { label: seriesDisplayLabel(s), ms: (existing?.ms ?? 0) + v })
        globalInferenceMs += v
      } else if (s.key === 'embed_passages') {
        passagesByDim.set(dimKey, (passagesByDim.get(dimKey) ?? 0) + v)
        globalPassages += v
      } else if (s.key === 'embed_inference_origin_ms') {
        globalOriginMs += v
      } else if (s.key === 'embed_passages_origin') {
        globalOriginPassages += v
      }
    }
    for (const [dimKey, { label, ms }] of msByDim.entries()) {
      const passages = passagesByDim.get(dimKey) ?? 0
      if (passages <= 0) {
        continue
      }
      let cur = byKey.get(dimKey)
      if (!cur) {
        cur = { label, totalMs: 0, totalPassages: 0 }
        byKey.set(dimKey, cur)
      }
      cur.totalMs += ms
      cur.totalPassages += passages
    }
  }

  const chartRows = Array.from(byKey.entries())
    .filter(([, v]) => v.totalPassages > 0)
    .map(([key, v]) => ({
      key,
      label: v.label,
      inferenceMeanMs: v.totalMs / v.totalPassages,
    }))
    .sort((a, b) => b.inferenceMeanMs - a.inferenceMeanMs)
  return {
    chartRows,
    globalAvgMs: globalPassages > 0 ? globalInferenceMs / globalPassages : null,
    globalE2eMs: globalOriginPassages > 0 ? globalOriginMs / globalOriginPassages : null,
  }
}

function sortApiNodeEntries(entries: Array<[string, NodeView]>): Array<[string, NodeView]> {
  const copy = [...entries]
  copy.sort((a, b) => a[0].localeCompare(b[0], undefined, { sensitivity: 'base' }))
  return copy
}

function formatEncoderRoleCell(role: string): string {
  return role === 'all' ? 'idx/qry' : role
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

function isIndexWorkerRole(role: string): boolean {
  return role === 'index' || role === 'all'
}

function partitionIndexRoleNodes(nodes: { [key: string]: NodeView }): Array<[string, NodeView]> {
  const rows: Array<[string, NodeView]> = []
  for (const entry of Object.entries(nodes)) {
    if (isIndexWorkerRole(entry[1].role)) {
      rows.push(entry)
    }
  }
  return sortEncoderNodeEntries(rows)
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

type ClusterThroughputHistoryPoint = {
  t: number
  byKey: Map<string, number>
  avgMs: number | null
  e2eMs: number | null
}

type AggregateApiChartMetrics = Omit<ApiMetricsHistoryPoint, 't'> & {
  err4xxRps: number | null
  err5xxRps: number | null
}

function aggregateApiChartMetrics(
  view: Metrics | null,
  windowSec: number = CLUSTER_METRICS_WINDOW_SEC,
): AggregateApiChartMetrics {
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

    const wsAllR = getMetricWindowSampleByFirstKey(list, 'api_requests', windowSec)
    if (wsAllR != null) {
      const v = Number(wsAllR.value)
      if (Number.isFinite(v)) {
        sumAllRps += v
        sawAllRps = true
      }
    }
    const wsAllM = getMetricWindowSampleByFirstKey(list, 'api_request_ms', windowSec)
    if (wsAllM != null) {
      const v = Number(wsAllM.value)
      const n = Number(wsAllM.n)
      if (Number.isFinite(v) && Number.isFinite(n) && n > 0) {
        allMsSumVN += v
        allMsSumN += Math.trunc(n)
      }
    }

    const wsSr = getMetricWindowSampleByFirstKey(list, 'api_search', windowSec)
    if (wsSr != null) {
      const v = Number(wsSr.value)
      if (Number.isFinite(v)) {
        sumSearchRps += v
        sawSearchRps = true
      }
    }
    const wsSm = getMetricWindowSampleByFirstKey(list, 'api_search_ms', windowSec)
    if (wsSm != null) {
      const v = Number(wsSm.value)
      const n = Number(wsSm.n)
      if (Number.isFinite(v) && Number.isFinite(n) && n > 0) {
        searchMsSumVN += v
        searchMsSumN += Math.trunc(n)
      }
    }

    for (const rk of ['api_async_upload', 'api_sync_upload', 'api_bulk_upload'] as const) {
      const ws = getMetricWindowSampleByFirstKey(list, rk, windowSec)
      if (ws != null) {
        const v = Number(ws.value)
        if (Number.isFinite(v)) {
          sumIngestRps += v
          sawIngestRps = true
        }
      }
    }
    for (const mk of ['api_async_upload_ms', 'api_sync_upload_ms', 'api_bulk_upload_ms'] as const) {
      const ws = getMetricWindowSampleByFirstKey(list, mk, windowSec)
      if (ws != null) {
        const v = Number(ws.value)
        const n = Number(ws.n)
        if (Number.isFinite(v) && Number.isFinite(n) && n > 0) {
          ingestMsSumVN += v
          ingestMsSumN += Math.trunc(n)
        }
      }
    }

    const ws4 = getMetricWindowSampleByFirstKey(list, 'api_error_4xx', windowSec)
    if (ws4 != null) {
      const v = Number(ws4.value)
      if (Number.isFinite(v)) {
        sumErr4xx += v
        sawErr4xx = true
      }
    }
    const ws5 = getMetricWindowSampleByFirstKey(list, 'api_error_5xx', windowSec)
    if (ws5 != null) {
      const v = Number(ws5.value)
      if (Number.isFinite(v)) {
        sumErr5xx += v
        sawErr5xx = true
      }
    }
  }

  const sawAnyErr = sawErr4xx || sawErr5xx
  const w = windowSec
  return {
    allRps: sawAllRps ? sumAllRps / w : null,
    allMs: allMsSumN > 0 ? allMsSumVN / allMsSumN : null,
    searchRps: sawSearchRps ? sumSearchRps / w : null,
    searchMs: searchMsSumN > 0 ? searchMsSumVN / searchMsSumN : null,
    ingestRps: sawIngestRps ? sumIngestRps / w : null,
    ingestMs: ingestMsSumN > 0 ? ingestMsSumVN / ingestMsSumN : null,
    err4xxRps: sawErr4xx ? sumErr4xx / w : null,
    err5xxRps: sawErr5xx ? sumErr5xx / w : null,
    errRps: sawAnyErr ? (sumErr4xx + sumErr5xx) / w : null,
  }
}

type ApiAggScratch = {
  allV: number
  sawAll: boolean
  allMsV: number
  allMsN: number
  searchV: number
  sawSearch: boolean
  searchMsV: number
  searchMsN: number
  ingestV: number
  sawIngest: boolean
  ingestMsV: number
  ingestMsN: number
  err4V: number
  saw4: boolean
  err5V: number
  saw5: boolean
}

function emptyApiScratch(): ApiAggScratch {
  return {
    allV: 0,
    sawAll: false,
    allMsV: 0,
    allMsN: 0,
    searchV: 0,
    sawSearch: false,
    searchMsV: 0,
    searchMsN: 0,
    ingestV: 0,
    sawIngest: false,
    ingestMsV: 0,
    ingestMsN: 0,
    err4V: 0,
    saw4: false,
    err5V: 0,
    saw5: false,
  }
}

function foldMetricsBucketIntoApiScratch(a: ApiAggScratch, b: MetricsBucket): void {
  const v = b.value
  const n = b.n != null ? Math.trunc(b.n) : 0
  if (!Number.isFinite(v)) {
    return
  }
  switch (b.key) {
    case 'api_requests':
      a.sawAll = true
      a.allV += v
      break
    case 'api_request_ms':
      if (n > 0) {
        a.allMsV += v
        a.allMsN += n
      }
      break
    case 'api_search':
      a.sawSearch = true
      a.searchV += v
      break
    case 'api_search_ms':
      if (n > 0) {
        a.searchMsV += v
        a.searchMsN += n
      }
      break
    case 'api_async_upload':
    case 'api_sync_upload':
    case 'api_bulk_upload':
      a.sawIngest = true
      a.ingestV += v
      break
    case 'api_async_upload_ms':
    case 'api_sync_upload_ms':
    case 'api_bulk_upload_ms':
      if (n > 0) {
        a.ingestMsV += v
        a.ingestMsN += n
      }
      break
    case 'api_error_4xx':
      a.saw4 = true
      a.err4V += v
      break
    case 'api_error_5xx':
      a.saw5 = true
      a.err5V += v
      break
    default:
      break
  }
}

function apiScratchToHistoryPoint(
  bucketStartSec: number,
  a: ApiAggScratch,
  bucketSec: number,
): ApiMetricsHistoryPoint {
  const w = bucketSec
  const sawAnyErr = a.saw4 || a.saw5
  return {
    t: bucketStartSec * 1000,
    allRps: a.sawAll ? a.allV / w : null,
    allMs: a.allMsN > 0 ? a.allMsV / a.allMsN : null,
    searchRps: a.sawSearch ? a.searchV / w : null,
    searchMs: a.searchMsN > 0 ? a.searchMsV / a.searchMsN : null,
    ingestRps: a.sawIngest ? a.ingestV / w : null,
    ingestMs: a.ingestMsN > 0 ? a.ingestMsV / a.ingestMsN : null,
    errRps: sawAnyErr ? (a.err4V + a.err5V) / w : null,
  }
}

function apiMetricTrendsToPointsByBucketStart(
  trends: MetricTrend[],
  expectedBucketSec: number,
): Map<number, ApiMetricsHistoryPoint> {
  const byStart = new Map<number, ApiAggScratch>()
  for (const tr of trends) {
    for (const b of tr.buckets ?? []) {
      if (b.bucket_seconds !== expectedBucketSec) {
        continue
      }
      const slot = b.bucket_start
      let acc = byStart.get(slot)
      if (!acc) {
        acc = emptyApiScratch()
        byStart.set(slot, acc)
      }
      foldMetricsBucketIntoApiScratch(acc, b)
    }
  }
  const out = new Map<number, ApiMetricsHistoryPoint>()
  for (const [slot, acc] of byStart) {
    out.set(slot, apiScratchToHistoryPoint(slot, acc, expectedBucketSec))
  }
  return out
}

function trimChartBucketMap(store: Map<number, unknown>, cutoffBucketStartSec: number): void {
  for (const k of [...store.keys()]) {
    if (k < cutoffBucketStartSec) {
      store.delete(k)
    }
  }
}

function bucketStartCutoffSec(historyMs: number, nowMs: number): number {
  return Math.floor((nowMs - historyMs) / 1000)
}

function sortedApiChartPoints(
  store: Map<number, ApiMetricsHistoryPoint>,
  live: ApiMetricsHistoryPoint | null,
  cutoffMs: number,
  nowMs: number,
): ApiMetricsHistoryPoint[] {
  const rows: ApiMetricsHistoryPoint[] = []
  const sortedKeys = Array.from(store.keys())
    .filter((k) => k * 1000 >= cutoffMs)
    .sort((a, b) => a - b)
  for (const k of sortedKeys) {
    const row = store.get(k)
    if (row != null) {
      rows.push({ ...row, t: k * 1000 })
    }
  }
  if (live != null) {
    rows.push({ ...live, t: nowMs })
  }
  return rows
}

function dimsKeyFromBucketDims(d: string[] | undefined): string {
  const arr = d ?? []
  return `${arr[0] ?? ''}\0${arr[1] ?? ''}\0${arr[2] ?? ''}`
}

function clusterThroughputPointFromEncoderBuckets(
  buckets: MetricsBucket[],
  bucketStartSec: number,
): ClusterThroughputHistoryPoint | null {
  const msByDim = new Map<string, number>()
  const passagesByDim = new Map<string, number>()
  let globalInferenceMs = 0
  let globalPassages = 0
  let globalOriginMs = 0
  let globalOriginPassages = 0

  for (const b of buckets) {
    const dimKey = dimsKeyFromBucketDims(b.dims)
    const v = b.value
    if (!Number.isFinite(v)) {
      continue
    }
    switch (b.key) {
      case 'embed_inference_ms':
        globalInferenceMs += v
        msByDim.set(dimKey, (msByDim.get(dimKey) ?? 0) + v)
        break
      case 'embed_passages':
        globalPassages += v
        passagesByDim.set(dimKey, (passagesByDim.get(dimKey) ?? 0) + v)
        break
      case 'embed_inference_origin_ms':
        globalOriginMs += v
        break
      case 'embed_passages_origin':
        globalOriginPassages += v
        break
      default:
        break
    }
  }

  const byKey = new Map<string, number>()
  for (const [dimKey, ms] of msByDim.entries()) {
    const passages = passagesByDim.get(dimKey) ?? 0
    if (passages > 0) {
      byKey.set(dimKey, ms / passages)
    }
  }

  const globalAvgMs = globalPassages > 0 ? globalInferenceMs / globalPassages : null
  const globalE2eMs = globalOriginPassages > 0 ? globalOriginMs / globalOriginPassages : null
  if (byKey.size === 0 && globalAvgMs == null && globalE2eMs == null) {
    return null
  }
  return {
    t: bucketStartSec * 1000,
    byKey,
    avgMs: globalAvgMs,
    e2eMs: globalE2eMs,
  }
}

function encoderMetricTrendsToPointsByBucketStart(
  trends: MetricTrend[],
  expectedBucketSec: number,
): Map<number, ClusterThroughputHistoryPoint> {
  const byStart = new Map<number, MetricsBucket[]>()
  for (const tr of trends) {
    for (const b of tr.buckets ?? []) {
      if (b.bucket_seconds !== expectedBucketSec) {
        continue
      }
      const arr = byStart.get(b.bucket_start) ?? []
      arr.push(b)
      byStart.set(b.bucket_start, arr)
    }
  }
  const out = new Map<number, ClusterThroughputHistoryPoint>()
  for (const [start, blist] of byStart) {
    const pt = clusterThroughputPointFromEncoderBuckets(blist, start)
    if (pt != null) {
      out.set(start, pt)
    }
  }
  return out
}

type IndexingLatencyHistoryPoint = {
  t: number
  queueJobMs: number | null
  bulkJobMs: number | null
}

function indexingLatencyPointFromBuckets(
  buckets: MetricsBucket[],
  bucketStartSec: number,
): IndexingLatencyHistoryPoint | null {
  let qV = 0
  let qN = 0
  let bV = 0
  let bN = 0
  for (const b of buckets) {
    if (b.bucket_start !== bucketStartSec) {
      continue
    }
    const v = b.value
    if (!Number.isFinite(v)) {
      continue
    }
    if (b.key === 'index_queue_job_ms') {
      const n = b.n ?? 0
      if (n > 0) {
        qV += v
        qN += n
      }
    } else if (b.key === 'index_bulk_job_ms') {
      const n = b.n ?? 0
      if (n > 0) {
        bV += v
        bN += n
      }
    }
  }
  const queueJobMs = qN > 0 ? qV / qN : null
  const bulkJobMs = bN > 0 ? bV / bN : null
  if (queueJobMs == null && bulkJobMs == null) {
    return null
  }
  return {
    t: bucketStartSec * 1000,
    queueJobMs,
    bulkJobMs,
  }
}

function indexMetricTrendsToPointsByBucketStart(
  trends: MetricTrend[],
  expectedBucketSec: number,
): Map<number, IndexingLatencyHistoryPoint> {
  const byStart = new Map<number, MetricsBucket[]>()
  for (const tr of trends) {
    for (const b of tr.buckets ?? []) {
      if (b.bucket_seconds !== expectedBucketSec) {
        continue
      }
      if (b.key !== 'index_queue_job_ms' && b.key !== 'index_bulk_job_ms') {
        continue
      }
      const arr = byStart.get(b.bucket_start) ?? []
      arr.push(b)
      byStart.set(b.bucket_start, arr)
    }
  }
  const out = new Map<number, IndexingLatencyHistoryPoint>()
  for (const [start, blist] of byStart) {
    const pt = indexingLatencyPointFromBuckets(blist, start)
    if (pt != null) {
      out.set(start, pt)
    }
  }
  return out
}

function sortedClusterThroughputPoints(
  store: Map<number, ClusterThroughputHistoryPoint>,
  live: ClusterThroughputHistoryPoint | null,
  cutoffMs: number,
  nowMs: number,
): ClusterThroughputHistoryPoint[] {
  const rows: ClusterThroughputHistoryPoint[] = []
  const sortedKeys = Array.from(store.keys())
    .filter((k) => k * 1000 >= cutoffMs)
    .sort((a, b) => a - b)
  for (const k of sortedKeys) {
    const row = store.get(k)
    if (row != null) {
      rows.push({ ...row, t: k * 1000 })
    }
  }
  if (live != null) {
    rows.push({ ...live, t: nowMs })
  }
  return rows
}

function weightedIndexingJobMsFromNodes(view: Metrics | null): {
  queueJobMs: number | null
  bulkJobMs: number | null
} {
  let qV = 0
  let qN = 0
  let bV = 0
  let bN = 0
  for (const node of Object.values(view?.nodes ?? {})) {
    if (!isIndexWorkerRole(node.role)) {
      continue
    }
    const list = node.metrics ?? []
    for (const key of ['index_queue_job_ms', 'index_bulk_job_ms'] as const) {
      const ws = getMetricWindowSampleByFirstKey(list, key)
      if (ws == null) {
        continue
      }
      const v = Number(ws.value)
      const n = Number(ws.n)
      if (!Number.isFinite(v) || !Number.isFinite(n) || n <= 0) {
        continue
      }
      if (key === 'index_queue_job_ms') {
        qV += v
        qN += n
      } else {
        bV += v
        bN += n
      }
    }
  }
  return {
    queueJobMs: qN > 0 ? qV / qN : null,
    bulkJobMs: bN > 0 ? bV / bN : null,
  }
}

function sortedIndexingLatencyPoints(
  store: Map<number, IndexingLatencyHistoryPoint>,
  live: IndexingLatencyHistoryPoint | null,
  cutoffMs: number,
  nowMs: number,
): IndexingLatencyHistoryPoint[] {
  const rows: IndexingLatencyHistoryPoint[] = []
  const sortedKeys = Array.from(store.keys())
    .filter((k) => k * 1000 >= cutoffMs)
    .sort((a, b) => a - b)
  for (const k of sortedKeys) {
    const row = store.get(k)
    if (row != null) {
      rows.push({ ...row, t: k * 1000 })
    }
  }
  if (live != null) {
    rows.push({ ...live, t: nowMs })
  }
  return rows
}

function readHomeMetricsTabFromDom($root: JQuery<HTMLElement>): HomeMetricsTabId {
  if ($root.find('[data-home-metrics-tab="api"]').attr('aria-selected') === 'true') {
    return 'api'
  }
  if ($root.find('[data-home-metrics-tab="indexing"]').attr('aria-selected') === 'true') {
    return 'indexing'
  }
  return 'encoder'
}

function readClusterChartThemeColors(): {
  grid: string
  tick: string
  tooltipBg: string
  tooltipBorder: string
} {
  const s = getComputedStyle(document.documentElement)
  const grid = s.getPropertyValue('--chart-cluster-grid').trim()
  const tick = s.getPropertyValue('--chart-cluster-tick').trim()
  const tooltipBg = s.getPropertyValue('--chart-cluster-tooltip-bg').trim()
  const tooltipBorder = s.getPropertyValue('--chart-cluster-tooltip-border').trim()
  return {
    grid: grid || 'rgba(120, 120, 120, 0.18)',
    tick: tick || '#555555',
    tooltipBg: tooltipBg || 'rgba(255, 255, 255, 0.78)',
    tooltipBorder: tooltipBorder || 'rgba(0, 0, 0, 0.1)',
  }
}

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

const HOME_LINE_CHART_ZOOM: ZoomPluginOptions = {
  pan: {
    enabled: true,
    mode: 'x',
  },
  zoom: {
    mode: 'x',
    wheel: { enabled: true },
    pinch: { enabled: true },
  },
}

function buildApiMetricsChartOptions(
  cutoff: number,
  now: number,
  gridColor: string,
  tickColor: string,
  tooltipBg: string,
  tooltipBorder: string,
  chartFont: ReturnType<typeof readClusterChartFont>,
  yAxisTitle: string,
  latencyTooltip: boolean,
): ChartOptions<'line'> {
  const spanMs = now - cutoff
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
        backgroundColor: tooltipBg,
        borderColor: tooltipBorder,
        borderWidth: 1,
        titleColor: tickColor,
        bodyColor: tickColor,
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
      zoom: HOME_LINE_CHART_ZOOM,
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
            return formatClusterChartXAxisTickLabel(n, spanMs)
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

const HOME_READINESS_KEYS: HomeReadinessKey[] = ['database', 'rabbitmq', 'index', 'query']

export class HomePanel extends DashboardPanel {
  private readyPollTimer: number | null = null
  private readyPollGeneration = 0
  /** Consecutive failed /ready samples per key; UI shows Not ready only after 2 failures in a row. */
  private readyFailureStreaks: Record<HomeReadinessKey, number> = {
    database: 0,
    rabbitmq: 0,
    index: 0,
    query: 0,
  }
  private homeApi: AmgixApi | null = null
  private clusterThroughputChart: Chart<'line'> | null = null
  private encoderChartBuckets = new Map<number, ClusterThroughputHistoryPoint>()
  private indexingLatencyChart: Chart<'line'> | null = null
  private indexingChartBuckets = new Map<number, IndexingLatencyHistoryPoint>()
  private clusterSeriesLabels = new Map<string, string>()
  private apiMetricsRequestsChart: Chart<'line'> | null = null
  private apiMetricsLatenciesChart: Chart<'line'> | null = null
  private apiChartBuckets = new Map<number, ApiMetricsHistoryPoint>()
  private metricsChartLiveView: Metrics | null = null
  private homeChartHistoryMs = DEFAULT_HOME_CHART_HISTORY_MS
  private homeChartRangeListenerAttached = false

  override deactivate(): void {
    this.clearReadyPoll()
    this.readyPollGeneration += 1
  }

  applyMetricsTabFromRoute(tab: HomeMetricsTabId): void {
    const $root = $('#panel-home [data-home-root]')
    if (!$root.length) {
      return
    }
    if (!$root.find('[data-home-metrics-tab]').length) {
      return
    }
    this.activateHomeMetricsTab($root, tab)
    const api = this.homeApi
    if (api != null) {
      void this.ensureChartTrendsForActiveTab(api, $root)
      void this.refreshHomeMetricsAfterTabChange(api, $root, tab)
    }
  }

  private async refreshHomeMetricsAfterTabChange(
    api: AmgixApi,
    $root: JQuery<HTMLElement>,
    tab: HomeMetricsTabId,
  ): Promise<void> {
    const gen = this.readyPollGeneration
    let metrics: Metrics | null = null
    try {
      metrics = await api.metricsCurrent({
        window: HOME_CHART_METRICS_LIVE_WINDOW_SEC,
        keys: homeMetricsCurrentKeys(tab),
      })
    } catch {
      metrics = null
    }
    if (gen !== this.readyPollGeneration) {
      return
    }
    const $still = $('#panel-home [data-home-root]')
    if (!$still.length || $still.get(0) !== $root.get(0)) {
      return
    }
    this.metricsChartLiveView = metrics
    this.applyClusterViewToDom($root, metrics)
    this.refreshVisibleHomeCharts($root)
  }

  init(api: AmgixApi): void {
    this.homeApi = api
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
    for (const k of HOME_READINESS_KEYS) {
      this.readyFailureStreaks[k] = 0
    }
    this.destroyApiMetricsChart()
    this.destroyClusterThroughputChart()
    this.destroyIndexingLatencyChart()
    this.encoderChartBuckets.clear()
    this.indexingChartBuckets.clear()
    this.clusterSeriesLabels.clear()
    this.apiChartBuckets.clear()
    this.metricsChartLiveView = null
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
    tooltipBg: string,
    tooltipBorder: string,
    chartFont: ReturnType<typeof readClusterChartFont>,
    yAxisTitle: string,
  ): void {
    const xScale = ch.options.scales?.x
    if (xScale && typeof xScale === 'object' && 'min' in xScale && 'max' in xScale && !ch.isZoomedOrPanned()) {
      ;(xScale as { min?: number; max?: number }).min = cutoff
      ;(xScale as { min?: number; max?: number }).max = now
    }
    ch.options.color = tickColor
    const plSync = ch.options.plugins as Record<string, unknown> | undefined
    if (plSync) {
      plSync.zoom = HOME_LINE_CHART_ZOOM
    }
    if (ch.options.font && typeof ch.options.font === 'object') {
      const f = ch.options.font as { family?: string; size?: number }
      f.family = chartFont.family
      f.size = chartFont.tickPx
    }
    const tip = ch.options.plugins?.tooltip
    if (tip && typeof tip === 'object') {
      const t = tip as {
        backgroundColor?: string
        borderColor?: string
        borderWidth?: number
        titleColor?: string
        bodyColor?: string
        titleFont?: { family?: string; size?: number }
        bodyFont?: { family?: string; size?: number }
      }
      t.backgroundColor = tooltipBg
      t.borderColor = tooltipBorder
      t.borderWidth = 1
      t.titleColor = tickColor
      t.bodyColor = tickColor
      if (t.titleFont) {
        t.titleFont.family = chartFont.family
        t.titleFont.size = chartFont.tooltipPx
      }
      if (t.bodyFont) {
        t.bodyFont.family = chartFont.family
        t.bodyFont.size = chartFont.tooltipPx
      }
    }
    const spanMs = now - cutoff
    const xs = ch.options.scales?.x
    if (xs && typeof xs === 'object') {
      const gx = (xs as { grid?: { color?: string }; ticks?: { color?: string; font?: { family?: string; size?: number } } }).grid
      if (gx) {
        gx.color = gridColor
      }
      const tx = (xs as {
        ticks?: {
          color?: string
          font?: { family?: string; size?: number }
          callback?: (tickValue: string | number, index: number, ticks: unknown) => string | string[]
        }
      }).ticks
      if (tx) {
        tx.color = tickColor
        if (tx.font) {
          tx.font.family = chartFont.family
          tx.font.size = chartFont.tickPx
        }
        tx.callback = (tickValue: string | number) => {
          const n = typeof tickValue === 'number' ? tickValue : Number(tickValue)
          if (!Number.isFinite(n)) {
            return ''
          }
          return formatClusterChartXAxisTickLabel(n, spanMs)
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

  private destroyIndexingLatencyChart(): void {
    this.indexingLatencyChart?.destroy()
    this.indexingLatencyChart = null
  }

  private refreshClusterThroughputChart($root: JQuery<HTMLElement>): void {
    const $canvas = $root.find('[data-home-cluster-throughput-chart]')
    const canvas = $canvas.get(0) as HTMLCanvasElement | undefined
    const $wrap = $root.find('[data-home-cluster-chart-wrap]')
    const $canvasWrap = $root.find('[data-home-cluster-chart-canvas-wrap]')
    const $legendUl = $root.find('[data-home-cluster-chart-legend]')
    if (!canvas || !$wrap.length || !$canvasWrap.length || !$legendUl.length) {
      return
    }

    const now = Date.now()
    const cutoff = now - this.homeChartHistoryMs
    const spanMs = now - cutoff
    const liveView = this.metricsChartLiveView
    const aggLive = aggregateClusterThroughput(liveView, HOME_CHART_METRICS_LIVE_WINDOW_SEC)
    for (const row of aggLive.chartRows) {
      this.clusterSeriesLabels.set(row.key, row.label)
    }
    const liveByKey = new Map<string, number>()
    for (const row of aggLive.chartRows) {
      liveByKey.set(row.key, row.inferenceMeanMs)
    }
    const livePoint: ClusterThroughputHistoryPoint | null =
      liveView != null
        ? {
            t: now,
            byKey: liveByKey,
            avgMs: aggLive.globalAvgMs,
            e2eMs: aggLive.globalE2eMs,
          }
        : null

    const clusterThroughputHistory = sortedClusterThroughputPoints(
      this.encoderChartBuckets,
      livePoint,
      cutoff,
      now,
    )

    for (const pt of clusterThroughputHistory) {
      for (const k of pt.byKey.keys()) {
        if (!this.clusterSeriesLabels.has(k)) {
          this.clusterSeriesLabels.set(k, labelFromVectorMetricsMergeKey(k))
        }
      }
    }

    const keySet = new Set<string>()
    for (const pt of clusterThroughputHistory) {
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
      data: clusterThroughputHistory.map((pt) => ({
        x: pt.t,
        y: typeof pt.avgMs === 'number' && Number.isFinite(pt.avgMs) ? pt.avgMs : null,
      })),
      borderColor: avgLineColor,
      backgroundColor: avgLineColor,
      borderWidth: 2,
      fill: false,
      tension: 0.4,
      pointRadius: 1,
      pointHoverRadius: 5,
      pointHitRadius: 12,
      spanGaps: false,
    }
    const e2eMsDataset = {
      label: 'Routed Latency',
      data: clusterThroughputHistory.map((pt) => ({
        x: pt.t,
        y: typeof pt.e2eMs === 'number' && Number.isFinite(pt.e2eMs) ? pt.e2eMs : null,
      })),
      borderColor: e2eLineColor,
      backgroundColor: e2eLineColor,
      borderWidth: 2,
      fill: false,
      tension: 0.4,
      pointRadius: 1,
      pointHoverRadius: 5,
      pointHitRadius: 12,
      spanGaps: false,
    }

    const perVectorDatasets = seriesKeys.map((key, i) => {
      const typeModel = normalizeVectorSeriesLabelForKey(key, this.clusterSeriesLabels.get(key))
      return {
        label: `Inference Latency (${typeModel})`,
        data: clusterThroughputHistory.map((pt) => {
          const y = pt.byKey.get(key)
          return {
            x: pt.t,
            y: typeof y === 'number' && Number.isFinite(y) ? y : null,
          }
        }),
        borderColor: batchColors[i]!,
        backgroundColor: batchColors[i]!,
        borderWidth: 2,
        fill: false,
        tension: 0.4,
        pointRadius: 1,
        pointHoverRadius: 5,
        pointHitRadius: 12,
        spanGaps: false,
      }
    })

    const datasets = [avgMsDataset, e2eMsDataset, ...perVectorDatasets]

    $canvasWrap.css('height', '150px')

    const { grid: gridColor, tick: tickColor, tooltipBg, tooltipBorder } = readClusterChartThemeColors()
    const chartFont = readClusterChartFont()

    const legendItems = [
      {
        color: avgLineColor,
        ...clusterChartLegendMsValue('Inference Latency', aggLive.globalAvgMs),
      },
      {
        color: e2eLineColor,
        ...clusterChartLegendMsValue('Routed Latency', aggLive.globalE2eMs),
      },
      ...seriesKeys.map((key, i) => {
        const typeModel = normalizeVectorSeriesLabelForKey(key, this.clusterSeriesLabels.get(key))
        return {
          color: batchColors[i]!,
          ...clusterChartLegendMsValue(`Inference Latency (${typeModel})`, liveByKey.get(key) ?? null),
        }
      }),
    ]
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
              backgroundColor: tooltipBg,
              borderColor: tooltipBorder,
              borderWidth: 1,
              titleColor: tickColor,
              bodyColor: tickColor,
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
            zoom: HOME_LINE_CHART_ZOOM,
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
                  return formatClusterChartXAxisTickLabel(n, spanMs)
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
      if (xScale && typeof xScale === 'object' && 'min' in xScale && 'max' in xScale && !ch.isZoomedOrPanned()) {
        ;(xScale as { min?: number; max?: number }).min = cutoff
        ;(xScale as { min?: number; max?: number }).max = now
      }
      const leg = ch.options.plugins?.legend
      if (leg && typeof leg === 'object' && 'display' in leg) {
        ;(leg as { display?: boolean }).display = false
      }
      ch.options.color = tickColor
      const plEnc = ch.options.plugins as Record<string, unknown> | undefined
      if (plEnc) {
        plEnc.zoom = HOME_LINE_CHART_ZOOM
      }
      const tipEnc = ch.options.plugins?.tooltip
      if (tipEnc && typeof tipEnc === 'object') {
        const te = tipEnc as {
          backgroundColor?: string
          borderColor?: string
          borderWidth?: number
          titleColor?: string
          bodyColor?: string
        }
        te.backgroundColor = tooltipBg
        te.borderColor = tooltipBorder
        te.borderWidth = 1
        te.titleColor = tickColor
        te.bodyColor = tickColor
      }
      const xs = ch.options.scales?.x
      if (xs && typeof xs === 'object') {
        const gx = (xs as { grid?: { color?: string }; ticks?: { color?: string } }).grid
        if (gx) {
          gx.color = gridColor
        }
        const tx = (xs as {
          ticks?: {
            color?: string
            callback?: (tickValue: string | number, index: number, ticks: unknown) => string | string[]
          }
        }).ticks
        if (tx) {
          tx.color = tickColor
          tx.callback = (tickValue: string | number) => {
            const n = typeof tickValue === 'number' ? tickValue : Number(tickValue)
            if (!Number.isFinite(n)) {
              return ''
            }
            return formatClusterChartXAxisTickLabel(n, spanMs)
          }
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

  private refreshIndexingLatencyChart($root: JQuery<HTMLElement>): void {
    const $canvas = $root.find('[data-home-indexing-latency-chart]')
    const canvas = $canvas.get(0) as HTMLCanvasElement | undefined
    const $wrap = $root.find('[data-home-indexing-chart-wrap]')
    const $canvasWrap = $root.find('[data-home-indexing-chart-canvas-wrap]')
    const $legendUl = $root.find('[data-home-indexing-chart-legend]')
    if (!canvas || !$wrap.length || !$canvasWrap.length || !$legendUl.length) {
      return
    }

    const now = Date.now()
    const cutoff = now - this.homeChartHistoryMs
    const liveView = this.metricsChartLiveView
    const liveMs = weightedIndexingJobMsFromNodes(liveView)
    const livePoint: IndexingLatencyHistoryPoint | null =
      liveView != null && (liveMs.queueJobMs != null || liveMs.bulkJobMs != null)
        ? {
            t: now,
            queueJobMs: liveMs.queueJobMs,
            bulkJobMs: liveMs.bulkJobMs,
          }
        : null

    const indexingHistory = sortedIndexingLatencyPoints(this.indexingChartBuckets, livePoint, cutoff, now)

    const queueLineColor = '#2563eb'
    const bulkLineColor = '#ea580c'

    const queueMsDataset = {
      label: 'Single-doc',
      data: indexingHistory.map((pt) => ({
        x: pt.t,
        y: pt.queueJobMs != null && Number.isFinite(pt.queueJobMs) ? pt.queueJobMs : null,
      })),
      borderColor: queueLineColor,
      backgroundColor: queueLineColor,
      borderWidth: 2,
      fill: false,
      tension: 0.4,
      pointRadius: 1,
      pointHoverRadius: 5,
      pointHitRadius: 12,
      spanGaps: false,
    }
    const bulkMsDataset = {
      label: 'Bulk batch',
      data: indexingHistory.map((pt) => ({
        x: pt.t,
        y: pt.bulkJobMs != null && Number.isFinite(pt.bulkJobMs) ? pt.bulkJobMs : null,
      })),
      borderColor: bulkLineColor,
      backgroundColor: bulkLineColor,
      borderWidth: 2,
      fill: false,
      tension: 0.4,
      pointRadius: 1,
      pointHoverRadius: 5,
      pointHitRadius: 12,
      spanGaps: false,
    }

    const datasets = [queueMsDataset, bulkMsDataset]

    $canvasWrap.css('height', '150px')

    const { grid: gridColor, tick: tickColor, tooltipBg, tooltipBorder } = readClusterChartThemeColors()
    const chartFont = readClusterChartFont()

    const legendItems = [
      {
        color: queueLineColor,
        ...clusterChartLegendMsValue('Single-doc', liveMs.queueJobMs),
      },
      {
        color: bulkLineColor,
        ...clusterChartLegendMsValue('Bulk batch', liveMs.bulkJobMs),
      },
    ]
    renderClusterChartHtmlLegend($legendUl, legendItems, tickColor)

    if (this.indexingLatencyChart == null) {
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        return
      }
      const config: ChartConfiguration<'line'> = {
        type: 'line',
        data: { datasets },
        options: buildApiMetricsChartOptions(
          cutoff,
          now,
          gridColor,
          tickColor,
          tooltipBg,
          tooltipBorder,
          chartFont,
          'Milliseconds',
          true,
        ),
      }
      this.indexingLatencyChart = new Chart(ctx, config)
    } else {
      const ch = this.indexingLatencyChart
      ch.data.datasets = datasets as typeof ch.data.datasets
      this.syncSingleYAxisApiMetricsChartTheme(
        ch,
        cutoff,
        now,
        gridColor,
        tickColor,
        tooltipBg,
        tooltipBorder,
        chartFont,
        'Milliseconds',
      )
    }
  }

  private refreshApiMetricsChart($root: JQuery<HTMLElement>): void {
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

    const now = Date.now()
    const cutoff = now - this.homeChartHistoryMs
    const liveView = this.metricsChartLiveView
    const aggLive = aggregateApiChartMetrics(liveView, HOME_CHART_METRICS_LIVE_WINDOW_SEC)
    const livePoint: ApiMetricsHistoryPoint | null =
      liveView != null
        ? {
            t: now,
            allRps: aggLive.allRps,
            allMs: aggLive.allMs,
            searchRps: aggLive.searchRps,
            searchMs: aggLive.searchMs,
            ingestRps: aggLive.ingestRps,
            ingestMs: aggLive.ingestMs,
            errRps: aggLive.errRps,
          }
        : null

    const apiMetricsHistory = sortedApiChartPoints(this.apiChartBuckets, livePoint, cutoff, now)

    const yNum = (v: number | null) => (typeof v === 'number' && Number.isFinite(v) ? v : null)

    const apiGroupColors = ['#8b5cf6', '#38bdf8', '#f59e0b']
    const apiErrorsReqColor = '#dc2626'

    const line = (label: string, color: string, pick: (pt: ApiMetricsHistoryPoint) => number | null) => ({
      label,
      data: apiMetricsHistory.map((pt) => ({ x: pt.t, y: pick(pt) })),
      borderColor: color,
      backgroundColor: color,
      borderWidth: 2,
      fill: false,
      tension: 0.4,
      pointRadius: 1,
      pointHoverRadius: 5,
      pointHitRadius: 12,
      spanGaps: false,
    })

    const datasetsRps = [
      line('Reqs', apiGroupColors[0]!, (pt) => yNum(pt.allRps)),
      line('Search', apiGroupColors[1]!, (pt) => yNum(pt.searchRps)),
      line('Doc Uploads', apiGroupColors[2]!, (pt) => yNum(pt.ingestRps)),
      line('Err/s', apiErrorsReqColor, (pt) => yNum(pt.errRps)),
    ]

    const datasetsMs = [
      line('Reqs', apiGroupColors[0]!, (pt) => yNum(pt.allMs)),
      line('Search', apiGroupColors[1]!, (pt) => yNum(pt.searchMs)),
      line('Doc Uploads', apiGroupColors[2]!, (pt) => yNum(pt.ingestMs)),
    ]

    $reqCanvasWrap.css('height', '150px')
    $latCanvasWrap.css('height', '150px')

    const { grid: gridColor, tick: tickColor, tooltipBg, tooltipBorder } = readClusterChartThemeColors()
    const chartFont = readClusterChartFont()

    const legendItemsRps = [
      { color: apiGroupColors[0]!, ...clusterChartLegendRpsValue('Reqs', aggLive.allRps) },
      { color: apiGroupColors[1]!, ...clusterChartLegendRpsValue('Search', aggLive.searchRps) },
      { color: apiGroupColors[2]!, ...clusterChartLegendRpsValue('Doc Uploads', aggLive.ingestRps) },
      { color: apiErrorsReqColor, ...clusterChartLegendRpsValue('Err/s', aggLive.errRps) },
    ]
    const legendItemsMs = [
      { color: apiGroupColors[0]!, ...clusterChartLegendMsValue('Reqs', aggLive.allMs) },
      { color: apiGroupColors[1]!, ...clusterChartLegendMsValue('Search', aggLive.searchMs) },
      { color: apiGroupColors[2]!, ...clusterChartLegendMsValue('Doc Uploads', aggLive.ingestMs) },
    ]
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
        options: buildApiMetricsChartOptions(
          cutoff,
          now,
          gridColor,
          tickColor,
          tooltipBg,
          tooltipBorder,
          chartFont,
          'Reqs/s',
          false,
        ),
      })
    } else {
      const ch = this.apiMetricsRequestsChart
      ch.data.datasets = datasetsRps as typeof ch.data.datasets
      this.syncSingleYAxisApiMetricsChartTheme(
        ch,
        cutoff,
        now,
        gridColor,
        tickColor,
        tooltipBg,
        tooltipBorder,
        chartFont,
        'Reqs/s',
      )
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
          tooltipBg,
          tooltipBorder,
          chartFont,
          'Milliseconds',
          true,
        ),
      })
    } else {
      const ch = this.apiMetricsLatenciesChart
      ch.data.datasets = datasetsMs as typeof ch.data.datasets
      this.syncSingleYAxisApiMetricsChartTheme(
        ch,
        cutoff,
        now,
        gridColor,
        tickColor,
        tooltipBg,
        tooltipBorder,
        chartFont,
        'Milliseconds',
      )
    }
  }

  private activateHomeMetricsTab($root: JQuery<HTMLElement>, panel: HomeMetricsTabId): void {
    const $apiPanel = $root.find('[data-home-metrics-tab-panel="api"]')
    const $indexingPanel = $root.find('[data-home-metrics-tab-panel="indexing"]')
    const $encPanel = $root.find('[data-home-metrics-tab-panel="encoder"]')
    const $btnApi = $root.find('[data-home-metrics-tab="api"]')
    const $btnIndexing = $root.find('[data-home-metrics-tab="indexing"]')
    const $btnEnc = $root.find('[data-home-metrics-tab="encoder"]')
    if (
      !$apiPanel.length ||
      !$indexingPanel.length ||
      !$encPanel.length ||
      !$btnApi.length ||
      !$btnIndexing.length ||
      !$btnEnc.length
    ) {
      return
    }
    const showApi = panel === 'api'
    const showIndexing = panel === 'indexing'
    const showEncoder = panel === 'encoder'
    $btnApi.attr('aria-selected', showApi ? 'true' : 'false')
    $btnIndexing.attr('aria-selected', showIndexing ? 'true' : 'false')
    $btnEnc.attr('aria-selected', showEncoder ? 'true' : 'false')
    $btnApi.attr('tabindex', showApi ? '0' : '-1')
    $btnIndexing.attr('tabindex', showIndexing ? '0' : '-1')
    $btnEnc.attr('tabindex', showEncoder ? '0' : '-1')
    $apiPanel.prop('hidden', !showApi)
    $indexingPanel.prop('hidden', !showIndexing)
    $encPanel.prop('hidden', !showEncoder)
    window.requestAnimationFrame(() => {
      this.refreshVisibleHomeCharts($root)
      if (showApi) {
        this.apiMetricsRequestsChart?.resize()
        this.apiMetricsLatenciesChart?.resize()
      } else if (showIndexing) {
        this.indexingLatencyChart?.resize()
      } else if (showEncoder) {
        this.clusterThroughputChart?.resize()
      }
    })
  }

  private applyClusterInfoToDom($root: JQuery<HTMLElement>, view: Metrics | null): void {
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
      $inferMs.text(`${formatClusterAvgMsCell(thr.globalAvgMs)} ms/passage`)
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

  private applyClusterViewToDom($root: JQuery<HTMLElement>, view: Metrics | null): void {
    try {
      this.applyClusterTablesAndChartsToDom($root, view)
    } finally {
      this.applyClusterInfoToDom($root, view)
    }
  }

  private refreshVisibleHomeCharts($root: JQuery<HTMLElement>): void {
    const $apiPanel = $root.find('[data-home-metrics-tab-panel="api"]')
    const $indexingPanel = $root.find('[data-home-metrics-tab-panel="indexing"]')
    const $encPanel = $root.find('[data-home-metrics-tab-panel="encoder"]')
    if ($apiPanel.length && !$apiPanel.prop('hidden')) {
      this.refreshApiMetricsChart($root)
    }
    if ($indexingPanel.length && !$indexingPanel.prop('hidden')) {
      this.refreshIndexingLatencyChart($root)
    }
    if ($encPanel.length && !$encPanel.prop('hidden')) {
      this.refreshClusterThroughputChart($root)
    }
  }

  private trendResolutionSec(): 60 | 300 {
    return homeChartTrendResolutionSec(this.homeChartHistoryMs)
  }

  private chartTrendPatchMs(): number {
    return homeChartTrendPatchMs(this.homeChartHistoryMs, this.trendResolutionSec())
  }

  private buildHomeChartHistorySelect(): JQuery<HTMLElement> {
    const $sel = $('<select>', {
      class: 'dashboard-home-chart-history-range',
      attr: { 'data-home-chart-history-range': '', 'aria-label': 'Historical chart range' },
    })
    for (const opt of HOME_CHART_HISTORY_OPTIONS) {
      $sel.append($('<option>', { value: String(opt.valueMs), text: opt.label }))
    }
    $sel.val(String(this.homeChartHistoryMs))
    return $sel
  }

  private syncHomeChartHistorySelects($root: JQuery<HTMLElement>): void {
    const v = String(this.homeChartHistoryMs)
    $root.find('[data-home-chart-history-range]').val(v)
  }

  private resetHomeChartZoom(): void {
    const mode = 'none' as const
    this.clusterThroughputChart?.resetZoom(mode)
    this.indexingLatencyChart?.resetZoom(mode)
    this.apiMetricsRequestsChart?.resetZoom(mode)
    this.apiMetricsLatenciesChart?.resetZoom(mode)
  }

  private resetHomeChartZoomFor(target: HomeChartZoomResetTarget): void {
    const mode = 'none' as const
    if (target === 'encoder') {
      this.clusterThroughputChart?.resetZoom(mode)
    } else if (target === 'indexing') {
      this.indexingLatencyChart?.resetZoom(mode)
    } else if (target === 'api-requests') {
      this.apiMetricsRequestsChart?.resetZoom(mode)
    } else {
      this.apiMetricsLatenciesChart?.resetZoom(mode)
    }
  }

  private async onHomeChartHistoryRangeChanged(
    api: AmgixApi,
    $root: JQuery<HTMLElement>,
    nextMs: number,
  ): Promise<void> {
    if (nextMs > HOME_CHART_HISTORY_SEVEN_DAYS_MS) {
      return
    }
    this.homeChartHistoryMs = nextMs
    this.syncHomeChartHistorySelects($root)
    this.resetHomeChartZoom()
    this.apiChartBuckets.clear()
    this.encoderChartBuckets.clear()
    this.indexingChartBuckets.clear()
    const tab = readHomeMetricsTabFromDom($root)
    if (tab === 'api' || tab === 'encoder' || tab === 'indexing') {
      await this.bootstrapHomeChartTrends(api, $root, tab, this.readyPollGeneration)
    }
    this.refreshVisibleHomeCharts($root)
  }

  private async bootstrapHomeChartTrends(
    api: AmgixApi,
    $root: JQuery<HTMLElement>,
    tab: HomeMetricsTabId,
    generation: number,
  ): Promise<void> {
    const until = new Date()
    const since = new Date(until.getTime() - this.homeChartHistoryMs)
    const keys =
      tab === 'api'
        ? [...API_CHART_TREND_KEYS]
        : tab === 'encoder'
          ? [...ENCODER_CHART_TREND_KEYS]
          : [...INDEXING_CHART_TREND_KEYS]
    const resolution = this.trendResolutionSec()
    try {
      const trends = await api.metricsTrends({
        since,
        until,
        resolution,
        keys,
      })
      if (generation !== this.readyPollGeneration) {
        return
      }
      const cutoffSec = bucketStartCutoffSec(this.homeChartHistoryMs, Date.now())
      if (tab === 'api') {
        const m = apiMetricTrendsToPointsByBucketStart(trends, resolution)
        for (const [k, v] of m) {
          this.apiChartBuckets.set(k, v)
        }
        trimChartBucketMap(this.apiChartBuckets, cutoffSec)
      } else if (tab === 'encoder') {
        const m = encoderMetricTrendsToPointsByBucketStart(trends, resolution)
        for (const [k, v] of m) {
          this.encoderChartBuckets.set(k, v)
        }
        trimChartBucketMap(this.encoderChartBuckets, cutoffSec)
      } else {
        const m = indexMetricTrendsToPointsByBucketStart(trends, resolution)
        for (const [k, v] of m) {
          this.indexingChartBuckets.set(k, v)
        }
        trimChartBucketMap(this.indexingChartBuckets, cutoffSec)
      }
      this.refreshVisibleHomeCharts($root)
    } catch {
    }
  }

  private async ensureChartTrendsForActiveTab(api: AmgixApi, $root: JQuery<HTMLElement>): Promise<void> {
    const generation = this.readyPollGeneration
    const tab = readHomeMetricsTabFromDom($root)
    if (tab === 'api' && this.apiChartBuckets.size === 0) {
      await this.bootstrapHomeChartTrends(api, $root, 'api', generation)
    } else if (tab === 'encoder' && this.encoderChartBuckets.size === 0) {
      await this.bootstrapHomeChartTrends(api, $root, 'encoder', generation)
    } else if (tab === 'indexing' && this.indexingChartBuckets.size === 0) {
      await this.bootstrapHomeChartTrends(api, $root, 'indexing', generation)
    }
  }

  private applyClusterTablesAndChartsToDom($root: JQuery<HTMLElement>, view: Metrics | null): void {
    const CLUSTER_API_COLSPAN = 13
    const CLUSTER_ENCODER_COLSPAN = 10
    const CLUSTER_ENCODER_BY_MODEL_COLSPAN = 7
    const CLUSTER_INDEXING_COLSPAN = 12
    const $apiTbody = $root.find('[data-home-api-cluster-tbody]')
    const $encTbody = $root.find('[data-home-cluster-tbody]')
    const $byModelTbody = $root.find('[data-home-encoder-by-model-tbody]')
    const $indexingTbody = $root.find('[data-home-indexing-cluster-tbody]')
    const $apiPanel = $root.find('[data-home-metrics-tab-panel="api"]')
    const $encPanel = $root.find('[data-home-metrics-tab-panel="encoder"]')
    const $indexingPanel = $root.find('[data-home-metrics-tab-panel="indexing"]')
    const showApiTable = $apiTbody.length > 0 && $apiPanel.length > 0 && !$apiPanel.prop('hidden')
    const showEncTable = $encTbody.length > 0 && $encPanel.length > 0 && !$encPanel.prop('hidden')
    const showIndexingTable =
      $indexingTbody.length > 0 && $indexingPanel.length > 0 && !$indexingPanel.prop('hidden')
    if (!showApiTable && !showEncTable && !showIndexingTable) {
      this.refreshVisibleHomeCharts($root)
      return
    }
    if (showApiTable) {
      $apiTbody.empty()
    }
    if (showEncTable) {
      $encTbody.empty()
      if ($byModelTbody.length) {
        $byModelTbody.empty()
      }
    }
    if (showIndexingTable) {
      $indexingTbody.empty()
    }
    const nodes = view?.nodes
    if (view == null || nodes === undefined) {
      if (showApiTable) {
        $apiTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_API_COLSPAN,
              text: 'Metrics unavailable.',
            }),
          ),
        )
      }
      if (showEncTable) {
        $encTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_ENCODER_COLSPAN,
              text: 'Metrics unavailable.',
            }),
          ),
        )
        if ($byModelTbody.length) {
          $byModelTbody.append(
            $('<tr>').append(
              $('<td>', {
                class: 'dashboard-home-cluster-placeholder',
                colspan: CLUSTER_ENCODER_BY_MODEL_COLSPAN,
                text: 'Metrics unavailable.',
              }),
            ),
          )
        }
      }
      if (showIndexingTable) {
        $indexingTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_INDEXING_COLSPAN,
              text: 'Metrics unavailable.',
            }),
          ),
        )
      }
      this.refreshVisibleHomeCharts($root)
      return
    }
    if (Object.keys(nodes).length === 0) {
      if (showApiTable) {
        $apiTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_API_COLSPAN,
              text: 'No API nodes reported.',
            }),
          ),
        )
      }
      if (showEncTable) {
        $encTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_ENCODER_COLSPAN,
              text: 'No encoder nodes reported.',
            }),
          ),
        )
        if ($byModelTbody.length) {
          $byModelTbody.append(
            $('<tr>').append(
              $('<td>', {
                class: 'dashboard-home-cluster-placeholder',
                colspan: CLUSTER_ENCODER_BY_MODEL_COLSPAN,
                text: 'No encoder nodes reported.',
              }),
            ),
          )
        }
      }
      if (showIndexingTable) {
        $indexingTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_INDEXING_COLSPAN,
              text: 'No index encoders reported.',
            }),
          ),
        )
      }
      this.refreshVisibleHomeCharts($root)
      return
    }
    const { api: apiEntries, encoders: encoderEntries } = partitionApiAndEncoderNodes(nodes)
    const indexEntries = partitionIndexRoleNodes(nodes)
    if (showApiTable) {
      if (apiEntries.length === 0) {
        $apiTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_API_COLSPAN,
              text: 'No API nodes reported.',
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
    if (showEncTable) {
      if (encoderEntries.length === 0) {
        $encTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_ENCODER_COLSPAN,
              text: 'No encoder nodes reported.',
            }),
          ),
        )
      } else {
        for (const [hostname, node] of encoderEntries) {
          const m = formatEmbeddingMetricsCells(node)
          const $nodeTd = $('<td>', { class: 'dashboard-home-v' })
          if (node.is_leader) {
            $nodeTd.append(
              $('<strong>', {
                class: 'dashboard-home-cluster-node-leader',
                text: `${hostname}*`,
                title: 'Current Leader',
              }),
            )
          } else {
            $nodeTd.text(hostname)
          }
          $encTbody.append(
            $('<tr>').append(
              $('<td>', { class: 'dashboard-home-v', text: formatEncoderRoleCell(node.role) }),
              $nodeTd,
              buildEncoderModelsCell(hostname, node),
              $('<td>', { class: 'dashboard-home-v', text: formatAtCapacityCell(node) }),
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
    if (showIndexingTable) {
      if (indexEntries.length === 0) {
        $indexingTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_INDEXING_COLSPAN,
              text: 'No index encoders reported.',
            }),
          ),
        )
      } else {
        for (const [hostname, node] of indexEntries) {
          const m = formatIndexingMetricsCells(node)
          const $nodeTd = $('<td>', { class: 'dashboard-home-v' })
          if (node.is_leader) {
            $nodeTd.append(
              $('<strong>', {
                class: 'dashboard-home-cluster-node-leader',
                text: `${hostname}*`,
                title: 'Current Leader',
              }),
            )
          } else {
            $nodeTd.text(hostname)
          }
          $indexingTbody.append(
            $('<tr>').append(
              $('<td>', { class: 'dashboard-home-v', text: formatEncoderRoleCell(node.role) }),
              $nodeTd,
              $('<td>', { class: 'dashboard-home-v', text: m.docsPerSec }),
              $('<td>', { class: 'dashboard-home-v', text: m.stalePerSec }),
              $('<td>', { class: 'dashboard-home-v', text: m.queueFailPerSec }),
              $('<td>', { class: 'dashboard-home-v', text: m.queueRequeuePerSec }),
              $('<td>', { class: 'dashboard-home-v', text: m.queueJobMs }),
              $('<td>', { class: 'dashboard-home-v', text: m.bulkBatchesPerSec }),
              $('<td>', { class: 'dashboard-home-v', text: m.bulkBatchSize }),
              $('<td>', { class: 'dashboard-home-v', text: m.bulkJobMs }),
              $('<td>', { class: 'dashboard-home-v', text: m.bulkFailPerSec }),
              $('<td>', { class: 'dashboard-home-v', text: m.bulkRequeuePerSec }),
            ),
          )
        }
      }
    }
    if (showEncTable && $byModelTbody.length) {
      if (encoderEntries.length === 0) {
        $byModelTbody.append(
          $('<tr>').append(
            $('<td>', {
              class: 'dashboard-home-cluster-placeholder',
              colspan: CLUSTER_ENCODER_BY_MODEL_COLSPAN,
              text: 'No encoder nodes reported.',
            }),
          ),
        )
      } else {
        const byModelRows = formatEmbeddingByModelTableRows(view)
        if (byModelRows.length === 0) {
          $byModelTbody.append(
            $('<tr>').append(
              $('<td>', {
                class: 'dashboard-home-cluster-placeholder',
                colspan: CLUSTER_ENCODER_BY_MODEL_COLSPAN,
                text: 'No data.',
              }),
            ),
          )
        } else {
          for (const r of byModelRows) {
            $byModelTbody.append(
              $('<tr>').append(
                $('<td>', { class: 'dashboard-home-v', text: r.type }),
                $('<td>', { class: 'dashboard-home-v', text: r.model }),
                $('<td>', { class: 'dashboard-home-v', text: r.requests }),
                $('<td>', { class: 'dashboard-home-v', text: r.rps }),
                $('<td>', { class: 'dashboard-home-v', text: r.avgMs }),
                $('<td>', { class: 'dashboard-home-v', text: r.e2eAvgMs }),
                $('<td>', { class: 'dashboard-home-v', text: r.errPerSec }),
              ),
            )
          }
        }
      }
    }
    this.refreshVisibleHomeCharts($root)
  }

  private readinessDisplayOk(rawOk: boolean, key: HomeReadinessKey): boolean {
    if (rawOk) {
      this.readyFailureStreaks[key] = 0
      return true
    }
    this.readyFailureStreaks[key] += 1
    return this.readyFailureStreaks[key] < 2
  }

  private applyReadinessToDom($root: JQuery<HTMLElement>, ready: ReadyResponse): void {
    for (const key of HOME_READINESS_KEYS) {
      const displayOk = this.readinessDisplayOk(ready[key], key)
      const $el = $root.find(`[data-home-ready="${key}"]`)
      if (!$el.length) {
        continue
      }
      $el.text(readinessLabel(displayOk))
      $el.toggleClass('dashboard-home-status--ready', displayOk)
      $el.toggleClass('dashboard-home-status--not-ready', !displayOk)
    }
  }

  private async pollHomeRefresh(api: AmgixApi, $root: JQuery<HTMLElement>, generation: number): Promise<void> {
    if (generation !== this.readyPollGeneration) {
      return
    }
    if ($('#panel-home').prop('hidden')) {
      return
    }
    const tab = readHomeMetricsTabFromDom($root)
    let metrics: Metrics | null = null
    try {
      metrics = await api.metricsCurrent({
        window: HOME_CHART_METRICS_LIVE_WINDOW_SEC,
        keys: homeMetricsCurrentKeys(tab),
      })
    } catch {
      metrics = null
    }
    if (generation !== this.readyPollGeneration) {
      return
    }
    this.metricsChartLiveView = metrics

    if (tab === 'api' || tab === 'encoder' || tab === 'indexing') {
      const until = new Date()
      const patchMs = this.chartTrendPatchMs()
      const since = new Date(until.getTime() - patchMs)
      const keys =
        tab === 'api'
          ? [...API_CHART_TREND_KEYS]
          : tab === 'encoder'
            ? [...ENCODER_CHART_TREND_KEYS]
            : [...INDEXING_CHART_TREND_KEYS]
      const resolution = this.trendResolutionSec()
      try {
        const trends = await api.metricsTrends({
          since,
          until,
          resolution,
          keys,
        })
        if (generation !== this.readyPollGeneration) {
          return
        }
        const cutoffSec = bucketStartCutoffSec(this.homeChartHistoryMs, Date.now())
        if (tab === 'api') {
          const patch = apiMetricTrendsToPointsByBucketStart(trends, resolution)
          for (const [k, v] of patch) {
            this.apiChartBuckets.set(k, v)
          }
          trimChartBucketMap(this.apiChartBuckets, cutoffSec)
        } else if (tab === 'encoder') {
          const patch = encoderMetricTrendsToPointsByBucketStart(trends, resolution)
          for (const [k, v] of patch) {
            this.encoderChartBuckets.set(k, v)
          }
          trimChartBucketMap(this.encoderChartBuckets, cutoffSec)
        } else {
          const patch = indexMetricTrendsToPointsByBucketStart(trends, resolution)
          for (const [k, v] of patch) {
            this.indexingChartBuckets.set(k, v)
          }
          trimChartBucketMap(this.indexingChartBuckets, cutoffSec)
        }
      } catch {
        if (generation !== this.readyPollGeneration) {
          return
        }
      }
    }

    this.applyClusterViewToDom($root, metrics)
    try {
      const ready = await fetchReadiness()
      if (generation !== this.readyPollGeneration) {
        return
      }
      this.applyReadinessToDom($root, ready)
    } catch {
    }
  }

  private async loadHome(api: AmgixApi, $root: JQuery<HTMLElement>): Promise<void> {
    this.clearReadyPoll()
    this.readyPollGeneration += 1
    const generation = this.readyPollGeneration

    $root.empty().append($('<p>', { class: 'dashboard-home-loading', text: 'Loading…' }))

    try {
      const route = parseDashboardRouteHash(window.location.hash.replace(/^#/, '').trim().toLowerCase())
      const [info, ready, metrics] = await Promise.all([
        api.systemInfo(),
        fetchReadiness(),
        api
          .metricsCurrent({
            window: HOME_CHART_METRICS_LIVE_WINDOW_SEC,
            keys: homeMetricsCurrentKeys(route.homeMetricsTab),
          })
          .catch((): null => null),
      ])
      if (generation !== this.readyPollGeneration) {
        return
      }
      this.metricsChartLiveView = metrics
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
        {
          key: 'Database',
          value: dbSummary,
          readiness: { ok: this.readinessDisplayOk(ready.database, 'database'), dataKey: 'database' },
        },
        {
          key: 'Broker',
          value: rmqSummary,
          readiness: { ok: this.readinessDisplayOk(ready.rabbitmq, 'rabbitmq'), dataKey: 'rabbitmq' },
        },
        {
          key: 'Index workers',
          value: '',
          readiness: { ok: this.readinessDisplayOk(ready.index, 'index'), dataKey: 'index' },
        },
        {
          key: 'Query workers',
          value: '',
          readiness: { ok: this.readinessDisplayOk(ready.query, 'query'), dataKey: 'query' },
        },
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
            text: 'Cluster Overview',
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
            $('<div>', { class: 'dashboard-home-cluster-chart-hint-actions' }).append(
              this.buildHomeChartHistorySelect(),
              homeChartZoomResetButton('encoder'),
              clusterHelpIcon(clusterChartHelpText()),
            ),
          ),
          $('<div>', {
            class: 'dashboard-home-cluster-chart-canvas-wrap',
            attr: { 'data-home-cluster-chart-canvas-wrap': '' },
          }).append(
            $('<canvas>', {
              attr: { 'data-home-cluster-throughput-chart': '' },
              'aria-label':
                'Cluster inference and routed embedding latency per passage over the selected time range.',
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
              text: 'API Nodes',
            }),
          ),
        )
        .append(
          $('<tr>', { class: 'dashboard-home-cluster-colhead' }).append(
            $('<th>', { text: 'Role' }),
            $('<th>', { text: 'Node' }),
            clusterThWithHelp('Reqs/s', clusterApiAllReqColumnHelpText()),
            clusterThWithHelp('ms/req', clusterApiAllMsColumnHelpText()),
            clusterThWithHelp('Async/s', clusterApiAsyncReqColumnHelpText()),
            clusterThWithHelp('ms/Async', clusterApiAsyncMsColumnHelpText()),
            clusterThWithHelp('Sync/s', clusterApiSyncReqColumnHelpText()),
            clusterThWithHelp('ms/Sync', clusterApiSyncMsColumnHelpText()),
            clusterThWithHelp('Bulk/s', clusterApiBulkReqColumnHelpText()),
            clusterThWithHelp('ms/Bulk', clusterApiBulkMsColumnHelpText()),
            clusterThWithHelp('Srch/s', clusterApiSearchReqColumnHelpText()),
            clusterThWithHelp('ms/Srch', clusterApiSearchMsColumnHelpText()),
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
              $('<div>', { class: 'dashboard-home-cluster-chart-hint-actions' }).append(
                this.buildHomeChartHistorySelect(),
                homeChartZoomResetButton('api-requests'),
                clusterHelpIcon(clusterApiRequestsChartHelpText()),
              ),
            ),
            $('<div>', {
              class: 'dashboard-home-cluster-chart-canvas-wrap',
              attr: { 'data-home-api-requests-chart-canvas-wrap': '' },
            }).append(
              $('<canvas>', {
                attr: { 'data-home-api-requests-metrics-chart': '' },
                'aria-label': 'Cluster API request rate over the selected time range.',
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
              $('<div>', { class: 'dashboard-home-cluster-chart-hint-actions' }).append(
                this.buildHomeChartHistorySelect(),
                homeChartZoomResetButton('api-latencies'),
                clusterHelpIcon(clusterApiLatenciesChartHelpText()),
              ),
            ),
            $('<div>', {
              class: 'dashboard-home-cluster-chart-canvas-wrap',
              attr: { 'data-home-api-latencies-chart-canvas-wrap': '' },
            }).append(
              $('<canvas>', {
                attr: { 'data-home-api-latencies-metrics-chart': '' },
                'aria-label': 'Cluster API mean request latency over the selected time range.',
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
              text: 'Encoder Nodes',
            }),
          ),
        )
        .append(
          $('<tr>', { class: 'dashboard-home-cluster-colhead' }).append(
            $('<th>', { text: 'Role' }),
            $('<th>', { text: 'Node' }),
            $('<th>', { text: 'Models' }),
            $('<th>', { text: 'At Capacity' }),
            $('<th>', { text: 'GPU' }),
            clusterThWithHelp('Batches', clusterBatchesColumnHelpText()),
            clusterThWithHelp('Passages/s', clusterRateColumnHelpText()),
            clusterThWithHelp('Infer ms/Passage', clusterInferMsColumnHelpText()),
            clusterThWithHelp('Routed ms/Passage', clusterPipeMsColumnHelpText()),
            clusterThWithHelp('Err/s', clusterErrPerSecColumnHelpText()),
          ),
        )

      const $encoderClusterTable = $('<table>', {
        class: 'dashboard-home-table dashboard-home-cluster-table dashboard-home-cluster-table--encoders',
      }).append($encoderClusterThead, $('<tbody>', { attr: { 'data-home-cluster-tbody': '' } }))

      const CLUSTER_ENCODER_BY_MODEL_COLSPAN = 7
      const $encoderByModelThead = $('<thead>')
        .append(
          $('<tr>').append(
            $('<th>', {
              class: 'dashboard-home-table-heading',
              colspan: CLUSTER_ENCODER_BY_MODEL_COLSPAN,
              text: 'Vector Metrics',
            }),
          ),
        )
        .append(
          $('<tr>', { class: 'dashboard-home-cluster-colhead' }).append(
            $('<th>', { text: 'Type' }),
            $('<th>', { text: 'Model' }),
            clusterThWithHelp('Batches', clusterByModelBatchesHelpText()),
            clusterThWithHelp('Passages/s', clusterByModelRateHelpText()),
            clusterThWithHelp('Infer ms/Passage', clusterByModelInferMsHelpText()),
            clusterThWithHelp('Routed ms/Passage', clusterByModelRoutedMsHelpText()),
            clusterThWithHelp('Err/s', clusterByModelErrHelpText()),
          ),
        )
      const $encoderByModelTable = $('<table>', {
        class:
          'dashboard-home-table dashboard-home-cluster-table dashboard-home-cluster-table--encoder-by-model',
      }).append($encoderByModelThead, $('<tbody>', { attr: { 'data-home-encoder-by-model-tbody': '' } }))

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
      const $btnIndexingMetrics = $('<button>', {
        type: 'button',
        role: 'tab',
        class: 'dashboard-home-metrics-tab',
        id: HOME_METRICS_TAB_INDEXING_ID,
        text: 'Indexing Metrics',
        attr: {
          'aria-selected': 'false',
          'aria-controls': HOME_METRICS_PANEL_INDEXING_ID,
          'data-home-metrics-tab': 'indexing',
          tabindex: '-1',
        },
      })
      const $btnEncoderMetrics = $('<button>', {
        type: 'button',
        role: 'tab',
        class: 'dashboard-home-metrics-tab',
        id: HOME_METRICS_TAB_ENCODER_ID,
        text: 'Embedding Metrics',
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
      }).append($btnApiMetrics, $btnIndexingMetrics, $btnEncoderMetrics)

      const $apiMetricsPanel = $('<div>', {
        class: 'dashboard-home-cluster-tables dashboard-home-metrics-tab-panel',
        id: HOME_METRICS_PANEL_API_ID,
        role: 'tabpanel',
        attr: {
          'data-home-metrics-tab-panel': 'api',
          'aria-labelledby': HOME_METRICS_TAB_API_ID,
        },
      }).append($apiChartsRow, $apiClusterTable)

      const CLUSTER_INDEXING_COLSPAN = 12
      const $indexingChartSection = $('<section>', { class: 'dashboard-home-cluster-metrics' }).append(
        $('<div>', {
          class: 'dashboard-home-cluster-chart-inner',
          attr: { 'data-home-indexing-chart-wrap': '' },
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
              text: 'Indexing Latency',
            }),
            $('<div>', { class: 'dashboard-home-cluster-chart-hint-actions' }).append(
              this.buildHomeChartHistorySelect(),
              homeChartZoomResetButton('indexing'),
              clusterHelpIcon(indexingClusterChartHelpText()),
            ),
          ),
          $('<div>', {
            class: 'dashboard-home-cluster-chart-canvas-wrap',
            attr: { 'data-home-indexing-chart-canvas-wrap': '' },
          }).append(
            $('<canvas>', {
              attr: { 'data-home-indexing-latency-chart': '' },
              'aria-label':
                'Mean time per single-document indexing item vs per bulk indexing batch over the selected time range.',
            }),
          ),
          $('<ul>', {
            class: 'dashboard-home-cluster-chart-legend',
            attr: { 'data-home-indexing-chart-legend': '' },
            'aria-label': 'Indexing chart series',
          }),
        ),
      )

      const $indexingClusterThead = $('<thead>')
        .append(
          $('<tr>').append(
            $('<th>', {
              class: 'dashboard-home-table-heading',
              colspan: CLUSTER_INDEXING_COLSPAN,
              text: 'Encoder Nodes',
            }),
          ),
        )
        .append(
          $('<tr>', { class: 'dashboard-home-cluster-colhead' }).append(
            $('<th>', { text: 'Role' }),
            $('<th>', { text: 'Node' }),
            clusterThWithHelp('Docs/s', clusterIndexingDocsPerSecHelpText()),
            clusterThWithHelp('Stale/s', clusterIndexingStalePerSecHelpText()),
            clusterThWithHelp('Doc fail/s', clusterIndexingQueueFailHelpText()),
            clusterThWithHelp('Doc rq/s', clusterIndexingQueueRequeueHelpText()),
            clusterThWithHelp('Doc ms', clusterIndexingQueueJobMsHelpText()),
            clusterThWithHelp('Bulk/s', clusterIndexingBulkBatchesHelpText()),
            clusterThWithHelp('Bulk sz', clusterIndexingBulkBatchSizeHelpText()),
            clusterThWithHelp('Bulk ms', clusterIndexingBulkJobMsHelpText()),
            clusterThWithHelp('Bulk fail/s', clusterIndexingBulkFailHelpText()),
            clusterThWithHelp('Bulk rq/s', clusterIndexingBulkRequeueHelpText()),
          ),
        )

      const $indexingClusterTable = $('<table>', {
        class: 'dashboard-home-table dashboard-home-cluster-table dashboard-home-cluster-table--indexing',
      }).append($indexingClusterThead, $('<tbody>', { attr: { 'data-home-indexing-cluster-tbody': '' } }))

      const $indexingMetricsPanel = $('<div>', {
        class: 'dashboard-home-cluster-tables dashboard-home-metrics-tab-panel',
        id: HOME_METRICS_PANEL_INDEXING_ID,
        role: 'tabpanel',
        attr: {
          'data-home-metrics-tab-panel': 'indexing',
          'aria-labelledby': HOME_METRICS_TAB_INDEXING_ID,
        },
      }).append($indexingChartSection, $indexingClusterTable)

      const $encoderMetricsPanel = $('<div>', {
        class: 'dashboard-home-cluster-tables dashboard-home-metrics-tab-panel',
        id: HOME_METRICS_PANEL_ENCODER_ID,
        role: 'tabpanel',
        attr: {
          'data-home-metrics-tab-panel': 'encoder',
          'aria-labelledby': HOME_METRICS_TAB_ENCODER_ID,
        },
      }).append($clusterChartSection, $encoderClusterTable, $encoderByModelTable)

      $indexingMetricsPanel.prop('hidden', true)
      $encoderMetricsPanel.prop('hidden', true)

      const $metricsShell = $('<div>', { class: 'dashboard-home-metrics-shell' }).append(
        $metricsTabList,
        $apiMetricsPanel,
        $indexingMetricsPanel,
        $encoderMetricsPanel,
      )

      const metricsTabOrder = ['api', 'indexing', 'encoder'] as const satisfies readonly HomeMetricsTabId[]

      $btnApiMetrics.on('click', () => {
        window.location.hash = formatDashboardRouteHash('home', 'api')
      })
      $btnIndexingMetrics.on('click', () => {
        window.location.hash = formatDashboardRouteHash('home', 'indexing')
      })
      $btnEncoderMetrics.on('click', () => {
        window.location.hash = formatDashboardRouteHash('home', 'encoder')
      })
      $metricsTabList.on('keydown', (e) => {
        const key = e.key
        if (key !== 'ArrowRight' && key !== 'ArrowLeft' && key !== 'Home' && key !== 'End') {
          return
        }
        e.preventDefault()
        let current: HomeMetricsTabId = 'encoder'
        if ($btnApiMetrics.attr('aria-selected') === 'true') {
          current = 'api'
        } else if ($btnIndexingMetrics.attr('aria-selected') === 'true') {
          current = 'indexing'
        }
        const idx = metricsTabOrder.indexOf(current)
        let nextIdx: number
        if (key === 'Home') {
          nextIdx = 0
        } else if (key === 'End') {
          nextIdx = metricsTabOrder.length - 1
        } else if (key === 'ArrowRight') {
          nextIdx = (idx + 1) % metricsTabOrder.length
        } else {
          nextIdx = (idx - 1 + metricsTabOrder.length) % metricsTabOrder.length
        }
        const next = metricsTabOrder[nextIdx]!
        window.location.hash = formatDashboardRouteHash('home', next)
        $root.find(`[data-home-metrics-tab="${next}"]`).get(0)?.focus()
      })

      $root.empty().append($topBand, $metricsShell)
      if (!this.homeChartRangeListenerAttached) {
        this.homeChartRangeListenerAttached = true
        $root.on('change', '[data-home-chart-history-range]', (e) => {
          const raw = Number((e.target as HTMLSelectElement).value)
          if (!Number.isFinite(raw)) {
            return
          }
          void this.onHomeChartHistoryRangeChanged(api, $root, raw)
        })
        $root.on('click', '[data-home-chart-zoom-reset]', (e) => {
          e.preventDefault()
          const v = (e.currentTarget as HTMLElement).getAttribute('data-home-chart-zoom-reset')
          if (v == null || !isHomeChartZoomResetTarget(v)) {
            return
          }
          this.resetHomeChartZoomFor(v)
        })
      }
      this.activateHomeMetricsTab($root, route.homeMetricsTab)
      this.applyClusterViewToDom($root, metrics)
      void this.bootstrapHomeChartTrends(api, $root, route.homeMetricsTab, generation)

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

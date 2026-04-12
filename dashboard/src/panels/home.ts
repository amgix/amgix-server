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
  type Scale,
  type Tick,
} from 'chart.js'
import $ from 'jquery'

import { hideDashboardError, showDashboardError } from '../error-bar'
import { DashboardPanel } from './panel-base'

Chart.register(LineController, LineElement, PointElement, CategoryScale, LinearScale, Tooltip, Legend)

const HOME_READY_POLL_MS = 10_000

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
  return `Embed requests that started on this node, sample count in the last ${w}s.`
}

function clusterRateColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Documents embedded per second that started on this node, over the last ${w}s.`
}

function clusterInferMsColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Local model inference time per document (weighted by document count per model), over the last ${w}s.`
}

function clusterPipeMsColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `End-to-end time per document from this node (including RPC to other encoders), over the last ${w}s.`
}

function clusterErrPerSecColumnHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Failed embed requests that started on this node, per second, over the last ${w}s.`
}

function clusterChartHelpText(): string {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return `Cluster-wide inference and pipeline latency over time, per document, ${w}s rolling window.`
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

function formatLastSeen(unixSeconds: number): string {
  const delta = Math.max(0, Math.floor(Date.now() / 1000 - unixSeconds))
  return `${delta}s`
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

function formatGbForCell(n: number): string {
  return n.toFixed(1)
}

function formatRamFreeTotalGb(node: NodeView): string {
  if (node.role === 'api') {
    return ''
  }
  if (!node.load_models) {
    return '-'
  }
  const free = node.free_ram_gb
  const total = node.total_ram_gb
  if (free == null || total == null || !Number.isFinite(free) || !Number.isFinite(total)) {
    return ''
  }
  return `${formatGbForCell(free)} / ${formatGbForCell(total)}`
}

function formatVramFreeTotalGb(node: NodeView): string {
  if (node.role === 'api') {
    return ''
  }
  if (!node.load_models) {
    return '-'
  }
  const free = node.free_vram_gb
  const total = node.total_vram_gb
  if (free == null || total == null || !Number.isFinite(free) || !Number.isFinite(total)) {
    return 'n/a'
  }
  return `${formatGbForCell(free)} / ${formatGbForCell(total)}`
}

const clusterMetricsWindowKey = String(CLUSTER_METRICS_WINDOW_SEC)

function getNodeMetricWindowSample(s: NodeMetricSeries): WindowSample | undefined {
  const w = s.windows
  if (w == null) {
    return undefined
  }
  return w[clusterMetricsWindowKey] ?? w[CLUSTER_METRICS_WINDOW_SEC]
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

function sortClusterNodeEntries(nodes: { [key: string]: NodeView }): Array<[string, NodeView]> {
  const entries = Object.entries(nodes)
  entries.sort((a, b) => {
    const byRole = a[1].role.localeCompare(b[1].role, undefined, { sensitivity: 'base' })
    if (byRole !== 0) {
      return byRole
    }
    return a[0].localeCompare(b[0], undefined, { sensitivity: 'base' })
  })
  return entries
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
    this.destroyClusterThroughputChart()
    this.clusterThroughputHistory = []
    this.clusterSeriesLabels.clear()
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
      label: 'Pipeline Latency',
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

  private applyClusterViewToDom($root: JQuery<HTMLElement>, view: ClusterView | null): void {
    const $tbody = $root.find('[data-home-cluster-tbody]')
    if (!$tbody.length) {
      this.refreshClusterThroughputChart($root, view)
      return
    }
    $tbody.empty()
    const nodes = view?.nodes
    if (view == null || nodes === undefined) {
      $tbody.append(
        $('<tr>').append(
          $('<td>', {
            class: 'dashboard-home-cluster-placeholder',
            colspan: 13,
            text: 'Cluster view unavailable.',
          }),
        ),
      )
      this.refreshClusterThroughputChart($root, view)
      return
    }
    if (Object.keys(nodes).length === 0) {
      $tbody.append(
        $('<tr>').append(
          $('<td>', {
            class: 'dashboard-home-cluster-placeholder',
            colspan: 13,
            text: 'No nodes in cluster view.',
          }),
        ),
      )
      this.refreshClusterThroughputChart($root, view)
      return
    }
    for (const [hostname, node] of sortClusterNodeEntries(nodes)) {
      const m = formatEmbeddingMetricsCells(node)
      const $nodeTd = $('<td>', { class: 'dashboard-home-v' })
      if (node.is_leader) {
        $nodeTd.append($('<strong>', { text: `${hostname}*` }))
      } else {
        $nodeTd.text(hostname)
      }
      $tbody.append(
        $('<tr>').append(
          $('<td>', { class: 'dashboard-home-v', text: node.role }),
          $nodeTd,
          $('<td>', { class: 'dashboard-home-v', text: formatLoadModelsCell(node) }),
          $('<td>', { class: 'dashboard-home-v', text: node.at_capacity ? 'Yes' : 'No' }),
          $('<td>', { class: 'dashboard-home-v', text: formatGpuStatus(node) }),
          $('<td>', { class: 'dashboard-home-v', text: formatRamFreeTotalGb(node) }),
          $('<td>', { class: 'dashboard-home-v', text: formatVramFreeTotalGb(node) }),
          $('<td>', { class: 'dashboard-home-v', text: m.requests }),
          $('<td>', { class: 'dashboard-home-v', text: m.rps }),
          $('<td>', { class: 'dashboard-home-v', text: m.avgMs }),
          $('<td>', { class: 'dashboard-home-v', text: m.e2eAvgMs }),
          $('<td>', { class: 'dashboard-home-v', text: m.errPerSec }),
          $('<td>', { class: 'dashboard-home-v', text: formatLastSeen(node.last_seen) }),
        ),
      )
    }
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

      const $clusterChartSection = $('<section>', { class: 'dashboard-home-cluster-metrics' }).append(
        $('<div>', {
          class: 'dashboard-home-cluster-chart-inner',
          attr: { 'data-home-cluster-chart-wrap': '' },
        }).append(
          $('<div>', { class: 'dashboard-home-cluster-chart-hint-row' }).append(clusterHelpIcon(clusterChartHelpText())),
          $('<div>', {
            class: 'dashboard-home-cluster-chart-canvas-wrap',
            attr: { 'data-home-cluster-chart-canvas-wrap': '' },
          }).append(
            $('<canvas>', {
              attr: { 'data-home-cluster-throughput-chart': '' },
              'aria-label': `Cluster inference and pipeline latency over time, per document, ${CLUSTER_METRICS_WINDOW_SEC}s rolling window`,
            }),
          ),
          $('<ul>', {
            class: 'dashboard-home-cluster-chart-legend',
            attr: { 'data-home-cluster-chart-legend': '' },
            'aria-label': 'Chart series',
          }),
        ),
      )

      const $clusterThead = $('<thead>')
        .append(
          $('<tr>').append(
            $('<th>', {
              class: 'dashboard-home-table-heading',
              colspan: 13,
              text: 'Cluster Nodes',
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
            $('<th>', { text: 'RAM (free/total, GB)' }),
            $('<th>', { text: 'VRAM (free/total, GB)' }),
            clusterThWithHelp('Batches', clusterBatchesColumnHelpText()),
            clusterThWithHelp('Docs/s', clusterRateColumnHelpText()),
            clusterThWithHelp('Infer ms/doc', clusterInferMsColumnHelpText()),
            clusterThWithHelp('Pipe ms/doc', clusterPipeMsColumnHelpText()),
            clusterThWithHelp('Err/s', clusterErrPerSecColumnHelpText()),
            $('<th>', { text: 'Last seen' }),
          ),
        )

      const $clusterTable = $('<table>', {
        class: 'dashboard-home-table dashboard-home-cluster-table',
      }).append($clusterThead, $('<tbody>', { attr: { 'data-home-cluster-tbody': '' } }))

      const $topRow = $('<div>', { class: 'dashboard-home-top-row' }).append($systemBlock, $clusterChartSection)
      $root.empty().append($topRow, $clusterTable)
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

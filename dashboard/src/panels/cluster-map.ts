import type { AmgixApi, Metrics, NodeMetricSeries, NodeView, SystemInfoResponse } from '@amgix/amgix-client'
import cytoscape, { type Core, type EdgeSingular, type ElementDefinition } from 'cytoscape'
import nodeHtmlLabel from 'cytoscape-node-html-label'
import $ from 'jquery'

import { DashboardPanel } from './panel-base'

nodeHtmlLabel(cytoscape)

const CLUSTER_MAP_POLL_MS = 10_000

/** Must match the `window` passed to GET /metrics/current for this panel. */
const CLUSTER_MAP_METRICS_CURRENT_KEYS = [
  'api_requests',
  'api_request_ms',
  'api_error_4xx',
  'api_error_5xx',
  'embed_passages',
  'embed_inference_ms',
] as const
const CLUSTER_MAP_ZOOM_STEP = 1.15
const CLUSTER_MAP_MIN_SCALE = 0.5
const CLUSTER_MAP_MAX_SCALE = 4
const CLUSTER_MAP_FIT_PADDING = 40
const CLUSTER_MAP_METRIC_WINDOW_SEC = 60

const CLUSTER_MAP_BROKER_NODE_ID = 'broker_rmq'
const CLUSTER_MAP_DB_NODE_ID = 'cluster_db'
const CLUSTER_MAP_DB_LABEL_FALLBACK = 'Database'
const CLUSTER_MAP_TITLE_API_ID = 'title_api_nodes'
const CLUSTER_MAP_GROUP_TITLE_WIDTH = 200
const CLUSTER_MAP_GROUP_TITLE_HEIGHT = 26
const CLUSTER_MAP_GROUP_TITLE_GAP = 12

const CLUSTER_MAP_API_ROW_Y = 0
/** API nodes: max per horizontal row (encoder-style grid uses3 per row). */
const CLUSTER_MAP_API_MAX_PER_ROW = 5
/** Min space from bottom of API block to RabbitMQ node center when the broker is shown. */
const CLUSTER_MAP_API_BROKER_GAP = 48
const CLUSTER_MAP_BROKER_Y = 170
const CLUSTER_MAP_SECTION_GAP_Y = 170
const CLUSTER_MAP_GROUP_GAP_Y = 175
const CLUSTER_MAP_ROW_GAP_Y = 120
const CLUSTER_MAP_COLUMN_GAP_X = 230
/** Min horizontal space between the closest edges of nodes in adjacent encoder-kind columns. */
const CLUSTER_MAP_ENCODER_KIND_COLUMN_CLEAR_X = 72

const CLUSTER_MAP_NODE_WIDTH = 184
const CLUSTER_MAP_NODE_HEIGHT = 88
const CLUSTER_MAP_COMPACT_NODE_WIDTH = 178
const CLUSTER_MAP_COMPACT_NODE_HEIGHT = 56
const CLUSTER_MAP_PLACEHOLDER_WIDTH = 300
const CLUSTER_MAP_PLACEHOLDER_HEIGHT = 72

const CLUSTER_MAP_NODE_API_FILL = '#4f3a78'
const CLUSTER_MAP_NODE_API_STROKE = '#cdb4fc'
const CLUSTER_MAP_NODE_ENCODER_INDEX_FILL = '#3d2848'
const CLUSTER_MAP_NODE_ENCODER_INDEX_STROKE = '#e0a8ec'
const CLUSTER_MAP_NODE_ENCODER_QUERY_FILL = '#172f45'
const CLUSTER_MAP_NODE_ENCODER_QUERY_STROKE = '#6ec8f0'
const CLUSTER_MAP_NODE_ENCODER_ALL_FILL = '#24244a'
const CLUSTER_MAP_NODE_ENCODER_ALL_STROKE = '#a8b4f5'
/** Muted from RabbitMQ brand orange (#ff6600) for dark diagram background. */
const CLUSTER_MAP_NODE_BROKER_FILL = '#5a3018'
const CLUSTER_MAP_NODE_BROKER_STROKE = '#c96d3d'
const CLUSTER_MAP_NODE_DB_FILL = '#263629'
const CLUSTER_MAP_NODE_DB_STROKE = '#8cc28d'
const CLUSTER_MAP_NODE_PLACEHOLDER_FILL = '#1e2129'
const CLUSTER_MAP_NODE_PLACEHOLDER_STROKE = '#585c68'
const CLUSTER_MAP_EDGE_COLOR = '#585c68'
/** Per-edge spread for hub routes (unbundled-bezier control-point-distance, px). */
const CLUSTER_MAP_EDGE_FAN_STEP = 38
const CLUSTER_MAP_NODE_STROKE_WIDTH = 2
const CLUSTER_MAP_NODE_HEAT_STROKE_WIDTH = 3
const API_LATENCY_HEAT_RATIO_CAP = 3
const API_LATENCY_HEAT_COOL_RGB = { r: 46, g: 125, b: 50 }
const API_LATENCY_HEAT_HOT_RGB = { r: 198, g: 40, b: 40 }

type EncoderKind = 'index' | 'query' | 'all'
type ClusterMapNodeBucket = 'api' | EncoderKind
type ClusterMapPosition = { x: number; y: number }
type CytoscapeWithNodeHtmlLabel = Core & {
  nodeHtmlLabel(
    params: Array<{
      query?: string
      halign?: 'left' | 'center' | 'right'
      valign?: 'top' | 'center' | 'bottom'
      halignBox?: 'left' | 'center' | 'right'
      valignBox?: 'top' | 'center' | 'bottom'
      cssClass?: string
      tpl?: (data: { labelHtml?: string }) => string
    }>,
    options?: { enablePointerEvents?: boolean },
  ): Core
}

type ClusterMapElementData = {
  id: string
  source?: string
  target?: string
  label?: string
  labelHtml?: string
  /** Unbundled-bezier: perpendicular offset of control point from source–target line. */
  cpd?: number
  /** Unbundled-bezier: control point position along edge (0–1). */
  cpw?: number
}

function encoderKindGroupTitle(kind: EncoderKind): string {
  switch (kind) {
    case 'index':
      return 'Index Nodes'
    case 'query':
      return 'Query Nodes'
    case 'all':
      return 'Index/Query Nodes'
  }
}

/** Y center for a group title node placed above a row whose node centers sit at `rowCenterY`. */
function groupTitleCenterY(rowCenterY: number): number {
  return (
    rowCenterY -
    CLUSTER_MAP_NODE_HEIGHT / 2 -
    CLUSTER_MAP_GROUP_TITLE_GAP -
    CLUSTER_MAP_GROUP_TITLE_HEIGHT / 2
  )
}

function groupTitleNode(id: string, title: string, position: ClusterMapPosition): ElementDefinition {
  const labelHtml = `<span class="dashboard-cluster-map-group-title">${escapeHtmlText(title)}</span>`
  return htmlNode(
    id,
    labelHtml,
    position,
    'rgba(0,0,0,0)',
    'rgba(0,0,0,0)',
    0,
    CLUSTER_MAP_GROUP_TITLE_WIDTH,
    CLUSTER_MAP_GROUP_TITLE_HEIGHT,
    'map-node html-node group-title-node',
  )
}

function encoderNodeBaseStyle(kind: EncoderKind): { fill: string; stroke: string } {
  switch (kind) {
    case 'index':
      return { fill: CLUSTER_MAP_NODE_ENCODER_INDEX_FILL, stroke: CLUSTER_MAP_NODE_ENCODER_INDEX_STROKE }
    case 'query':
      return { fill: CLUSTER_MAP_NODE_ENCODER_QUERY_FILL, stroke: CLUSTER_MAP_NODE_ENCODER_QUERY_STROKE }
    case 'all':
      return { fill: CLUSTER_MAP_NODE_ENCODER_ALL_FILL, stroke: CLUSTER_MAP_NODE_ENCODER_ALL_STROKE }
  }
}

function windowSampleAtSec(
  windows: NodeMetricSeries['windows'] | undefined,
  sec: number,
): { value: number; n: number } | null {
  if (windows == null) {
    return null
  }
  const sample = windows[String(sec) as keyof typeof windows]
  if (sample == null) {
    return null
  }
  if (sample.n != null && sample.n <= 0) {
    return null
  }
  return { value: sample.value, n: sample.n ?? 0 }
}

function ratioToLatencyHeatStroke(ratio: number): string {
  const t = Math.max(
    0,
    Math.min(
      1,
      (Math.min(ratio, API_LATENCY_HEAT_RATIO_CAP) - 1) / (API_LATENCY_HEAT_RATIO_CAP - 1),
    ),
  )
  const a = API_LATENCY_HEAT_COOL_RGB
  const b = API_LATENCY_HEAT_HOT_RGB
  const r = Math.round(a.r + t * (b.r - a.r))
  const g = Math.round(a.g + t * (b.g - a.g))
  const bl = Math.round(a.b + t * (b.b - a.b))
  const hex = (n: number) => n.toString(16).padStart(2, '0')
  return `#${hex(r)}${hex(g)}${hex(bl)}`
}

function sanitizeClusterMapNodeId(value: string): string {
  return `n_${value.replace(/[^a-zA-Z0-9_]/g, '_')}`
}

function escapeHtmlText(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function formatClusterMapAvgMs(ms: number): string {
  if (!Number.isFinite(ms)) {
    return ''
  }
  return new Intl.NumberFormat(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 2 }).format(ms)
}

function formatClusterMapRps(rps: number): string {
  if (!Number.isFinite(rps)) {
    return ''
  }
  return new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(rps)
}

function clusterMapFirstWindowSampleForMetric(
  node: NodeView,
  metricName: string,
): { value: number; n: number } | null {
  const metrics = node.metrics
  if (metrics == null || metrics.length === 0) {
    return null
  }
  for (const series of metrics) {
    if (series.key !== metricName) {
      continue
    }
    const sample = windowSampleAtSec(series.windows, CLUSTER_MAP_METRIC_WINDOW_SEC)
    if (sample != null) {
      return sample
    }
  }
  return null
}

function roleMaterialLigature(bucket: ClusterMapNodeBucket): string {
  if (bucket === 'api') {
    return 'api'
  }
  switch (bucket) {
    case 'index':
      return 'database'
    case 'query':
      return 'search'
    case 'all':
      return 'layers'
  }
}

function apiClusterMapRequestsRateText(node: NodeView): string {
  const sample = clusterMapFirstWindowSampleForMetric(node, 'api_requests')
  if (sample == null || !Number.isFinite(sample.value)) {
    return ''
  }
  return formatClusterMapRps(sample.value / CLUSTER_MAP_METRIC_WINDOW_SEC)
}

function apiClusterMapErrorsRateText(node: NodeView): string {
  let sum = 0
  let saw = false
  for (const metricName of ['api_error_4xx', 'api_error_5xx'] as const) {
    const sample = clusterMapFirstWindowSampleForMetric(node, metricName)
    if (sample != null && Number.isFinite(sample.value)) {
      sum += sample.value
      saw = true
    }
  }
  if (!saw) {
    return ''
  }
  return formatClusterMapRps(sum / CLUSTER_MAP_METRIC_WINDOW_SEC)
}

function encoderClusterMapDocsRateText(node: NodeView): string {
  let sum = 0
  let saw = false
  const metrics = node.metrics
  if (metrics != null) {
    for (const series of metrics) {
      if (series.key !== 'embed_passages') {
        continue
      }
      const sample = windowSampleAtSec(series.windows, CLUSTER_MAP_METRIC_WINDOW_SEC)
      if (sample != null && Number.isFinite(sample.value)) {
        sum += sample.value
        saw = true
      }
    }
  }
  if (!saw) {
    return ''
  }
  return formatClusterMapRps(sum / CLUSTER_MAP_METRIC_WINDOW_SEC)
}

function buildNodeRow(iconName: string, caption: string): string {
  const icon = `<span class='material-symbols-outlined dashboard-cluster-map-node-icon' aria-hidden='true'>${iconName}</span>`
  const text = `<span class='dashboard-cluster-map-node-caption'>${escapeHtmlText(caption)}</span>`
  return `<span class='dashboard-cluster-map-node-row'>${icon}${text}</span>`
}

function buildMaterialIconNodeLabel(host: string, node: NodeView, bucket: ClusterMapNodeBucket): string {
  const row = buildNodeRow(roleMaterialLigature(bucket), host)
  const latMs = nodeLocalAvgLatencyMsForMap(node, bucket)
  let latencyLine = `<span class='dashboard-cluster-map-node-latency dashboard-cluster-map-node-latency--empty'>— ${bucket === 'api' ? 'ms avg' : 'ms/passage'}</span>`
  if (latMs != null && Number.isFinite(latMs)) {
    const latText = formatClusterMapAvgMs(latMs)
    if (latText !== '') {
      latencyLine = `<span class='dashboard-cluster-map-node-latency'>${escapeHtmlText(latText)} ${bucket === 'api' ? 'ms avg' : 'ms/passage'}</span>`
    }
  }

  if (bucket === 'api') {
    const reqText = apiClusterMapRequestsRateText(node)
    const errText = apiClusterMapErrorsRateText(node)
    const reqLine =
      reqText !== ''
        ? `<span class='dashboard-cluster-map-node-latency'>${escapeHtmlText(reqText)} reqs/s</span>`
        : `<span class='dashboard-cluster-map-node-latency dashboard-cluster-map-node-latency--empty'>— reqs/s</span>`
    const errLine =
      errText !== ''
        ? `<span class='dashboard-cluster-map-node-latency'>${escapeHtmlText(errText)} err/s</span>`
        : `<span class='dashboard-cluster-map-node-latency dashboard-cluster-map-node-latency--empty'>— err/s</span>`
    return `<span class='dashboard-cluster-map-node-label'>${row}${latencyLine}${reqLine}${errLine}</span>`
  }

  const docText = encoderClusterMapDocsRateText(node)
  const docLine =
    docText !== ''
      ? `<span class='dashboard-cluster-map-node-latency'>${escapeHtmlText(docText)} passage/s</span>`
      : `<span class='dashboard-cluster-map-node-latency dashboard-cluster-map-node-latency--empty'>— passage/s</span>`
  return `<span class='dashboard-cluster-map-node-label'>${row}${latencyLine}${docLine}</span>`
}

function buildCompactNodeLabel(iconName: string, caption: string): string {
  return `<span class='dashboard-cluster-map-node-label dashboard-cluster-map-node-label--compact'>${buildNodeRow(iconName, caption)}</span>`
}

function encoderKindFromRole(role: string): EncoderKind {
  if (role === 'index' || role === 'query' || role === 'all') {
    return role
  }
  return 'all'
}

function partitionClusterNodes(nodes: { [key: string]: NodeView }): {
  apiEntries: Array<[string, NodeView]>
  encoderByKind: Record<EncoderKind, Array<[string, NodeView]>>
} {
  const apiEntries: Array<[string, NodeView]> = []
  const encoderByKind: Record<EncoderKind, Array<[string, NodeView]>> = {
    index: [],
    query: [],
    all: [],
  }
  for (const [host, node] of Object.entries(nodes)) {
    if (node.role === 'api') {
      apiEntries.push([host, node])
    } else {
      encoderByKind[encoderKindFromRole(node.role)].push([host, node])
    }
  }
  apiEntries.sort((a, b) => a[0].localeCompare(b[0], undefined, { sensitivity: 'base' }))
  for (const kind of Object.keys(encoderByKind) as EncoderKind[]) {
    encoderByKind[kind].sort((a, b) => a[0].localeCompare(b[0], undefined, { sensitivity: 'base' }))
  }
  return { apiEntries, encoderByKind }
}

function encoderEntriesInEdgeOrder(
  encoderByKind: Record<EncoderKind, Array<[string, NodeView]>>,
): Array<[string, NodeView]> {
  return [...encoderByKind.index, ...encoderByKind.query, ...encoderByKind.all]
}

function apiRequestAvgMs(node: NodeView): number | null {
  const metrics = node.metrics
  if (metrics == null || metrics.length === 0) {
    return null
  }
  for (const series of metrics) {
    if (series.key !== 'api_request_ms') {
      continue
    }
    const sample = windowSampleAtSec(series.windows, CLUSTER_MAP_METRIC_WINDOW_SEC)
    if (sample != null && sample.n > 0) {
      return sample.value / sample.n
    }
  }
  return null
}

function apiLatencyHeatStrokeByHost(apiEntries: Array<[string, NodeView]>): Map<string, string> {
  const out = new Map<string, string>()
  const samples: Array<{ host: string; value: number }> = []
  for (const [host, node] of apiEntries) {
    const avgMs = apiRequestAvgMs(node)
    if (avgMs != null) {
      samples.push({ host, value: avgMs })
    }
  }
  if (samples.length === 0) {
    return out
  }
  const mean = samples.reduce((sum, sample) => sum + sample.value, 0) / samples.length
  if (mean <= 0) {
    return out
  }
  for (const { host, value } of samples) {
    out.set(host, ratioToLatencyHeatStroke(value / mean))
  }
  return out
}

function encoderLocalDocLatencyMs(node: NodeView): number | null {
  const metrics = node.metrics
  if (metrics == null || metrics.length === 0) {
    return null
  }
  let totalMs = 0
  let totalPassages = 0
  for (const series of metrics) {
    const sample = windowSampleAtSec(series.windows, CLUSTER_MAP_METRIC_WINDOW_SEC)
    if (sample == null) {
      continue
    }
    if (series.key === 'embed_inference_ms') {
      totalMs += sample.value
    } else if (series.key === 'embed_passages') {
      totalPassages += sample.value
    }
  }
  return totalPassages > 0 ? totalMs / totalPassages : null
}

function nodeLocalAvgLatencyMsForMap(node: NodeView, bucket: ClusterMapNodeBucket): number | null {
  if (bucket === 'api') {
    return apiRequestAvgMs(node)
  }
  return encoderLocalDocLatencyMs(node)
}

function encoderDocLatencyHeatForGroup(encoderEntries: Array<[string, NodeView]>): Map<string, string> {
  const out = new Map<string, string>()
  const samples: Array<{ host: string; value: number }> = []
  for (const [host, node] of encoderEntries) {
    const value = encoderLocalDocLatencyMs(node)
    if (value != null) {
      samples.push({ host, value })
    }
  }
  if (samples.length === 0) {
    return out
  }
  const mean = samples.reduce((sum, sample) => sum + sample.value, 0) / samples.length
  if (mean <= 0) {
    return out
  }
  for (const { host, value } of samples) {
    out.set(host, ratioToLatencyHeatStroke(value / mean))
  }
  return out
}

function encoderDocLatencyHeatByKind(
  encoderByKind: Record<EncoderKind, Array<[string, NodeView]>>,
): Map<string, string> {
  const out = new Map<string, string>()
  for (const kind of ['index', 'query', 'all'] as const) {
    for (const [host, stroke] of encoderDocLatencyHeatForGroup(encoderByKind[kind])) {
      out.set(host, stroke)
    }
  }
  return out
}

function centeredPositions(count: number, gap: number): number[] {
  if (count <= 0) {
    return []
  }
  const start = -((count - 1) * gap) / 2
  return Array.from({ length: count }, (_, index) => start + index * gap)
}

/** Rows of up to `maxPerRow` nodes, centered horizontally within each row. */
function groupGridNodePositions(count: number, baseY: number, maxPerRow: number): ClusterMapPosition[] {
  if (count <= 0) {
    return []
  }
  const gap = CLUSTER_MAP_COLUMN_GAP_X
  const rowGap = CLUSTER_MAP_ROW_GAP_Y
  const out: ClusterMapPosition[] = []
  let index = 0
  let y = baseY
  while (index < count) {
    const inRow = Math.min(maxPerRow, count - index)
    const xs = centeredPositions(inRow, gap)
    for (let column = 0; column < inRow; column++) {
      out.push({ x: xs[column] ?? 0, y })
    }
    index += inRow
    if (index < count) {
      y += rowGap
    }
  }
  return out
}

/** Center-to-center X distance between encoder-kind columns (avoids overlap for up to 3 nodes wide per kind). */
function encoderKindColumnsCenterGap(): number {
  const maxCentersFromColumnAxis = CLUSTER_MAP_COLUMN_GAP_X
  const halfNode = CLUSTER_MAP_NODE_WIDTH / 2
  const halfSpan = maxCentersFromColumnAxis + halfNode
  return 2 * halfSpan + CLUSTER_MAP_ENCODER_KIND_COLUMN_CLEAR_X
}

function fanControlPointDistance(edgeIndex: number, edgeCount: number, step: number): number {
  if (edgeCount <= 1) {
    return 0
  }
  const mid = (edgeCount - 1) / 2
  return (edgeIndex - mid) * step
}

/** Vertical span (delta from first row to last) for one encoder subgroup. */
function encoderGroupHeight(count: number): number {
  if (count <= 0) {
    return 0
  }
  if (count === 3) {
    return CLUSTER_MAP_ROW_GAP_Y
  }
  const rows = Math.ceil(count / 3)
  return (rows - 1) * CLUSTER_MAP_ROW_GAP_Y
}

/**
 * 1: center. 2: side by side. 3: two on top, one centered below.
 * 4+: rows of up to 3, horizontal within each row (same y for that row).
 */
function encoderGroupNodePositions(count: number, baseY: number): ClusterMapPosition[] {
  if (count <= 0) {
    return []
  }
  const gap = CLUSTER_MAP_COLUMN_GAP_X
  const rowGap = CLUSTER_MAP_ROW_GAP_Y
  if (count === 1) {
    return [{ x: 0, y: baseY }]
  }
  if (count === 2) {
    const xs = centeredPositions(2, gap)
    return [
      { x: xs[0] ?? 0, y: baseY },
      { x: xs[1] ?? 0, y: baseY },
    ]
  }
  if (count === 3) {
    const topXs = centeredPositions(2, gap)
    return [
      { x: topXs[0] ?? 0, y: baseY },
      { x: topXs[1] ?? 0, y: baseY },
      { x: 0, y: baseY + rowGap },
    ]
  }
  const out: ClusterMapPosition[] = []
  let index = 0
  let y = baseY
  while (index < count) {
    const inRow = Math.min(3, count - index)
    const xs = centeredPositions(inRow, gap)
    for (let column = 0; column < inRow; column++) {
      out.push({ x: xs[column] ?? 0, y })
    }
    index += inRow
    if (index < count) {
      y += rowGap
    }
  }
  return out
}

function htmlNode(
  id: string,
  labelHtml: string,
  position: ClusterMapPosition,
  fillColor: string,
  borderColor: string,
  borderWidth: number,
  width: number,
  height: number,
  classes: string,
): ElementDefinition {
  return {
    group: 'nodes',
    data: { id, labelHtml } satisfies ClusterMapElementData,
    position,
    classes,
    style: {
      width,
      height,
      'background-color': fillColor,
      'border-color': borderColor,
      'border-width': borderWidth,
    },
  }
}

function placeholderNode(label: string): ElementDefinition {
  return {
    group: 'nodes',
    data: { id: 'placeholder', label } satisfies ClusterMapElementData,
    position: { x: 0, y: 0 },
    classes: 'placeholder-node',
    style: {
      width: CLUSTER_MAP_PLACEHOLDER_WIDTH,
      height: CLUSTER_MAP_PLACEHOLDER_HEIGHT,
      'background-color': CLUSTER_MAP_NODE_PLACEHOLDER_FILL,
      'border-color': CLUSTER_MAP_NODE_PLACEHOLDER_STROKE,
      'border-width': CLUSTER_MAP_NODE_STROKE_WIDTH,
    },
  }
}

function edgeElement(
  id: string,
  source: string,
  target: string,
  lineColor: string,
  lineWidth: number,
  bezierFan?: { index: number; count: number },
): ElementDefinition {
  const cpd =
    bezierFan != null
      ? -fanControlPointDistance(bezierFan.index, bezierFan.count, CLUSTER_MAP_EDGE_FAN_STEP)
      : undefined
  const data: ClusterMapElementData =
    cpd != null && cpd !== 0 ? { id, source, target, cpd, cpw: 0.5 } : { id, source, target }
  return {
    group: 'edges',
    data,
    style: {
      width: lineWidth,
      'line-color': lineColor,
    },
  }
}

function buildClusterGraph(view: Metrics | null, systemInfo: SystemInfoResponse | null): ElementDefinition[] {
  const nodes = view?.nodes
  if (nodes == null) {
    return [placeholderNode('Metrics unavailable')]
  }
  if (Object.keys(nodes).length === 0) {
    return [placeholderNode('No nodes reported')]
  }

  const { apiEntries, encoderByKind } = partitionClusterNodes(nodes)
  const encodersAll = encoderEntriesInEdgeOrder(encoderByKind)
  if (apiEntries.length === 0 && encodersAll.length === 0) {
    return [placeholderNode('No nodes reported')]
  }

  const apiHeat = apiLatencyHeatStrokeByHost(apiEntries)
  const encoderHeat = encoderDocLatencyHeatByKind(encoderByKind)
  const elements: ElementDefinition[] = []

  const hasBroker = apiEntries.length > 0 && encodersAll.length > 0
  let brokerY = CLUSTER_MAP_BROKER_Y

  if (apiEntries.length > 0) {
    const apiPositions = groupGridNodePositions(
      apiEntries.length,
      CLUSTER_MAP_API_ROW_Y,
      CLUSTER_MAP_API_MAX_PER_ROW,
    )
    const maxApiY =
      apiPositions.length > 0 ? Math.max(...apiPositions.map((p) => p.y)) : CLUSTER_MAP_API_ROW_Y
    const apiBlockBottom = maxApiY + CLUSTER_MAP_NODE_HEIGHT / 2
    if (hasBroker) {
      brokerY = Math.max(
        CLUSTER_MAP_BROKER_Y,
        apiBlockBottom + CLUSTER_MAP_API_BROKER_GAP + CLUSTER_MAP_COMPACT_NODE_HEIGHT / 2,
      )
    }

    apiEntries.forEach(([host, node], index) => {
      const pos = apiPositions[index]
      if (pos == null) {
        return
      }
      const nodeId = sanitizeClusterMapNodeId(host)
      const heatStroke = apiHeat.get(host) ?? CLUSTER_MAP_NODE_API_STROKE
      const borderWidth =
        apiHeat.has(host) ? CLUSTER_MAP_NODE_HEAT_STROKE_WIDTH : CLUSTER_MAP_NODE_STROKE_WIDTH
      elements.push(
        htmlNode(
          nodeId,
          buildMaterialIconNodeLabel(host, node, 'api'),
          pos,
          CLUSTER_MAP_NODE_API_FILL,
          heatStroke,
          borderWidth,
          CLUSTER_MAP_NODE_WIDTH,
          CLUSTER_MAP_NODE_HEIGHT,
          'map-node html-node api-node',
        ),
      )
    })
    elements.push(
      groupTitleNode(CLUSTER_MAP_TITLE_API_ID, 'API Nodes', {
        x: 0,
        y: groupTitleCenterY(CLUSTER_MAP_API_ROW_Y),
      }),
    )
  }

  if (hasBroker) {
    elements.push(
      htmlNode(
        CLUSTER_MAP_BROKER_NODE_ID,
        buildCompactNodeLabel('device_hub', 'RabbitMQ'),
        { x: 0, y: brokerY },
        CLUSTER_MAP_NODE_BROKER_FILL,
        CLUSTER_MAP_NODE_BROKER_STROKE,
        CLUSTER_MAP_NODE_STROKE_WIDTH,
        CLUSTER_MAP_COMPACT_NODE_WIDTH,
        CLUSTER_MAP_COMPACT_NODE_HEIGHT,
        'map-node html-node broker-node compact-node',
      ),
    )
  }

  const encoderKindsPresent = (['index', 'query', 'all'] as const).filter((k) => encoderByKind[k].length > 0)
  let currentGroupY = hasBroker ? brokerY + CLUSTER_MAP_SECTION_GAP_Y : CLUSTER_MAP_API_ROW_Y
  if (encoderKindsPresent.length > 0) {
    const encoderRowBaseY = currentGroupY
    const columnCenters = centeredPositions(encoderKindsPresent.length, encoderKindColumnsCenterGap())
    let maxEncoderBlockHeight = 0
    encoderKindsPresent.forEach((kind, columnIndex) => {
      const entries = encoderByKind[kind]
      const columnCenterX = columnCenters[columnIndex] ?? 0
      elements.push(
        groupTitleNode(
          `title_encoder_${kind}`,
          encoderKindGroupTitle(kind),
          { x: columnCenterX, y: groupTitleCenterY(encoderRowBaseY) },
        ),
      )
      const positions = encoderGroupNodePositions(entries.length, encoderRowBaseY)
      maxEncoderBlockHeight = Math.max(maxEncoderBlockHeight, encoderGroupHeight(entries.length))
      entries.forEach(([host, node], index) => {
        const local = positions[index]
        if (local == null) {
          return
        }
        const pos = { x: local.x + columnCenterX, y: local.y }
        const nodeId = sanitizeClusterMapNodeId(host)
        const { fill, stroke } = encoderNodeBaseStyle(kind)
        const heatStroke = encoderHeat.get(host) ?? stroke
        const borderWidth =
          encoderHeat.has(host) ? CLUSTER_MAP_NODE_HEAT_STROKE_WIDTH : CLUSTER_MAP_NODE_STROKE_WIDTH
        elements.push(
          htmlNode(
            nodeId,
            buildMaterialIconNodeLabel(host, node, kind),
            pos,
            fill,
            heatStroke,
            borderWidth,
            CLUSTER_MAP_NODE_WIDTH,
            CLUSTER_MAP_NODE_HEIGHT,
            `map-node html-node encoder-node encoder-node-${kind}`,
          ),
        )
      })
    })
    currentGroupY = encoderRowBaseY + maxEncoderBlockHeight + CLUSTER_MAP_GROUP_GAP_Y
  }

  if (encodersAll.length > 0) {
    const dbKindRaw = systemInfo?.database_kind?.trim() ?? ''
    const dbKind = dbKindRaw.length > 0 ? dbKindRaw : CLUSTER_MAP_DB_LABEL_FALLBACK
    elements.push(
      htmlNode(
        CLUSTER_MAP_DB_NODE_ID,
        buildCompactNodeLabel('storage', dbKind),
        { x: 0, y: currentGroupY },
        CLUSTER_MAP_NODE_DB_FILL,
        CLUSTER_MAP_NODE_DB_STROKE,
        CLUSTER_MAP_NODE_STROKE_WIDTH,
        CLUSTER_MAP_COMPACT_NODE_WIDTH,
        CLUSTER_MAP_COMPACT_NODE_HEIGHT,
        'map-node html-node db-node compact-node',
      ),
    )
  }

  if (hasBroker) {
    const apiFan = { count: apiEntries.length }
    for (let i = 0; i < apiEntries.length; i++) {
      const [host] = apiEntries[i]!
      const nodeId = sanitizeClusterMapNodeId(host)
      const stroke = apiHeat.get(host) ?? CLUSTER_MAP_EDGE_COLOR
      const lineWidth = apiHeat.has(host) ? CLUSTER_MAP_NODE_HEAT_STROKE_WIDTH : CLUSTER_MAP_NODE_STROKE_WIDTH
      elements.push(
        edgeElement(
          `edge_${nodeId}_${CLUSTER_MAP_BROKER_NODE_ID}`,
          nodeId,
          CLUSTER_MAP_BROKER_NODE_ID,
          stroke,
          lineWidth,
          { index: i, count: apiFan.count },
        ),
      )
    }
    const encBrokerFan = { count: encodersAll.length }
    for (let i = 0; i < encodersAll.length; i++) {
      const [host] = encodersAll[i]!
      const nodeId = sanitizeClusterMapNodeId(host)
      const stroke = encoderHeat.get(host) ?? CLUSTER_MAP_EDGE_COLOR
      const lineWidth =
        encoderHeat.has(host) ? CLUSTER_MAP_NODE_HEAT_STROKE_WIDTH : CLUSTER_MAP_NODE_STROKE_WIDTH
      elements.push(
        edgeElement(
          `edge_${CLUSTER_MAP_BROKER_NODE_ID}_${nodeId}`,
          CLUSTER_MAP_BROKER_NODE_ID,
          nodeId,
          stroke,
          lineWidth,
          { index: i, count: encBrokerFan.count },
        ),
      )
    }
  }

  if (encodersAll.length > 0) {
    const encDbFan = { count: encodersAll.length }
    for (let i = 0; i < encodersAll.length; i++) {
      const [host] = encodersAll[i]!
      const nodeId = sanitizeClusterMapNodeId(host)
      const stroke = encoderHeat.get(host) ?? CLUSTER_MAP_EDGE_COLOR
      const lineWidth =
        encoderHeat.has(host) ? CLUSTER_MAP_NODE_HEAT_STROKE_WIDTH : CLUSTER_MAP_NODE_STROKE_WIDTH
      elements.push(
        edgeElement(
          `edge_${nodeId}_${CLUSTER_MAP_DB_NODE_ID}`,
          nodeId,
          CLUSTER_MAP_DB_NODE_ID,
          stroke,
          lineWidth,
          { index: i, count: encDbFan.count },
        ),
      )
    }
  }

  return elements
}

/** Stable identity of the graph shape: node ids + edge ids (order-independent). */
function clusterMapTopologySignature(elements: ElementDefinition[]): string {
  const nodeIds: string[] = []
  const edgeIds: string[] = []
  for (const el of elements) {
    if (el.group === 'nodes') {
      const id = el.data != null && 'id' in el.data ? String((el.data as { id: string }).id) : ''
      if (id !== '') {
        nodeIds.push(id)
      }
    } else if (el.group === 'edges') {
      const id = el.data != null && 'id' in el.data ? String((el.data as { id: string }).id) : ''
      if (id !== '') {
        edgeIds.push(id)
      }
    }
  }
  nodeIds.sort()
  edgeIds.sort()
  return `${nodeIds.join('\0')}|${edgeIds.join('\0')}`
}

function clusterMapEdgeCpDistance(ele: EdgeSingular): number {
  const v = ele.data('cpd')
  return typeof v === 'number' && Number.isFinite(v) ? v : 0
}

function clusterMapEdgeCpWeight(ele: EdgeSingular): number {
  const v = ele.data('cpw')
  return typeof v === 'number' && Number.isFinite(v) ? v : 0.5
}

export class ClusterMapPanel extends DashboardPanel {
  private pollTimer: number | null = null
  private pollGeneration = 0
  private cy: Core | null = null
  /** Last applied topology; used to decide fit vs preserve pan/zoom on poll. */
  private clusterMapTopologySignature = ''
  private windowResizeHandler: (() => void) | null = null
  private zoomToolHandlersBound = false

  override deactivate(): void {
    this.clearPoll()
    this.pollGeneration += 1
    this.clusterMapTopologySignature = ''
    this.destroyGraph()
  }

  init(api: AmgixApi): void {
    const $root = $('#panel-cluster-map [data-cluster-map-root]')
    if (!$root.length) {
      return
    }
    this.clearPoll()
    this.pollGeneration += 1
    const generation = this.pollGeneration
    this.ensureShell($root)
    this.ensureZoomTools($root)
    this.applyDiagramHeight($root)
    this.attachWindowResizeHandler($root)
    this.bindZoomToolHandlers($root)
    void this.loadOnceThenPoll(api, $root, generation)
  }

  private clearPoll(): void {
    if (this.pollTimer != null) {
      window.clearInterval(this.pollTimer)
      this.pollTimer = null
    }
    this.detachWindowResizeHandler()
  }

  private detachWindowResizeHandler(): void {
    if (this.windowResizeHandler != null) {
      $(window).off('resize', this.windowResizeHandler)
      this.windowResizeHandler = null
    }
  }

  private attachWindowResizeHandler($root: JQuery<HTMLElement>): void {
    this.detachWindowResizeHandler()
    this.windowResizeHandler = () => {
      this.applyDiagramHeight($root)
      this.resizeGraph()
    }
    $(window).on('resize', this.windowResizeHandler)
  }

  private computeDiagramHeight($root: JQuery<HTMLElement>): number {
    const minHeight = 240
    const elHeight = ($el: JQuery): number => $el.get(0)?.getBoundingClientRect().height ?? 0
    const padV = ($el: JQuery): number =>
      (parseFloat($el.css('paddingTop')) || 0) + (parseFloat($el.css('paddingBottom')) || 0)
    const borderV = ($el: JQuery): number =>
      (parseFloat($el.css('borderTopWidth')) || 0) + (parseFloat($el.css('borderBottomWidth')) || 0)

    const topBarH = elHeight($('#dashboard-top-bar'))
    const $errorBar = $('#dashboard-error-bar')
    const errorBarH = !$errorBar.prop('hidden') ? elHeight($errorBar) : 0
    const footerH = elHeight($('.dashboard-site-footer'))
    const panelsPadH = padV($('#dashboard-panels'))
    const mapPadH = padV($root)

    const $heading = $root.find('.dashboard-cluster-map-heading')
    const headingH =
      $heading.length > 0
        ? ($heading.get(0)?.getBoundingClientRect().height ?? 0) +
          (parseFloat($heading.css('marginBottom')) || 0)
        : 0

    const $stage = $root.find('.dashboard-cluster-map-stage')
    const stagePadH = padV($stage)
    const stageBorderH = borderV($stage)

    const used =
      topBarH +
      errorBarH +
      footerH +
      panelsPadH +
      mapPadH +
      headingH +
      stagePadH +
      stageBorderH

    return Math.max(minHeight, Math.floor(($(window).height() ?? 0) - used))
  }

  private applyDiagramHeight($root: JQuery<HTMLElement>): void {
    const height = this.computeDiagramHeight($root)
    $root.find('[data-cluster-map-diagram]').css('height', `${height}px`)
  }

  private ensureShell($root: JQuery<HTMLElement>): void {
    if ($root.children().length > 0 && !$root.find('[data-cluster-map-canvas]').length) {
      $root.empty()
    }
    if ($root.children().length > 0) {
      return
    }
    $root.append(
      $('<div>', { class: 'dashboard-cluster-map-inner' }).append(
        $('<h2>', {
          class: 'mdc-typography--headline6 dashboard-cluster-map-heading',
          text: 'Cluster Map',
        }),
        $('<div>', { class: 'dashboard-cluster-map-stage' }).append(
          this.buildZoomToolsElement(),
          $('<div>', {
            class: 'dashboard-cluster-map-diagram',
            attr: { 'data-cluster-map-diagram': '' },
          }).append(
            $('<div>', {
              class: 'dashboard-cluster-map-canvas',
              attr: { 'data-cluster-map-canvas': '' },
              'aria-live': 'polite',
            }),
          ),
        ),
      ),
    )
  }

  private buildZoomToolsElement(): JQuery<HTMLElement> {
    const mkBtn = (extraClass: string, icon: string, label: string): JQuery<HTMLElement> =>
      $('<button>', {
        type: 'button',
        class: `dashboard-cluster-map-zoom-btn ${extraClass}`,
        title: label,
        'aria-label': label,
      }).append(
        $('<span>', {
          class: 'material-symbols-outlined',
          text: icon,
          'aria-hidden': 'true',
        }),
      )

    return $('<div>', {
      class: 'dashboard-cluster-map-zoom-tools',
      attr: { 'data-cluster-map-zoom-tools': '' },
      'aria-label': 'Diagram zoom',
    })
      .append(
        mkBtn('dashboard-cluster-map-zoom-in', 'add', 'Zoom in'),
        mkBtn('dashboard-cluster-map-zoom-reset', 'restart_alt', 'Reset pan and zoom'),
        mkBtn('dashboard-cluster-map-zoom-out', 'remove', 'Zoom out'),
      )
      .prop('hidden', true)
  }

  private ensureZoomTools($root: JQuery<HTMLElement>): void {
    const $stage = $root.find('.dashboard-cluster-map-stage')
    if (!$stage.length || $stage.find('[data-cluster-map-zoom-tools]').length) {
      return
    }
    $stage.prepend(this.buildZoomToolsElement())
  }

  private bindZoomToolHandlers($root: JQuery<HTMLElement>): void {
    if (this.zoomToolHandlersBound) {
      return
    }
    this.zoomToolHandlersBound = true
    $root.on('click', '.dashboard-cluster-map-zoom-in', (event) => {
      event.preventDefault()
      this.adjustZoom(CLUSTER_MAP_ZOOM_STEP)
    })
    $root.on('click', '.dashboard-cluster-map-zoom-out', (event) => {
      event.preventDefault()
      this.adjustZoom(1 / CLUSTER_MAP_ZOOM_STEP)
    })
    $root.on('click', '.dashboard-cluster-map-zoom-reset', (event) => {
      event.preventDefault()
      this.fitGraph()
    })
  }

  private destroyGraph(): void {
    if (this.cy == null) {
      return
    }
    this.cy.destroy()
    this.cy = null
  }

  private updateGraphElements(elements: ElementDefinition[]): void {
    const cy = this.cy
    if (cy == null) {
      return
    }
    const nextSig = clusterMapTopologySignature(elements)
    const topologyChanged = nextSig !== this.clusterMapTopologySignature
    this.clusterMapTopologySignature = nextSig

    const prevZoom = cy.zoom()
    const prevPan = cy.pan()

    cy.batch(() => {
      cy.elements().remove()
      cy.add(elements)
    })

    requestAnimationFrame(() => {
      cy.resize()
      if (topologyChanged) {
        this.fitGraph()
      } else {
        cy.zoom(prevZoom)
        cy.pan(prevPan)
      }
    })
  }

  private createGraph($canvas: JQuery<HTMLElement>, elements: ElementDefinition[]): void {
    const canvasHost = $canvas.get(0)
    if (canvasHost == null) {
      return
    }
    $canvas.empty()
    this.clusterMapTopologySignature = clusterMapTopologySignature(elements)
    const cy = cytoscape({
      container: canvasHost,
      elements,
      layout: { name: 'preset', fit: false, padding: CLUSTER_MAP_FIT_PADDING },
      minZoom: CLUSTER_MAP_MIN_SCALE,
      maxZoom: CLUSTER_MAP_MAX_SCALE,
      wheelSensitivity: 1,
      boxSelectionEnabled: false,
      autoungrabify: true,
      autounselectify: true,
      style: [
        {
          selector: 'node.map-node',
          style: {
            shape: 'round-rectangle',
            label: '',
            'overlay-opacity': 0,
            'text-opacity': 0,
          },
        },
        {
          selector: 'node.group-title-node',
          style: {
            'background-opacity': 0,
            'border-width': 0,
            'border-opacity': 0,
          },
        },
        {
          selector: 'node.db-node',
          style: {
            shape: 'barrel',
          },
        },
        {
          selector: 'node.placeholder-node',
          style: {
            shape: 'round-rectangle',
            label: 'data(label)',
            color: '#c4c6ce',
            'font-size': 14,
            'font-family': 'var(--sans)',
            'text-wrap': 'wrap',
            'text-max-width': '260px',
            'text-valign': 'center',
            'text-halign': 'center',
            'overlay-opacity': 0,
          },
        },
        {
          selector: 'edge',
          style: {
            'curve-style': 'unbundled-bezier',
            'control-point-distances': clusterMapEdgeCpDistance,
            'control-point-weights': clusterMapEdgeCpWeight,
            'line-style': 'dashed',
            'overlay-opacity': 0,
          },
        },
      ],
    })

    ;(cy as unknown as CytoscapeWithNodeHtmlLabel).nodeHtmlLabel(
      [
        {
          query: 'node[labelHtml]',
          halign: 'center',
          valign: 'center',
          halignBox: 'center',
          valignBox: 'center',
          cssClass: 'dashboard-cluster-map-html-label',
          tpl: (data) => data.labelHtml ?? '',
        },
      ],
      { enablePointerEvents: false },
    )

    this.cy = cy
    requestAnimationFrame(() => {
      cy.resize()
      this.fitGraph()
    })
  }

  private fitGraph(): void {
    if (this.cy == null) {
      return
    }
    const fitTargets = this.cy.nodes(':childless')
    this.cy.fit(fitTargets.length > 0 ? fitTargets : this.cy.elements(), CLUSTER_MAP_FIT_PADDING)
  }

  private resizeGraph(): void {
    if (this.cy == null) {
      return
    }
    this.cy.resize()
  }

  private adjustZoom(factor: number): void {
    if (this.cy == null) {
      return
    }
    const container = this.cy.container()
    if (container == null) {
      return
    }
    const nextZoom = Math.min(CLUSTER_MAP_MAX_SCALE, Math.max(CLUSTER_MAP_MIN_SCALE, this.cy.zoom() * factor))
    if (!Number.isFinite(nextZoom)) {
      return
    }
    this.cy.zoom({
      level: nextZoom,
      renderedPosition: {
        x: container.clientWidth / 2,
        y: container.clientHeight / 2,
      },
    })
  }

  private async loadOnceThenPoll(
    api: AmgixApi,
    $root: JQuery<HTMLElement>,
    generation: number,
  ): Promise<void> {
    const systemInfo = await api.systemInfo().catch((): null => null)
    if (generation !== this.pollGeneration) {
      return
    }
    await this.fetchAndRender(api, $root, generation, systemInfo)
    this.pollTimer = window.setInterval(() => {
      void this.fetchAndRender(api, $root, generation, systemInfo)
    }, CLUSTER_MAP_POLL_MS)
  }

  private async fetchAndRender(
    api: AmgixApi,
    $root: JQuery<HTMLElement>,
    generation: number,
    systemInfo: SystemInfoResponse | null,
  ): Promise<void> {
    if (generation !== this.pollGeneration || $('#panel-cluster-map').prop('hidden')) {
      return
    }
    const $canvas = $root.find('[data-cluster-map-canvas]')
    if (!$canvas.length) {
      return
    }

    const metrics = await api
      .metricsCurrent({
        window: CLUSTER_MAP_METRIC_WINDOW_SEC,
        keys: [...CLUSTER_MAP_METRICS_CURRENT_KEYS],
      })
      .catch((): null => null)
    if (generation !== this.pollGeneration || $('#panel-cluster-map').prop('hidden')) {
      return
    }

    this.applyDiagramHeight($root)
    const elements = buildClusterGraph(metrics, systemInfo)

    try {
      const canvasHost = $canvas.get(0)
      if (this.cy != null && canvasHost != null && this.cy.container() !== canvasHost) {
        this.destroyGraph()
      }
      if (this.cy == null) {
        this.createGraph($canvas, elements)
      } else {
        this.updateGraphElements(elements)
      }
      $root.find('[data-cluster-map-zoom-tools]').prop('hidden', false)
    } catch {
      this.clusterMapTopologySignature = ''
      this.destroyGraph()
      $root.find('[data-cluster-map-zoom-tools]').prop('hidden', true)
      $canvas.empty().append(
        $('<p>', {
          class: 'dashboard-cluster-map-render-error',
          text: 'Could not render the cluster diagram.',
        }),
      )
    }
  }
}

import {
  ResponseError,
  type AmgixApi,
  type ClusterView,
  type NodeView,
  type ReadyResponse,
} from '@amgix/amgix-client'
import $ from 'jquery'

import { hideDashboardError, showDashboardError } from '../error-bar'
import { DashboardPanel } from './panel-base'

const HOME_READY_POLL_MS = 10_000

/** Cluster table embedding columns use this rolling window (seconds). */
const CLUSTER_METRICS_WINDOW_SEC = 60

function buildClusterNodesFootnotes(): JQuery<HTMLElement> {
  const w = CLUSTER_METRICS_WINDOW_SEC
  return $('<div>', { class: 'dashboard-home-cluster-footnotes' }).append(
    $('<p>', { text: '* Current encoder leader.' }),
    $('<p>', { text: 'Batches: total embed batches in that window on this node.' }),
    $('<p>', { text: `Rate/s: embed batches per second over the last ${w}s.` }),
    $('<p>', { text: 'Avg ms: mean local inference time per batch, weighted by batch count per vector type.' }),
    $('<p>', { text: 'E2E ms: mean end-to-end time per batch including routing; only reported on originating nodes.' }),
  )
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
  const s = n.toFixed(1)
  return s.endsWith('.0') ? String(Math.round(n)) : s
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

function formatClusterRpsCell(rps: number): string {
  if (!Number.isFinite(rps)) {
    return ''
  }
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 0 }).format(rps)
}

function formatClusterAvgMsCell(ms: number): string {
  if (!Number.isFinite(ms)) {
    return ''
  }
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 0 }).format(ms)
}

/** Sum batch rates and counts across vector types; latency fields are n-weighted averages over the window. */
function formatEmbeddingMetricsCells60s(node: NodeView): { rps: string; avgMs: string; e2eAvgMs: string; requests: string } {
  const list = node.metrics ?? []
  let sawWindow = false
  let totalN = 0
  let sumRps = 0
  let weightedMs = 0
  let weightedE2eMs = 0
  let e2eN = 0
  for (const vm of list) {
    const wm = vm.windows?.[clusterMetricsWindowKey]
    if (wm == null) {
      continue
    }
    sawWindow = true
    const n = Number(wm.n)
    const rps = wm.rps
    const avgMs = wm.avg_ms
    if (Number.isFinite(rps)) {
      sumRps += rps
    }
    if (Number.isFinite(n) && n > 0) {
      const ni = Math.trunc(n)
      totalN += ni
      if (Number.isFinite(avgMs)) {
        weightedMs += avgMs * ni
      }
      const e2e = (wm as unknown as Record<string, unknown>)['e2e_avg_ms'] as number | null | undefined
      if (e2e != null && Number.isFinite(e2e)) {
        weightedE2eMs += e2e * ni
        e2eN += ni
      }
    }
  }
  if (!sawWindow) {
    return { rps: '', avgMs: '', e2eAvgMs: '', requests: '' }
  }
  return {
    rps: formatClusterRpsCell(sumRps),
    avgMs: totalN > 0 ? formatClusterAvgMsCell(weightedMs / totalN) : '',
    e2eAvgMs: e2eN > 0 ? formatClusterAvgMsCell(weightedE2eMs / e2eN) : '',
    requests: String(totalN),
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

function $readinessStatus(ok: boolean, readinessKey: HomeReadinessKey): JQuery<HTMLElement> {
  return $('<span>', {
    class: ok ? 'dashboard-home-status dashboard-home-status--ready' : 'dashboard-home-status dashboard-home-status--not-ready',
    text: readinessLabel(ok),
    attr: { 'data-home-ready': readinessKey },
  })
}

export class HomePanel extends DashboardPanel {
  private readyPollTimer: number | null = null
  private readyPollGeneration = 0

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
  }

  private applyClusterViewToDom($root: JQuery<HTMLElement>, view: ClusterView | null): void {
    const $tbody = $root.find('[data-home-cluster-tbody]')
    if (!$tbody.length) {
      return
    }
    $tbody.empty()
    const nodes = view?.nodes
    if (view == null || nodes === undefined) {
      $tbody.append(
        $('<tr>').append(
          $('<td>', {
            class: 'dashboard-home-cluster-placeholder',
            colspan: 12,
            text: 'Cluster view unavailable.',
          }),
        ),
      )
      return
    }
    if (Object.keys(nodes).length === 0) {
      $tbody.append(
        $('<tr>').append(
          $('<td>', {
            class: 'dashboard-home-cluster-placeholder',
            colspan: 12,
            text: 'No nodes in cluster view.',
          }),
        ),
      )
      return
    }
    for (const [hostname, node] of sortClusterNodeEntries(nodes)) {
      const m = formatEmbeddingMetricsCells60s(node)
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
          $('<td>', { class: 'dashboard-home-v', text: formatLastSeen(node.last_seen) }),
        ),
      )
    }
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

      const $clusterThead = $('<thead>')
        .append(
          $('<tr>').append(
            $('<th>', {
              class: 'dashboard-home-table-heading',
              colspan: 12,
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
            $('<th>', { text: 'Batches' }),
            $('<th>', { text: 'Rate/s' }),
            $('<th>', { text: 'Avg ms' }),
            $('<th>', { text: 'E2E ms' }),
            $('<th>', { text: 'Last seen' }),
          ),
        )

      const $clusterTable = $('<table>', {
        class: 'dashboard-home-table dashboard-home-cluster-table',
      }).append($clusterThead, $('<tbody>', { attr: { 'data-home-cluster-tbody': '' } }))

      $root.empty().append($systemTable, $clusterTable, buildClusterNodesFootnotes())
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

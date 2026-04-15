import {
  CollectionConfigToJSON,
  ResponseError,
  type AmgixApi,
  type CollectionConfig,
  type QueueInfo,
} from '@amgix/amgix-client'
import {
  ArcElement,
  Chart,
  Legend,
  PieController,
  Tooltip,
  type ChartConfiguration,
} from 'chart.js'
import $ from 'jquery'

import { hideDashboardError, showDashboardError } from '../error-bar'
import { DashboardPanel } from './panel-base'

Chart.register(PieController, ArcElement, Tooltip, Legend)

const DASH = '-'

const COLLECTION_STATS_POLL_MS = 10_000

function formatInt(n: number): string {
  return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
}

/** Pie segment colors: defined only in `style.css` (`--chart-pie-*`). */
function pieChartPalette(): { backgrounds: string[]; border: string } {
  const style = getComputedStyle(document.documentElement)
  const v = (name: string) => style.getPropertyValue(name).trim()
  return {
    backgrounds: [
      v('--chart-pie-indexed'),
      v('--chart-pie-queued'),
      v('--chart-pie-requeued'),
      v('--chart-pie-failed'),
    ],
    border: v('--chart-pie-border'),
  }
}

function formatRequestError(context: string, err: unknown): string {
  if (err instanceof ResponseError) {
    return `${context} (HTTP ${err.response.status})`
  }
  if (err instanceof Error) {
    return err.message
  }
  return context
}

function normalizeFastApiDetail(detail: unknown): string {
  if (detail == null) {
    return ''
  }
  if (typeof detail === 'string') {
    return detail.trim()
  }
  if (Array.isArray(detail)) {
    const lines: string[] = []
    for (const item of detail) {
      if (item && typeof item === 'object' && 'msg' in item) {
        const msg = String((item as { msg: unknown }).msg)
        const loc = (item as { loc?: unknown }).loc
        if (Array.isArray(loc) && loc.length > 0) {
          lines.push(`${loc.map(String).join('.')}: ${msg}`)
        } else {
          lines.push(msg)
        }
      } else {
        lines.push(JSON.stringify(item))
      }
    }
    return lines.join('\n')
  }
  if (typeof detail === 'object') {
    return JSON.stringify(detail)
  }
  return String(detail)
}

async function buildErrorMessage(context: string, err: unknown): Promise<string> {
  if (err instanceof ResponseError) {
    const { response } = err
    let body = ''
    try {
      body = (await response.clone().text()).trim()
    } catch {
      body = ''
    }
    let fromBody = ''
    if (body) {
      try {
        const j = JSON.parse(body) as { detail?: unknown; message?: unknown }
        if ('detail' in j) {
          fromBody = normalizeFastApiDetail(j.detail)
        } else if (typeof j.message === 'string') {
          fromBody = j.message.trim()
        }
      } catch {
        fromBody = body.length > 800 ? `${body.slice(0, 800)}…` : body
      }
    }
    const statusBit = `${response.status}${response.statusText ? ` ${response.statusText}` : ''}`
    const head = `${context}\n\nHTTP ${statusBit}`.trim()
    return fromBody ? `${head}\n\n${fromBody}` : head
  }
  if (err instanceof Error && err.message) {
    return `${context}\n\n${err.message}`
  }
  return context
}

function openConfirmDialog(options: {
  title: string
  message: string
  confirmLabel: string
  danger?: boolean
}): Promise<boolean> {
  const { title, message, confirmLabel, danger } = options

  return new Promise((resolve) => {
    const titleId = `collections-confirm-title-${Date.now()}`
    const $dialog = $('<dialog>', {
      class: 'dashboard-collections-confirm-dialog',
      'aria-labelledby': titleId,
    })

    let settled = false
    const finish = (confirmed: boolean): void => {
      if (settled) {
        return
      }
      settled = true
      ;($dialog.get(0) as HTMLDialogElement | undefined)?.close()
      $dialog.remove()
      resolve(confirmed)
    }

    $dialog.on('close', () => {
      if (!settled) {
        finish(false)
      }
    })

    const $cancel = $('<button>', {
      type: 'button',
      class: 'dashboard-collections-confirm-dialog-btn dashboard-collections-confirm-dialog-btn--secondary',
      text: 'Cancel',
    })
    $cancel.on('click', () => {
      finish(false)
    })

    const $confirm = $('<button>', {
      type: 'button',
      class: danger
        ? 'dashboard-collections-confirm-dialog-btn dashboard-collections-confirm-dialog-btn--danger'
        : 'dashboard-collections-confirm-dialog-btn dashboard-collections-confirm-dialog-btn--primary',
      text: confirmLabel,
    })
    $confirm.on('click', () => {
      finish(true)
    })

    $dialog.append(
      $('<h4>', {
        id: titleId,
        class: 'dashboard-collections-confirm-dialog-title',
        text: title,
      }),
      $('<p>', {
        class: 'dashboard-collections-confirm-dialog-message',
        text: message,
      }),
      $('<div>', { class: 'dashboard-collections-confirm-dialog-actions' }).append($cancel, $confirm),
    )

    $('body').append($dialog)
    ;($dialog.get(0) as HTMLDialogElement).showModal()
  })
}

function openFullConfigModal(collectionName: string, json: string): void {
  $('#collections-full-config-dialog').remove()

  const $dialog = $('<dialog>', {
    id: 'collections-full-config-dialog',
    class: 'dashboard-collections-config-dialog',
    'aria-labelledby': 'collections-full-config-title',
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

  $dialog.append(
    $('<h4>', {
      id: 'collections-full-config-title',
      class: 'dashboard-collections-config-dialog-title',
      text: `Configuration for ${collectionName}`,
    }),
    $('<textarea>', {
      readonly: true,
      class: 'dashboard-collections-config-dialog-textarea',
      text: json,
    }),
    $('<div>', { class: 'dashboard-collections-config-dialog-actions' }).append($close),
  )

  $('body').append($dialog)
  ;($dialog.get(0) as HTMLDialogElement).showModal()
}

function buildVectorsSection(config: CollectionConfig, collectionName: string): JQuery<HTMLElement> {
  const fullJson = JSON.stringify(CollectionConfigToJSON(config), null, 2)

  const $headerRow = $('<div>', { class: 'dashboard-collections-vectors-header' }).append(
    $('<span>', { class: 'dashboard-collections-vectors-title', text: 'Vectors:' }),
    $('<a>', {
      href: '#',
      class: 'dashboard-collections-full-config-link',
      text: 'Full Configuration',
    }).on('click', (e) => {
      e.preventDefault()
      openFullConfigModal(collectionName, fullJson)
    }),
  )

  const $thead = $('<thead>').append(
    $('<tr>').append(
      ['Name', 'Type', 'Fields', 'Model', 'Dimensions'].map((label) =>
        $('<th>', { scope: 'col', text: label }),
      ),
    ),
  )

  const $tbody = $('<tbody>')
  if (config.vectors.length === 0) {
    $tbody.append(
      $('<tr>').append(
        $('<td>', { colSpan: 5, text: 'No vectors configured.' }),
      ),
    )
  }
  for (const v of config.vectors) {
    const fields =
      v.index_fields && v.index_fields.length > 0 ? v.index_fields.join(', ') : 'content'
    const model = v.model != null && String(v.model).length > 0 ? String(v.model) : DASH
    const dimensions =
      v.dimensions != null && Number.isFinite(v.dimensions) ? formatInt(v.dimensions) : DASH

    $tbody.append(
      $('<tr>').append(
        $('<td>', { text: v.name }),
        $('<td>', { text: v.type }),
        $('<td>', { text: fields }),
        $('<td>', { text: model }),
        $('<td>', { text: dimensions }),
      ),
    )
  }

  const $table = $('<table>', { class: 'dashboard-collections-vectors-table' }).append($thead, $tbody)

  return $('<section>', { class: 'dashboard-collections-detail-section' }).append(
    $headerRow,
    $('<div>', { class: 'dashboard-collections-vectors-table-wrap' }).append($table),
  )
}

const DOC_STATS_PIE_KINDS = ['indexed', 'queued', 'requeued', 'failed'] as const

function statRow(kind: string, label: string, value: number): JQuery<HTMLTableRowElement> {
  const $tr = $('<tr>', { 'data-stat': kind }).append(
    $('<th>', { scope: 'row', text: label }),
    $('<td>', { 'data-stat-value': '1', text: formatInt(value) }),
  ) as JQuery<HTMLTableRowElement>
  if ((DOC_STATS_PIE_KINDS as readonly string[]).includes(kind)) {
    $tr.toggleClass('dashboard-collections-doc-stats-stat--inactive', value === 0)
  }
  return $tr
}

function queueTotalRateRow(): JQuery<HTMLTableRowElement> {
  const $icon = $('<span>', {
    class: 'material-symbols-outlined material-symbols--btn dashboard-collections-queue-rate-icon',
    'aria-hidden': 'true',
  })
  const $val = $('<span>', { class: 'dashboard-collections-queue-rate-value' })
  const $td = $('<td>').append(
    $('<span>', { class: 'dashboard-collections-queue-rate-wrap' }).append($icon, ' ', $val),
  )
  return $('<tr>', {
    'data-stat': 'queue-total-rate',
    title: 'Rate of change of queue entry total over the poll interval (docs/s).',
  }).append(
    $('<th>', { scope: 'row', text: 'Queue total Δ' }),
    $td,
  ) as JQuery<HTMLTableRowElement>
}

function formatQueueRateMagnitude(docsPerSec: number): string {
  const a = Math.abs(docsPerSec)
  if (a >= 100) {
    return a.toFixed(0)
  }
  if (a >= 10) {
    return a.toFixed(1)
  }
  return a.toFixed(2)
}

function applyQueueTotalRateToDom(
  $detail: JQuery<HTMLElement>,
  rate: number | null,
  trend: 'up' | 'down' | 'flat' | 'none',
): void {
  const $root = $detail.find('[data-collections-doc-stats]')
  if (!$root.length) {
    return
  }
  const $row = $root.find('tr[data-stat="queue-total-rate"]')
  const $icon = $row.find('.dashboard-collections-queue-rate-icon')
  const $val = $row.find('.dashboard-collections-queue-rate-value')
  $icon.removeClass(
    'dashboard-collections-queue-rate-icon--up dashboard-collections-queue-rate-icon--down dashboard-collections-queue-rate-icon--flat dashboard-collections-queue-rate-icon--hidden',
  )
  if (trend === 'none' || rate === null) {
    $icon.addClass('dashboard-collections-queue-rate-icon--hidden').text('')
    $val.text(DASH)
    $row.removeClass('dashboard-collections-queue-rate-row--nonzero')
    return
  }
  $val.text(`${formatQueueRateMagnitude(rate)} /s`)
  if (trend === 'up') {
    $icon.text('arrow_upward').addClass('dashboard-collections-queue-rate-icon--up')
  } else if (trend === 'down') {
    $icon.text('arrow_downward').addClass('dashboard-collections-queue-rate-icon--down')
  } else {
    $icon.text('horizontal_rule').addClass('dashboard-collections-queue-rate-icon--flat')
  }
  $row.toggleClass('dashboard-collections-queue-rate-row--nonzero', Math.abs(rate) > 1e-9)
}

function applyCollectionStatsToDom(
  $detail: JQuery<HTMLElement>,
  docCount: number,
  queueInfo: QueueInfo,
  grandTotal: number,
): void {
  const $root = $detail.find('[data-collections-doc-stats]')
  if (!$root.length) {
    return
  }
  const set = (kind: string, n: number) => {
    $root.find(`tr[data-stat="${kind}"] td[data-stat-value]`).text(formatInt(n))
    if ((DOC_STATS_PIE_KINDS as readonly string[]).includes(kind)) {
      $root.find(`tr[data-stat="${kind}"]`).toggleClass('dashboard-collections-doc-stats-stat--inactive', n === 0)
    }
  }
  set('indexed', docCount)
  set('queued', queueInfo.queued)
  set('requeued', queueInfo.requeued)
  set('failed', queueInfo.failed)
  set('total', grandTotal)
}

export class CollectionsPanel extends DashboardPanel {
  private queuePieChart: Chart | null = null
  private statsPollTimer: number | null = null
  private statsPollGeneration = 0
  private queueTotalSample: { t: number; total: number } | null = null

  override deactivate(): void {
    this.clearStatsPoll()
    this.statsPollGeneration += 1
  }

  init(api: AmgixApi): void {
    const $nav = $('#panel-collections [data-collections-nav]')
    const $detail = $('#panel-collections [data-collections-detail]')
    if (!$nav.length || !$detail.length) {
      return
    }
    void this.loadCollectionsNav(api, $nav, $detail, null)
  }

  override refresh(api: AmgixApi): void {
    const $nav = $('#panel-collections [data-collections-nav]')
    const $detail = $('#panel-collections [data-collections-detail]')
    if (!$nav.length || !$detail.length) {
      return
    }
    const prefer = this.readSelectedCollectionName($nav)
    void this.loadCollectionsNav(api, $nav, $detail, prefer)
  }

  private readSelectedCollectionName($nav: JQuery<HTMLElement>): string | null {
    const t = $nav.find('.dashboard-collections-nav-item--selected').first().text().trim()
    return t.length > 0 ? t : null
  }

  private async loadCollectionsNav(
    api: AmgixApi,
    $nav: JQuery<HTMLElement>,
    $detail: JQuery<HTMLElement>,
    preferName: string | null,
  ): Promise<void> {
    this.clearStatsPoll()
    this.destroyQueuePieChart()
    this.queueTotalSample = null

    $nav.empty()
    $detail.empty().text('Loading collections…')

    try {
      const names = (await api.listCollections()).slice().sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }))
      hideDashboardError()
      $nav.empty()
      $detail.empty()

      if (names.length === 0) {
        $nav.append($('<p>', { class: 'dashboard-collections-empty', text: 'No collections yet.' }))
        return
      }

      const $ul = $('<ul>', { class: 'dashboard-collections-nav-list', role: 'list' })

      for (const name of names) {
        const $btn = $('<button>', {
          type: 'button',
          class: 'dashboard-collections-nav-item',
          text: name,
        })
        $btn.on('click', () => {
          void this.selectCollection(api, name, $nav, $detail)
        })
        $ul.append($('<li>', { role: 'none' }).append($btn))
      }
      $nav.append($ul)

      const chosen =
        preferName != null && names.includes(preferName) ? preferName : names[0]
      await this.selectCollection(api, chosen, $nav, $detail)
    } catch (err) {
      $nav.empty()
      $detail.empty()
      showDashboardError(formatRequestError('Could not load collections.', err))
    }
  }

  private clearStatsPoll(): void {
    if (this.statsPollTimer != null) {
      clearInterval(this.statsPollTimer)
      this.statsPollTimer = null
    }
  }

  private destroyQueuePieChart(): void {
    this.queuePieChart?.destroy()
    this.queuePieChart = null
  }

  private updateQueueTotalRate($detail: JQuery<HTMLElement>, queueTotal: number): void {
    const now = performance.now()
    let trend: 'up' | 'down' | 'flat' | 'none' = 'none'
    let rate: number | null = null
    const prev = this.queueTotalSample
    if (prev) {
      const dt = (now - prev.t) / 1000
      if (dt >= 0.5) {
        rate = (queueTotal - prev.total) / dt
        if (rate > 1e-9) {
          trend = 'up'
        } else if (rate < -1e-9) {
          trend = 'down'
        } else {
          trend = 'flat'
        }
      }
    }
    this.queueTotalSample = { t: now, total: queueTotal }
    applyQueueTotalRateToDom($detail, rate, trend)
  }

  private mountQueuePieChart(canvas: HTMLCanvasElement, docCount: number, queueInfo: QueueInfo): void {
    this.destroyQueuePieChart()
    const data = [docCount, queueInfo.queued, queueInfo.requeued, queueInfo.failed]

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }

    const { backgrounds, border } = pieChartPalette()
    const labels = ['Indexed', 'Queued', 'Requeued', 'Failed']

    const config: ChartConfiguration<'pie'> = {
      type: 'pie',
      data: {
        labels,
        datasets: [
          {
            data,
            backgroundColor: backgrounds,
            borderColor: border,
            borderWidth: 1,
            hoverOffset: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        layout: {
          padding: { top: 2, right: 2, bottom: 2, left: 2 },
        },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              boxWidth: 14,
              padding: 10,
              font: { size: 15, family: 'Roboto, system-ui, sans-serif' },
              color: getComputedStyle(document.documentElement).color.trim() || '#000',
            },
          },
          tooltip: {
            callbacks: {
              label: (item) => {
                const v = typeof item.raw === 'number' ? item.raw : Number(item.raw)
                const label = item.label ?? ''
                return `${label}: ${formatInt(v)}`
              },
            },
          },
        },
      },
    }

    this.queuePieChart = new Chart(ctx, config)
  }

  private async pollCollectionStats(
    api: AmgixApi,
    name: string,
    $nav: JQuery<HTMLElement>,
    $detail: JQuery<HTMLElement>,
    generation: number,
  ): Promise<void> {
    if (generation !== this.statsPollGeneration) {
      return
    }
    if ($('#panel-collections').prop('hidden')) {
      return
    }
    try {
      const collStats = await api.getCollectionStats({ collectionName: name })
      if (generation !== this.statsPollGeneration) {
        return
      }
      if ($('#panel-collections').prop('hidden')) {
        return
      }
      const docCount = collStats.doc_count
      const queueInfo = collStats.queue
      const grandTotal = docCount + queueInfo.queued + queueInfo.requeued + queueInfo.failed
      const chartVisible = this.queuePieChart !== null
      const shouldShowChart = grandTotal > 0
      if (chartVisible !== shouldShowChart) {
        this.clearStatsPoll()
        await this.selectCollection(api, name, $nav, $detail)
        return
      }
      applyCollectionStatsToDom($detail, docCount, queueInfo, grandTotal)
      this.updateQueueTotalRate($detail, queueInfo.total)
      if (this.queuePieChart) {
        this.queuePieChart.data.datasets[0].data = [
          docCount,
          queueInfo.queued,
          queueInfo.requeued,
          queueInfo.failed,
        ]
        this.queuePieChart.update('none')
      }
    } catch (err) {
      if (err instanceof ResponseError && err.response.status === 404) {
        if (generation !== this.statsPollGeneration) {
          return
        }
        void this.loadCollectionsNav(api, $nav, $detail, null)
        return
      }
      // Ignore transient poll failures; next tick retries.
    }
  }

  private async selectCollection(
    api: AmgixApi,
    name: string,
    $nav: JQuery<HTMLElement>,
    $detail: JQuery<HTMLElement>,
  ): Promise<void> {
    this.clearStatsPoll()
    this.statsPollGeneration += 1
    const pollGeneration = this.statsPollGeneration
    this.queueTotalSample = null
    this.destroyQueuePieChart()

    $nav.find('.dashboard-collections-nav-item').each(function () {
      const $btn = $(this)
      const selected = $btn.text() === name
      $btn.attr('aria-pressed', selected ? 'true' : 'false')
      $btn.toggleClass('dashboard-collections-nav-item--selected', selected)
    })

    $detail.empty().append(
      $('<p>', { class: 'dashboard-collections-detail-loading', text: `Loading “${name}”…` }),
    )

    try {
      const [config, collStats] = await Promise.all([
        api.getCollectionConfig({ collectionName: name }),
        api.getCollectionStats({ collectionName: name }),
      ])
      hideDashboardError()

      const docCount = collStats.doc_count
      const queueInfo = collStats.queue
      const grandTotal = docCount + queueInfo.queued + queueInfo.requeued + queueInfo.failed

      const $vectorsSection = buildVectorsSection(config, name)

      const $statsBody = $('<tbody>')
      $statsBody.append(
        statRow('indexed', 'Indexed', docCount),
        statRow('queued', 'Queued', queueInfo.queued),
        statRow('requeued', 'Requeued', queueInfo.requeued),
        statRow('failed', 'Failed', queueInfo.failed),
        queueTotalRateRow(),
      )
      $statsBody.append(
        $('<tr>', { class: 'dashboard-collections-doc-stats-total', 'data-stat': 'total' }).append(
          $('<th>', { scope: 'row', text: 'Total' }),
          $('<td>', { 'data-stat-value': '1', text: formatInt(grandTotal) }),
        ),
      )

      const $statsTable = $('<table>', {
        class: 'dashboard-collections-doc-stats-table',
        role: 'table',
        'data-collections-doc-stats': '1',
      }).append(
        $('<thead>').append(
          $('<tr>').append(
            $('<th>', {
              class: 'dashboard-collections-doc-stats-table-heading',
              colspan: 2,
              scope: 'colgroup',
              text: 'Document Stats',
            }),
          ),
        ),
        $statsBody,
      )

      const $statsTableWrap = $('<div>', {
        class: 'dashboard-collections-doc-stats-table-wrap',
      }).append($statsTable)

      const $statsRow = $('<div>', { class: 'dashboard-collections-doc-stats-row' }).append($statsTableWrap)

      let chartCanvas: HTMLCanvasElement | undefined
      if (grandTotal > 0) {
        const $chartInner = $('<div>', { class: 'dashboard-collections-doc-stats-chart-inner' }).append(
          $('<canvas>', {
            'aria-label': 'Pie chart of document stats by state',
            role: 'img',
          }),
        )
        const $chartWrap = $('<div>', { class: 'dashboard-collections-doc-stats-chart-wrap' }).append($chartInner)
        $statsRow.append($chartWrap)
        const el = $chartInner.find('canvas').get(0)
        if (el) {
          chartCanvas = el
        }
      }

      const $actionButtons = $('<div>', { class: 'dashboard-collections-doc-stats-actions' })

      const mkAction = (label: string, iconName: string): JQuery<HTMLButtonElement> => {
        const $btn = $('<button>', {
          type: 'button',
          class: 'dashboard-collections-stats-action',
        }) as JQuery<HTMLButtonElement>
        $btn.append(
          $('<span>', {
            class: 'material-symbols-outlined material-symbols--btn',
            'aria-hidden': 'true',
            text: iconName,
          }),
          $('<span>').text(label),
        )
        return $btn
      }

      const $btnEmpty = mkAction('Empty', 'delete_sweep')
      const $btnDeleteQueue = mkAction('Delete Queue', 'playlist_remove')
      const $btnDelete = mkAction('Delete', 'delete_forever')

      const setActionsBusy = (busy: boolean): void => {
        $actionButtons.find('button').prop('disabled', busy)
      }

      $btnEmpty.on('click', () => {
        void (async () => {
          const ok = await openConfirmDialog({
            title: 'Empty collection',
            message: `Remove all indexed documents and vectors from “${name}”? The collection configuration will be kept.`,
            confirmLabel: 'Empty collection',
            danger: true,
          })
          if (!ok) {
            return
          }
          setActionsBusy(true)
          try {
            await api.emptyCollection({ collectionName: name })
            await this.selectCollection(api, name, $nav, $detail)
          } catch (err) {
            showDashboardError(await buildErrorMessage('Could not empty the collection.', err))
          } finally {
            setActionsBusy(false)
          }
        })()
      })

      $btnDeleteQueue.on('click', () => {
        void (async () => {
          const ok = await openConfirmDialog({
            title: 'Delete queue',
            message: `Remove all processing-queue entries for “${name}”? Documents already indexed are not affected.`,
            confirmLabel: 'Delete queue',
            danger: true,
          })
          if (!ok) {
            return
          }
          setActionsBusy(true)
          try {
            await api.deleteCollectionQueue({ collectionName: name })
            await this.selectCollection(api, name, $nav, $detail)
          } catch (err) {
            showDashboardError(await buildErrorMessage('Could not clear the queue.', err))
          } finally {
            setActionsBusy(false)
          }
        })()
      })

      $btnDelete.on('click', () => {
        void (async () => {
          const ok = await openConfirmDialog({
            title: 'Delete collection',
            message: `Permanently delete collection “${name}” and all of its data? This cannot be undone.`,
            confirmLabel: 'Delete collection',
            danger: true,
          })
          if (!ok) {
            return
          }
          setActionsBusy(true)
          try {
            await api.deleteCollection({ collectionName: name })
            await this.loadCollectionsNav(api, $nav, $detail, null)
          } catch (err) {
            showDashboardError(await buildErrorMessage('Could not delete the collection.', err))
          } finally {
            setActionsBusy(false)
          }
        })()
      })

      $actionButtons.append($btnEmpty, $btnDeleteQueue, $btnDelete)

      const $docStatsSection = $('<section>', { class: 'dashboard-collections-detail-section' }).append(
        $statsRow,
        $actionButtons,
      )

      const $title = $('<h3>', { class: 'mdc-typography--headline6 dashboard-collections-detail-title' }).append(
        $('<span>', { class: 'dashboard-collections-detail-name-label', text: 'Name:' }),
        $('<span>', { class: 'dashboard-collections-detail-name-value', text: name }),
      )

      $detail.empty().append($title, $vectorsSection, $docStatsSection)

      applyQueueTotalRateToDom($detail, null, 'none')
      this.queueTotalSample = { t: performance.now(), total: queueInfo.total }

      if (chartCanvas) {
        const tableH = $statsTableWrap.get(0)?.offsetHeight ?? 0
        if (tableH > 0) {
          chartCanvas.parentElement!.style.height = `${tableH}px`
        }
        this.mountQueuePieChart(chartCanvas, docCount, queueInfo)
      }

      this.statsPollTimer = window.setInterval(() => {
        void this.pollCollectionStats(api, name, $nav, $detail, pollGeneration)
      }, COLLECTION_STATS_POLL_MS)
    } catch (err) {
      $detail.empty()
      showDashboardError(formatRequestError(`Could not load details for “${name}”.`, err))
    }
  }
}

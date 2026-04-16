import type { AmgixApi } from '@amgix/amgix-client'
import $ from 'jquery'

import type { DashboardPanel } from './panels/panel-base'
import { HomePanel } from './panels/home'
import { ClusterMapPanel } from './panels/cluster-map'
import { CollectionsPanel } from './panels/collections'
import { QueryPanel } from './panels/query'
import {
  formatDashboardRouteHash,
  parseDashboardRouteHash,
  type DashboardPanelId,
} from './route-hash'

export type PanelId = DashboardPanelId

function isPanelId(value: string | undefined): value is PanelId {
  return (
    value === 'home' ||
    value === 'clustermap' ||
    value === 'collections' ||
    value === 'query'
  )
}

const panels: Record<PanelId, DashboardPanel> = {
  home: new HomePanel(),
  clustermap: new ClusterMapPanel(),
  collections: new CollectionsPanel(),
  query: new QueryPanel(),
}

let activePanel: PanelId | null = null
let dashboardApi: AmgixApi | null = null

function hashPanelId(): string {
  return window.location.hash.replace(/^#/, '').trim().toLowerCase()
}

function renderPanelSwitch(id: PanelId): void {
  const api = dashboardApi
  if (!api) {
    throw new Error('initDashboardNav(api) must run before applyRouteFromHash')
  }

  if (activePanel != null) {
    panels[activePanel].deactivate()
  }

  $('[data-panel]').each(function () {
    const $el = $(this)
    const panelKey = $el.data('panel') as string | undefined
    $el.prop('hidden', panelKey !== id)
  })

  $('[data-panel-link]').each(function () {
    const $btn = $(this)
    if ($btn.data('panelLink') === id) {
      $btn.attr('aria-current', 'page')
    } else {
      $btn.removeAttr('aria-current')
    }
  })

  activePanel = id
  panels[id].init(api)
}

export function applyRouteFromHash(): void {
  const raw = hashPanelId()
  const parsed = parseDashboardRouteHash(raw)
  if (raw !== parsed.canonicalFragment) {
    window.location.hash = parsed.canonicalFragment
    return
  }

  if (parsed.panel !== activePanel) {
    renderPanelSwitch(parsed.panel)
  } else if (parsed.panel === 'home') {
    ;(panels.home as HomePanel).applyMetricsTabFromRoute(parsed.homeMetricsTab)
  }
}

export function showPanel(id: PanelId): void {
  const frag = formatDashboardRouteHash(id, 'api')
  if (hashPanelId() !== frag) {
    window.location.hash = frag
  }
}

export function initDashboardNav(api: AmgixApi): void {
  dashboardApi = api
  $('[data-panel-link]').on('click', function () {
    const panelId = $(this).data('panelLink') as string | undefined
    if (!isPanelId(panelId)) {
      return
    }
    if (activePanel === panelId) {
      if (panelId === 'home') {
        const desired = formatDashboardRouteHash('home', 'api')
        if (hashPanelId() !== desired) {
          window.location.hash = desired
        }
      }
      const apiRef = dashboardApi
      if (apiRef) {
        panels[panelId].refresh(apiRef)
      }
      return
    }
    showPanel(panelId)
  })

  $(window).on('hashchange', () => {
    applyRouteFromHash()
  })

  applyRouteFromHash()
}

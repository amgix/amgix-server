import type { AmgixApi } from '@amgix/amgix-client'
import $ from 'jquery'

import type { DashboardPanel } from './panels/panel-base'
import { HomePanel } from './panels/home'
import { ClusterMapPanel } from './panels/cluster-map'
import { CollectionsPanel } from './panels/collections'
import { QueryPanel } from './panels/query'

export type PanelId = 'home' | 'cluster-map' | 'collections' | 'query'

function isPanelId(value: string | undefined): value is PanelId {
  return (
    value === 'home' ||
    value === 'cluster-map' ||
    value === 'collections' ||
    value === 'query'
  )
}

const panels: Record<PanelId, DashboardPanel> = {
  home: new HomePanel(),
  'cluster-map': new ClusterMapPanel(),
  collections: new CollectionsPanel(),
  query: new QueryPanel(),
}

let activePanel: PanelId | null = null
let dashboardApi: AmgixApi | null = null

function hashPanelId(): string {
  return window.location.hash.replace(/^#/, '').trim().toLowerCase()
}

/** Panel to show from the current URL hash (`#home`, `#cluster-map`, …). Invalid or missing → `home`. */
export function initialPanelIdFromHash(): PanelId {
  const raw = hashPanelId()
  if (isPanelId(raw)) {
    return raw
  }
  return 'home'
}

function renderPanel(id: PanelId): void {
  if (activePanel === id) {
    return
  }

  const api = dashboardApi
  if (!api) {
    throw new Error('initDashboardNav(api) must run before showPanel')
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

export function showPanel(id: PanelId): void {
  renderPanel(id)
  const canonical = `#${id}`
  if (window.location.hash !== canonical) {
    window.location.hash = id
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
      const api = dashboardApi
      if (api) {
        panels[panelId].refresh(api)
      }
      return
    }
    showPanel(panelId)
  })

  $(window).on('hashchange', () => {
    const raw = hashPanelId()
    const id: PanelId = isPanelId(raw) ? raw : 'home'
    if (!isPanelId(raw)) {
      if (window.location.hash !== `#${id}`) {
        window.location.hash = id
        return
      }
    }
    renderPanel(id)
  })
}

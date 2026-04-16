export type DashboardPanelId = 'home' | 'clustermap' | 'collections' | 'query'

export type HomeMetricsTabId = 'api' | 'indexing' | 'encoder'

/**
 * Parse the URL fragment (without leading `#`), lowercase.
 * Returns a canonical fragment (no `#`) suitable for `location.hash`.
 */
export function parseDashboardRouteHash(raw: string): {
  panel: DashboardPanelId
  homeMetricsTab: HomeMetricsTabId
  canonicalFragment: string
} {
  const s = raw.trim().toLowerCase()
  if (!s) {
    return { panel: 'home', homeMetricsTab: 'api', canonicalFragment: 'home' }
  }
  if (s === 'home') {
    return { panel: 'home', homeMetricsTab: 'api', canonicalFragment: 'home' }
  }
  if (s.startsWith('home-')) {
    const rest = s.slice('home-'.length)
    if (rest === 'api' || rest === '') {
      return { panel: 'home', homeMetricsTab: 'api', canonicalFragment: 'home' }
    }
    if (rest === 'indexing' || rest === 'encoder') {
      return { panel: 'home', homeMetricsTab: rest, canonicalFragment: s }
    }
    return { panel: 'home', homeMetricsTab: 'api', canonicalFragment: 'home' }
  }
  if (s === 'clustermap' || s === 'collections' || s === 'query') {
    return { panel: s, homeMetricsTab: 'api', canonicalFragment: s }
  }
  return { panel: 'home', homeMetricsTab: 'api', canonicalFragment: 'home' }
}

export function formatDashboardRouteHash(panel: DashboardPanelId, homeMetricsTab: HomeMetricsTabId): string {
  if (panel === 'home') {
    return homeMetricsTab === 'api' ? 'home' : `home-${homeMetricsTab}`
  }
  return panel
}

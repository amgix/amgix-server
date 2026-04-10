import '@fontsource/roboto/latin-400.css'
import '@fontsource/roboto/latin-500.css'
import '@fontsource/roboto/latin-700.css'
import '@fontsource/material-symbols-outlined/400.css'

// `material-components-web` in package.json pulls all @material/* for dev/CI; only import what we ship.
import '@material/typography/dist/mdc.typography.min.css'
import '@material/top-app-bar/dist/mdc.top-app-bar.min.css'

import './style.css'

import logoUrl from './assets/amgix-logo.png'

import $ from 'jquery'
import { MDCTopAppBar } from '@material/top-app-bar/component'
import { AmgixApi, Configuration } from '@amgix/amgix-client'

import { initDashboardErrorBar } from './error-bar'
import { initDashboardNav, initialPanelIdFromHash, showPanel } from './panel-nav'
import { initDashboardThemeToggle } from './theme'

function amgixApiBasePath(): string {
  if (import.meta.env.DEV) {
    // Same origin as the Vite dev server; `vite.config` proxies `/v1` → API (avoids CORS).
    return ''
  }
  return window.location.origin
}

export const amgixConfiguration = new Configuration({
  basePath: amgixApiBasePath(),
})

export const amgixApi = new AmgixApi(amgixConfiguration)

$(() => {
  $('.dashboard-logo-slot').attr('src', logoUrl)
  const barEl = $('#dashboard-top-bar').get(0)
  if (barEl) {
    MDCTopAppBar.attachTo(barEl)
  }
  initDashboardErrorBar()
  initDashboardNav(amgixApi)
  initDashboardThemeToggle($('#theme-toggle'))
  showPanel(initialPanelIdFromHash())
})

import '@fontsource/roboto/latin-400.css'
import '@fontsource/roboto/latin-500.css'
import '@fontsource/roboto/latin-700.css'
import '@fontsource/material-symbols-outlined/400.css'

// `material-components-web` in package.json pulls all @material/* for dev/CI; only import what we ship.
import '@material/typography/dist/mdc.typography.min.css'
import '@material/top-app-bar/dist/mdc.top-app-bar.min.css'
import '@material/icon-button/dist/mdc.icon-button.min.css'
import '@material/snackbar/dist/mdc.snackbar.min.css'

import './style.css'

import logoUrl from './assets/amgix-logo.png'

import $ from 'jquery'
import { MDCTopAppBar } from '@material/top-app-bar/component'
import { AmgixApi, Configuration } from '@amgix/amgix-client'

import {
  ApiKeyRequiredError,
  createApiKeyMiddleware,
  ensureApiKeyAccess,
  initDashboardApiKeyButton,
} from './api-key-gate'
import { amgixApiBasePath } from './api-key'
import { initDashboardErrorBar } from './error-bar'
import { initDashboardNav, refreshActivePanel } from './panel-nav'
import { initDashboardThemeToggle } from './theme'

export const amgixConfiguration = new Configuration({
  basePath: amgixApiBasePath(),
  middleware: [createApiKeyMiddleware()],
})

export const amgixApi = new AmgixApi(amgixConfiguration)

async function bootstrapDashboard(): Promise<void> {
  try {
    await ensureApiKeyAccess()
  } catch (err) {
    if (!(err instanceof ApiKeyRequiredError)) {
      throw err
    }
  }
  initDashboardNav(amgixApi)
}

$(() => {
  $('.dashboard-logo-slot').attr('src', logoUrl)
  const barEl = $('#dashboard-top-bar').get(0)
  if (barEl) {
    MDCTopAppBar.attachTo(barEl)
  }
  initDashboardErrorBar()
  initDashboardThemeToggle($('#theme-toggle'))
  initDashboardApiKeyButton($('#api-key-toggle'))

  $(document).on('amgix-dashboard-api-key-changed', () => {
    refreshActivePanel()
  })

  void bootstrapDashboard()
})

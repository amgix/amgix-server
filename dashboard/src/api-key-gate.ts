import type { Middleware } from '@amgix/amgix-client'
import $ from 'jquery'

import {
  clearStoredApiKey,
  getStoredApiKey,
  isProtectedV1Url,
  probeApiAccess,
  requestInitSkipsReauthOn401,
  setStoredApiKey,
  stripDashboardInternalHeaders,
} from './api-key'

const DIALOG_ID = 'dashboard-api-key-dialog'

const skipReauthByInit = new WeakMap<RequestInit, boolean>()

export class ApiKeyRequiredError extends Error {
  override name = 'ApiKeyRequiredError'
}

let unauthorizedHandling: Promise<void> | null = null

function syncApiKeyButtonState(): void {
  const $btn = $('#api-key-toggle')
  if (!$btn.length) {
    return
  }
  const hasKey = Boolean(getStoredApiKey())
  const tip = hasKey ? 'Change API key' : 'Set API key'
  $btn.attr({ title: tip, 'aria-label': tip })
  $btn.toggleClass('dashboard-api-key-toggle--active', hasKey)
}

export function promptForApiKey(opts?: {
  title?: string
  message?: string
  errorMessage?: string
}): Promise<string | null> {
  return new Promise((resolve) => {
    $(`#${DIALOG_ID}`).remove()

    const title = opts?.title ?? 'API key'
    const message =
      opts?.message ??
      'Enter an Amgix API key. Use an admin or read key for full dashboard access.'

    const $dialog = $('<dialog>', {
      id: DIALOG_ID,
      class: 'dashboard-collections-config-dialog dashboard-api-key-dialog',
      'aria-labelledby': 'dashboard-api-key-dialog-title',
    })

    const $input = $('<input>', {
      type: 'password',
      class: 'dashboard-api-key-input',
      autocomplete: 'off',
      spellcheck: false,
      'aria-label': 'API key',
    })

    const $error = $('<p>', {
      class: 'dashboard-api-key-error',
      text: opts?.errorMessage ?? '',
      hidden: !opts?.errorMessage,
    })

    const $connect = $('<button>', {
      type: 'button',
      class: 'dashboard-collections-config-dialog-close',
      text: 'Connect',
    })
    const $cancel = $('<button>', {
      type: 'button',
      class: 'dashboard-api-key-cancel',
      text: 'Cancel',
    })

    let settled = false
    const finish = (value: string | null): void => {
      if (settled) {
        return
      }
      settled = true
      const el = $dialog.get(0) as HTMLDialogElement | undefined
      if (el?.open) {
        el.close()
      }
      $dialog.remove()
      resolve(value)
    }

    $connect.on('click', () => {
      const value = String($input.val() ?? '').trim()
      if (!value) {
        $error.text('API key must not be empty.').prop('hidden', false)
        $input.trigger('focus')
        return
      }
      finish(value)
    })

    $cancel.on('click', () => finish(null))

    $dialog.on('close', () => finish(null))

    $dialog.on('cancel', (event) => {
      event.preventDefault()
      finish(null)
    })

    $input.on('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault()
        $connect.trigger('click')
      }
    })

    $dialog.append(
      $('<h4>', {
        id: 'dashboard-api-key-dialog-title',
        class: 'dashboard-collections-config-dialog-title',
        text: title,
      }),
      $('<p>', { class: 'dashboard-api-key-message', text: message }),
      $input,
      $error,
      $('<div>', { class: 'dashboard-collections-config-dialog-actions' }).append($connect, $cancel),
    )

    $('body').append($dialog)
    const el = $dialog.get(0) as HTMLDialogElement
    el.showModal()
    $input.trigger('focus')
  })
}

export async function ensureApiKeyAccess(): Promise<void> {
  const stored = getStoredApiKey()
  const initial = await probeApiAccess(stored)
  if (initial === 'ok' || initial === 'unavailable') {
    syncApiKeyButtonState()
    return
  }

  let errorMessage: string | undefined
  while (true) {
    const key = await promptForApiKey({
      title: 'API key required',
      message: 'Enter an Amgix API key to access this dashboard.',
      errorMessage,
    })
    if (!key) {
      throw new ApiKeyRequiredError()
    }

    const probe = await probeApiAccess(key)
    if (probe === 'ok') {
      setStoredApiKey(key)
      syncApiKeyButtonState()
      return
    }
    if (probe === 'unavailable') {
      setStoredApiKey(key)
      syncApiKeyButtonState()
      return
    }

    errorMessage = 'Authentication failed. Check your API key and try again.'
    clearStoredApiKey()
  }
}

async function handleApiUnauthorized(): Promise<void> {
  clearStoredApiKey()
  syncApiKeyButtonState()
  await ensureApiKeyAccess()
  $(document).trigger('amgix-dashboard-api-key-changed')
}

export function notifyApiUnauthorized(): void {
  if (unauthorizedHandling) {
    return
  }
  unauthorizedHandling = handleApiUnauthorized().catch(() => {
    /* user dismissed the dialog */
  }).finally(() => {
    unauthorizedHandling = null
  })
}

export function createApiKeyMiddleware(): Middleware {
  return {
    pre: async ({ url, init }) => {
      const skipReauth = requestInitSkipsReauthOn401(init)
      let nextInit = stripDashboardInternalHeaders(init)
      const key = getStoredApiKey()
      if (key) {
        const headers = new Headers(nextInit.headers as HeadersInit)
        headers.set('api-key', key)
        nextInit = { ...nextInit, headers }
      }
      skipReauthByInit.set(nextInit, skipReauth)
      return { url, init: nextInit }
    },
    post: async ({ response, url, init }) => {
      const skipReauth = skipReauthByInit.get(init) ?? false
      skipReauthByInit.delete(init)
      if (response.status === 401 && isProtectedV1Url(url) && !skipReauth) {
        notifyApiUnauthorized()
      }
      return response
    },
  }
}

export function initDashboardApiKeyButton($button: JQuery<HTMLButtonElement>): void {
  if (!$button.length) {
    return
  }
  syncApiKeyButtonState()
  $button.on('click', async () => {
    const current = getStoredApiKey()
    const key = await promptForApiKey({
      title: current ? 'Change API key' : 'Set API key',
      message: 'Enter an Amgix API key. Use an admin or read key for full dashboard access.',
    })
    if (!key) {
      return
    }

    const probe = await probeApiAccess(key)
    if (probe === 'unauthorized') {
      await promptForApiKey({
        title: current ? 'Change API key' : 'Set API key',
        message: 'Enter an Amgix API key. Use an admin or read key for full dashboard access.',
        errorMessage: 'Authentication failed. Check your API key and try again.',
      })
      return
    }

    setStoredApiKey(key)
    syncApiKeyButtonState()
    $(document).trigger('amgix-dashboard-api-key-changed')
  })
}

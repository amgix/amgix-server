import { ResponseError } from '@amgix/amgix-client'
import $ from 'jquery'

export function formatRequestError(context: string, err: unknown): string {
  if (err instanceof ResponseError) {
    return `${context} (HTTP ${err.response.status})`
  }
  if (err instanceof Error && err.message) {
    return `${context}: ${err.message}`
  }
  return context
}

export function stripModelNamespaceForDisplay(rawModel: string): string {
  const t = rawModel.trim()
  if (!t) {
    return ''
  }
  const i = t.lastIndexOf('/')
  if (i < 0 || i >= t.length - 1) {
    return t
  }
  const tail = t.slice(i + 1).trim()
  return tail || t
}

export function openReadonlyJsonDialog(opts: {
  dialogId: string
  titleId: string
  title: string
  json: string
}): void {
  const { dialogId, titleId, title, json } = opts
  $(`#${dialogId}`).remove()

  const $dialog = $('<dialog>', {
    id: dialogId,
    class: 'dashboard-collections-config-dialog',
    'aria-labelledby': titleId,
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
      id: titleId,
      class: 'dashboard-collections-config-dialog-title',
      text: title,
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

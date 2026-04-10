import $ from 'jquery'

let inited = false

function $bar(): JQuery<HTMLElement> {
  return $('#dashboard-error-bar')
}

export function initDashboardErrorBar(): void {
  if (inited) {
    return
  }
  const $el = $bar()
  if (!$el.length) {
    return
  }
  inited = true
  $el.find('.dashboard-error-bar__dismiss').on('click', () => {
    hideDashboardError()
  })
}

export function showDashboardError(message: string): void {
  initDashboardErrorBar()
  const $el = $bar()
  if (!$el.length) {
    return
  }
  const text = message.trim() || 'Something went wrong.'
  $el.find('.dashboard-error-bar__message').text(text)
  $el.removeAttr('hidden')
}

export function hideDashboardError(): void {
  const $el = $bar()
  if (!$el.length) {
    return
  }
  $el.attr('hidden', '')
  $el.find('.dashboard-error-bar__message').empty()
}

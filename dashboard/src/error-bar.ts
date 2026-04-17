import { MDCSnackbar } from '@material/snackbar/component'

let snackbar: MDCSnackbar | null = null

function getSnackbarEl(): HTMLElement | null {
  return document.getElementById('dashboard-error-snackbar')
}

export function initDashboardErrorBar(): void {
  if (snackbar) {
    return
  }
  const el = getSnackbarEl()
  if (!el) {
    return
  }
  snackbar = MDCSnackbar.attachTo(el)
  snackbar.timeoutMs = 5000
}

export function showDashboardError(message: string): void {
  initDashboardErrorBar()
  if (!snackbar) {
    return
  }
  const text = message.trim() || 'Something went wrong.'
  snackbar.labelText = text
  if (!snackbar.isOpen) {
    snackbar.open()
  }
}

export function hideDashboardError(): void {
  if (!snackbar?.isOpen) {
    return
  }
  snackbar.close('dismiss')
}

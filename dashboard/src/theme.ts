import $ from 'jquery'

export type DashboardTheme = 'light' | 'dark'

const STORAGE_KEY = 'amgix-dashboard-theme'

export function getStoredTheme(): DashboardTheme {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'dark' || v === 'light') {
      return v
    }
  } catch {
    /* localStorage unavailable */
  }
  return 'dark'
}

export function setStoredTheme(theme: DashboardTheme): void {
  try {
    localStorage.setItem(STORAGE_KEY, theme)
  } catch {
    /* ignore */
  }
}

export function applyDashboardTheme(theme: DashboardTheme): void {
  $('html').attr('data-theme', theme)
}

export function syncThemeToggle($button: JQuery<HTMLButtonElement>): void {
  const dark = $('html').attr('data-theme') === 'dark'
  const $icon = $button.find('[data-theme-icon]')
  if ($icon.length) {
    $icon.text(dark ? 'light_mode' : 'dark_mode')
  }
  const tip = dark ? 'Switch to light theme' : 'Switch to dark theme'
  $button.attr({ title: tip, 'aria-label': tip })
}

export function initDashboardThemeToggle($button: JQuery<HTMLButtonElement>): void {
  if (!$button.length) {
    return
  }
  applyDashboardTheme(getStoredTheme())
  syncThemeToggle($button)
  $button.on('click', () => {
    const next: DashboardTheme = $('html').attr('data-theme') === 'dark' ? 'light' : 'dark'
    applyDashboardTheme(next)
    setStoredTheme(next)
    syncThemeToggle($button)
    $(document).trigger('amgix-dashboard-theme-changed')
  })
}

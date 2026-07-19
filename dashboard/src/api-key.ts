const STORAGE_KEY = 'amgix-dashboard-api-key'

export const DASHBOARD_SKIP_REAUTH_HEADER = 'X-Amgix-Dashboard-Skip-Reauth'

const PUBLIC_V1_PATHS = new Set(['/v1/version', '/v1/health/check', '/v1/health/ready'])

export function amgixApiBasePath(): string {
  if (import.meta.env.DEV) {
    return ''
  }
  return window.location.origin
}

export function getStoredApiKey(): string | null {
  try {
    const value = sessionStorage.getItem(STORAGE_KEY)?.trim()
    return value || null
  } catch {
    return null
  }
}

export function setStoredApiKey(apiKey: string): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, apiKey.trim())
  } catch {
    /* sessionStorage unavailable */
  }
}

export function clearStoredApiKey(): void {
  try {
    sessionStorage.removeItem(STORAGE_KEY)
  } catch {
    /* sessionStorage unavailable */
  }
}

/** Pass as the OpenAPI client initOverrides on dashboard write ops that may 401 with a read-only key. */
export function dashboardSkipReauthOn401(): RequestInit {
  return { headers: { [DASHBOARD_SKIP_REAUTH_HEADER]: '1' } }
}

export function requestInitSkipsReauthOn401(init: RequestInit): boolean {
  const headers = new Headers(init.headers as HeadersInit)
  return headers.get(DASHBOARD_SKIP_REAUTH_HEADER) === '1'
}

export function stripDashboardInternalHeaders(init: RequestInit): RequestInit {
  const headers = new Headers(init.headers as HeadersInit)
  headers.delete(DASHBOARD_SKIP_REAUTH_HEADER)
  return { ...init, headers }
}

export function isProtectedV1Url(url: string): boolean {
  try {
    const path = new URL(url, window.location.origin).pathname
    return path.startsWith('/v1/') && !PUBLIC_V1_PATHS.has(path)
  } catch {
    const path = url.split('?')[0] ?? url
    return path.startsWith('/v1/') && !PUBLIC_V1_PATHS.has(path)
  }
}

export function amgixFetch(path: string, opts?: { apiKey?: string; init?: RequestInit }): Promise<Response> {
  const base = amgixApiBasePath()
  const url = path.startsWith('http') ? path : `${base}${path}`
  const headers = new Headers(opts?.init?.headers)
  const key = opts?.apiKey ?? getStoredApiKey()
  if (key) {
    headers.set('api-key', key)
  }
  return fetch(url, { ...opts?.init, headers })
}

export type ApiProbeResult = 'ok' | 'unauthorized' | 'unavailable'

export async function probeApiAccess(apiKey?: string | null): Promise<ApiProbeResult> {
  try {
    const res = await amgixFetch('/v1/collections', { apiKey: apiKey ?? undefined })
    if (res.status === 401) {
      return 'unauthorized'
    }
    return 'ok'
  } catch {
    return 'unavailable'
  }
}

import type { AmgixApi } from '@amgix/amgix-client'

export abstract class DashboardPanel {
  abstract init(api: AmgixApi): void

  /** Re-run when the user clicks the already-active top-nav item. */
  refresh(api: AmgixApi): void {
    this.init(api)
  }
}

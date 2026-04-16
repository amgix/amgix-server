import {
  ResponseError,
  VectorSearchWeightFieldEnum,
  type AmgixApi,
  type CollectionConfig,
  type SearchQuery,
  type SearchResult,
  type VectorConfig,
  type VectorSearchWeight,
} from '@amgix/amgix-client'
import $ from 'jquery'

import { hideDashboardError, showDashboardError } from '../error-bar'
import { DashboardPanel } from './panel-base'

const QUERY_SEARCH_LIMIT = 10

function formatSearchError(context: string, err: unknown): string {
  if (err instanceof ResponseError) {
    return `${context} (HTTP ${err.response.status})`
  }
  if (err instanceof Error && err.message) {
    return `${context}: ${err.message}`
  }
  return context
}

function formatResultTimestamp(ts: Date): string {
  if (!(ts instanceof Date) || Number.isNaN(ts.getTime())) {
    return ''
  }
  return ts.toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'medium' })
}

function truncateCell(s: string, maxLen: number): string {
  const t = s.trim()
  if (t.length <= maxLen) {
    return t
  }
  return `${t.slice(0, maxLen)}…`
}

function indexFieldsForVector(v: VectorConfig): string[] {
  if (v.index_fields != null && v.index_fields.length > 0) {
    return v.index_fields.map((f) => String(f))
  }
  return ['content']
}

function vectorNameWithOptionalModel(v: VectorConfig): string {
  const name = String(v.name ?? '').trim() || '(unnamed)'
  const raw = v.model != null ? String(v.model).trim() : ''
  if (!raw) {
    return name
  }
  const model = raw.length > 96 ? `${raw.slice(0, 96)}…` : raw
  return `${name} - ${model}`
}

type QueryVectorArm = {
  vectorName: string
  field: VectorSearchWeightFieldEnum
  label: string
}

function parseSearchField(raw: string): VectorSearchWeightFieldEnum | null {
  const s = String(raw).trim().toLowerCase()
  if (s === 'name') {
    return VectorSearchWeightFieldEnum.Name
  }
  if (s === 'description') {
    return VectorSearchWeightFieldEnum.Description
  }
  if (s === 'content') {
    return VectorSearchWeightFieldEnum.Content
  }
  return null
}

function vectorArmsFromConfig(config: CollectionConfig): QueryVectorArm[] {
  const out: QueryVectorArm[] = []
  for (const v of config.vectors) {
    const vectorName = String(v.name ?? '').trim()
    if (!vectorName) {
      continue
    }
    const head = vectorNameWithOptionalModel(v)
    for (const rawField of indexFieldsForVector(v)) {
      const field = parseSearchField(rawField)
      if (field == null) {
        continue
      }
      out.push({
        vectorName,
        field,
        label: `${head} (${field})`,
      })
    }
  }
  return out
}

export class QueryPanel extends DashboardPanel {
  private queryApi: AmgixApi | null = null
  private searchGeneration = 0
  private vectorListGeneration = 0
  private domBuilt = false

  deactivate(): void {
    this.searchGeneration += 1
    this.vectorListGeneration += 1
    this.queryApi = null
  }

  init(api: AmgixApi): void {
    this.queryApi = api
    const $root = $('#panel-query [data-query-root]')
    if (!$root.length) {
      return
    }
    if (!this.domBuilt) {
      this.buildQueryDom($root)
      this.domBuilt = true
    }
    void this.loadCollections(api, $root)
  }

  private buildQueryDom($root: JQuery<HTMLElement>): void {
    const selectId = 'dashboard-query-collection-select'
    const queryId = 'dashboard-query-text'

    const $form = $('<form>', {
      class: 'dashboard-query-form',
      attr: { novalidate: '', 'data-query-form': '' },
    })

    const $vectorsBlock = $('<div>', {
      class: 'dashboard-query-vectors',
      attr: { 'data-query-vectors-wrap': '', hidden: '' },
    }).append(
      $('<span>', {
        class: 'dashboard-query-vectors-heading',
        text: 'Indexed vectors (uncheck to exclude from search)',
      }),
      $('<ul>', {
        class: 'dashboard-query-vectors-list',
        attr: { 'data-query-vectors-list': '', role: 'list' },
      }),
    )

    const $select = $('<select>', {
      id: selectId,
      class: 'dashboard-query-select',
      attr: { 'data-query-collection': '', 'aria-required': 'true' },
    })
    const $textarea = $('<textarea>', {
      id: queryId,
      class: 'dashboard-query-textarea',
      attr: {
        'data-query-text': '',
        rows: '2',
        spellcheck: 'true',
        autocomplete: 'off',
        'aria-required': 'true',
      },
      title: 'Enter runs search. Shift+Enter inserts a new line.',
    })

    const $grid = $('<div>', { class: 'dashboard-query-form-grid' })
    const $actions = $('<div>', { class: 'dashboard-query-actions' }).append(
      $('<button>', {
        type: 'submit',
        class: 'dashboard-query-submit',
        text: 'Search',
        attr: { 'data-query-submit': '' },
      }),
    )

    $grid.append(
      $('<label>', {
        class: 'dashboard-query-side-label',
        attr: { for: selectId },
        text: 'Collections:',
      }),
      $('<div>', { class: 'dashboard-query-control-stack' }).append($select),
      $('<label>', {
        class: 'dashboard-query-side-label',
        attr: { for: queryId },
        text: 'Query:',
      }),
      $('<div>', { class: 'dashboard-query-control-stack' }).append($textarea, $vectorsBlock),
      $actions,
    )

    $form.append($grid)

    const $tableWrap = $('<div>', {
      class: 'dashboard-query-results-wrap',
      attr: { 'data-query-results-wrap': '' },
    })
    const $table = $('<table>', {
      class: 'dashboard-query-results',
      attr: { 'data-query-results': '' },
    })
    const $tbodyInit = $('<tbody>', { attr: { 'data-query-results-body': '' } })
    $tbodyInit.append(
      $('<tr>').append(
        $('<td>', {
          class: 'dashboard-query-results-placeholder',
          colspan: 6,
          text: 'Run a search to see matching documents.',
        }),
      ),
    )
    $table.append(
      $('<thead>').append(
        $('<tr>').append(
          $('<th>', { scope: 'col', text: 'Score' }),
          $('<th>', { scope: 'col', text: 'ID' }),
          $('<th>', { scope: 'col', text: 'Name' }),
          $('<th>', { scope: 'col', text: 'Description' }),
          $('<th>', { scope: 'col', text: 'Tags' }),
          $('<th>', { scope: 'col', text: 'Updated' }),
        ),
      ),
      $tbodyInit,
    )
    $tableWrap.append($table)

    $root.empty().append($form, $tableWrap)

    $form.on('submit', (ev) => {
      ev.preventDefault()
      const apiRef = this.queryApi
      if (apiRef != null) {
        void this.runSearch(apiRef, $root)
      }
    })

    $form.find('[data-query-collection]').on('change', () => {
      const apiRef = this.queryApi
      if (apiRef != null) {
        void this.refreshVectorsForSelection(apiRef, $root)
      }
    })

    $form.find('[data-query-text]').on('keydown', (ev) => {
      if (ev.key !== 'Enter' || ev.shiftKey) {
        return
      }
      ev.preventDefault()
      const apiRef = this.queryApi
      if (apiRef != null) {
        void this.runSearch(apiRef, $root)
      }
    })
  }

  private async loadCollections(api: AmgixApi, $root: JQuery<HTMLElement>): Promise<void> {
    const $sel = $root.find('[data-query-collection]')
    if (!$sel.length) {
      return
    }
    const prev = String($sel.val() ?? '')
    $sel.prop('disabled', true)
    try {
      const names = (await api.listCollections()).slice().sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }))
      hideDashboardError()
      $sel.empty()
      $sel.append($('<option>', { value: '', text: 'Select a collection…' }))
      for (const name of names) {
        $sel.append($('<option>', { value: name, text: name }))
      }
      if (prev && names.includes(prev)) {
        $sel.val(prev)
      }
    } catch (err) {
      $sel.empty()
      $sel.append($('<option>', { value: '', text: 'Select a collection…' }))
      showDashboardError(formatSearchError('Could not load collections', err))
    } finally {
      $sel.prop('disabled', false)
    }
    const apiRef = this.queryApi
    if (apiRef != null) {
      void this.refreshVectorsForSelection(apiRef, $root)
    }
  }

  private setVectorsUi(
    $root: JQuery<HTMLElement>,
    mode: 'hidden' | 'loading' | 'lines' | 'empty' | 'error',
    rows?: QueryVectorArm[],
    errorMessage?: string,
  ): void {
    const $wrap = $root.find('[data-query-vectors-wrap]')
    const $ul = $root.find('[data-query-vectors-list]')
    if (!$wrap.length || !$ul.length) {
      return
    }
    if (mode === 'hidden') {
      $wrap.attr('hidden', '')
      $ul.empty()
      return
    }
    $wrap.removeAttr('hidden')
    $ul.empty()
    if (mode === 'loading') {
      $ul.append(
        $('<li>', {
          class: 'dashboard-query-vectors-item dashboard-query-vectors-item--muted',
          role: 'listitem',
          text: 'Loading…',
        }),
      )
      return
    }
    if (mode === 'error') {
      $ul.append(
        $('<li>', {
          class: 'dashboard-query-vectors-item dashboard-query-vectors-item--error',
          role: 'listitem',
          text: errorMessage != null && errorMessage.trim() ? errorMessage.trim() : 'Could not load collection config.',
        }),
      )
      return
    }
    if (mode === 'empty') {
      $ul.append(
        $('<li>', {
          class: 'dashboard-query-vectors-item dashboard-query-vectors-item--muted',
          role: 'listitem',
          text: 'No vectors configured for this collection.',
        }),
      )
      return
    }
    for (let i = 0; i < (rows ?? []).length; i++) {
      const row = rows![i]!
      const inputId = `dashboard-query-vw-${i}`
      const $cb = $('<input>', {
        type: 'checkbox',
        id: inputId,
        class: 'dashboard-query-vectors-checkbox',
        attr: {
          'data-vector-name': row.vectorName,
          'data-vector-field': row.field,
        },
      })
      $cb.prop('checked', true)
      const $text = $('<span>', { class: 'dashboard-query-vectors-check-text', text: row.label })
      const $lab = $('<label>', {
        class: 'dashboard-query-vectors-check-label',
        attr: { for: inputId },
      })
      $lab.append($cb, $text)
      $ul.append($('<li>', { class: 'dashboard-query-vectors-item', role: 'listitem' }).append($lab))
    }
  }

  private vectorWeightsFromCheckedBoxes($root: JQuery<HTMLElement>): 'default' | 'none' | VectorSearchWeight[] {
    const $boxes = $root.find('input.dashboard-query-vectors-checkbox')
    if (!$boxes.length) {
      return 'default'
    }
    const $checked = $root.find('input.dashboard-query-vectors-checkbox:checked')
    if (!$checked.length) {
      return 'none'
    }
    const weights: VectorSearchWeight[] = []
    $checked.each((_, el) => {
      const $e = $(el)
      const vector_name = String($e.attr('data-vector-name') ?? '')
      const fieldAttr = String($e.attr('data-vector-field') ?? '')
      const field =
        fieldAttr === VectorSearchWeightFieldEnum.Name
          ? VectorSearchWeightFieldEnum.Name
          : fieldAttr === VectorSearchWeightFieldEnum.Description
            ? VectorSearchWeightFieldEnum.Description
            : fieldAttr === VectorSearchWeightFieldEnum.Content
              ? VectorSearchWeightFieldEnum.Content
              : null
      if (!vector_name || field == null) {
        return
      }
      weights.push({ vector_name, field, weight: 1.0 })
    })
    return weights.length > 0 ? weights : 'none'
  }

  private async refreshVectorsForSelection(api: AmgixApi, $root: JQuery<HTMLElement>): Promise<void> {
    const gen = ++this.vectorListGeneration
    const $sel = $root.find('[data-query-collection]')
    const collectionName = String($sel.val() ?? '').trim()

    if (!collectionName) {
      if (gen === this.vectorListGeneration) {
        this.setVectorsUi($root, 'hidden')
      }
      return
    }

    this.setVectorsUi($root, 'loading')

    try {
      const config = await api.getCollectionConfig({ collectionName })
      if (gen !== this.vectorListGeneration) {
        return
      }
      const arms = vectorArmsFromConfig(config)
      if (arms.length === 0) {
        this.setVectorsUi($root, 'empty')
      } else {
        this.setVectorsUi($root, 'lines', arms)
      }
    } catch (err) {
      if (gen !== this.vectorListGeneration) {
        return
      }
      this.setVectorsUi($root, 'error', undefined, formatSearchError('Could not load collection config', err))
    }
  }

  private async runSearch(api: AmgixApi, $root: JQuery<HTMLElement>): Promise<void> {
    const gen = ++this.searchGeneration
    const $sel = $root.find('[data-query-collection]')
    const $ta = $root.find('[data-query-text]')
    const $btn = $root.find('[data-query-submit]')
    const $tbody = $root.find('[data-query-results-body]')

    const collectionName = String($sel.val() ?? '').trim()
    const queryText = String($ta.val() ?? '').trim()

    if (!collectionName) {
      showDashboardError('Choose a collection before searching.')
      return
    }
    if (!queryText) {
      showDashboardError('Enter a search query.')
      return
    }

    const vectorWeights = this.vectorWeightsFromCheckedBoxes($root)
    if (vectorWeights === 'none') {
      showDashboardError('Select at least one indexed vector to search.')
      return
    }

    hideDashboardError()
    $btn.prop('disabled', true)
    $tbody.empty().append(
      $('<tr>').append(
        $('<td>', {
          class: 'dashboard-query-results-placeholder',
          colspan: 6,
          text: 'Searching…',
        }),
      ),
    )

    try {
      const searchQuery: SearchQuery = {
        query: queryText,
        limit: QUERY_SEARCH_LIMIT,
      }
      if (vectorWeights !== 'default') {
        searchQuery.vector_weights = vectorWeights
      }
      // console.log(searchQuery)
      const results = await api.search({
        collectionName,
        searchQuery,
      })
      if (gen !== this.searchGeneration) {
        return
      }
      this.renderResults($tbody, results)
    } catch (err) {
      if (gen !== this.searchGeneration) {
        return
      }
      $tbody.empty().append(
        $('<tr>').append(
          $('<td>', {
            class: 'dashboard-query-results-placeholder dashboard-query-results-placeholder--error',
            colspan: 6,
            text: 'Search failed. See the error bar above.',
          }),
        ),
      )
      showDashboardError(formatSearchError('Search failed', err))
    } finally {
      if (gen === this.searchGeneration) {
        $btn.prop('disabled', false)
      }
    }
  }

  private renderResults($tbody: JQuery<HTMLElement>, results: SearchResult[]): void {
    $tbody.empty()
    if (results.length === 0) {
      $tbody.append(
        $('<tr>').append(
          $('<td>', {
            class: 'dashboard-query-results-placeholder',
            colspan: 6,
            text: 'No documents matched your query.',
          }),
        ),
      )
      return
    }

    for (const r of results) {
      const name = r.name != null ? String(r.name) : ''
      const desc = r.description != null ? String(r.description) : ''
      const tags = r.tags != null && r.tags.length > 0 ? r.tags.join(', ') : ''
      const score = Number.isFinite(r.score) ? r.score.toFixed(4) : String(r.score)
      const updated = formatResultTimestamp(r.timestamp)

      $tbody.append(
        $('<tr>').append(
          $('<td>', { class: 'dashboard-query-cell-num', text: score }),
          $('<td>', { class: 'dashboard-query-cell-mono', text: r.id }),
          $('<td>', { text: truncateCell(name, 200) }),
          $('<td>', { class: 'dashboard-query-cell-desc', text: truncateCell(desc, 400) }),
          $('<td>', { text: truncateCell(tags, 200) }),
          $('<td>', { class: 'dashboard-query-cell-nowrap', text: updated }),
        ),
      )
    }
  }
}

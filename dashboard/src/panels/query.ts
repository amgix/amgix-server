import {
  CollectionConfigToJSON,
  DocumentToJSON,
  SearchQueryFusionModeEnum,
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
import { formatRequestError, openReadonlyJsonDialog, stripModelNamespaceForDisplay } from './common'
import { DashboardPanel } from './panel-base'

const QUERY_SEARCH_LIMIT = 10

const VECTOR_WEIGHT_SLIDER_MIN = 0
const VECTOR_WEIGHT_SLIDER_MAX = 10
const VECTOR_WEIGHT_SLIDER_STEP = 0.25
const VECTOR_WEIGHT_SLIDER_DEFAULT = 1

function vectorWeightSliderDisplayLabel(value: string): string {
  const n = Number.parseFloat(value)
  if (!Number.isFinite(n)) {
    return value
  }
  const t = Math.round(n / VECTOR_WEIGHT_SLIDER_STEP) * VECTOR_WEIGHT_SLIDER_STEP
  return String(parseFloat(t.toFixed(2)))
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
  const display = stripModelNamespaceForDisplay(raw)
  const model = display.length > 96 ? `${display.slice(0, 96)}…` : display
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

    const fusionSelectId = 'dashboard-query-fusion-select'
    const $fusionSelect = $('<select>', {
      id: fusionSelectId,
      class: 'dashboard-query-select dashboard-query-fusion-select',
      attr: { 'data-query-fusion-mode': '' },
    }).append(
      $('<option>', { value: SearchQueryFusionModeEnum.Rrf, text: 'RRF' }),
      $('<option>', { value: SearchQueryFusionModeEnum.Linear, text: 'Linear' }),
    )
    $fusionSelect.val(SearchQueryFusionModeEnum.Rrf)

    const limitSliderId = 'dashboard-query-limit-slider'
    const limitDefault = String(QUERY_SEARCH_LIMIT)
    const $limitSlider = $('<input>', {
      type: 'range',
      id: limitSliderId,
      class: 'dashboard-query-vectors-limit-slider',
      attr: {
        min: '1',
        max: '100',
        step: '1',
        value: limitDefault,
        'data-query-limit-slider': '',
        'aria-valuemin': '1',
        'aria-valuemax': '100',
        'aria-valuenow': limitDefault,
        'aria-label': 'Results limit',
      },
    })
    const $limitValue = $('<span>', {
      class: 'dashboard-query-limit-value',
      text: limitDefault,
    })

    const $vectorsList = $('<ul>', {
      class: 'dashboard-query-vectors-list',
      attr: { 'data-query-vectors-list': '', role: 'list' },
    })

    const $vectorsExtras = $('<div>', { class: 'dashboard-query-vectors-extras' }).append(
      $('<div>', { class: 'dashboard-query-vectors-extras-row' }).append(
        $('<label>', {
          class: 'dashboard-query-vectors-extras-label',
          attr: { for: fusionSelectId },
          text: 'Fusion Mode:',
        }),
        $('<div>', { class: 'dashboard-query-vectors-extras-control' }).append($fusionSelect),
      ),
      $('<div>', { class: 'dashboard-query-vectors-extras-row' }).append(
        $('<label>', {
          class: 'dashboard-query-vectors-extras-label',
          attr: { for: limitSliderId },
          text: 'Results Limit:',
        }),
        $('<div>', {
          class: 'dashboard-query-vectors-extras-control dashboard-query-vectors-extras-control--limit',
        }).append($limitSlider, $limitValue),
      ),
    )

    const $vectorsBlock = $('<div>', {
      class: 'dashboard-query-vectors',
      attr: { 'data-query-vectors-wrap': '', hidden: '' },
    }).append(
      $('<div>', { class: 'dashboard-query-vectors-header' }).append(
        $('<span>', { class: 'dashboard-query-vectors-col-title', text: 'Indexed Vectors' }),
        $('<span>', { class: 'dashboard-query-vectors-col-title dashboard-query-vectors-col-title--weights', text: 'Weights' }),
      ),
      $vectorsList,
      $vectorsExtras,
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
        rows: '1',
        spellcheck: 'true',
        autocomplete: 'off',
        'aria-required': 'true',
      },
      title: 'Enter runs the search. Shift+Enter inserts a new line.',
    })

    const $fullConfigLink = $('<a>', {
      href: '#',
      class: 'dashboard-collections-full-config-link',
      text: 'Full Configuration',
      attr: { 'data-query-full-config': '' },
    })
    $fullConfigLink.on('click', (e) => {
      e.preventDefault()
      const apiRef = this.queryApi
      if (apiRef != null) {
        void this.showQueryFullConfiguration(apiRef, $root)
      }
    })

    const $collectionRow = $('<div>', { class: 'dashboard-query-collection-row' }).append($select, $fullConfigLink)

    const $submitBtn = $('<button>', {
      type: 'submit',
      class: 'dashboard-query-submit',
      text: 'Search',
      attr: { 'data-query-submit': '' },
    })
    const $queryRow = $('<div>', { class: 'dashboard-query-query-row' }).append($textarea, $submitBtn)

    const $grid = $('<div>', { class: 'dashboard-query-form-grid' })

    $grid.append(
      $('<label>', {
        class: 'dashboard-query-side-label',
        attr: { for: selectId },
        text: 'Collections:',
      }),
      $('<div>', { class: 'dashboard-query-control-stack' }).append($collectionRow),
      $('<label>', {
        class: 'dashboard-query-side-label',
        attr: { for: queryId },
        text: 'Query:',
      }),
      $('<div>', { class: 'dashboard-query-control-stack' }).append($queryRow, $vectorsBlock),
    )

    $form.append($grid)

    const $tableWrap = $('<div>', {
      class: 'dashboard-query-results-wrap',
      attr: { 'data-query-results-wrap': '' },
    })
    const $searchTiming = $('<p>', {
      class: 'dashboard-query-search-timing dashboard-query-search-timing--empty',
      attr: { 'data-query-search-timing': '', 'aria-hidden': 'true' },
      text: '',
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
          colspan: 5,
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
          $('<th>', { scope: 'col', text: 'Updated' }),
        ),
      ),
      $tbodyInit,
    )
    $tableWrap.append($table)

    $root.empty().append($form, $searchTiming, $tableWrap)

    $form.on('submit', (ev) => {
      ev.preventDefault()
      const apiRef = this.queryApi
      if (apiRef != null) {
        void this.runSearch(apiRef, $root)
      }
    })

    $form.find('[data-query-collection]').on('change', () => {
      this.searchGeneration += 1
      this.resetQueryResultsToPlaceholder($root)
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

    $root.on('input', '.dashboard-query-vectors-weight-slider', function (this: HTMLInputElement) {
      this.setAttribute('aria-valuenow', this.value)
      const label = vectorWeightSliderDisplayLabel(this.value)
      $(this).closest('.dashboard-query-vectors-weight').find('.dashboard-query-vectors-weight-value').text(label)
    })

    $root.on('change', 'input.dashboard-query-vectors-checkbox', function (this: HTMLInputElement) {
      $(this).closest('li.dashboard-query-vectors-item').find('.dashboard-query-vectors-weight-slider').prop('disabled', !this.checked)
    })

    $root.on('input', '.dashboard-query-vectors-limit-slider', function (this: HTMLInputElement) {
      this.setAttribute('aria-valuenow', this.value)
      $(this)
        .closest('.dashboard-query-vectors-extras-control--limit')
        .find('.dashboard-query-limit-value')
        .text(this.value)
    })

    $root.on('click', '[data-query-doc-id]', (ev) => {
      ev.preventDefault()
      const $t = $(ev.currentTarget)
      const documentId = String($t.attr('data-query-doc-id') ?? '').trim()
      const collectionName = String($t.attr('data-query-doc-collection') ?? '').trim()
      if (!documentId || !collectionName) {
        return
      }
      const apiRef = this.queryApi
      if (apiRef != null) {
        void this.openDocumentFromQueryTable(apiRef, collectionName, documentId)
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
      showDashboardError(formatRequestError('Could not load collections', err))
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
      const sliderId = `dashboard-query-vw-w-${i}`
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
      const defVal = String(VECTOR_WEIGHT_SLIDER_DEFAULT)
      const $slider = $('<input>', {
        type: 'range',
        id: sliderId,
        class: 'dashboard-query-vectors-weight-slider',
        attr: {
          min: String(VECTOR_WEIGHT_SLIDER_MIN),
          max: String(VECTOR_WEIGHT_SLIDER_MAX),
          step: String(VECTOR_WEIGHT_SLIDER_STEP),
          value: defVal,
          'aria-valuemin': String(VECTOR_WEIGHT_SLIDER_MIN),
          'aria-valuemax': String(VECTOR_WEIGHT_SLIDER_MAX),
          'aria-valuenow': defVal,
          'aria-label': `Weight for ${row.label}`,
        },
      })
      const $weightVal = $('<span>', {
        class: 'dashboard-query-vectors-weight-value',
        text: vectorWeightSliderDisplayLabel(defVal),
      })
      const $weightCol = $('<div>', { class: 'dashboard-query-vectors-weight' }).append($slider, $weightVal)
      const $armCol = $('<div>', { class: 'dashboard-query-vectors-arm' }).append($lab)
      const $row = $('<li>', {
        class: 'dashboard-query-vectors-item dashboard-query-vectors-item--with-weight',
        role: 'listitem',
      }).append($armCol, $weightCol)
      $ul.append($row)
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
      const $slider = $e.closest('li.dashboard-query-vectors-item').find('.dashboard-query-vectors-weight-slider')
      let weight = Number.parseFloat(String($slider.val() ?? ''))
      if (!Number.isFinite(weight)) {
        weight = VECTOR_WEIGHT_SLIDER_DEFAULT
      }
      weight = Math.min(VECTOR_WEIGHT_SLIDER_MAX, Math.max(VECTOR_WEIGHT_SLIDER_MIN, weight))
      weights.push({ vector_name, field, weight })
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
      this.setVectorsUi($root, 'error', undefined, formatRequestError('Could not load collection config', err))
    }
  }

  private async openDocumentFromQueryTable(
    api: AmgixApi,
    collectionName: string,
    documentId: string,
  ): Promise<void> {
    try {
      const doc = await api.getDocument({ collectionName, documentId })
      hideDashboardError()
      const json = JSON.stringify(DocumentToJSON(doc), null, 2)
      openReadonlyJsonDialog({
        dialogId: 'query-document-dialog',
        titleId: 'query-document-title',
        title: `Document ${documentId} (${collectionName})`,
        json,
      })
    } catch (err) {
      showDashboardError(formatRequestError('Could not load document', err))
    }
  }

  private resetQueryResultsToPlaceholder($root: JQuery<HTMLElement>): void {
    const $tbody = $root.find('[data-query-results-body]')
    const $timing = $root.find('[data-query-search-timing]')
    $tbody.empty().append(
      $('<tr>').append(
        $('<td>', {
          class: 'dashboard-query-results-placeholder',
          colspan: 5,
          text: 'Run a search to see matching documents.',
        }),
      ),
    )
    $timing.text('').addClass('dashboard-query-search-timing--empty').attr('aria-hidden', 'true')
  }

  private async showQueryFullConfiguration(api: AmgixApi, $root: JQuery<HTMLElement>): Promise<void> {
    const collectionName = String($root.find('[data-query-collection]').val() ?? '').trim()
    if (!collectionName) {
      showDashboardError('Choose a collection to view its configuration.')
      return
    }
    try {
      const config = await api.getCollectionConfig({ collectionName })
      hideDashboardError()
      const json = JSON.stringify(CollectionConfigToJSON(config), null, 2)
      openReadonlyJsonDialog({
        dialogId: 'query-full-config-dialog',
        titleId: 'query-full-config-title',
        title: `Configuration for ${collectionName}`,
        json,
      })
    } catch (err) {
      showDashboardError(formatRequestError('Could not load collection configuration', err))
    }
  }

  private async runSearch(api: AmgixApi, $root: JQuery<HTMLElement>): Promise<void> {
    const gen = ++this.searchGeneration
    const $sel = $root.find('[data-query-collection]')
    const $ta = $root.find('[data-query-text]')
    const $btn = $root.find('[data-query-submit]')
    const $tbody = $root.find('[data-query-results-body]')
    const $timing = $root.find('[data-query-search-timing]')

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
    $timing.text('').addClass('dashboard-query-search-timing--empty').attr('aria-hidden', 'true')
    $btn.prop('disabled', true)
    $tbody.empty().append(
      $('<tr>').append(
        $('<td>', {
          class: 'dashboard-query-results-placeholder',
          colspan: 5,
          text: 'Searching…',
        }),
      ),
    )

    try {
      const fusionVal = String($root.find('[data-query-fusion-mode]').val() ?? '')
      const fusion_mode =
        fusionVal === SearchQueryFusionModeEnum.Linear
          ? SearchQueryFusionModeEnum.Linear
          : SearchQueryFusionModeEnum.Rrf

      let limit = Number.parseInt(String($root.find('[data-query-limit-slider]').val() ?? ''), 10)
      if (!Number.isFinite(limit)) {
        limit = QUERY_SEARCH_LIMIT
      }
      limit = Math.min(100, Math.max(1, limit))

      const searchQuery: SearchQuery = {
        query: queryText,
        limit,
        raw_scores: true,
        fusion_mode,
      }
      if (vectorWeights !== 'default') {
        searchQuery.vector_weights = vectorWeights
      }
      // console.log(searchQuery)
      const t0 = performance.now()
      const results = await api.search({
        collectionName,
        searchQuery,
      })
      const elapsedMs = Math.max(0, Math.round(performance.now() - t0))
      if (gen !== this.searchGeneration) {
        return
      }
      const timingText =
        results.length > 0 ? `Found in ${elapsedMs} ms` : `No results in ${elapsedMs} ms`
      $timing.removeClass('dashboard-query-search-timing--empty').removeAttr('aria-hidden').text(timingText)
      this.renderResults($tbody, results, collectionName)
    } catch (err) {
      if (gen !== this.searchGeneration) {
        return
      }
      $timing.text('').addClass('dashboard-query-search-timing--empty').attr('aria-hidden', 'true')
      $tbody.empty().append(
        $('<tr>').append(
          $('<td>', {
            class: 'dashboard-query-results-placeholder dashboard-query-results-placeholder--error',
            colspan: 5,
            text: 'Search failed. See the error bar above.',
          }),
        ),
      )
      showDashboardError(formatRequestError('Search failed', err))
    } finally {
      if (gen === this.searchGeneration) {
        $btn.prop('disabled', false)
      }
    }
  }

  private formatRawScoresDetail(r: SearchResult): string {
    const items = r.vector_scores
    if (items == null || items.length === 0) {
      return ''
    }
    return items
      .map((vs) => {
        const v = String(vs.vector ?? '').trim() || '(vector)'
        const f = String(vs.field ?? '').trim() || '(field)'
        const s = Number.isFinite(vs.score) ? vs.score.toFixed(4) : String(vs.score)
        return `${v} (${f}): ${s}`
      })
      .join('; ')
  }

  private renderResults(
    $tbody: JQuery<HTMLElement>,
    results: SearchResult[],
    searchCollectionName: string,
  ): void {
    $tbody.empty()
    if (results.length === 0) {
      $tbody.append(
        $('<tr>').append(
          $('<td>', {
            class: 'dashboard-query-results-placeholder',
            colspan: 5,
            text: 'No documents matched your query.',
          }),
        ),
      )
      return
    }

    for (const r of results) {
      const name = r.name != null ? String(r.name) : ''
      const desc = r.description != null ? String(r.description) : ''
      const score = Number.isFinite(r.score) ? r.score.toFixed(4) : String(r.score)
      const updated = formatResultTimestamp(r.timestamp)

      const $idBtn = $('<button>', {
        type: 'button',
        class: 'dashboard-query-doc-id-btn',
        text: r.id,
        attr: {
          'data-query-doc-id': r.id,
          'data-query-doc-collection': searchCollectionName,
          'aria-label': `Open full document ${r.id}`,
        },
      })

      $tbody.append(
        $('<tr>', { class: 'dashboard-query-result-main-row' }).append(
          $('<td>', { class: 'dashboard-query-cell-num', text: score }),
          $('<td>', { class: 'dashboard-query-cell-mono' }).append($idBtn),
          $('<td>', { text: truncateCell(name, 200) }),
          $('<td>', { class: 'dashboard-query-cell-desc', text: truncateCell(desc, 400) }),
          $('<td>', { class: 'dashboard-query-cell-nowrap', text: updated }),
        ),
      )
      const rawDetail = this.formatRawScoresDetail(r)
      if (rawDetail) {
        const $rawCell = $('<td>', {
          class: 'dashboard-query-raw-scores-cell',
          colspan: 4,
        })
        $rawCell.append(
          $('<strong>', { class: 'dashboard-query-raw-scores-label', text: 'Raw Scores' }),
          document.createTextNode(' -> '),
          document.createTextNode(rawDetail),
        )
        $tbody.append(
          $('<tr>', { class: 'dashboard-query-raw-scores-row' }).append(
            $('<td>', { class: 'dashboard-query-raw-scores-spacer' }),
            $rawCell,
          ),
        )
      }
    }
  }
}

import {
  ENTRY_FILTERS,
  type EntryFilter,
} from '../../stores/transcript'

interface TranscriptSearchProps {
  matches: number
  total: number
  active: boolean
  search: string
  filter: EntryFilter
  onSearch: (value: string) => void
  onFilter: (value: EntryFilter) => void
}

/**
 * Search and filter across the transcript.
 *
 * The store keeps thousands of entries for a restored session, so
 * the query narrows what gets rendered rather than filtering the
 * DOM after the fact. The parent owns the filtered entries; this
 * component only owns the controls.
 */
export function TranscriptSearch({
  matches,
  total,
  active,
  search,
  filter,
  onSearch,
  onFilter,
}: TranscriptSearchProps) {
  return (
    <div className="tsearch">
      <div className="tsearch-row">
        <input
          className="tsearch-input"
          type="search"
          value={search}
          placeholder="Search transcript…"
          aria-label="Search transcript"
          onChange={(event) => onSearch(event.target.value)}
        />

        {active && (
          <span className="tsearch-count" aria-live="polite">
            {matches}/{total}
          </span>
        )}
      </div>

      <div className="tsearch-filters" role="group" aria-label="Filter">
        {ENTRY_FILTERS.map((option) => (
          <button
            key={option.value}
            type="button"
            className={`tsearch-chip${
              filter === option.value ? ' is-active' : ''
            }`}
            aria-pressed={filter === option.value}
            onClick={() => onFilter(option.value)}
          >
            {option.label}
          </button>
        ))}

        {active && (
          <button
            type="button"
            className="tsearch-clear"
            onClick={() => {
              onSearch('')
              onFilter('all')
            }}
          >
            Clear
          </button>
        )}
      </div>
    </div>
  )
}
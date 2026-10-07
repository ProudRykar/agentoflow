import { budgetSeverity } from '../../stores/transcript'
import '../../components/primitives.css'

interface ContextMeterProps {
  used: number | null
  limit: number | null
  /** The assembler had to drop or trim something to fit. */
  trimmed: boolean
  /**
   * Whether the count came from a real tokenizer.
   *
   * Shown because the fallback divides by character length, which
   * undercounts tool output and code by around a third. A meter that
   * looks equally confident either way is how that becomes a confusing
   * rejection from the provider.
   */
  exact?: boolean
  /** Named blocks that were dropped or cut. */
  dropped?: string[]
  /** The provider's own count for the last request, if it gave one. */
  measured?: number | null
  /** Our estimate for that same request. */
  estimate?: number | null
}

/** "memory" -> "memory", "conversation" -> "conversation". */
function describeDropped(blocks: string[] | undefined): string {
  if (!blocks || blocks.length === 0) {
    return ''
  }

  const names = blocks.join(', ')

  return `Dropped: ${names}`
}

function compact(value: number): string {
  if (value >= 1000) {
    return `${Math.round(value / 100) / 10}k`
  }

  return String(value)
}

/**
 * How full the context window is.
 *
 * The raw token count was previously shown as a bare number, which
 * tells the reader nothing about whether the agent is close to the
 * limit or why the visible history looks shorter than it used to.
 */
export function ContextMeter({
  used,
  limit,
  trimmed,
  exact = true,
  dropped,
  measured = null,
  estimate = null,
}: ContextMeterProps) {
  const { ratio, tone } = budgetSeverity(used, limit)

  const percent = Math.min(100, Math.round(ratio * 100))

  const notes: string[] = []

  const droppedNote = describeDropped(dropped)

  if (droppedNote) {
    notes.push(droppedNote)
  } else if (trimmed) {
    notes.push('Trimmed to fit the budget')
  }

  if (tone === 'over') {
    notes.push('Over budget')
  } else if (tone === 'warn') {
    notes.push('Close to the limit')
  }

  if (!exact) {
    notes.push('Estimated, not measured')
  }

  // Both figures side by side. Where they disagree the estimate is
  // what the budget is computed from, so the gap is worth reading.
  if (measured !== null && estimate !== null && estimate > 0) {
    const ratio = measured / estimate
    const drift = Math.abs(ratio - 1)

    if (drift > 0.2) {
      notes.push(
        `Provider counted ${compact(measured)}, we estimated ${compact(
          estimate,
        )}`,
      )
    }
  }

  if (notes.length === 0) {
    notes.push('Within budget')
  }

  return (
    <div className={`budget is-${tone}`}>
      <div className="budget-head">
        <span>context</span>
        <span className="budget-value">
          {used === null ? '—' : compact(used)}
          {limit === null ? '' : ` / ${compact(limit)}`}
        </span>
      </div>

      <div
        className="budget-track"
        role="meter"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Context budget used"
      >
        <div className="budget-fill" style={{ width: `${percent}%` }} />
      </div>

      {/* Each note its own element: "Dropped: memory" and "Over budget"
          are different facts, and one joined string is both harder to
          read and impossible to assert on. */}
      <p
        className={
          notes.length === 1 && notes[0] === 'Within budget'
            ? 'budget-note'
            : 'budget-note is-warn'
        }
      >
        {notes.map((item) => (
          <span key={item} className="budget-note-line">
            {item}
          </span>
        ))}
      </p>
    </div>
  )
}
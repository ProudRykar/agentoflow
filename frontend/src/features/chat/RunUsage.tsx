import {
  emptyRunUsage,
  isMeasured,
  totalTokens,
  type RunUsageView,
} from './runUsage'
import '../../components/primitives.css'

interface RunUsagePanelProps {
  // Defaults rather than required: this is a view of state that may
  // legitimately not have arrived yet, and a panel that throws on a
  // missing figure takes the whole sidebar with it.
  usage?: RunUsageView
  /** Token allowance for the run, when one is configured. */
  ceiling?: number | null
}

function compact(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`
  }

  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}k`
  }

  return String(value)
}

function money(value: number): string {
  // Two decimals is the wrong unit for small totals: $0.0123 printed
  // as $0.01 is understated by a fifth while still looking precise.
  // A local run costs cents and fractions, so those get shown.
  if (value === 0) {
    return '$0.00'
  }

  if (value < 0.1) {
    return `$${value.toFixed(4)}`
  }

  return `$${value.toFixed(2)}`
}

/**
 * What the run has spent, beside the ceiling it was given.
 *
 * Shown rather than logged because the decision it supports is
 * immediate: keep going, or stop. A number available only after the
 * fact answers neither.
 *
 * When the provider reported nothing, that is stated instead of a
 * total. Zero tokens and no measurement are different facts, and the
 * second one is the one that decides whether to trust the first.
 */
export function RunUsagePanel({
  usage = emptyRunUsage(),
  ceiling,
}: RunUsagePanelProps) {
  if (usage.calls === 0) {
    return null
  }

  const measured = isMeasured(usage)
  const total = totalTokens(usage)

  const parts: string[] = [`${usage.calls} call${usage.calls === 1 ? '' : 's'}`]

  if (measured) {
    parts.push(`${compact(total)} tok`)

    if (usage.completionTokens > 0) {
      parts.push(`${compact(usage.completionTokens)} out`)
    }

    if (usage.estimatedCost !== null) {
      parts.push(money(usage.estimatedCost))
    }
  } else {
    parts.push('not measured')
  }

  const ratio =
    ceiling && ceiling > 0 && measured ? total / ceiling : null

  const tone =
    ratio === null ? 'none' : ratio >= 1 ? 'over' : ratio >= 0.75 ? 'warn' : 'none'

  return (
    <div className={`run-usage is-${tone}`}>
      <span className="run-usage-label">run</span>

      <span className="run-usage-figures">{parts.join(' · ')}</span>

      {ceiling ? (
        <span className="run-usage-ceiling">
          of {compact(ceiling)} tok
        </span>
      ) : null}

      {measured && usage.callsWithoutUsage > 0 ? (
        <span className="run-usage-note">
          {usage.callsWithoutUsage} call
          {usage.callsWithoutUsage === 1 ? '' : 's'} unreported
        </span>
      ) : null}
    </div>
  )
}

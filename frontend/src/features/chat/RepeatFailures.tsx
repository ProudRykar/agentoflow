import { repeatedFailures } from '../../stores/transcript'

interface RepeatFailuresProps {
  failures: Record<string, number>
  threshold?: number
}

/**
 * Surfaces a tool that keeps failing.
 *
 * A failed tool used to be visible only as a card inside a long
 * transcript, so a run that retried the same broken call four times
 * looked like ordinary progress. This states it plainly, because the
 * fix is on the operator's side: the tool or its arguments are wrong.
 */
export function RepeatFailures({
  failures,
  threshold = 2,
}: RepeatFailuresProps) {
  const offenders = repeatedFailures(failures, threshold)

  if (offenders.length === 0) {
    return null
  }

  return (
    <div className="repeat-banner" role="status">
      <span className="repeat-title">
        {offenders.length === 1
          ? 'A tool keeps failing'
          : `${offenders.length} tools keep failing`}
      </span>

      <ul className="repeat-list">
        {offenders.map(({ tool, count }) => (
          <li key={tool}>
            <code>{tool}</code>
            <span className="repeat-count">
              failed {count}×
            </span>
          </li>
        ))}
      </ul>

      <p className="repeat-note">
        The agent stopped retrying. Check the tool output below, then
        change the arguments or the server configuration.
      </p>
    </div>
  )
}
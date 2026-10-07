import { StatusChip } from '../../components/StatusChip'
import { CodeBlock } from '../markdown/CodeBlock'
import { codeArgument } from './codeArgument'
import type { ToolEntry } from '../../stores/transcript'

interface ToolCardProps {
  entry: ToolEntry
  /** How many times this tool has failed in the current run. */
  failures?: number
  onToggle: () => void
}

function formatDuration(seconds: number | null): string {
  if (seconds === null) {
    return ''
  }

  if (seconds < 1) {
    return `${Math.round(seconds * 1000)}ms`
  }

  return `${seconds.toFixed(1)}s`
}

/** Past this many failures the badge is worth showing. */
const FAILURE_BADGE_THRESHOLD = 2

export function ToolCard({
  entry,
  failures = 0,
  onToggle,
}: ToolCardProps) {
  const hasOutput = Boolean(entry.output)
  const isOpen = entry.expanded || entry.status === 'error'

  return (
    <div className={`tool-card tool-${entry.status}`}>
      <button
        type="button"
        className="tool-head"
        onClick={onToggle}
        aria-expanded={isOpen}
      >
        <span className="tool-symbol" aria-hidden="true">
          {entry.status === 'running' ? '●' : entry.status === 'error' ? '✗' : '▸'}
        </span>
        <span className="tool-name">{entry.name}</span>

        {failures >= FAILURE_BADGE_THRESHOLD && (
          <span
            className="tool-failures"
            title={`Failed ${failures} times in this run`}
          >
            failed {failures}×
          </span>
        )}

        <span className="tool-status">
          <StatusChip status={entry.status} />
          {formatDuration(entry.durationSeconds)}
        </span>
      </button>

      <div className="tool-args">
        {Object.entries(entry.arguments).map(([name, value]) => {
          const language = codeArgument(entry.name, name, value)

          if (language !== undefined) {
            return (
              <div key={name} className="tool-arg">
                <span className="tool-arg-name">{name}</span>
                <CodeBlock language={language} text={value as string} />
              </div>
            )
          }

          return (
            <div key={name} className="tool-arg">
              <span className="tool-arg-name">{name}</span>
              <span className="tool-arg-value">{formatValue(value)}</span>
            </div>
          )
        })}
      </div>

      {entry.status === 'error' && (
        <div className="tool-error">
          {entry.errorCode && (
            <div className="tool-error-code">{entry.errorCode}</div>
          )}
          {entry.errorMessage && (
            <pre className="tool-error-message">{entry.errorMessage}</pre>
          )}
        </div>
      )}

      {entry.status === 'success' && hasOutput && (
        <button
          type="button"
          className="tool-output-toggle"
          onClick={onToggle}
        >
          {isOpen ? 'hide output' : 'show output'}
        </button>
      )}

      {entry.status === 'success' && hasOutput && isOpen && (
        <pre className="tool-output">{entry.output}</pre>
      )}
    </div>
  )
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) {
    return '—'
  }

  if (typeof value === 'string') {
    return value
  }

  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

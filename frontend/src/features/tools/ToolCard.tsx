import type { ToolEntry } from '../../stores/transcript'

interface ToolCardProps {
  entry: ToolEntry
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

const LABEL: Record<ToolEntry['status'], string> = {
  running: 'running',
  success: 'done',
  error: 'error',
}

export function ToolCard({ entry, onToggle }: ToolCardProps) {
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
        <span className="tool-status">
          {LABEL[entry.status]}
          {formatDuration(entry.durationSeconds)}
        </span>
      </button>

      <div className="tool-args">
        {Object.entries(entry.arguments).map(([name, value]) => (
          <div key={name} className="tool-arg">
            <span className="tool-arg-name">{name}</span>
            <span className="tool-arg-value">{formatValue(value)}</span>
          </div>
        ))}
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

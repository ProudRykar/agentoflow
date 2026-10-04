import type { ThinkingEntry } from '../../stores/transcript'

interface ThinkingCardProps {
  entry: ThinkingEntry
  onToggle: () => void
}

export function ThinkingCard({ entry, onToggle }: ThinkingCardProps) {
  return (
    <div className="thinking-card">
      <button
        type="button"
        className="thinking-head"
        onClick={onToggle}
        aria-expanded={entry.expanded}
      >
        <span aria-hidden="true">{entry.expanded ? '▾' : '▸'}</span>
        <span>Thinking</span>
        <span className="thinking-iteration">
          iteration {entry.iteration}
        </span>
      </button>

      {entry.expanded && (
        <pre className="thinking-body">{entry.content}</pre>
      )}
    </div>
  )
}

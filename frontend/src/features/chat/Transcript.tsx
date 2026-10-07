import { useCallback, useState } from 'react'
import { VirtualList } from '../../components/VirtualList'
import type { Entry, RunInfo, RunState } from '../../stores/transcript'
import { groupByRun } from '../../stores/transcript'
import { Markdown } from '../markdown/Markdown'
import { ToolCard } from '../tools/ToolCard'
import { ThinkingCard } from '../tools/ThinkingCard'

interface TranscriptProps {
  entries: Entry[]
  runs: RunState['runs']
  streaming: boolean
  repeatFailures: Record<string, number>
  onToggle: (id: string) => void
}

/**
 * Measured row heights, tallest first.
 *
 * Rows are near-uniform because every entry is a card with the same
 * collapsed shape; anything expanded is passed through unmeasured so
 * a long tool output never desynchronises the window.
 */
const ROW_HEIGHT = 56

export function Transcript({
  entries,
  runs,
  streaming,
  repeatFailures,
  onToggle,
}: TranscriptProps) {
  const [viewport, setViewport] = useState<HTMLDivElement | null>(
    null,
  )
  const [pinned, setPinned] = useState(true)

  const groups = groupByRun(entries)

  // The list scrolls itself, so the handler is attached there rather
  // than to a wrapper. Reading a wrapper's scroll position always
  // reported 0, which made the view look stuck.
  const onScroll = useCallback((node: HTMLDivElement) => {
    // "Pinned" means the reader is at the newest content; anything
    // else and streaming must not yank the view away.
    const distance =
      node.scrollHeight - node.scrollTop - node.clientHeight

    setPinned(distance < ROW_HEIGHT * 2)
  }, [])

  const jumpToLatest = useCallback(() => {
    if (!viewport) {
      return
    }

    viewport.scrollTop = viewport.scrollHeight
    setPinned(true)
  }, [viewport])

  return (
    <div className="transcript-shell">
      <div className="transcript">
        <VirtualList
          className="transcript-viewport"
          onViewport={setViewport}
          onScroll={onScroll}
          items={groups}
          itemHeight={ROW_HEIGHT}
          renderItem={(group, index) => (
            <GroupView
              key={
                (group as { runId: string }).runId === ''
                  ? `main-${index}`
                  : `sub-${(group as { runId: string }).runId}-${index}`
              }
              group={group as {
                runId: string
                entries: Entry[]
              }}
              info={runs[(group as { runId: string }).runId]}
              repeatFailures={repeatFailures}
              onToggle={onToggle}
            />
          )}
          empty={
            streaming ? (
              <div className="entry entry-assistant">
                <span className="spinner" />
              </div>
            ) : null
          }
        />
      </div>

      {(streaming || !pinned) && (
        <button
          type="button"
          className="jump-latest"
          onClick={jumpToLatest}
        >
          {pinned ? 'Following output ↓' : 'Jump to latest ↓'}
        </button>
      )}
    </div>
  )
}

function GroupView({
  group,
  info,
  repeatFailures,
  onToggle,
}: {
  group: { runId: string; entries: Entry[] }
  info: RunInfo | undefined
  repeatFailures: Record<string, number>
  onToggle: (id: string) => void
}) {
  if (group.runId === '') {
    return (
      <div className="group">
        {group.entries.map((entry) => (
          <EntryView
            key={entry.id}
            entry={entry}
            repeatFailures={repeatFailures}
            onToggle={onToggle}
          />
        ))}
      </div>
    )
  }

  return (
    <SubagentGroup
      info={info}
      entries={group.entries}
      repeatFailures={repeatFailures}
      onToggle={onToggle}
    />
  )
}

function SubagentGroup({
  info,
  entries,
  repeatFailures,
  onToggle,
}: {
  info: RunInfo | undefined
  entries: Entry[]
  repeatFailures: Record<string, number>
  onToggle: (id: string) => void
}) {
  const label = info?.role && info.role !== 'main' ? info.role : 'subagent'

  return (
    <section className="subagent-group">
      <header className="subagent-head">
        <span className="subagent-mark" aria-hidden="true">
          ⎇
        </span>
        <span className="subagent-role">{label}</span>
        {info?.model && (
          <span className="subagent-model">{info.model}</span>
        )}
        <span
          className={`subagent-state${
            info?.finished ? ' is-done' : ' is-running'
          }`}
        >
          {info?.finished ? 'finished' : 'running'}
        </span>
      </header>

      <div className="subagent-body">
        {entries.map((entry) => (
          <EntryView
            key={entry.id}
            entry={entry}
            repeatFailures={repeatFailures}
            onToggle={onToggle}
          />
        ))}
      </div>
    </section>
  )
}

function EntryView({
  entry,
  repeatFailures,
  onToggle,
}: {
  entry: Entry
  repeatFailures: Record<string, number>
  onToggle: (id: string) => void
}) {
  switch (entry.kind) {
    case 'user':
      return (
        <div className="entry entry-user">
          <div className="entry-role">User</div>
          <div className="entry-body">{entry.content}</div>
        </div>
      )

    case 'assistant':
      return (
        <div className="entry entry-assistant">
          <div className="entry-role">Agent</div>
          <div className="entry-body">
            <Markdown source={entry.content} />
            {entry.streaming && (
              <span className="caret" aria-label="streaming" />
            )}
          </div>
        </div>
      )

    case 'thinking':
      return (
        <ThinkingCard
          entry={entry}
          onToggle={() => onToggle(entry.id)}
        />
      )

    case 'tool':
      return (
        <ToolCard
          entry={entry}
          failures={repeatFailures[entry.name] ?? 0}
          onToggle={() => onToggle(entry.id)}
        />
      )

    case 'error':
      return (
        <div className="entry entry-error">
          <div className="entry-role">Error</div>
          <div className="entry-body">{entry.content}</div>
        </div>
      )

    default:
      return null
  }
}
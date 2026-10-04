import type { Entry, RunInfo, RunState } from '../../stores/transcript'
import { groupByRun } from '../../stores/transcript'
import { Markdown } from '../markdown/Markdown'
import { ToolCard } from '../tools/ToolCard'
import { ThinkingCard } from '../tools/ThinkingCard'

interface TranscriptProps {
  entries: Entry[]
  runs: RunState['runs']
  streaming: boolean
  onToggle: (id: string) => void
}

export function Transcript({
  entries,
  runs,
  streaming,
  onToggle,
}: TranscriptProps) {
  const groups = groupByRun(entries)

  return (
    <div className="transcript">
      {groups.map((group, index) => {
        if (group.runId === '') {
          return (
            <div className="group" key={`main-${index}`}>
              {group.entries.map((entry) => (
                <EntryView
                  key={entry.id}
                  entry={entry}
                  onToggle={onToggle}
                />
              ))}
            </div>
          )
        }

        return (
          <SubagentGroup
            key={`sub-${group.runId}-${index}`}
            info={runs[group.runId]}
            entries={group.entries}
            onToggle={onToggle}
          />
        )
      })}

      {streaming && entries.length === 0 && (
        <div className="entry entry-assistant">
          <span className="spinner" />
        </div>
      )}
    </div>
  )
}

function SubagentGroup({
  info,
  entries,
  onToggle,
}: {
  info: RunInfo | undefined
  entries: Entry[]
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
            onToggle={onToggle}
          />
        ))}
      </div>
    </section>
  )
}

function EntryView({
  entry,
  onToggle,
}: {
  entry: Entry
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

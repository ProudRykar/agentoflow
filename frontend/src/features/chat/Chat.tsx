import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import {
  allowApproval,
  cancelRun,
  denyApproval,
  listPlugins,
  listSkills,
  resumeRun,
} from '../../api/client'
import type { PluginInfo, SkillInfo } from '../../api/types'
import {
  queryTranscript,
  type ConnectionState,
  type Entry,
  type EntryFilter,
  type RunState,
} from '../../stores/transcript'
import { ApprovalPanel } from '../approval/ApprovalPanel'
import type { PendingApprovalView } from '../approval/types'
import { PluginsPanel } from '../plugins/PluginsPanel'
import { SkillsPanel } from '../skills/SkillsPanel'
import { ContextMeter } from './ContextMeter'
import { PlanPanel } from './PlanPanel'
import { RepeatFailures } from './RepeatFailures'
import { RunStatus } from './RunStatus'
import { RunUsagePanel } from './RunUsage'
import { Transcript } from './Transcript'
import { TranscriptSearch } from './TranscriptSearch'
import { CommandPalette } from '../../components/CommandPalette'
import { useShortcuts } from '../../hooks/useShortcuts'
import '../../components/primitives.css'

interface ChatProps {
  sessionId: string
  entries: Entry[]
  approval: PendingApprovalView | null
  run: RunState
  connection: ConnectionState
  onSend: (prompt: string) => void
  onToggle: (id: string) => void
  onRegenerate: (hint: string) => void
}

export function Chat({
  sessionId,
  entries,
  approval,
  run,
  connection,
  onSend,
  onToggle,
  onRegenerate,
}: ChatProps) {
  const [draft, setDraft] = useState('')
  const [skills, setSkills] = useState<SkillInfo[]>([])
  const [plugins, setPlugins] = useState<PluginInfo[]>([])
  const [approvalBusy, setApprovalBusy] = useState(false)
  const [approvalError, setApprovalError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState<EntryFilter>('all')
  const [paletteOpen, setPaletteOpen] = useState(false)

  const inputRef = useRef<HTMLTextAreaElement | null>(null)

  const busy = run.running
  const canSend = draft.trim().length > 0 && !busy && !approval

  // Only meaningful when the last word was the agent's: there has to
  // be an answer to replace. Offered while the session is idle, since
  // regenerating mid-run would drop tool results still in flight.
  const canRegenerate =
    !busy && !approval && entries.at(-1)?.kind === 'assistant'

  // The query narrows what is rendered. A restored session can hold
  // thousands of entries, so filtering after render would leave the
  // virtual window measuring the wrong content.
  const slice = useMemo(
    () => queryTranscript(entries, { search, filter }),
    [entries, search, filter],
  )

  // Skills and plugins are read-only projections of the managers
  // the agent already uses, so they refresh when a session changes
  // or a run starts rather than streaming their own events.
  const refreshObservability = useCallback(() => {
    let cancelled = false

    const load = async () => {
      try {
        const [nextSkills, nextPlugins] = await Promise.all([
          listSkills(sessionId),
          listPlugins(sessionId),
        ])

        if (!cancelled) {
          setSkills(nextSkills)
          setPlugins(nextPlugins)
        }
      } catch {
        if (!cancelled) {
          setSkills([])
          setPlugins([])
        }
      }
    }

    void load()

    return () => {
      cancelled = true
    }
  }, [sessionId])

  useEffect(() => refreshObservability(), [refreshObservability, busy])

  const submit = useCallback(() => {
    const prompt = draft.trim()

    if (!prompt || busy || approval) {
      return
    }

    setDraft('')
    setActionError(null)
    onSend(prompt)
  }, [approval, busy, draft, onSend])

  const decide = useCallback(
    async (approved: boolean) => {
      if (!approval) {
        return
      }

      setApprovalBusy(true)
      setApprovalError(null)

      try {
        // The Python future lives on the backend; the browser
        // only sends a command.
        if (approved) {
          await allowApproval(sessionId, approval.approvalId)
        } else {
          await denyApproval(sessionId, approval.approvalId)
        }
      } catch (error) {
        setApprovalError(
          error instanceof Error ? error.message : 'decision failed',
        )
      } finally {
        setApprovalBusy(false)
      }
    },
    [approval, sessionId],
  )

  const cancel = useCallback(async () => {
    try {
      await cancelRun(sessionId)
    } catch (error) {
      setActionError(
        error instanceof Error ? error.message : 'cancel failed',
      )
    }
  }, [sessionId])

  // Search box lives inside TranscriptSearch; focus it by id so the
  // shortcut works without threading a ref through two components.
  const focusSearch = useCallback(() => {
    document
      .querySelector<HTMLInputElement>('.tsearch-input')
      ?.focus()
  }, [])

  useShortcuts({
    paletteOpen,
    stopEnabled: busy,
    onPalette: () => setPaletteOpen((previous) => !previous),
    onClosePalette: () => setPaletteOpen(false),
    onSearch: focusSearch,
    onStop: () => void cancel(),
  })

  const commands = useMemo(
    () => [
      {
        id: 'search',
        label: 'Search transcript',
        hint: '⌘F',
        run: focusSearch,
      },
      {
        id: 'clear-search',
        label: 'Clear transcript filters',
        run: () => {
          setSearch('')
          setFilter('all')
        },
      },
      {
        id: 'filter-errors',
        label: 'Show only errors',
        run: () => setFilter('errors'),
      },
      {
        id: 'filter-tools',
        label: 'Show only tool calls',
        run: () => setFilter('tools'),
      },
      {
        id: 'resume',
        label: 'Resume blocked run',
        run: () => void resume(),
      },
      {
        id: 'stop',
        label: 'Stop the current run',
        hint: busy ? 'Esc' : undefined,
        run: () => void cancel(),
      },
    ],
    [busy, focusSearch],
  )

  const resume = useCallback(async () => {
    try {
      await resumeRun(sessionId)
    } catch (error) {
      setActionError(
        error instanceof Error ? error.message : 'resume failed',
      )
    }
  }, [sessionId])

  return (
    <div className="chat">
      <div className="chat-main">
        <RepeatFailures failures={run.repeatFailures} />

        <TranscriptSearch
          matches={slice.matches}
          total={slice.total}
          active={slice.active}
          search={search}
          filter={filter}
          onSearch={setSearch}
          onFilter={setFilter}
        />

        <Transcript
          entries={slice.entries}
          runs={run.runs}
          streaming={busy}
          repeatFailures={run.repeatFailures}
          onToggle={onToggle}
        />
      </div>

      <aside className="chat-side">
        <RunUsagePanel
          usage={run.runUsage}
          ceiling={run.maxPromptTokens}
        />

        <RunStatus
          phase={run.phase}
          iteration={run.iteration}
          state={run.sessionState}
          model={run.model}
          link={connection.status}
          blocked={run.sessionState === 'blocked'}
          busy={busy}
          onResume={() => void resume()}
          onCancel={() => void cancel()}
        />

        <section className="panel">
          <ContextMeter
            used={run.contextUsed}
            limit={run.contextLimit}
            trimmed={run.contextTrimmed}
            exact={run.contextCounterExact}
            dropped={run.contextDropped}
            measured={run.contextMeasured}
            estimate={run.contextEstimate}
          />
        </section>

        {run.plan && (
          <PlanPanel
            objective={run.plan.objective}
            steps={run.plan.steps}
            currentStepId={run.plan.currentStepId}
            revision={run.plan.revision}
            todos={run.todos}
          />
        )}

        {actionError && (
          <div className="inline-error">{actionError}</div>
        )}

        <ApprovalPanel
          approval={approval}
          busy={approvalBusy}
          error={approvalError}
          onAllow={() => void decide(true)}
          onDeny={() => void decide(false)}
        />

        <SkillsPanel skills={skills} />
        <PluginsPanel plugins={plugins} />
      </aside>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault()
          submit()
        }}
      >
        <textarea
          ref={inputRef}
          className="composer-input"
          rows={1}
          value={draft}
          placeholder={
            approval
              ? 'Waiting for your decision…'
              : busy
                ? 'Agent is working…'
                : 'Type a message…  (Enter to send, Shift+Enter for a new line)'
          }
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key !== 'Enter') {
              return
            }

            // Shift+Enter is a newline; plain Enter sends.
            if (event.shiftKey) {
              return
            }

            event.preventDefault()
            submit()
          }}
          disabled={busy || Boolean(approval)}
        />

        {canRegenerate ? (
          <button
            type="button"
            className="btn btn-regenerate"
            title="Answer the last question again"
            onClick={() => {
              // Whatever is in the box is the hint. Empty is a plain
              // retry and is fine, so it is sent rather than blocked.
              onRegenerate(draft.trim())
              setDraft('')
            }}
          >
            Regenerate
          </button>
        ) : null}

        <button
          type="submit"
          className="btn btn-send"
          disabled={!canSend}
        >
          Send
        </button>

        {busy && (
          <button
            type="button"
            className="btn btn-cancel"
            onClick={() => void cancel()}
          >
            Cancel
          </button>
        )}
      </form>

      <CommandPalette
        open={paletteOpen}
        commands={commands}
        onClose={() => setPaletteOpen(false)}
      />
    </div>
  )
}

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  allowApproval,
  cancelRun,
  denyApproval,
  listPlugins,
  listSkills,
  resumeRun,
} from '../../api/client'
import type { PluginInfo, SkillInfo } from '../../api/types'
import type {
  ConnectionState,
  Entry,
  RunState,
} from '../../stores/transcript'
import { ApprovalPanel } from '../approval/ApprovalPanel'
import type { PendingApprovalView } from '../approval/types'
import { PluginsPanel } from '../plugins/PluginsPanel'
import { SkillsPanel } from '../skills/SkillsPanel'
import { Transcript } from './Transcript'

interface ChatProps {
  sessionId: string
  entries: Entry[]
  approval: PendingApprovalView | null
  run: RunState
  connection: ConnectionState
  onSend: (prompt: string) => void
  onToggle: (id: string) => void
}

export function Chat({
  sessionId,
  entries,
  approval,
  run,
  connection,
  onSend,
  onToggle,
}: ChatProps) {
  const [draft, setDraft] = useState('')
  const [skills, setSkills] = useState<SkillInfo[]>([])
  const [plugins, setPlugins] = useState<PluginInfo[]>([])
  const [approvalBusy, setApprovalBusy] = useState(false)
  const [approvalError, setApprovalError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  const bottomRef = useRef<HTMLDivElement | null>(null)

  const busy = run.running
  const canSend = draft.trim().length > 0 && !busy && !approval

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

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [entries])

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
        <Transcript
          entries={entries}
          runs={run.runs}
          streaming={busy}
          onToggle={onToggle}
        />
        <div ref={bottomRef} />
      </div>

      <aside className="chat-side">
        <section className="panel">
          <h3>Session</h3>

          <dl className="stats">
            <dt>state</dt>
            <dd>{run.sessionState}</dd>

            <dt>phase</dt>
            <dd>{run.phase ?? '—'}</dd>

            <dt>iteration</dt>
            <dd>{run.iteration || '—'}</dd>

            <dt>model</dt>
            <dd className="truncate" title={run.model}>
              {run.model || '—'}
            </dd>

            <dt>context</dt>
            <dd>{run.estimatedTokens ?? '—'}</dd>

            <dt>link</dt>
            <dd className={`link-${connection.status}`}>
              {connection.status}
            </dd>
          </dl>

          {run.sessionState === 'blocked' && (
            <button
              type="button"
              className="btn btn-resume"
              onClick={() => void resume()}
            >
              Resume
            </button>
          )}

          {busy && (
            <button
              type="button"
              className="btn btn-cancel"
              onClick={() => void cancel()}
            >
              Cancel run
            </button>
          )}

          {actionError && (
            <div className="inline-error">{actionError}</div>
          )}
        </section>

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
        <input
          className="composer-input"
          value={draft}
          placeholder={
            approval
              ? 'Waiting for your decision…'
              : busy
                ? 'Agent is working…'
                : 'Type a message…'
          }
          onChange={(event) => setDraft(event.target.value)}
          disabled={busy || Boolean(approval)}
        />

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
    </div>
  )
}

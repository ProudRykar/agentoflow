import { useEffect, useRef, useState } from 'react'
import type { SessionInfo } from '../../api/types'

interface SessionSidebarProps {
  sessions: SessionInfo[]
  currentId: string | null
  collapsed: boolean
  connection: string
  activeView: 'chat' | 'tools'
  busy: boolean
  onSelect: (id: string) => void
  onCreate: () => void
  onClose: (id: string) => void
  onRename: (id: string, title: string) => Promise<void>
  onToggle: () => void
  onView: (view: 'chat' | 'tools') => void
  onOpenSettings: () => void
}

function relativeTime(seconds: number): string {
  const delta = Date.now() / 1000 - seconds

  if (!Number.isFinite(delta) || delta < 60) {
    return 'just now'
  }

  if (delta < 3600) {
    return `${Math.floor(delta / 60)}m ago`
  }

  if (delta < 86400) {
    return `${Math.floor(delta / 3600)}h ago`
  }

  return `${Math.floor(delta / 86400)}d ago`
}

export function SessionSidebar({
  sessions,
  currentId,
  collapsed,
  connection,
  activeView,
  busy,
  onSelect,
  onCreate,
  onClose,
  onRename,
  onToggle,
  onView,
  onOpenSettings,
}: SessionSidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(
    null,
  )
  const [draft, setDraft] = useState('')
  const [confirmId, setConfirmId] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    if (editingId) {
      inputRef.current?.select()
    }
  }, [editingId])

  const beginRename = (session: SessionInfo) => {
    setConfirmId(null)
    setEditingId(session.session_id)
    setDraft(session.title || session.session_id.slice(0, 8))
  }

  const commitRename = async (id: string) => {
    const title = draft.trim()

    setEditingId(null)

    if (!title) {
      return
    }

    try {
      await onRename(id, title)
    } catch {
      // The banner in App reports the failure.
    }
  }

  if (collapsed) {
    return (
      <aside className="sidebar sidebar-collapsed">
        <button
          type="button"
          className="sidebar-expand"
          onClick={onToggle}
          aria-label="Expand sidebar"
          title="Expand sidebar"
        >
          »
        </button>

        <button
          type="button"
          className="rail-icon"
          onClick={() => onView('chat')}
          title="Chat"
          aria-label="Chat"
        >
          💬
        </button>

        <button
          type="button"
          className="rail-icon"
          onClick={() => onView('tools')}
          title="Tools"
          aria-label="Tools"
        >
          🛠
        </button>

        <button
          type="button"
          className="rail-icon rail-new"
          onClick={onCreate}
          title="New session"
          aria-label="New session"
        >
          +
        </button>

        <div className="rail-spacer" />

        <button
          type="button"
          className="rail-icon"
          onClick={onOpenSettings}
          title="Settings"
          aria-label="Settings"
        >
          ⚙
        </button>

        <span
          className={`rail-dot rail-dot-${connection}`}
          title={connection}
        />
      </aside>
    )
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        <span className="brand">agentoflow</span>

        <button
          type="button"
          className="icon-button"
          onClick={onToggle}
          aria-label="Collapse sidebar"
          title="Collapse sidebar"
        >
          «
        </button>
      </div>

      <button
        type="button"
        className="new-session"
        onClick={onCreate}
      >
        <span aria-hidden="true">+</span> New session
      </button>

      <nav className="view-switch" aria-label="Views">
        <button
          type="button"
          className={activeView === 'chat' ? 'is-active' : ''}
          onClick={() => onView('chat')}
        >
          Chat
        </button>
        <button
          type="button"
          className={activeView === 'tools' ? 'is-active' : ''}
          onClick={() => onView('tools')}
        >
          Tools
        </button>
      </nav>

      <div className="session-scroll">
        {sessions.length === 0 ? (
          <p className="sidebar-empty">No sessions yet.</p>
        ) : (
          <ul className="session-list">
            {sessions.map((session) => {
              const active = session.session_id === currentId
              const editing = editingId === session.session_id
              const confirming = confirmId === session.session_id

              return (
                <li
                  key={session.session_id}
                  className={`session-row${
                    active ? ' is-active' : ''
                  }${confirming ? ' is-confirming' : ''}`}
                >
                  {editing ? (
                    <input
                      ref={inputRef}
                      className="session-rename"
                      value={draft}
                      onChange={(event) =>
                        setDraft(event.target.value)
                      }
                      onBlur={() => void commitRename(session.session_id)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          void commitRename(session.session_id)
                        }

                        if (event.key === 'Escape') {
                          setEditingId(null)
                        }
                      }}
                      aria-label="Session title"
                    />
                  ) : (
                    <button
                      type="button"
                      className="session-main"
                      onClick={() => onSelect(session.session_id)}
                      onDoubleClick={() =>
                        beginRename(session)
                      }
                      title={`${session.title}\n${session.working_directory}`}
                    >
                      <span className="session-title">
                        {session.title ||
                          session.session_id.slice(0, 8)}
                      </span>
                      <span className="session-meta">
                        {session.session.state}
                        {' · '}
                        {relativeTime(session.created_at)}
                      </span>
                    </button>
                  )}

                  {!editing && (
                    <div className="session-tools">
                      {confirming ? (
                        <>
                          <button
                            type="button"
                            className="session-delete confirm"
                            onClick={() => {
                              setConfirmId(null)
                              onClose(session.session_id)
                            }}
                            aria-label="Confirm delete"
                            title="Confirm delete"
                          >
                            ✓
                          </button>
                          <button
                            type="button"
                            className="session-delete"
                            onClick={() => setConfirmId(null)}
                            aria-label="Cancel delete"
                          >
                            ×
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            type="button"
                            className="session-tool"
                            onClick={() =>
                              beginRename(session)
                            }
                            aria-label={`Rename ${session.title}`}
                            title="Rename"
                          >
                            ✎
                          </button>
                          <button
                            type="button"
                            className="session-tool"
                            onClick={() =>
                              setConfirmId(session.session_id)
                            }
                            aria-label={`Delete ${session.title}`}
                            title="Delete"
                          >
                            🗑
                          </button>
                        </>
                      )}
                    </div>
                  )}
                </li>
              )
            })}
          </ul>
        )}
      </div>

      <div className="sidebar-foot">
        <button
          type="button"
          className="gear"
          onClick={onOpenSettings}
          title="Settings"
          aria-label="Settings"
        >
          <span aria-hidden="true">⚙</span>
          <span className="gear-label">Settings</span>
        </button>

        <span className="sidebar-status">
          <span
            className={`rail-dot rail-dot-${connection}`}
          />
          {connection}
          {busy ? ' · working' : ''}
        </span>
      </div>
    </aside>
  )
}

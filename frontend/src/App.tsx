import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ApiError,
  closeSession,
  createSession,
  listSessions,
  renameSession,
  sendMessage,
  setApiToken,
} from './api/client'
import type { SessionInfo, WireEnvelope } from './api/types'
import type { PendingApprovalView } from './features/approval/types'
import { Chat } from './features/chat/Chat'
import { SettingsModal } from './features/settings/SettingsModal'
import { ToolsPage } from './features/tools/ToolsPage'
import { SessionSidebar } from './features/sessions/SessionSidebar'
import { TokenDialog } from './features/sessions/TokenDialog'
import {
  initialState,
  reduce,
  toggleEntry,
} from './stores/transcript'
import type { AppState } from './stores/transcript'
import { SessionSocket } from './websocket/SessionSocket'
import './App.css'
import './features/tools/ToolsPage.css'

const TOKEN_KEY = 'agentoflow.token'

function loadToken(): string {
  try {
    return window.localStorage.getItem(TOKEN_KEY) ?? ''
  } catch {
    return ''
  }
}

function persistToken(value: string): void {
  try {
    if (value) {
      window.localStorage.setItem(TOKEN_KEY, value)
    } else {
      window.localStorage.removeItem(TOKEN_KEY)
    }
  } catch {
    // Private-mode storage failures are non-fatal.
  }
}

export default function App() {
  const [token, setToken] = useState(loadToken)
  const [needsToken, setNeedsToken] = useState(false)
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [currentId, setCurrentId] = useState<string | null>(null)
  const [state, setState] = useState<AppState>(initialState)
  const [error, setError] = useState<string | null>(null)
  const [collapsed, setCollapsed] = useState(false)
  const [view, setView] = useState<'chat' | 'tools'>('chat')
  const [settingsOpen, setSettingsOpen] = useState(false)

  const socketRef = useRef<SessionSocket | null>(null)
  const cursorRef = useRef(-1)

  useEffect(() => {
    setApiToken(token)
  }, [token])

  const attachSession = useCallback((sessionId: string) => {
    socketRef.current?.close()
    socketRef.current = null

    // Reset per-session state so two sessions never share a view.
    setState(initialState())
    cursorRef.current = -1
    setCurrentId(sessionId || null)
    setError(null)
  }, [])

  const handleEvent = useCallback((event: WireEnvelope) => {
    cursorRef.current = Math.max(cursorRef.current, event.seq)

    setState((previous) => reduce(previous, event))
  }, [])

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await listSessions())
      setNeedsToken(false)
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 401) {
        setNeedsToken(true)
        return
      }

      setError(describe(cause))
    }
  }, [])

  useEffect(() => {
    void refreshSessions()
  }, [refreshSessions, token])

  useEffect(() => {
    socketRef.current?.close()
    socketRef.current = null

    if (!currentId) {
      return
    }

    const socket = new SessionSocket(
      currentId,
      cursorRef.current,
      {
        onEvent: handleEvent,
        onStatus: (status, cause) => {
          setState((previous) => ({
            ...previous,
            connection: {
              ...previous.connection,
              status,
              lastError: cause ?? null,
            },
          }))
        },
      },
    )

    socketRef.current = socket
    socket.connect()

    return () => {
      socket.close()

      if (socketRef.current === socket) {
        socketRef.current = null
      }
    }
  }, [currentId, handleEvent])

  const handleCreate = useCallback(async () => {
    try {
      const created = await createSession()

      await refreshSessions()
      attachSession(created.session_id)
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 401) {
        setNeedsToken(true)
        return
      }

      setError(describe(cause))
    }
  }, [attachSession, refreshSessions])

  const handleClose = useCallback(
    async (sessionId: string) => {
      try {
        await closeSession(sessionId)
        await refreshSessions()

        if (sessionId === currentId) {
          attachSession('')
        }
      } catch (cause) {
        setError(describe(cause))
      }
    },
    [attachSession, currentId, refreshSessions],
  )

  const handleSend = useCallback(
    (prompt: string) => {
      if (!currentId) {
        return
      }

      // The session owns the run/continue decision.
      void sendMessage(currentId, prompt, 'auto')
        .then(() => refreshSessions())
        .catch((cause) => setError(describe(cause)))
    },
    [currentId, refreshSessions],
  )

  const handleRename = useCallback(
    async (sessionId: string, title: string) => {
      try {
        await renameSession(sessionId, title)

        setSessions(
          await listSessions(),
        )
      } catch (cause) {
        setError(describe(cause))
        throw cause
      }
    },
    [],
  )

  const handleToggle = useCallback((id: string) => {
    setState((previous) => toggleEntry(previous, id))
  }, [])

  const approval: PendingApprovalView | null = useMemo(
    () => state.transcript.approval,
    [state.transcript.approval],
  )

  return (
    <div className="app">
      <div className="workspace">
        <SessionSidebar
          sessions={sessions}
          currentId={currentId}
          collapsed={collapsed}
          connection={state.connection.status}
          activeView={view}
          busy={state.run.running}
          onSelect={attachSession}
          onCreate={() => void handleCreate()}
          onClose={(id) => void handleClose(id)}
          onRename={handleRename}
          onToggle={() => setCollapsed(!collapsed)}
          onView={setView}
          onOpenSettings={() => setSettingsOpen(true)}
        />

        <main className="workspace-main">
          {error && (
            <div className="banner" role="alert">
              <span>{error}</span>
              <button
                type="button"
                className="banner-close"
                onClick={() => setError(null)}
                aria-label="Dismiss"
              >
                ×
              </button>
            </div>
          )}

          {view === 'tools' ? (
            <ToolsPage
              sessionId={currentId}
              refreshKey={state.run.sessionState}
            />
          ) : currentId ? (
            <Chat
              sessionId={currentId}
              entries={state.transcript.entries}
              approval={approval}
              run={state.run}
              connection={state.connection}
              onSend={handleSend}
              onToggle={handleToggle}
            />
          ) : (
            <div className="empty">
              <p>Select a session, or create one to begin.</p>
            </div>
          )}
        </main>
      </div>

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSaved={() => {
          void refreshSessions()
        }}
      />

      {needsToken && (
        <TokenDialog
          onSubmit={(value) => {
            persistToken(value)
            setToken(value)
            setApiToken(value)
            setNeedsToken(false)
            void refreshSessions()
          }}
          onCancel={() => {
            persistToken('')
            setToken('')
            setApiToken('')
            setNeedsToken(false)
          }}
        />
      )}
    </div>
  )
}

function describe(cause: unknown): string {
  if (cause instanceof ApiError) {
    return `${cause.status}: ${cause.message}`
  }

  if (cause instanceof Error) {
    return cause.message
  }

  return 'request failed'
}

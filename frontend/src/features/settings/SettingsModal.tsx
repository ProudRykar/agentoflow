import { useCallback, useEffect, useState } from 'react'
import { ConfigEditor } from './ConfigEditor'
import { ModelsEditor } from './ModelsEditor'
import './SettingsModal.css'

type Tab = 'config' | 'models'

interface SettingsModalProps {
  open: boolean
  onClose: () => void
  onSaved: () => void
}

export function SettingsModal({
  open,
  onClose,
  onSaved,
}: SettingsModalProps) {
  const [tab, setTab] = useState<Tab>('config')
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  useEffect(() => {
    if (!open) {
      setError(null)
      setNotice(null)
    }
  }, [open])

  useEffect(() => {
    if (!open) {
      return
    }

    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    window.addEventListener('keydown', onKey)

    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  const handleSaved = useCallback(
    (message: string) => {
      setNotice(message)
      setError(null)
      onSaved()
    },
    [onSaved],
  )

  const handleError = useCallback((cause: unknown) => {
    setError(
      cause instanceof Error ? cause.message : 'save failed',
    )
    setNotice(null)
  }, [])

  if (!open) {
    return null
  }

  return (
    <div
      className="overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Settings"
    >
      <div className="settings">
        <header className="settings-head">
          <nav className="settings-tabs">
            <button
              type="button"
              className={tab === 'config' ? 'is-active' : ''}
              onClick={() => setTab('config')}
            >
              Configuration
            </button>
            <button
              type="button"
              className={tab === 'models' ? 'is-active' : ''}
              onClick={() => setTab('models')}
            >
              Models
            </button>
          </nav>

          <button
            type="button"
            className="icon-button"
            onClick={onClose}
            aria-label="Close settings"
          >
            ×
          </button>
        </header>

        {error && (
          <div className="settings-banner settings-banner-error">
            {error}
          </div>
        )}

        {notice && (
          <div className="settings-banner settings-banner-ok">
            {notice}
          </div>
        )}

        <div className="settings-body">
          {tab === 'config' ? (
            <ConfigEditor
              onSaved={handleSaved}
              onError={handleError}
            />
          ) : (
            <ModelsEditor
              onSaved={handleSaved}
              onError={handleError}
            />
          )}
        </div>

        <footer className="settings-foot">
          <p className="muted small">
            Changes apply to new sessions. Secret values are stored
            on the server and are never sent to the browser.
          </p>
          <button
            type="button"
            className="btn"
            onClick={onClose}
          >
            Close
          </button>
        </footer>
      </div>
    </div>
  )
}

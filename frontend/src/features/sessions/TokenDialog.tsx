import { useState } from 'react'

interface TokenDialogProps {
  onSubmit: (token: string) => void
  onCancel: () => void
}

/**
 * Shown only after the backend rejects a request with 401.
 *
 * The backend runs unauthenticated by default, so this is not a
 * startup gate — it appears when a token is actually required.
 */
export function TokenDialog({ onSubmit, onCancel }: TokenDialogProps) {
  const [value, setValue] = useState('')

  return (
    <div className="overlay">
      <form
        className="dialog"
        onSubmit={(event) => {
          event.preventDefault()
          onSubmit(value.trim())
        }}
      >
        <h2>API token required</h2>

        <p className="muted">
          The server was started with{' '}
          <code>AGENTOFLOW_API_TOKEN</code>. Paste the token to
          continue. It is stored in this browser only.
        </p>

        <input
          className="dialog-input"
          type="password"
          value={value}
          autoFocus
          placeholder="token"
          onChange={(event) => setValue(event.target.value)}
        />

        <div className="dialog-actions">
          <button
            type="submit"
            className="btn btn-send"
            disabled={!value.trim()}
          >
            Connect
          </button>
          <button
            type="button"
            className="btn"
            onClick={onCancel}
          >
            Cancel
          </button>
        </div>
      </form>
    </div>
  )
}

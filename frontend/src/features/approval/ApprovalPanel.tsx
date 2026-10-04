interface ApprovalPanelProps {
  approval: {
    approvalId: string
    toolName: string
    permission: string
    reason: string
    arguments: Record<string, unknown>
  } | null
  busy: boolean
  error: string | null
  onAllow: () => void
  onDeny: () => void
}

export function ApprovalPanel({
  approval,
  busy,
  error,
  onAllow,
  onDeny,
}: ApprovalPanelProps) {
  if (!approval) {
    return null
  }

  return (
    <div className="approval" role="alertdialog" aria-label="Approval required">
      <div className="approval-head">Permission required</div>

      <dl className="approval-grid">
        <dt>tool</dt>
        <dd className="approval-tool">{approval.toolName}</dd>

        <dt>permission</dt>
        <dd className="approval-permission">{approval.permission}</dd>
      </dl>

      {approval.reason && (
        <p className="approval-reason">{approval.reason}</p>
      )}

      {Object.keys(approval.arguments).length > 0 && (
        <div className="approval-args">
          {Object.entries(approval.arguments).map(([name, value]) => (
            <div key={name} className="approval-arg">
              <span className="approval-arg-name">{name}</span>
              <span className="approval-arg-value">{stringify(value)}</span>
            </div>
          ))}
        </div>
      )}

      {error && <div className="approval-error">{error}</div>}

      <div className="approval-actions">
        <button
          type="button"
          className="btn btn-allow"
          onClick={onAllow}
          disabled={busy}
        >
          Allow
        </button>
        <button
          type="button"
          className="btn btn-deny"
          onClick={onDeny}
          disabled={busy}
        >
          Deny
        </button>
      </div>
    </div>
  )
}

function stringify(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }

  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

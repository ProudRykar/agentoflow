import { useState } from 'react'
import { StatusChip } from '../../components/StatusChip'

interface RunStatusProps {
  phase: string | null
  iteration: number
  state: string
  model: string
  link: string
  blocked: boolean
  onResume: () => void
  busy: boolean
  onCancel: () => void
}

const PHASE_LABELS: Record<string, string> = {
  idle: 'Idle',
  planning: 'Planning',
  executing: 'Running tools',
  reflecting: 'Reflecting',
  synthesis: 'Writing the answer',
  blocked: 'Blocked',
}

/**
 * One line for "what is the run doing", details behind a toggle.
 *
 * The raw phase, iteration and state names were previously the
 * primary presentation, which is debug output styled as a feature.
 */
export function RunStatus({
  phase,
  iteration,
  state,
  model,
  link,
  blocked,
  onResume,
  busy,
  onCancel,
}: RunStatusProps) {
  const [details, setDetails] = useState(false)

  const phaseLabel = phase
    ? (PHASE_LABELS[phase] ?? phase)
    : state === 'running'
      ? 'Working'
      : 'Idle'

  const step = iteration > 0 ? ` · step ${iteration}` : ''

  return (
    <section className="panel">
      <div className="status-line">
        <span className="status-phase">
          {phaseLabel}
          {step}
        </span>

        <StatusChip status={link} tone="progress" label="live" />
      </div>

      <div className="status-actions">
        {blocked && (
          <button
            type="button"
            className="btn btn-resume"
            onClick={onResume}
          >
            Resume
          </button>
        )}

        {busy && (
          <button
            type="button"
            className="btn btn-cancel"
            onClick={onCancel}
          >
            Stop
          </button>
        )}

        <button
          type="button"
          className="btn btn-quiet"
          aria-expanded={details}
          onClick={() => setDetails((previous) => !previous)}
        >
          {details ? 'Hide details' : 'Details'}
        </button>
      </div>

      {details && (
        <dl className="stats">
          <dt>state</dt>
          <dd>{state}</dd>

          <dt>phase</dt>
          <dd>{phase ?? '—'}</dd>

          <dt>iteration</dt>
          <dd>{iteration || '—'}</dd>

          <dt>model</dt>
          <dd className="truncate" title={model}>
            {model || '—'}
          </dd>

          <dt>link</dt>
          <dd className={`link-${link}`}>{link}</dd>
        </dl>
      )}
    </section>
  )
}
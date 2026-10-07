/**
 * Shared status vocabulary.
 *
 * Four different status vocabularies used to sit on one screen:
 * tool cards said running/done/error, the socket said
 * open/closed/connecting, sessions said idle/completed/blocked, and
 * plan steps said pending/active/completed. This maps all of them
 * onto one set of tones so the UI reads consistently.
 */

import './primitives.css'

export type StatusTone =
  | 'neutral'
  | 'progress'
  | 'ok'
  | 'warn'
  | 'bad'

const TONE_BY_STATUS: Record<string, StatusTone> = {
  // tool states
  running: 'progress',
  success: 'ok',
  done: 'ok',
  error: 'bad',
  failed: 'bad',

  // connection states
  open: 'ok',
  connecting: 'progress',
  closed: 'warn',
  lost: 'warn',

  // session states
  idle: 'neutral',
  completed: 'ok',
  cancelled: 'warn',
  blocked: 'warn',
  waiting_approval: 'warn',

  // plan step states
  pending: 'neutral',
  active: 'progress',

  // plugin / skill / mcp states
  loaded: 'ok',
  activated: 'ok',
  connected: 'ok',
  disabled: 'neutral',
}

export function toneFor(status: string | null | undefined): StatusTone {
  if (!status) {
    return 'neutral'
  }

  return TONE_BY_STATUS[status.toLowerCase()] ?? 'neutral'
}

/** Short label shown next to the tone. */
export function labelFor(status: string | null | undefined): string {
  return status ? status.replace(/_/g, ' ') : '—'
}

interface StatusChipProps {
  status: string | null | undefined
  label?: string
  /** Overrides the mapped tone, for one-off cases. */
  tone?: StatusTone
  title?: string
  className?: string
}

export function StatusChip({
  status,
  label,
  tone,
  title,
  className,
}: StatusChipProps) {
  const resolved = tone ?? toneFor(status)

  return (
    <span
      className={`chip chip-${resolved}${className ? ` ${className}` : ''}`}
      title={title}
    >
      {label ?? labelFor(status)}
    </span>
  )
}
import type { PlanStepInfo } from '../../api/types'

export interface StepSummary {
  /** The short line under a step's description. */
  detail: string
  /** An advisory note, only when there is something worth saying. */
  hint: string
}

/**
 * How far along the plan is.
 *
 * Skipped steps count as done. They are not outstanding work, and a
 * bar that stalls because a step was deliberately passed over reads as
 * a stalled agent.
 */
export function planProgress(steps: PlanStepInfo[]): {
  done: number
  total: number
  percent: number
} {
  const total = steps.length

  const done = steps.filter((step) =>
    ['completed', 'skipped'].includes(step.status),
  ).length

  return {
    done,
    total,
    percent: total === 0 ? 0 : Math.round((done / total) * 100),
  }
}

/**
 * One step's secondary line.
 *
 * An error is shown before anything else, because a failed step is the
 * only thing here that needs acting on. Retries are worth a line of
 * their own: "attempt 3" is the difference between a tool that is
 * struggling and one that has failed once and moved on.
 */
export function describePlan(
  step: PlanStepInfo,
  isCurrent: boolean,
): StepSummary {
  const detail = (() => {
    if (step.status === 'failed' && step.error) {
      return step.error
    }

    if (step.attempts > 1) {
      return `attempt ${step.attempts}`
    }

    if (step.status === 'active') {
      return isCurrent ? 'in progress' : 'active'
    }

    return ''
  })()

  const hint = (() => {
    if (step.delegation) {
      return `may delegate to ${step.delegation.role}`
    }

    return ''
  })()

  return { detail, hint }
}

import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { PlanStepInfo, TodoInfo } from '../../api/types'
import { PlanPanel } from './PlanPanel'
import { describePlan, planProgress } from './planView'

function step(over: Partial<PlanStepInfo> = {}): PlanStepInfo {
  return {
    id: 'execution',
    description: 'do the thing',
    phase: 'execution',
    status: 'pending',
    attempts: 0,
    result: null,
    error: null,
    delegation: null,
    ...over,
  }
}

describe('planProgress', () => {
  it('counts only finished steps as done', () => {
    const { done, total } = planProgress([
      step({ id: 'a', status: 'completed' }),
      step({ id: 'b', status: 'active' }),
      step({ id: 'c', status: 'pending' }),
    ])

    expect(done).toBe(1)
    expect(total).toBe(3)
  })

  it('treats a skipped step as done rather than outstanding', () => {
    // Otherwise the bar stalls on work that was deliberately passed
    // over, which reads as a stalled agent.
    const { done, percent } = planProgress([
      step({ id: 'a', status: 'completed' }),
      step({ id: 'b', status: 'skipped' }),
    ])

    expect(done).toBe(2)
    expect(percent).toBe(100)
  })

  it('does not divide by zero on an empty plan', () => {
    expect(planProgress([])).toEqual({ done: 0, total: 0, percent: 0 })
  })

  it('reports zero percent while nothing is finished', () => {
    const { percent } = planProgress([
      step({ status: 'active' }),
      step({ id: 'b' }),
    ])

    expect(percent).toBe(0)
  })
})

describe('describePlan', () => {
  it('leads with the error on a failed step', () => {
    const summary = describePlan(
      step({ status: 'failed', error: 'tool crashed' }),
      false,
    )

    expect(summary.detail).toBe('tool crashed')
  })

  it('reports a retry rather than hiding it', () => {
    const summary = describePlan(step({ attempts: 3 }), false)

    expect(summary.detail).toBe('attempt 3')
  })

  it('says nothing extra on a plain pending step', () => {
    const summary = describePlan(step(), false)

    expect(summary.detail).toBe('')
    expect(summary.hint).toBe('')
  })

  it('surfaces a delegation hint', () => {
    const summary = describePlan(
      step({
        delegation: {
          role: 'explorer',
          power: 'read',
          reason: 'many files to look at',
        },
      }),
      false,
    )

    expect(summary.hint).toBe('may delegate to explorer')
  })
})

describe('PlanPanel', () => {
  it('renders nothing for an empty plan', () => {
    // "Not planned yet" and "planned, nothing to do" are different
    // situations; only the second deserves an empty heading.
    const { container } = render(
      <PlanPanel
        objective="do the thing"
        steps={[]}
        currentStepId={null}
        revision={1}
      />,
    )

    expect(container.querySelector('.plan-panel')).toBeNull()
  })

  it('lists every step with its status', () => {
    render(
      <PlanPanel
        objective="do the thing"
        steps={[
          step({ id: 'research', description: 'look around', status: 'completed' }),
          step({ id: 'execution', description: 'write code', status: 'active' }),
          step({ id: 'synthesis', description: 'report back' }),
        ]}
        currentStepId="execution"
        revision={1}
      />,
    )

    expect(screen.getByText('look around')).toBeTruthy()
    expect(screen.getByText('write code')).toBeTruthy()
    expect(screen.getByText('report back')).toBeTruthy()
    expect(screen.getByText('1/3')).toBeTruthy()
  })

  it('marks the current step for assistive tech', () => {
    render(
      <PlanPanel
        objective="do the thing"
        steps={[
          step({ id: 'execution', status: 'active' }),
        ]}
        currentStepId="execution"
        revision={1}
      />,
    )

    const current = document.querySelector('[aria-current="step"]')

    expect(current).not.toBeNull()
  })

  it('shows a failure with its reason', () => {
    render(
      <PlanPanel
        objective="do the thing"
        steps={[
          step({
            id: 'execution',
            status: 'failed',
            error: 'timeout after 30s',
          }),
        ]}
        currentStepId={null}
        revision={1}
      />,
    )

    expect(screen.getByText('timeout after 30s')).toBeTruthy()
  })

  it('mentions the revision only when the plan was revised', () => {
    const { rerender } = render(
      <PlanPanel
        objective="x"
        steps={[step()]}
        currentStepId={null}
        revision={1}
      />,
    )

    expect(screen.queryByText(/revised/)).toBeNull()

    rerender(
      <PlanPanel
        objective="x"
        steps={[step()]}
        currentStepId={null}
        revision={3}
      />,
    )

    expect(screen.getByText('revised 3×')).toBeTruthy()
  })
})

describe('plan as a todo list', () => {
  it('is headed as a Todo', () => {
    render(
      <PlanPanel
        objective="изучить документацию"
        steps={[step()]}
        currentStepId="execution"
        revision={1}
      />,
    )

    // The heading the model is shown, so both views name the same
    // thing in the same words.
    expect(screen.getByText('▼Todo')).toBeTruthy()
  })

  it('uses checkbox markers, not status words', () => {
    render(
      <PlanPanel
        objective="изучить документацию"
        steps={[
          step({ id: 'a', status: 'completed' }),
          step({ id: 'b', status: 'active' }),
          step({ id: 'c', status: 'pending' }),
        ]}
        currentStepId="b"
        revision={1}
      />,
    )

    for (const marker of ['[✓]', '[•]', '[○]']) {
      expect(screen.getByLabelText(
        marker === '[✓]' ? 'completed' : marker === '[•]' ? 'active' : 'pending',
      ).textContent).toBe(marker)
    }
  })

  it('distinguishes failed from completed', () => {
    render(
      <PlanPanel
        objective="изучить документацию"
        steps={[
          step({ id: 'a', status: 'completed' }),
          step({ id: 'b', status: 'failed' }),
        ]}
        currentStepId="b"
        revision={1}
      />,
    )

    // A tick/cross pair would have made a failure read as a retryable
    // step at a glance.
    expect(screen.getByLabelText('failed').textContent).toBe('[✗]')
    expect(screen.getByLabelText('completed').textContent).toBe('[✓]')
  })
})

describe('the model\'s own checklist', () => {
  const todo = (
    description: string,
    status: TodoInfo['status'],
  ): TodoInfo => ({
    id: `todo-${description}`,
    description,
    status,
  })

  it('is shown above the plan when present', () => {
    render(
      <PlanPanel
        objective="изучить документацию"
        steps={[step()]}
        currentStepId="execution"
        revision={1}
        todos={[
          todo('Read the config', 'completed'),
          todo('Fetch the list', 'in_progress'),
          todo('Write it up', 'pending'),
          todo('Skip the mirror', 'cancelled'),
        ]}
      />,
    )

    expect(screen.getByText('Read the config')).toBeTruthy()
    expect(screen.getByLabelText('in_progress').textContent).toBe(
      '[•]',
    )
  })

  // The harness saying which phase it is in is not the agent saying
  // what it is doing, so the two must not render identically.
  it('distinguishes the model vocabulary from the plan one', () => {
    render(
      <PlanPanel
        objective="изучить документацию"
        steps={[step({ id: 'e', status: 'active' })]}
        currentStepId="e"
        revision={1}
        todos={[todo('Something', 'in_progress')]}
      />,
    )

    expect(screen.getByLabelText('in_progress')).toBeTruthy()
    expect(screen.getByLabelText('active')).toBeTruthy()
  })

  it('shows the plan alone when the model has not planned', () => {
    render(
      <PlanPanel
        objective="изучить документацию"
        steps={[step()]}
        currentStepId="execution"
        revision={1}
      />,
    )

    expect(screen.queryByLabelText('in_progress')).toBeNull()
    expect(screen.getByText('do the thing')).toBeTruthy()
  })
})

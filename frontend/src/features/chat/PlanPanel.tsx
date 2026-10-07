import type { PlanStepInfo, TodoInfo } from '../../api/types'
import { describePlan, planProgress } from './planView'
import '../../components/primitives.css'

interface PlanPanelProps {
  objective: string
  steps: PlanStepInfo[]
  currentStepId: string | null
  revision: number
  /** The model's own checklist, when it has written one. */
  todos?: TodoInfo[]
}

/**
 * The model's vocabulary, mapped onto the plan's glyphs.
 *
 * Mapped rather than reused: "in_progress" is the agent saying what it
 * is doing, "active" is the harness saying which phase it is in. They
 * look alike and are not the same claim, so the union stays separate
 * and only the rendering is shared.
 */
const TODO_MARK: Record<TodoInfo['status'], string> = {
  completed: '[✓]',
  in_progress: '[•]',
  pending: '[○]',
  cancelled: '[-]',
}

/**
 * The same vocabulary the model is shown.
 *
 * These were a second, different set of glyphs from the ones rendered
 * into the prompt, so the checklist the user read and the checklist the
 * agent acted on disagreed about what "done" looked like. One set,
 * defined once on the backend and mirrored here.
 */
const STATUS_MARK: Record<PlanStepInfo['status'], string> = {
  pending: '[○]',
  active: '[•]',
  completed: '[✓]',
  failed: '[✗]',
  skipped: '[-]',
}

/**
 * The agent's plan, as it stands right now.
 *
 * This answers the question the transcript cannot: what is the agent
 * trying to do, and how much of it is left. Without it the only plan
 * information on screen was a "step 3/7" counter in the status bar,
 * which says the same thing as a page number.
 *
 * A run that never planned anything shows nothing at all rather than an
 * empty heading, because an empty plan and no plan are different
 * situations and only one of them is a problem.
 */
export function PlanPanel({
  objective,
  steps,
  currentStepId,
  revision,
  todos = [],
}: PlanPanelProps) {
  if (steps.length === 0) {
    return null
  }

  const { done, total, percent } = planProgress(steps)

  const currentIndex = steps.findIndex(
    (step) => step.id === currentStepId,
  )

  return (
    <section className="panel plan-panel" aria-label="Execution plan">
      <header className="plan-head">
        <span className="plan-title">▼Todo</span>
        <span className="plan-count">
          {done}/{total}
        </span>
      </header>

      {objective && <p className="plan-objective">{objective}</p>}

      <div
        className="plan-track"
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Plan progress"
      >
        <div className="plan-fill" style={{ width: `${percent}%` }} />
      </div>

      {/*
        The model's checklist first, because it is what the agent
        intends and the plan below is what the harness decided. Both
        are shown: collapsing them into one list would hide the
        difference between "I will do this" and "the runtime is
        checking this".
      */}
      {todos.length > 0 ? (
        <ol className="plan-steps plan-todos">
          {todos.map((todo) => (
            <li
              key={todo.id}
              className={`plan-step is-${todo.status}`}
            >
              <span
                className="plan-mark"
                aria-label={todo.status}
              >
                {TODO_MARK[todo.status]}
              </span>

              <span className="plan-body">
                <span className="plan-desc">
                  {todo.description}
                </span>
              </span>
            </li>
          ))}
        </ol>
      ) : null}

      <ol className="plan-steps">
        {steps.map((step, index) => {
          const summary = describePlan(step, currentIndex === index)

          return (
            <li
              key={step.id}
              className={`plan-step is-${step.status}`}
              aria-current={
                step.id === currentStepId ? 'step' : undefined
              }
            >
              <span
                className="plan-mark"
                aria-label={step.status}
              >
                {STATUS_MARK[step.status]}
              </span>

              <span className="plan-body">
                <span className="plan-desc">{step.description}</span>

                {summary.detail && (
                  <span className="plan-detail">{summary.detail}</span>
                )}

                {summary.hint && (
                  <span className="plan-hint">{summary.hint}</span>
                )}
              </span>
            </li>
          )
        })}
      </ol>

      {revision > 1 && (
        <footer className="plan-foot">
          revised {revision}×
        </footer>
      )}
    </section>
  )
}

import { describe, expect, it } from 'vitest'
import type { WireEnvelope } from '../api/types'
import { initialState, reduce } from './transcript'

/**
 * The plan arriving from the server.
 *
 * These exist because the store is the only place the plan is kept, and
 * a bug here is invisible: the panel simply stops updating, which looks
 * exactly like an agent that stopped planning.
 */

function envelope(data: unknown, seq = 1): WireEnvelope {
  return {
    type: 'plan.updated',
    session_id: 's1',
    seq,
    timestamp: '2026-01-01T00:00:00+00:00',
    data,
  } as WireEnvelope
}

const PLAN = {
  objective: 'find her other accounts',
  steps: [
    {
      id: 'research',
      description: 'read the pages on file',
      phase: 'research',
      status: 'completed',
      attempts: 1,
      result: null,
      error: null,
      delegation: null,
    },
    {
      id: 'execution',
      description: 'expand the sources',
      phase: 'execution',
      status: 'active',
      attempts: 2,
      result: null,
      error: null,
      delegation: null,
    },
  ],
  revision: 2,
  current_step_id: 'execution',
}

describe('plan.updated', () => {
  it('starts as no plan at all', () => {
    // Null and empty are different: null means the runtime has not
    // planned yet, which is normal before a run starts.
    expect(initialState().run.plan).toBeNull()
  })

  it('stores the plan from snake_case fields', () => {
    const state = reduce(initialState(), envelope(PLAN))

    expect(state.run.plan?.objective).toBe('find her other accounts')
    expect(state.run.plan?.revision).toBe(2)
    expect(state.run.plan?.currentStepId).toBe('execution')
    expect(state.run.plan?.steps).toHaveLength(2)
    expect(state.run.plan?.steps[1]?.attempts).toBe(2)
  })

  it('replaces the plan instead of merging it', () => {
    // A revision can drop steps. Merging would resurrect them and the
    // panel would show work that is no longer planned.
    const first = reduce(initialState(), envelope(PLAN))

    const revised = {
      ...PLAN,
      steps: [PLAN.steps[0]!],
      current_step_id: 'research',
      revision: 3,
    }

    const second = reduce(first, envelope(revised, 2))

    expect(second.run.plan?.steps).toHaveLength(1)
    expect(second.run.plan?.revision).toBe(3)
  })

  it('advances the sequence so the cursor moves past it', () => {
    const state = reduce(initialState(), envelope(PLAN, 7))

    expect(state.transcript.latestSeq).toBe(7)
  })

  it('survives a plan with no steps', () => {
    const state = reduce(
      initialState(),
      envelope({ ...PLAN, steps: [], current_step_id: null }),
    )

    expect(state.run.plan?.steps).toEqual([])
    expect(state.run.plan?.currentStepId).toBeNull()
  })

  it('tolerates missing fields rather than throwing', () => {
    const state = reduce(initialState(), envelope({}))

    expect(state.run.plan?.steps).toEqual([])
    expect(state.run.plan?.objective).toBe('')
  })
})

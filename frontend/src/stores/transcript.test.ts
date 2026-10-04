import { describe, expect, it } from 'vitest'
import type { WireEnvelope } from '../api/types'
import {
  groupByRun,
  initialState,
  reduce,
  type AppState,
} from './transcript'

function event(
  type: string,
  data: Record<string, unknown>,
  seq: number,
  runId?: string,
): WireEnvelope {
  return {
    type,
    session_id: 's1',
    seq,
    timestamp: '2026-01-01T00:00:00+00:00',
    data,
    ...(runId ? { run_id: runId } : {}),
  } as WireEnvelope
}

/** One LLM turn: thinking chunks then the answer, as the agent emits. */
function turn(
  state: AppState,
  base: number,
  iteration: number,
  prompt: string,
  answer: string,
): AppState {
  const started = reduce(
    state,
    event('agent.started', { prompt }, base),
  )

  const requested = reduce(
    started,
    event('llm.requested', { iteration, message_count: 1, tool_count: 0 }, base + 1),
  )

  let next = reduce(
    requested,
    event(
      'llm.thinking_chunk',
      { iteration, content: 'think ' },
      base + 2,
    ),
  )

  next = reduce(
    next,
    event('llm.thinking_chunk', { iteration, content: 'hard' }, base + 3),
  )

  next = reduce(
    next,
    event('llm.content_chunk', { iteration, content: answer }, base + 4),
  )

  next = reduce(
    next,
    event(
      'llm.responded',
      {
        iteration,
        content: answer,
        thinking: 'think hard',
        tool_call_count: 0,
      },
      base + 5,
    ),
  )

  return reduce(
    next,
    event('agent.finished', { result: answer }, base + 6),
  )
}

function kinds(state: AppState) {
  return state.transcript.entries.map((entry) => entry.kind)
}

function contents(state: AppState, kind: string) {
  return state.transcript.entries
    .filter((entry) => entry.kind === kind)
    .map((entry) => (entry as { content: string }).content)
}

describe('transcript reducer: repeated LLM requests', () => {
  it('starts a new thinking block for the next request', () => {
    // Regression: iteration restarts at 1 for every run, so matching
    // on iteration merged "thinking 2" into "thinking 1".
    const first = turn(initialState(), 1, 1, 'first question', 'answer one')
    const second = turn(first, 20, 1, 'second question', 'answer two')

    expect(contents(second, 'thinking')).toEqual([
      'think hard',
      'think hard',
    ])
    expect(contents(second, 'assistant')).toEqual([
      'answer one',
      'answer two',
    ])
  })

  it('does not duplicate the final answer', () => {
    // Regression: llm.responded cleared `streaming` before
    // agent.finished ran, so the result was appended a second time.
    const state = turn(initialState(), 1, 1, 'question', 'the answer')

    expect(contents(state, 'assistant')).toEqual(['the answer'])
  })

  it('shows the order request, thinking, answer, request, thinking, answer', () => {
    const state = turn(
      turn(initialState(), 1, 1, 'first question', 'answer one'),
      20,
      1,
      'second question',
      'answer two',
    )

    expect(kinds(state)).toEqual([
      'user',
      'thinking',
      'assistant',
      'user',
      'thinking',
      'assistant',
    ])
  })

  it('keeps separate blocks inside one run with several iterations', () => {
    let state = reduce(
      initialState(),
      event('agent.started', { prompt: 'do things' }, 1),
    )

    state = reduce(
      state,
      event('llm.requested', { iteration: 1 }, 2),
    )

    state = reduce(
      state,
      event('llm.thinking_chunk', { iteration: 1, content: 'plan' }, 3),
    )

    state = reduce(
      state,
      event('llm.responded', { iteration: 1, content: '' }, 4),
    )

    state = reduce(
      state,
      event('llm.requested', { iteration: 2 }, 5),
    )

    state = reduce(
      state,
      event('llm.thinking_chunk', { iteration: 2, content: 'act' }, 6),
    )

    state = reduce(
      state,
      event('llm.content_chunk', { iteration: 2, content: 'done' }, 7),
    )

    state = reduce(
      state,
      event('llm.responded', { iteration: 2, content: 'done' }, 8),
    )

    state = reduce(state, event('agent.finished', { result: 'done' }, 9))

    expect(contents(state, 'thinking')).toEqual(['plan', 'act'])
    expect(contents(state, 'assistant')).toEqual(['done'])
  })

  it('still appends the answer when nothing was streamed', () => {
    // A turn with no content chunks has to render from the result.
    let state = reduce(
      initialState(),
      event('agent.started', { prompt: 'hi' }, 1),
    )

    state = reduce(
      state,
      event('llm.requested', { iteration: 1 }, 2),
    )

    state = reduce(
      state,
      event('llm.responded', { iteration: 1, content: 'only result' }, 3),
    )

    state = reduce(
      state,
      event('agent.finished', { result: 'only result' }, 4),
    )

    expect(contents(state, 'assistant')).toEqual(['only result'])
  })
})

describe('transcript reducer: subagent isolation', () => {
  it('does not merge a subagent answer into the main answer', () => {
    let state = reduce(
      initialState(),
      event('agent.started', { prompt: 'delegate' }, 1),
    )

    state = reduce(
      state,
      event('llm.requested', { iteration: 1 }, 2),
    )

    state = reduce(
      state,
      event('llm.content_chunk', { iteration: 1, content: 'main text' }, 3),
    )

    state = reduce(
      state,
      event(
        'llm.content_chunk',
        { iteration: 1, content: 'child text' },
        4,
        'child-1',
      ),
    )

    const assistants = state.transcript.entries.filter(
      (entry) => entry.kind === 'assistant',
    )

    expect(assistants).toHaveLength(2)
    expect(
      assistants.map((entry) => (entry as { content: string }).content),
    ).toEqual(['main text', 'child text'])
  })

  it('does not merge subagent thinking into main thinking', () => {
    let state = reduce(
      initialState(),
      event('agent.started', { prompt: 'delegate' }, 1),
    )

    state = reduce(
      state,
      event('llm.requested', { iteration: 1 }, 2),
    )

    state = reduce(
      state,
      event('llm.thinking_chunk', { iteration: 1, content: 'main' }, 3),
    )

    state = reduce(
      state,
      event(
        'llm.thinking_chunk',
        { iteration: 1, content: 'child' },
        4,
        'child-1',
      ),
    )

    expect(contents(state, 'thinking')).toEqual(['main', 'child'])
  })

  it('groups entries by the run that produced them', () => {
    let state = reduce(
      initialState(),
      event('agent.started', { prompt: 'delegate' }, 1),
    )

    state = reduce(
      state,
      event('llm.content_chunk', { iteration: 1, content: 'main' }, 2),
    )

    state = reduce(
      state,
      event(
        'llm.content_chunk',
        { iteration: 1, content: 'child' },
        3,
        'child-1',
      ),
    )

    const groups = groupByRun(state.transcript.entries)

    expect(groups.map((group) => group.runId)).toEqual(['', 'child-1'])
  })
})

describe('transcript reducer: replay converges', () => {
  const stream: WireEnvelope[] = [
    event('agent.started', { prompt: 'first question' }, 1),
    event('llm.requested', { iteration: 1 }, 2),
    event('llm.thinking_chunk', { iteration: 1, content: 'think' }, 3),
    event('llm.content_chunk', { iteration: 1, content: 'answer one' }, 4),
    event('llm.responded', { iteration: 1, content: 'answer one' }, 5),
    event('agent.finished', { result: 'answer one' }, 6),
    event('agent.started', { prompt: 'second question' }, 7),
    event('llm.requested', { iteration: 1 }, 8),
    event('llm.thinking_chunk', { iteration: 1, content: 'rethink' }, 9),
    event('llm.content_chunk', { iteration: 1, content: 'answer two' }, 10),
    event('llm.responded', { iteration: 1, content: 'answer two' }, 11),
    event('agent.finished', { result: 'answer two' }, 12),
  ]

  it('rebuilds the same transcript from a full replay', () => {
    // This is what opening a restored session does: replay the
    // whole stream into an empty state.
    const once = stream.reduce(reduce, initialState())

    const replayed = [...stream].reduce(reduce, initialState())

    expect(contents(replayed, 'thinking')).toEqual(
      contents(once, 'thinking'),
    )
    expect(contents(replayed, 'assistant')).toEqual(
      contents(once, 'assistant'),
    )
  })

  it('converges when a client resumes from a cursor', () => {
    // The client keeps its view and receives only what it missed.
    const all = stream.reduce(reduce, initialState())
    const cursor = 6

    const partial = stream
      .filter((item) => item.seq <= cursor)
      .reduce(reduce, initialState())

    const resumed = stream
      .filter((item) => item.seq > cursor)
      .reduce(reduce, partial)

    expect(contents(resumed, 'thinking')).toEqual(
      contents(all, 'thinking'),
    )
    expect(contents(resumed, 'assistant')).toEqual(
      contents(all, 'assistant'),
    )
  })
})

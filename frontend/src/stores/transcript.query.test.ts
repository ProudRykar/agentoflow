import { describe, expect, it } from 'vitest'
import type { WireEnvelope } from '../api/types'
import {
  budgetSeverity,
  queryTranscript,
  repeatedFailures,
  searchableText,
  initialState,
  reduce,
  type Entry,
} from './transcript'

function event(
  type: string,
  data: Record<string, unknown>,
  seq: number,
): WireEnvelope {
  return {
    type,
    session_id: 's1',
    seq,
    timestamp: '2026-01-01T00:00:00+00:00',
    data,
  } as WireEnvelope
}

const ENTRIES: Entry[] = [
  { kind: 'user', id: 'u1', content: 'fetch the tags' },
  { kind: 'thinking', id: 't1', iteration: 1, content: 'need a tag tool', expanded: false, runId: '' },
  {
    kind: 'tool',
    id: 'c1',
    callId: 'call1',
    name: 'get_tags',
    arguments: { page: 1 },
    status: 'error',
    output: null,
    errorCode: 'invalid_type',
    errorMessage: 'expected int',
    durationSeconds: 1.2,
    expanded: false,
    runId: '',
  },
  { kind: 'assistant', id: 'a1', content: 'The tool failed, sorry.', streaming: false, runId: '' },
]

// ======================================================================
// Budget meter
// ======================================================================


describe('context budget', () => {
  it('reads the limit from the request event', () => {
    const state = reduce(
      initialState(),
      event(
        'llm.requested',
        {
          iteration: 1,
          estimated_tokens: 24000,
          context_limit: 28000,
          context_trimmed: true,
        },
        2,
      ),
    )

    expect(state.run.contextUsed).toBe(24000)
    expect(state.run.contextLimit).toBe(28000)
    expect(state.run.contextTrimmed).toBe(true)
  })

  it('keeps the limit when a later request omits it', () => {
    let state = reduce(
      initialState(),
      event('llm.requested', { estimated_tokens: 10, context_limit: 100 }, 1),
    )

    state = reduce(
      state,
      event('llm.requested', { estimated_tokens: 20 }, 2),
    )

    expect(state.run.contextLimit).toBe(100)
    expect(state.run.contextUsed).toBe(20)
  })

  it.each([
    [50, 100, 'ok'],
    [85, 100, 'warn'],
    [120, 100, 'over'],
  ])('rates %i/%i as %s', (used, limit, tone) => {
    expect(budgetSeverity(used, limit).tone).toBe(tone)
  })

  it('is neutral without numbers', () => {
    expect(budgetSeverity(null, null)).toEqual({ ratio: 0, tone: 'ok' })
  })
})

// ======================================================================
// Repeated failures
// ======================================================================


describe('repeat failures', () => {
  it('counts failures per tool and clears on success', () => {
    let state = reduce(
      initialState(),
      event('tool.started', { tool_call_id: 'c1', tool_name: 'get_tags', arguments: {} }, 1),
    )

    state = reduce(
      state,
      event(
        'tool.finished',
        {
          tool_call_id: 'c1',
          tool_name: 'get_tags',
          error_code: 'invalid_type',
          error_message: 'boom',
          output: null,
          duration_seconds: 1,
        },
        2,
      ),
    )

    expect(state.run.repeatFailures.get_tags).toBe(1)

    state = reduce(
      state,
      event(
        'tool.finished',
        {
          tool_call_id: 'c1',
          tool_name: 'get_tags',
          error_code: 'invalid_type',
          error_message: 'boom',
          output: null,
          duration_seconds: 1,
        },
        3,
      ),
    )

    expect(state.run.repeatFailures.get_tags).toBe(2)

    state = reduce(
      state,
      event(
        'tool.finished',
        {
          tool_call_id: 'c1',
          tool_name: 'get_tags',
          error_code: null,
          error_message: null,
          output: 'ok',
          duration_seconds: 1,
        },
        4,
      ),
    )

    expect(state.run.repeatFailures.get_tags).toBeUndefined()
  })

  it('reports only the offenders, worst first', () => {
    const list = repeatedFailures({
      once_failed: 1,
      noisy: 5,
      also_noisy: 3,
    })

    expect(list.map((item) => item.tool)).toEqual([
      'noisy',
      'also_noisy',
    ])
  })

  it('ignores a single failure', () => {
    expect(repeatedFailures({ flaky: 1 })).toEqual([])
  })
})

// ======================================================================
// Search and filter
// ======================================================================


describe('transcript query', () => {
  it('filters by kind', () => {
    const slice = queryTranscript(ENTRIES, {
      search: '',
      filter: 'errors',
    })

    expect(slice.entries).toHaveLength(1)
    expect(slice.entries[0].kind).toBe('tool')
  })

  it('searches message text', () => {
    const slice = queryTranscript(ENTRIES, {
      search: 'tags',
      filter: 'all',
    })

    // "fetch the tags" matches the user message, "get_tags" the
    // tool card. "need a tag tool" has no plural, so it does not.
    expect(slice.entries.map((e) => e.id)).toEqual(['u1', 'c1'])
    expect(slice.matches).toBe(2)
    expect(slice.total).toBe(4)
    expect(slice.active).toBe(true)
  })

  it('is case insensitive', () => {
    const slice = queryTranscript(ENTRIES, {
      search: 'SORRY',
      filter: 'all',
    })

    expect(slice.entries.map((e) => e.id)).toEqual(['a1'])
  })

  it('searches tool output and errors', () => {
    expect(searchableText(ENTRIES[2])).toContain('expected int')
    expect(searchableText(ENTRIES[2])).toContain('get_tags')
  })

  it('returns everything when inactive', () => {
    const slice = queryTranscript(ENTRIES, { search: '', filter: 'all' })

    expect(slice.entries).toHaveLength(4)
    expect(slice.active).toBe(false)
  })

  it('combines a search with a filter', () => {
    const slice = queryTranscript(ENTRIES, {
      search: 'tags',
      filter: 'tools',
    })

    expect(slice.entries.map((e) => e.id)).toEqual(['c1'])
  })

  it('reports no matches honestly', () => {
    const slice = queryTranscript(ENTRIES, {
      search: 'nothing here',
      filter: 'all',
    })

    expect(slice.entries).toHaveLength(0)
    expect(slice.matches).toBe(0)
    expect(slice.total).toBe(4)
  })
})
// ======================================================================
// Card defaults
// ======================================================================


describe('card expansion defaults', () => {
  it('opens a thinking card expanded', () => {
    // The reasoning is the substance of a run. Collapsed by default
    // it read as absent, and a wrong turn stayed hidden until after
    // the answer had been read.
    const state = reduce(
      initialState(),
      event(
        'llm.thinking_chunk',
        { iteration: 1, content: 'weighing the options' },
        2,
      ),
    )

    const thinking = state.transcript.entries.find(
      (entry) => entry.kind === 'thinking',
    )

    expect(thinking).toBeDefined()
    expect(
      thinking?.kind === 'thinking' && thinking.expanded,
    ).toBe(true)
  })

  it('opens a tool card', () => {
    // A hidden result is the one thing in the transcript that cannot
    // be read without another click, and it is exactly what explains
    // what the agent just did.
    const state = reduce(
      initialState(),
      event(
        'tool.started',
        {
          tool_call_id: 'c1',
          tool_name: 'read_file',
          arguments: {},
        },
        2,
      ),
    )

    const tool = state.transcript.entries.find(
      (entry) => entry.kind === 'tool',
    )

    expect(tool).toBeDefined()
    expect(
      tool?.kind === 'tool' && tool.expanded,
    ).toBe(true)
  })
})

// ======================================================================
// A subagent's prompt is not a user turn
// ======================================================================


describe('subagent prompts', () => {
  it('does not turn a delegation prompt into a user message', () => {
    // The role, objective and instructions are already visible as the
    // subagent.run arguments. Repeating them as a user turn makes the
    // delegation look like the user said it.
    const parent = reduce(
      initialState(),
      event(
        'agent.started',
        {
          prompt: 'find the tags',
          run_id: 'run-parent',
          parent_run_id: null,
          agent_id: 'main',
          role: 'main',
          model: 'gemma4:e4b-it-qat',
        },
        2,
      ),
    )

    const state = reduce(
      parent,
      event(
        'agent.started',
        {
          prompt: 'Your role: writer\n\nYour objective:\nwrite it',
          run_id: 'run-child',
          parent_run_id: 'run-parent',
          agent_id: 'subagent:writer',
          role: 'writer',
          model: 'gemma4:12b',
        },
        3,
      ),
    )

    const users = state.transcript.entries.filter(
      (entry) => entry.kind === 'user',
    )

    expect(users).toHaveLength(1)
    expect(users[0]?.content).toBe('find the tags')
  })
})

describe('replayed subagent prompts', () => {
  it('does not repeat a delegation prompt on a second agent.started', () => {
    // A reconnect can replay the event. Without the run already being
    // known as a child, the missing parent_run_id turns the prompt
    // back into a user turn.
    const first = reduce(
      initialState(),
      event(
        'agent.started',
        {
          prompt: 'find the tags',
          run_id: 'run-parent',
          parent_run_id: null,
          agent_id: 'main',
          role: 'main',
          model: 'gemma4:e4b-it-qat',
        },
        2,
      ),
    )

    const child = reduce(
      first,
      event(
        'agent.started',
        {
          prompt: 'Your role: writer',
          run_id: 'run-child',
          parent_run_id: 'run-parent',
          agent_id: 'subagent:writer',
          role: 'writer',
          model: 'gemma4:12b',
        },
        3,
      ),
    )

    const replayed = reduce(
      child,
      event(
        'agent.started',
        {
          prompt: 'Your role: writer',
          run_id: 'run-child',
          parent_run_id: null,
          agent_id: 'subagent:writer',
          role: 'writer',
          model: 'gemma4:12b',
        },
        4,
      ),
    )

    const users = replayed.transcript.entries.filter(
      (entry) => entry.kind === 'user',
    )

    expect(users).toHaveLength(1)
    expect(users[0]?.content).toBe('find the tags')
  })
})

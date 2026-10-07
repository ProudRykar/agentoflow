import type {
  PlanStepInfo,
  TodoInfo,
  WireEnvelope,
} from '../api/types'
import {
  addRunUsage,
  emptyRunUsage,
  type RunUsageView,
} from '../features/chat/runUsage'

export type ToolStatus = 'running' | 'success' | 'error'

export interface UserEntry {
  kind: 'user'
  id: string
  content: string
}

export interface AssistantEntry {
  kind: 'assistant'
  id: string
  content: string
  streaming: boolean
  /** run that produced the text; empty for the main agent */
  runId: string
}

export interface ThinkingEntry {
  kind: 'thinking'
  id: string
  iteration: number
  content: string
  expanded: boolean
  /** run that produced the reasoning; empty for the main agent */
  runId: string
}

export interface ErrorEntry {
  kind: 'error'
  id: string
  content: string
}

export type Entry =
  | UserEntry
  | AssistantEntry
  | ThinkingEntry
  | ToolEntry
  | ErrorEntry

export interface RunInfo {
  runId: string
  parentRunId: string | null
  role: string
  model: string
  agentId: string
  finished: boolean
  /** Entry indices produced by this run, for tree rendering. */
  entryIds: string[]
}

export interface ToolEntry {
  kind: 'tool'
  id: string
  callId: string
  name: string
  arguments: Record<string, unknown>
  status: ToolStatus
  output: string | null
  errorCode: string | null
  errorMessage: string | null
  durationSeconds: number | null
  expanded: boolean
  /** run that produced the call; empty for the main agent */
  runId: string
}

/**
 * Which transcript entries the current LLM request is filling.
 *
 * Streaming chunks cannot be matched by iteration alone: iteration
 * restarts at 1 for every run, so the second user message would
 * merge its reasoning into the first message's block. Tracking the
 * entry explicitly is also what lets `agent.finished` tell "the
 * answer was already streamed" from "nothing was streamed yet".
 */
export interface RunBlocks {
  /** entry receiving thinking chunks for the current request */
  thinkingId: string | null
  /** entry receiving content chunks for the current request */
  contentId: string | null
  /** this run has produced assistant text at least once */
  producedContent: boolean
}

export interface RunState {
  phase: string | null
  iteration: number
  /** Tokens in the last assembled request, and the budget for it. */
  contextUsed: number | null
  contextLimit: number | null
  /** The assembler dropped or truncated blocks to fit. */
  contextTrimmed: boolean
  /** True when the count came from a real tokenizer, not a heuristic. */
  contextCounterExact: boolean
  /** Which blocks were dropped or cut, by name. */
  contextDropped: string[]
  /** What the provider last reported consuming, when it said. */
  contextMeasured: number | null

  /**
   * What the whole run cost, so far.
   *
   * Per run, not per session: one figure across a whole conversation
   * cannot be compared against anything, because it grows for reasons
   * that have nothing to do with the turn being looked at.
   */
  runUsage: RunUsageView

  /** The run's token allowance, null when none is configured. */
  maxPromptTokens: number | null
  /** The local estimate for that same request, beside the real figure. */
  contextEstimate: number | null
  /** tool name -> how many times it failed with the same arguments. */
  repeatFailures: Record<string, number>
  streamingRunId: string | null
  thinkingRunId: string | null
  runs: Record<string, RunInfo>
  blocks: Record<string, RunBlocks>
  sessionState: string
  running: boolean
  estimatedTokens: number | null
  lastError: string
  model: string
  workingDirectory: string
  /**
   * The runtime plan, replaced wholesale on every `plan.updated`.
   * Null until the first one arrives, which is distinct from an empty
   * plan: one means "not planned yet", the other means "planned, and
   * there is nothing to do".
   */
  plan: {
    objective: string
    steps: PlanStepInfo[]
    revision: number
    currentStepId: string | null
  } | null

  /** The checklist the model wrote, when it has written one. */
  todos: TodoInfo[]
}

export interface TranscriptState {
  entries: Entry[]
  latestSeq: number
  approval: {
    approvalId: string
    toolName: string
    permission: string
    reason: string
    arguments: Record<string, unknown>
  } | null
  phaseReason: string
}

export interface ConnectionState {
  status: 'idle' | 'connecting' | 'open' | 'closed'
  attempt: number
  lastError: string | null
}

export interface AppState {
  transcript: TranscriptState
  run: RunState
  connection: ConnectionState
}

const EMPTY_BLOCKS: RunBlocks = {
  thinkingId: null,
  contentId: null,
  producedContent: false,
}

function blockKey(runKey: string): string {
  return runKey || 'main'
}

function blocksOf(
  run: RunState,
  runKey: string,
): RunBlocks {
  return run.blocks[blockKey(runKey)] ?? EMPTY_BLOCKS
}

/** Replace the streaming block bookkeeping for one run. */
function withBlocks(
  run: RunState,
  runKey: string,
  patch: Partial<RunBlocks>,
): RunState {
  const key = blockKey(runKey)

  return {
    ...run,
    blocks: {
      ...run.blocks,
      [key]: { ...blocksOf(run, runKey), ...patch },
    },
  }
}

/** Append a chunk to a known entry, or report that it is gone. */
function appendToEntry(
  entries: Entry[],
  id: string | null,
  chunk: string,
): Entry[] | null {
  if (!id) {
    return null
  }

  const index = entries.findIndex((entry) => entry.id === id)

  if (index === -1) {
    return null
  }

  const next = [...entries]
  const entry = next[index] as AssistantEntry | ThinkingEntry

  next[index] = { ...entry, content: entry.content + chunk }

  return next
}

/** Register an entry id on a run, creating a stub run if needed. */
function appendToRun(
  run: RunState,
  runKey: string,
  entryId: string,
  patch: Partial<RunState> = {},
): RunState {
  const key = runKey || 'main'

  const previous = run.runs[key]

  const next: RunInfo = previous ?? {
    runId: runKey,
    parentRunId: null,
    role: key === 'main' ? 'main' : 'subagent',
    model: '',
    agentId: '',
    finished: false,
    entryIds: [],
  }

  return {
    ...run,
    ...patch,
    runs: {
      ...run.runs,
      [key]: {
        ...next,
        entryIds: next.entryIds.includes(entryId)
          ? next.entryIds
          : [...next.entryIds, entryId],
      },
    },
  }
}

/**
 * Group transcript entries by the run that produced them.
 *
 * `parent_run_id` links a child run to its parent *run*, not to the
 * tool call that spawned it, so the backend cannot express a nested
 * "call -> subagent" tree today. Grouping by run is the most
 * specific structure the event stream actually supports.
 */
export function groupByRun(
  entries: Entry[],
): Array<{ runId: string; entries: Entry[] }> {
  const groups: Array<{ runId: string; entries: Entry[] }> = []

  for (const entry of entries) {
    const runId = runIdOf(entry)

    const last = groups[groups.length - 1]

    if (last && last.runId === runId) {
      last.entries.push(entry)
      continue
    }

    groups.push({ runId, entries: [entry] })
  }

  return groups
}

export function runIdOf(entry: Entry): string {
  if (
    entry.kind === 'tool' ||
    entry.kind === 'assistant' ||
    entry.kind === 'thinking'
  ) {
    return entry.runId
  }

  return ''
}

export function initialState(): AppState {
  return {
    transcript: {
      entries: [],
      latestSeq: -1,
      approval: null,
      phaseReason: '',
    },
    run: {
      phase: null,
      iteration: 0,
      contextUsed: null,
      contextLimit: null,
      contextTrimmed: false,
      contextCounterExact: true,
      contextDropped: [],
      contextMeasured: null,
      runUsage: emptyRunUsage(),
      maxPromptTokens: null,
      contextEstimate: null,
      repeatFailures: {},
      streamingRunId: null,
      thinkingRunId: null,
      runs: {},
      blocks: {},
      sessionState: 'idle',
      running: false,
      estimatedTokens: null,
      lastError: '',
      model: '',
      plan: null,
      todos: [],
      workingDirectory: '',
    },
    connection: {
      status: 'idle',
      attempt: 0,
      lastError: null,
    },
  }
}

let counter = 0

/**
 * Remove the assistant's last answer, and anything it produced.
 *
 * Tool entries are dropped with it: they belong to the answer being
 * replaced, and leaving them would show tool calls whose result the
 * new answer never refers to.
 */
function dropTrailingAssistant(entries: Entry[]): Entry[] {
  let cutoff = entries.length

  while (cutoff > 0) {
    const kind = entries[cutoff - 1].kind

    if (kind === 'assistant' || kind === 'tool') {
      cutoff -= 1
      continue
    }

    break
  }

  return cutoff === entries.length ? entries : entries.slice(0, cutoff)
}

function nextId(prefix: string): string {
  counter += 1

  return `${prefix}-${counter}`
}

/**
 * Fold one wire event into the view model.
 *
 * This is a pure reducer: the backend owns the domain state and the
 * reducer only mirrors it. Frontend state is rebuilt from the event
 * stream, so a reconnect that replays history converges to the same
 * result.
 */
export function reduce(
  state: AppState,
  event: WireEnvelope,
): AppState {
  const runKey = event.run_id ?? ''

  switch (event.type) {
    case 'session.snapshot': {
      const data = event.data as unknown as {
        state: string
        model: string
        max_prompt_tokens?: number | null
        working_directory: string
        phase: string | null
        iteration: number | null
        running: boolean
        last_error: string
        approval: {
          approval_id: string
          tool_name: string
          permission: string
          reason: string
          arguments: Record<string, unknown>
        } | null
      }

      return {
        ...state,
        transcript: {
          ...state.transcript,
          latestSeq: event.seq,
          approval: data.approval
            ? {
                approvalId: data.approval.approval_id,
                toolName: data.approval.tool_name,
                permission: data.approval.permission,
                reason: data.approval.reason,
                arguments: data.approval.arguments,
              }
            : null,
        },
        run: {
          ...state.run,
          sessionState: data.state,
          running: data.running,
          phase: data.phase ?? state.run.phase,
          iteration: data.iteration ?? state.run.iteration,
          lastError: data.last_error,
          model: data.model || state.run.model,
          workingDirectory:
            data.working_directory || state.run.workingDirectory,
          maxPromptTokens:
            data.max_prompt_tokens ?? state.run.maxPromptTokens,
        },
      }
    }

    case 'agent.started': {
      const data = event.data as unknown as {
        prompt: string
        run_id: string
        parent_run_id: string | null
        agent_id: string
        role: string
        model: string
        regenerated?: boolean
      }

      const runId = data.run_id || 'main'

      // A run already known to be a child is a subagent, even if this
      // event carries no parent_run_id: a replayed or duplicated
      // agent.started would otherwise repeat the delegation prompt as
      // a fresh user turn.
      const isSubagent =
        Boolean(data.parent_run_id) ||
        Boolean(state.run.runs[runId]?.parentRunId)

      // A retry repeats a question already on screen, so echoing the
      // prompt would show it twice. The discarded answer goes instead,
      // which is what the user asked for: a different answer, not a
      // second copy of the same question.
      const entries = isSubagent
        ? state.transcript.entries
        : data.regenerated
          ? dropTrailingAssistant(state.transcript.entries)
          : [
              ...state.transcript.entries,
              { kind: 'user', id: nextId('user'), content: data.prompt } as UserEntry,
            ]


      const previous = state.run.runs[runId]

      return {
        ...state,
        transcript: {
          ...state.transcript,
          entries,
          latestSeq: event.seq,
        },
        run: {
          ...state.run,
          running: true,
          sessionState: 'running',
          runs: {
            ...state.run.runs,
            [runId]: {
              runId: data.run_id,
              parentRunId: data.parent_run_id,
              role: data.role,
              model: data.model,
              agentId: data.agent_id,
              finished: false,
              entryIds: previous?.entryIds ?? [],
            },
          },
        },
      }
    }

    case 'agent.phase_changed': {
      const data = event.data as unknown as { phase: string; reason: string }

      return {
        ...state,
        transcript: {
          ...state.transcript,
          latestSeq: event.seq,
          phaseReason: data.reason ?? '',
        },
        run: {
          ...state.run,
          phase: data.phase,
        },
      }
    }

    case 'plan.updated': {
      const data = event.data as unknown as {
        objective: string
        steps: PlanStepInfo[]
        revision: number
        current_step_id: string | null
        todos?: TodoInfo[]
      }

      return {
        ...state,
        transcript: {
          ...state.transcript,
          latestSeq: event.seq,
        },
        run: {
          ...state.run,
          // Replaced, never merged: the server sends the whole plan,
          // and merging would keep steps that a revision has dropped.
          plan: {
            objective: data.objective ?? '',
            steps: data.steps ?? [],
            revision: data.revision ?? 0,
            currentStepId: data.current_step_id ?? null,
          },
          // Replaced wholesale, like the plan: the model sends the
          // whole list, and an item it dropped must disappear rather
          // than linger as a row the user thinks is still planned.
          todos: data.todos ?? state.run.todos,
        },
      }
    }

    case 'llm.requested': {
      const data = event.data as unknown as {
        iteration: number
        estimated_tokens: number | null
        context_limit?: number | null
        context_trimmed?: boolean
        counter_exact?: boolean
        dropped_blocks?: string[]
      }

      return {
        ...state,
        transcript: { ...state.transcript, latestSeq: event.seq },
        run: withBlocks(
          {
            ...state.run,
            iteration: data.iteration,
            contextUsed: data.estimated_tokens ?? null,
            contextLimit:
              data.context_limit ?? state.run.contextLimit,
            contextTrimmed: Boolean(data.context_trimmed),
            contextCounterExact: data.counter_exact !== false,
            contextDropped: data.dropped_blocks ?? [],
            thinkingRunId: runKey || state.run.thinkingRunId,
            estimatedTokens: data.estimated_tokens,
          },
          runKey,
          // A new request must not extend the previous one's
          // reasoning or answer.
          { thinkingId: null, contentId: null },
        ),
      }
    }

    case 'llm.thinking_chunk': {
      const data = event.data as unknown as {
        content: string
        iteration: number
      }

      const blocks = blocksOf(state.run, runKey)

      const merged = appendToEntry(
        state.transcript.entries,
        blocks.thinkingId,
        data.content,
      )

      if (merged) {
        return {
          ...state,
          transcript: {
            ...state.transcript,
            entries: merged,
            latestSeq: event.seq,
          },
          run: withBlocks(state.run, runKey, {
            thinkingId: blocks.thinkingId,
          }),
        }
      }

      const id = nextId('thinking')

      const entries: Entry[] = [
        ...state.transcript.entries,
        {
          kind: 'thinking',
          id,
          iteration: data.iteration,
          content: data.content,
          // Open by default. The reasoning is the substance of a
          // run, and a collapsed card hides a wrong turn until after
          // the answer has been read.
          expanded: true,
          runId: runKey,
        },
      ]

      return {
        ...state,
        transcript: {
          ...state.transcript,
          entries,
          latestSeq: event.seq,
        },
        run: appendToRun(
          withBlocks(state.run, runKey, { thinkingId: id }),
          runKey,
          id,
          {
            thinkingRunId: runKey || null,
          },
        ),
      }
    }

    case 'llm.content_chunk': {
      const data = event.data as unknown as {
        content: string
        iteration: number
      }

      const blocks = blocksOf(state.run, runKey)

      const merged = appendToEntry(
        state.transcript.entries,
        blocks.contentId,
        data.content,
      )

      if (merged) {
        return {
          ...state,
          transcript: {
            ...state.transcript,
            entries: merged.map((entry) =>
              entry.id === blocks.contentId &&
              entry.kind === 'assistant'
                ? { ...entry, streaming: true }
                : entry,
            ),
            latestSeq: event.seq,
          },
          run: withBlocks(state.run, runKey, {
            contentId: blocks.contentId,
            producedContent: true,
          }),
        }
      }

      const id = nextId('assistant')

      const entries: Entry[] = [
        ...state.transcript.entries,
        {
          kind: 'assistant',
          id,
          content: data.content,
          streaming: true,
          runId: runKey,
        },
      ]

      return {
        ...state,
        transcript: {
          ...state.transcript,
          entries,
          latestSeq: event.seq,
        },
        run: appendToRun(
          withBlocks(state.run, runKey, {
            contentId: id,
            producedContent: true,
          }),
          runKey,
          id,
          {
            streamingRunId: runKey || null,
            thinkingRunId: null,
          },
        ),
      }
    }

    case 'llm.responded': {
      const usage = event.data as unknown as {
        prompt_tokens?: number | null
        estimated_prompt_tokens?: number | null
      }

      const counted = addRunUsage(state.run.runUsage, runKey, {
        promptTokens: usage.prompt_tokens ?? null,
        completionTokens: 0,
        estimatedCost: null,
      })

      // Only this run's entries: clearing every assistant entry
      // would also close a subagent's in-flight answer.
      const entries = state.transcript.entries.map((entry) =>
        entry.kind === 'assistant' && entry.runId === runKey
          ? { ...entry, streaming: false }
          : entry,
      )

      return {
        ...state,
        transcript: { ...state.transcript, entries, latestSeq: event.seq },
        run: withBlocks(
          {
            ...state.run,
            streamingRunId: null,
            thinkingRunId: null,
            // The provider's own figure beside our estimate. Having
            // both is what makes the meter trustworthy: a guess shown
            // as a measurement is indistinguishable from a wrong one.
            contextMeasured: usage.prompt_tokens ?? null,
            contextEstimate:
              usage.estimated_prompt_tokens ?? null,
            runUsage: counted,
          },
          runKey,
          { thinkingId: null, contentId: null },
        ),
      }
    }

    case 'tool.started': {
      const data = event.data as unknown as {
        tool_call_id: string
        tool_name: string
        arguments: Record<string, unknown>
      }

      const entries: Entry[] = [...state.transcript.entries]

      const tool: ToolEntry = {
        kind: 'tool',
        id: nextId('tool'),
        callId: data.tool_call_id,
        name: data.tool_name,
        arguments: data.arguments ?? {},
        status: 'running',
        output: null,
        errorCode: null,
        errorMessage: null,
        durationSeconds: null,
        // Open by default. A hidden result is the one thing in the
        // transcript that cannot be read without another click, and it
        // is exactly the part that explains what the agent just did.
        expanded: true,
        runId: runKey,
      }

      entries.push(tool)

      return {
        ...state,
        transcript: {
          ...state.transcript,
          entries,
          latestSeq: event.seq,
        },
        run: appendToRun(state.run, runKey, tool.id, {
          thinkingRunId: null,
        }),
      }
    }

    case 'tool.finished': {
      const data = event.data as unknown as {
        tool_call_id: string
        tool_name: string
        error_code: string | null
        error_message: string | null
        output: string | null
        duration_seconds: number | null
      }

      const entries: Entry[] = state.transcript.entries.map((entry) =>
        entry.kind === 'tool' && entry.callId === data.tool_call_id
          ? {
              ...entry,
              status:
                data.error_code === null
                  ? ('success' as const)
                  : ('error' as const),
              output: data.output,
              errorCode: data.error_code,
              errorMessage: data.error_message,
              durationSeconds: data.duration_seconds,
            }
          : entry,
      )

      const failed = data.error_code !== null

      if (failed) {
        // A repeat count per tool is what turns "something failed"
        // into something the user can act on: one hiccup is noise,
        // five identical failures is a broken tool.
        const previous = state.run.repeatFailures[data.tool_name] ?? 0
        const next = previous + 1

        return {
          ...state,
          transcript: { ...state.transcript, entries, latestSeq: event.seq },
          run: {
            ...state.run,
            repeatFailures: {
              ...state.run.repeatFailures,
              [data.tool_name]: next,
            },
          },
        }
      }

      // A success means the tool is healthy again.
      if (state.run.repeatFailures[data.tool_name]) {
        const next = { ...state.run.repeatFailures }

        delete next[data.tool_name]

        return {
          ...state,
          transcript: { ...state.transcript, entries, latestSeq: event.seq },
          run: { ...state.run, repeatFailures: next },
        }
      }

      return {
        ...state,
        transcript: { ...state.transcript, entries, latestSeq: event.seq },
      }
    }

    case 'agent.finished': {
      const data = event.data as unknown as { result: string }

      // Regression: this used to look for an assistant entry with
      // ``streaming`` still true. ``llm.responded`` always clears
      // that flag just before this event, so the check was never
      // true and the result was appended after the streamed text,
      // showing the same answer twice. What actually matters is
      // whether this run produced any assistant text at all.
      const producedContent = blocksOf(state.run, runKey).producedContent

      let entries = state.transcript.entries.map((entry) =>
        entry.kind === 'assistant' && entry.runId === runKey
          ? { ...entry, streaming: false }
          : entry,
      )

      if (!runKey && !producedContent && data.result) {
        entries = [
          ...entries,
          {
            kind: 'assistant',
            id: nextId('assistant'),
            content: data.result,
            streaming: false,
            runId: runKey,
          },
        ]
      }

      return {
        ...state,
        transcript: { ...state.transcript, entries, latestSeq: event.seq },
        run: {
          ...state.run,
          blocks: {
            ...state.run.blocks,
            [blockKey(runKey)]: EMPTY_BLOCKS,
          },
          running: false,
          sessionState: 'completed',
          streamingRunId: null,
          thinkingRunId: null,
          runs: runKey
            ? {
                ...state.run.runs,
                [runKey]: {
                  ...(state.run.runs[runKey] ?? {
                    runId: runKey,
                    parentRunId: null,
                    role: 'subagent',
                    model: '',
                    agentId: '',
                    entryIds: [],
                  }),
                  finished: true,
                },
              }
            : state.run.runs,
        },
      }
    }

    case 'approval.requested': {
      const data = event.data as unknown as {
        approval_id: string
        tool_name: string
        permission: string
        reason: string
        arguments: Record<string, unknown>
      }

      return {
        ...state,
        transcript: {
          ...state.transcript,
          latestSeq: event.seq,
          approval: {
            approvalId: data.approval_id,
            toolName: data.tool_name,
            permission: data.permission,
            reason: data.reason,
            arguments: data.arguments ?? {},
          },
        },
        run: { ...state.run, sessionState: 'waiting_approval' },
      }
    }

    case 'approval.resolved': {
      return {
        ...state,
        transcript: {
          ...state.transcript,
          latestSeq: event.seq,
          approval: null,
        },
        run: {
          ...state.run,
          sessionState: state.run.running ? 'running' : state.run.sessionState,
        },
      }
    }

    case 'run.failed': {
      const data = event.data as unknown as { error: string }

      const id = nextId('error')

      return {
        ...state,
        transcript: {
          ...state.transcript,
          latestSeq: event.seq,
          entries: [
            ...state.transcript.entries,
            {
              kind: 'error',
              id,
              content: data.error,
            },
          ],
        },
        run: {
          ...state.run,
          running: false,
          sessionState: 'error',
          lastError: data.error,
          streamingRunId: null,
          thinkingRunId: null,
        },
      }
    }

    case 'heartbeat':
    case 'ping':
    case 'pong':
      return state

    default:
      return state
  }
}

export function toggleEntry(state: AppState, id: string): AppState {
  let changed = false

  const entries = state.transcript.entries.map((entry) => {
    if (entry.id !== id) {
      return entry
    }

    if (entry.kind !== 'tool' && entry.kind !== 'thinking') {
      return entry
    }

    changed = true

    return { ...entry, expanded: !entry.expanded }
  })

  // Preserve identity when nothing toggled, so consumers relying on
  // reference equality do not re-render.
  if (!changed) {
    return state
  }

  return {
    ...state,
    transcript: { ...state.transcript, entries },
  }
}

/* ==================================================================
 * Transcript queries
 * ================================================================== */

/** Entry kinds that can be filtered out of the transcript. */
export type EntryFilter = 'all' | 'thinking' | 'tools' | 'errors'

export const ENTRY_FILTERS: {
  value: EntryFilter
  label: string
}[] = [
  { value: 'all', label: 'All' },
  { value: 'thinking', label: 'Thinking' },
  { value: 'tools', label: 'Tools' },
  { value: 'errors', label: 'Errors' },
]

function matchesFilter(entry: Entry, filter: EntryFilter): boolean {
  switch (filter) {
    case 'thinking':
      return entry.kind === 'thinking'
    case 'tools':
      return entry.kind === 'tool'
    case 'errors':
      return (
        entry.kind === 'error' ||
        (entry.kind === 'tool' && entry.status === 'error')
      )
    case 'all':
    default:
      return true
  }
}

/** The text a search term is matched against, per entry kind. */
export function searchableText(entry: Entry): string {
  switch (entry.kind) {
    case 'user':
    case 'assistant':
    case 'error':
      return entry.content
    case 'thinking':
      return entry.content
    case 'tool':
      return [
        entry.name,
        JSON.stringify(entry.arguments ?? {}),
        entry.output ?? '',
        entry.errorMessage ?? '',
      ].join('\n')
    default:
      return ''
  }
}

export interface TranscriptQuery {
  search: string
  filter: EntryFilter
}

export interface TranscriptSlice {
  /** Entries to render, after filter and search. */
  entries: Entry[]
  /** How many entries matched, before search narrowed them. */
  total: number
  matches: number
  search: string
  filter: EntryFilter
  active: boolean
}

/**
 * Filter and search the transcript.
 *
 * A long restored session can hold thousands of entries, so the
 * search runs on the flat view model rather than the DOM: the result
 * is what gets rendered.
 */
export function queryTranscript(
  entries: Entry[],
  query: TranscriptQuery,
): TranscriptSlice {
  const needle = query.search.trim().toLowerCase()

  const filtered = entries.filter((entry) =>
    matchesFilter(entry, query.filter),
  )

  const searched = needle
    ? filtered.filter((entry) =>
        searchableText(entry).toLowerCase().includes(needle),
      )
    : filtered

  return {
    entries: searched,
    total: entries.length,
    matches: needle ? searched.length : filtered.length,
    search: needle ? query.search : '',
    filter: query.filter,
    active: needle.length > 0 || query.filter !== 'all',
  }
}

/** Tools that failed at least twice, worst first. */
export function repeatedFailures(
  failures: Record<string, number>,
  threshold = 2,
): { tool: string; count: number }[] {
  return Object.entries(failures)
    .filter(([, count]) => count >= threshold)
    .map(([tool, count]) => ({ tool, count }))
    .sort((left, right) => right.count - left.count)
}

/** A 0-1 ratio plus a severity, for the context meter. */
export function budgetSeverity(
  used: number | null,
  limit: number | null,
): { ratio: number; tone: 'ok' | 'warn' | 'over' } {
  if (used === null || limit === null || limit <= 0) {
    return { ratio: 0, tone: 'ok' }
  }

  const ratio = used / limit

  if (ratio > 1) {
    return { ratio, tone: 'over' }
  }

  if (ratio >= 0.8) {
    return { ratio, tone: 'warn' }
  }

  return { ratio, tone: 'ok' }
}

import type { WireEnvelope } from '../api/types'

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

const MAX_OUTPUT_CHARS = 8000
const COLLAPSED_OUTPUT_CHARS = 500

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
      streamingRunId: null,
      thinkingRunId: null,
      runs: {},
      blocks: {},
      sessionState: 'idle',
      running: false,
      estimatedTokens: null,
      lastError: '',
      model: '',
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

function nextId(prefix: string): string {
  counter += 1

  return `${prefix}-${counter}`
}

export function truncateOutput(output: string | null): string | null {
  if (!output) {
    return null
  }

  if (output.length <= MAX_OUTPUT_CHARS) {
    return output
  }

  return `${output.slice(0, COLLAPSED_OUTPUT_CHARS)}\n… (${output.length} chars total, expand to fetch full output)`
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
      }

      const isSubagent = Boolean(data.parent_run_id)

      const entries = isSubagent
        ? state.transcript.entries
        : [
            ...state.transcript.entries,
            { kind: 'user', id: nextId('user'), content: data.prompt } as UserEntry,
          ]

      const runId = data.run_id || 'main'

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

    case 'llm.requested': {
      const data = event.data as unknown as {
        iteration: number
        estimated_tokens: number | null
      }

      return {
        ...state,
        transcript: { ...state.transcript, latestSeq: event.seq },
        run: withBlocks(
          {
            ...state.run,
            iteration: data.iteration,
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
          expanded: false,
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
        expanded: false,
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
        output: string | null
        error_code: string | null
        error_message: string | null
        duration_seconds: number | null
      }

      const entries = state.transcript.entries.map((entry) => {
        if (entry.kind !== 'tool') {
          return entry
        }

        const tool = entry as ToolEntry

        if (tool.callId !== data.tool_call_id) {
          return entry
        }

        return {
          ...tool,
          status: data.error_code ? ('error' as const) : ('success' as const),
          output: truncateOutput(data.output),
          errorCode: data.error_code,
          errorMessage: data.error_message,
          durationSeconds: data.duration_seconds,
          expanded: Boolean(data.error_code),
        }
      })

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

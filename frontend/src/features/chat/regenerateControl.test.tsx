import { describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Chat } from './Chat'
import type {
  ConnectionState,
  Entry,
  RunState,
} from '../../stores/transcript'
import type { PendingApprovalView } from '../approval/types'

function assistantEntry(content = 'Paris.'): Entry {
  return {
    kind: 'assistant',
    id: 'a1',
    content,
    streaming: false,
    runId: 'main',
  }
}

function userEntry(content = 'Capital of France?'): Entry {
  return { kind: 'user', id: 'u1', content }
}

function run(over: Partial<RunState> = {}): RunState {
  return {
    running: false,
    sessionState: 'idle',
    phase: 'idle',
    iteration: 0,
    model: 'test',
    runs: {},
    repeatFailures: 0,
    ...over,
  } as RunState
}

function busyRun(): RunState {
  return run({ running: true, sessionState: 'running' })
}

const connection: ConnectionState = {
  status: 'open',
  attempt: 1,
  lastError: null,
}

function renderChat(
  entries: Entry[],
  overrides: Partial<Parameters<typeof Chat>[0]> = {},
) {
  const props: Parameters<typeof Chat>[0] = {
    sessionId: 's1',
    entries,
    approval: null,
    run: run(),
    connection,
    onSend: vi.fn(),
    onToggle: vi.fn(),
    onRegenerate: vi.fn(),
    ...overrides,
  }

  return { props, ...render(<Chat {...props} />) }
}

describe('regenerate control', () => {
  it('is offered when the agent spoke last', async () => {
    const { props } = renderChat([userEntry(), assistantEntry()])

    await userEvent.click(
      screen.getByRole('button', { name: 'Regenerate' }),
    )

    expect(props.onRegenerate).toHaveBeenCalledWith('')
  })

  it('sends the typed text as the hint', async () => {
    const { props } = renderChat([userEntry(), assistantEntry()])

    await userEvent.type(screen.getByRole('textbox'), 'be shorter')
    await userEvent.click(
      screen.getByRole('button', { name: 'Regenerate' }),
    )

    expect(props.onRegenerate).toHaveBeenCalledWith('be shorter')
  })

  it('clears the box so the hint is not sent twice', async () => {
    renderChat([userEntry(), assistantEntry()])

    const box = screen.getByRole('textbox')
    await userEvent.type(box, 'be shorter')
    await userEvent.click(
      screen.getByRole('button', { name: 'Regenerate' }),
    )

    await waitFor(() => expect(box).toHaveValue(''))
  })

  it('is absent when the user spoke last', () => {
    renderChat([assistantEntry(), userEntry('Actually, two words')])

    expect(
      screen.queryByRole('button', { name: 'Regenerate' }),
    ).toBeNull()
  })

  it('is absent on an empty conversation', () => {
    renderChat([])

    expect(
      screen.queryByRole('button', { name: 'Regenerate' }),
    ).toBeNull()
  })

  // Regenerating mid-run would drop tool results still in flight.
  it('is absent while the run is in flight', () => {
    renderChat([userEntry(), assistantEntry()], { run: busyRun() })

    expect(
      screen.queryByRole('button', { name: 'Regenerate' }),
    ).toBeNull()
  })

  it('is absent while an approval is pending', () => {
    renderChat([userEntry(), assistantEntry()], {
      approval: {
        approvalId: 'ap1',
        toolName: 'delete_file',
        permission: 'filesystem.write',
        reason: 'writes outside the workspace',
        arguments: {},
      } as PendingApprovalView,
    })

    expect(
      screen.queryByRole('button', { name: 'Regenerate' }),
    ).toBeNull()
  })
})

import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { CommandPalette } from './CommandPalette'
import { StatusChip, toneFor } from './StatusChip'
import { Field } from './Field'

describe('StatusChip vocabulary', () => {
  it('maps every status vocabulary onto one tone set', () => {
    // Tool, connection, session and plan states used to carry four
    // different vocabularies.
    expect(toneFor('running')).toBe('progress')
    expect(toneFor('success')).toBe('ok')
    expect(toneFor('error')).toBe('bad')

    expect(toneFor('open')).toBe('ok')
    expect(toneFor('connecting')).toBe('progress')
    expect(toneFor('closed')).toBe('warn')

    expect(toneFor('idle')).toBe('neutral')
    expect(toneFor('completed')).toBe('ok')
    expect(toneFor('blocked')).toBe('warn')
    expect(toneFor('cancelled')).toBe('warn')

    expect(toneFor('pending')).toBe('neutral')
    expect(toneFor('active')).toBe('progress')
  })

  it('is neutral for unknown and missing states', () => {
    expect(toneFor('something_new')).toBe('neutral')
    expect(toneFor(null)).toBe('neutral')
    expect(toneFor(undefined)).toBe('neutral')
  })

  it('renders a readable label', () => {
    render(<StatusChip status="waiting_approval" />)

    expect(screen.getByText('waiting approval')).toBeInTheDocument()
    expect(screen.getByText('waiting approval').className).toContain(
      'chip-warn',
    )
  })
})

describe('Field', () => {
  it('shows a hint under the control', () => {
    render(
      <Field label="Max messages" hint="counts objects, not turns">
        <input aria-label="max" />
      </Field>,
    )

    expect(
      screen.getByText('counts objects, not turns'),
    ).toBeInTheDocument()
  })

  it('prefers a warning over the hint', () => {
    render(
      <Field
        label="Divisor"
        hint="characters per token"
        warning="Above 4 undercounts code"
      >
        <input aria-label="d" />
      </Field>,
    )

    expect(
      screen.getByText('Above 4 undercounts code'),
    ).toBeInTheDocument()

    expect(
      screen.queryByText('characters per token'),
    ).not.toBeInTheDocument()

    expect(
      screen.getByText('Divisor').closest('.field'),
    ).toHaveClass('has-warning')
  })
})

describe('CommandPalette', () => {
  const commands = [
    { id: 'a', label: 'Search transcript', hint: '⌘F', run: vi.fn() },
    { id: 'b', label: 'Show only errors', run: vi.fn() },
    { id: 'c', label: 'Stop the current run', run: vi.fn() },
  ]

  it('renders nothing when closed', () => {
    render(
      <CommandPalette open={false} commands={commands} onClose={vi.fn()} />,
    )

    expect(
      screen.queryByRole('dialog'),
    ).not.toBeInTheDocument()
  })

  it('lists commands and runs the chosen one', async () => {
    const user = userEvent.setup()
    const run = vi.fn()
    const onClose = vi.fn()

    render(
      <CommandPalette
        open
        commands={[{ id: 'a', label: 'Do it', run }]}
        onClose={onClose}
      />,
    )

    await user.click(screen.getByRole('button', { name: /Do it/ }))

    expect(run).toHaveBeenCalledTimes(1)
    expect(onClose).toHaveBeenCalled()
  })

  it('filters as you type', async () => {
    const user = userEvent.setup()

    render(
      <CommandPalette
        open
        commands={commands}
        onClose={vi.fn()}
      />,
    )

    await user.type(screen.getByLabelText('Command'), 'stop')

    expect(
      screen.getByRole('button', { name: /Stop the current run/ }),
    ).toBeInTheDocument()

    expect(
      screen.queryByRole('button', { name: /Show only errors/ }),
    ).not.toBeInTheDocument()
  })

  it('says so when nothing matches', async () => {
    const user = userEvent.setup()

    render(
      <CommandPalette open commands={commands} onClose={vi.fn()} />,
    )

    await user.type(screen.getByLabelText('Command'), 'zzzz')

    expect(screen.getByText('No matching command')).toBeInTheDocument()
  })

  it('runs on Enter', async () => {
    const user = userEvent.setup()
    const run = vi.fn()

    render(
      <CommandPalette
        open
        commands={[{ id: 'a', label: 'Do it', run }]}
        onClose={vi.fn()}
      />,
    )

    await user.type(screen.getByLabelText('Command'), '{Enter}')

    expect(run).toHaveBeenCalledTimes(1)
  })

  it('moves the selection with the arrows', async () => {
    const user = userEvent.setup()
    const second = vi.fn()

    render(
      <CommandPalette
        open
        commands={[
          { id: 'a', label: 'First', run: vi.fn() },
          { id: 'b', label: 'Second', run: second },
        ]}
        onClose={vi.fn()}
      />,
    )

    await user.type(screen.getByLabelText('Command'), '{ArrowDown}{Enter}')

    expect(second).toHaveBeenCalledTimes(1)
  })
})
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ContextMeter } from './ContextMeter'
import { RepeatFailures } from './RepeatFailures'
import { RunStatus } from './RunStatus'
import { TranscriptSearch } from './TranscriptSearch'
import type { EntryFilter } from '../../stores/transcript'

// ======================================================================
// Context meter
// ======================================================================


describe('ContextMeter', () => {
  it('shows used and limit', () => {
    // 14k of 28k is half, comfortably inside the budget.
    render(<ContextMeter used={14000} limit={28000} trimmed={false} />)

    expect(screen.getByText('14k / 28k')).toBeInTheDocument()
    expect(screen.getByText('Within budget')).toBeInTheDocument()
  })

  it('warns near the limit', () => {
    const { container } = render(
      <ContextMeter used={24000} limit={28000} trimmed={false} />,
    )

    // 24k/28k is above the 80% warning line.
    expect(
      container.querySelector('.budget')?.className,
    ).toContain('is-warn')

    expect(screen.getByText('Close to the limit')).toBeInTheDocument()
  })

  it('reports going over', () => {
    const { container } = render(
      <ContextMeter used={30000} limit={28000} trimmed={false} />,
    )

    expect(
      container.querySelector('.budget')?.className,
    ).toContain('is-over')

    expect(screen.getByText('Over budget')).toBeInTheDocument()
  })

  it('explains why history looks shorter', () => {
    render(<ContextMeter used={28000} limit={28000} trimmed />)

    expect(
      screen.getByText('Trimmed to fit the budget'),
    ).toBeInTheDocument()
  })

  it('is honest without numbers', () => {
    render(<ContextMeter used={null} limit={null} trimmed={false} />)

    expect(screen.getByText('—')).toBeInTheDocument()
  })

  it('exposes the ratio to assistive tech', () => {
    render(<ContextMeter used={7000} limit={28000} trimmed={false} />)

    expect(screen.getByRole('meter')).toHaveAttribute(
      'aria-valuenow',
      '25',
    )
  })
})

// ======================================================================
// Repeat failures
// ======================================================================


describe('RepeatFailures', () => {
  it('stays quiet for a single failure', () => {
    const { container } = render(
      <RepeatFailures failures={{ get_tags: 1 }} />,
    )

    expect(container.querySelector('.repeat-banner')).toBeNull()
  })

  it('names the tool and the count', () => {
    render(<RepeatFailures failures={{ get_tags: 4 }} />)

    expect(
      screen.getByText('A tool keeps failing'),
    ).toBeInTheDocument()

    expect(screen.getByText('get_tags')).toBeInTheDocument()
    expect(screen.getByText('failed 4×')).toBeInTheDocument()
  })

  it('counts several offenders', () => {
    render(
      <RepeatFailures failures={{ get_tags: 3, read_file: 2 }} />,
    )

    expect(
      screen.getByText('2 tools keep failing'),
    ).toBeInTheDocument()
  })

  it('tells the reader what to do', () => {
    render(<RepeatFailures failures={{ get_tags: 2 }} />)

    expect(
      screen.getByText(/agent stopped retrying/i),
    ).toBeInTheDocument()
  })

  it('renders nothing with no failures at all', () => {
    const { container } = render(<RepeatFailures failures={{}} />)

    expect(container.querySelector('.repeat-banner')).toBeNull()
  })
})

// ======================================================================
// Run status
// ======================================================================


describe('RunStatus', () => {
  it('leads with a readable phase, not raw state names', () => {
    render(
      <RunStatus
        phase="executing"
        iteration={3}
        state="running"
        model="gemma4:12b"
        link="open"
        blocked={false}
        busy
        onResume={vi.fn()}
        onCancel={vi.fn()}
      />,
    )

    expect(screen.getByText('Running tools · step 3')).toBeInTheDocument()
  })

  it('hides the internals until asked', async () => {
    const user = userEvent.setup()

    render(
      <RunStatus
        phase="executing"
        iteration={3}
        state="running"
        model="gemma4:12b"
        link="open"
        blocked={false}
        busy
        onResume={vi.fn()}
        onCancel={vi.fn()}
      />,
    )

    expect(screen.queryByText('gemma4:12b')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Details' }))

    expect(screen.getByText('gemma4:12b')).toBeInTheDocument()
    expect(screen.getByText('iteration')).toBeInTheDocument()
  })

  it('offers resume only when blocked', () => {
    const onResume = vi.fn()

    render(
      <RunStatus
        phase={null}
        iteration={0}
        state="blocked"
        model=""
        link="open"
        blocked
        busy={false}
        onResume={onResume}
        onCancel={vi.fn()}
      />,
    )

    screen.getByRole('button', { name: 'Resume' })
    expect(
      screen.queryByRole('button', { name: 'Stop' }),
    ).not.toBeInTheDocument()
  })

  it('offers stop only while running', () => {
    render(
      <RunStatus
        phase="planning"
        iteration={1}
        state="running"
        model=""
        link="open"
        blocked={false}
        busy
        onResume={vi.fn()}
        onCancel={vi.fn()}
      />,
    )

    screen.getByRole('button', { name: 'Stop' })
  })
})

// ======================================================================
// Transcript search
// ======================================================================


describe('TranscriptSearch', () => {
  function setup(
    overrides: Partial<
      React.ComponentProps<typeof TranscriptSearch>
    > = {},
  ) {
    const onSearch = vi.fn()
    const onFilter = vi.fn()

    render(
      <TranscriptSearch
        matches={3}
        total={10}
        active
        search="tags"
        filter="all"
        onSearch={onSearch}
        onFilter={onFilter}
        {...overrides}
      />,
    )

    return { onSearch, onFilter }
  }

  it('reports the match count', () => {
    setup()

    expect(screen.getByText('3/10')).toBeInTheDocument()
  })

  it('hides the count when idle', () => {
    setup({ active: false, matches: 10, total: 10, search: '' })

    expect(screen.queryByText('10/10')).not.toBeInTheDocument()
  })

  it('offers the filter kinds', () => {
    setup()

    for (const label of ['All', 'Thinking', 'Tools', 'Errors']) {
      screen.getByRole('button', { name: label })
    }
  })

  it('marks the active filter', () => {
    setup({ filter: 'errors' })

    expect(
      screen.getByRole('button', { name: 'Errors' }),
    ).toHaveAttribute('aria-pressed', 'true')

    expect(
      screen.getByRole('button', { name: 'All' }),
    ).toHaveAttribute('aria-pressed', 'false')
  })

  it('clears both query and filter', async () => {
    const user = userEvent.setup()
    const { onSearch, onFilter } = setup({ filter: 'tools' })

    await user.click(screen.getByRole('button', { name: 'Clear' }))

    expect(onSearch).toHaveBeenCalledWith('')
    expect(onFilter).toHaveBeenCalledWith('all' as EntryFilter)
  })

  it('does not offer clear when nothing is applied', () => {
    setup({ active: false })

    expect(
      screen.queryByRole('button', { name: 'Clear' }),
    ).not.toBeInTheDocument()
  })
})

describe('ContextMeter fidelity', () => {
  it('says the number is measured when a real tokenizer was used', () => {
    render(
      <ContextMeter used={1000} limit={8000} trimmed={false} exact />,
    )

    expect(screen.queryByText(/estimated, not measured/i)).toBeNull()
    expect(screen.getByText('Within budget')).toBeTruthy()
  })

  it('admits when the count is a heuristic', () => {
    // The fallback divides by character length, which undercounts
    // tool output and code by around a third. Showing that number
    // with the same confidence as a measurement is what turns a
    // silent underestimate into a confusing rejection.
    render(
      <ContextMeter
        used={1000}
        limit={8000}
        trimmed={false}
        exact={false}
      />,
    )

    expect(screen.getByText('Estimated, not measured')).toBeTruthy()
  })

  it('names which block was dropped, not just that something was', () => {
    // "Trimmed to fit" leaves the reader unable to tell memory being
    // dropped from history being cut, and the remedy differs.
    render(
      <ContextMeter
        used={7000}
        limit={8000}
        trimmed={true}
        exact={true}
        dropped={['memory']}
      />,
    )

    expect(screen.getByText('Dropped: memory')).toBeTruthy()
    expect(screen.queryByText('Trimmed to fit the budget')).toBeNull()
  })

  it('shows the drift when the provider disagrees with the estimate', () => {
    render(
      <ContextMeter
        used={5000}
        limit={8000}
        trimmed={false}
        exact={true}
        measured={9000}
        estimate={5000}
      />,
    )

    expect(screen.getByText(/Provider counted 9k/)).toBeTruthy()
  })

  it('says nothing about drift when the two agree', () => {
    render(
      <ContextMeter
        used={5000}
        limit={8000}
        trimmed={false}
        exact={true}
        measured={5200}
        estimate={5000}
      />,
    )

    expect(screen.queryByText(/Provider counted/)).toBeNull()
  })
})

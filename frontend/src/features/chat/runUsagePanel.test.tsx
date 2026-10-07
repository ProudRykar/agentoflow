import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { RunUsagePanel } from './RunUsage'
import { emptyRunUsage, type RunUsageView } from './runUsage'

function usage(over: Partial<RunUsageView> = {}): RunUsageView {
  return { ...emptyRunUsage(), ...over }
}

describe('run usage panel', () => {
  it('shows nothing before the run has made a call', () => {
    const { container } = render(<RunUsagePanel usage={usage()} />)

    expect(container.querySelector('.run-usage')).toBeNull()
  })

  it('renders nothing at all rather than crashing on absent state', () => {
    const { container } = render(<RunUsagePanel />)

    expect(container.querySelector('.run-usage')).toBeNull()
  })

  it('shows tokens and cost when measured', () => {
    render(
      <RunUsagePanel
        usage={usage({
          promptTokens: 3400,
          completionTokens: 250,
          calls: 3,
          estimatedCost: 0.0123,
        })}
      />,
    )

    // One joined string, so matched as a whole.
    expect(
      screen.getByText(/3 calls .* 3\.6k tok .* \$0\.0123/),
    ).toBeTruthy()
  })

  // Zero tokens and no measurement are different facts, and the second
  // is the one that decides whether to trust the first.
  it('says the run was not measured instead of showing zero', () => {
    render(
      <RunUsagePanel
        usage={usage({ calls: 2, callsWithoutUsage: 2 })}
      />,
    )

    expect(screen.getByText(/not measured/)).toBeTruthy()
    expect(screen.queryByText(/0 tok/)).toBeNull()
  })

  it('notes partial reporting without hiding the measured total', () => {
    render(
      <RunUsagePanel
        usage={usage({
          promptTokens: 1200,
          calls: 4,
          callsWithoutUsage: 2,
        })}
      />,
    )

    expect(screen.getByText(/2 calls unreported/)).toBeTruthy()
  })

  it('shows progress toward a configured ceiling', () => {
    render(
      <RunUsagePanel
        usage={usage({ promptTokens: 5000, calls: 2 })}
        ceiling={10000}
      />,
    )

    expect(screen.getByText(/of 10\.0k tok/)).toBeTruthy()
  })

  it('does not round a small total away', () => {
    render(
      <RunUsagePanel
        usage={usage({
          promptTokens: 4000,
          calls: 2,
          estimatedCost: 0.0123,
        })}
      />,
    )

    // $0.01 would understate it by a fifth while still looking exact.
    expect(screen.getByText(/\$0\.0123/)).toBeTruthy()
  })

  it('omits the ceiling when none is configured', () => {
    render(
      <RunUsagePanel usage={usage({ promptTokens: 5000, calls: 2 })} />,
    )

    expect(screen.queryByText(/of /)).toBeNull()
  })

  it('does not claim progress toward a ceiling it cannot measure', () => {
    render(
      <RunUsagePanel
        usage={usage({ calls: 2, callsWithoutUsage: 2 })}
        ceiling={10000}
      />,
    )

    expect(screen.getByText(/not measured/)).toBeTruthy()
  })
})

import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { useShortcuts } from './useShortcuts'

function Probe(props: Parameters<typeof useShortcuts>[0]) {
  useShortcuts(props)

  return (
    <div>
      <span data-testid="palette">
        {String(props.paletteOpen)}
      </span>
    </div>
  )
}

function setup(
  overrides: Partial<Parameters<typeof useShortcuts>[0]> = {},
) {
  const handlers = {
    onPalette: vi.fn(),
    onSearch: vi.fn(),
    onStop: vi.fn(),
    onClosePalette: vi.fn(),
    paletteOpen: false,
    stopEnabled: false,
    ...overrides,
  }

  render(<Probe {...handlers} />)

  return handlers
}

describe('keyboard shortcuts', () => {
  it('opens the palette with the meta key', async () => {
    const user = userEvent.setup()
    const { onPalette } = setup()

    await user.keyboard('{Meta>}k{/Meta}')

    expect(onPalette).toHaveBeenCalledTimes(1)
  })

  it('accepts ctrl as well', async () => {
    const user = userEvent.setup()
    const { onPalette } = setup()

    await user.keyboard('{Control>}k{/Control}')

    expect(onPalette).toHaveBeenCalledTimes(1)
  })

  it('focuses search on meta-f', async () => {
    const user = userEvent.setup()
    const { onSearch } = setup()

    await user.keyboard('{Meta>}f{/Control}')

    expect(onSearch).toHaveBeenCalledTimes(1)
  })

  it('focuses search on slash outside a field', async () => {
    const user = userEvent.setup()
    const { onSearch } = setup()

    await user.keyboard('/')

    expect(onSearch).toHaveBeenCalledTimes(1)
  })

  it('ignores slash while typing', async () => {
    const user = userEvent.setup()
    const { onSearch } = setup()

    render(<input aria-label="draft" />)

    await user.click(screen.getByLabelText('draft'))
    await user.keyboard('/')

    expect(onSearch).not.toHaveBeenCalled()
  })

  it('stops a run on escape', async () => {
    const user = userEvent.setup()
    const { onStop } = setup({ stopEnabled: true })

    await user.keyboard('{Escape}')

    expect(onStop).toHaveBeenCalledTimes(1)
  })

  it('does nothing on escape with no run to stop', async () => {
    const user = userEvent.setup()
    const { onStop } = setup({ stopEnabled: false })

    await user.keyboard('{Escape}')

    expect(onStop).not.toHaveBeenCalled()
  })

  it('closes the palette before stopping the run', async () => {
    const user = userEvent.setup()
    const { onStop, onClosePalette } = setup({
      paletteOpen: true,
      stopEnabled: true,
    })

    await user.keyboard('{Escape}')

    expect(onClosePalette).toHaveBeenCalledTimes(1)
    expect(onStop).not.toHaveBeenCalled()
  })

  it('detaches its listener on unmount', async () => {
    const user = userEvent.setup()
    const onPalette = vi.fn()

    const { unmount } = render(
      <Probe
        onPalette={onPalette}
        onSearch={vi.fn()}
        onStop={vi.fn()}
        onClosePalette={vi.fn()}
        paletteOpen={false}
        stopEnabled={false}
      />,
    )

    unmount()

    // A leaked listener would keep firing after the component is
    // gone, so every shortcut would fire twice.
    await user.keyboard('{Meta>}k{/Meta}')

    expect(onPalette).not.toHaveBeenCalled()
  })
})
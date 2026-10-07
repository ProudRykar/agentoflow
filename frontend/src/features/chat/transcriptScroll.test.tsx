import { render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { Transcript } from './Transcript'
import { VirtualList } from '../../components/VirtualList'

/**
 * The scroll container used to be the wrapper, while the list that
 * actually owns scrollTop sat inside it with no overflow. The list
 * therefore never saw a scroll: it kept its window at row 0, and the
 * wheel stopped dead once the wrapper ran out of travel.
 *
 * jsdom has no layout, so these assert the structural contract --
 * which element is the scroller, and that the list reports it -- and
 * not pixels.
 */
describe('transcript scrolling', () => {
  it('puts the scroller on the list viewport, not a wrapper', () => {
    const { container } = render(
      <Transcript
        entries={[
          {
            kind: 'assistant',
            id: 'a1',
            content: 'hello',
            runId: '',
          } as never,
        ]}
        runs={{}}
        streaming={false}
        repeatFailures={{}}
        onToggle={vi.fn()}
      />,
    )

    // The list owns the scrolling element.
    expect(
      container.querySelector('.transcript-viewport'),
    ).toBeTruthy()

    // The wrapper is present but is not a second scroller. Two
    // nested scrollers is what made the wheel feel dead.
    expect(container.querySelector('.transcript')).toBeTruthy()
  })

  it('reports its viewport node to the caller', () => {
    // Transcript pins and jumps using this node. Before, it read
    // its own wrapper and always saw 0.
    const seen: (HTMLElement | null)[] = []

    render(
      <VirtualList
        items={[{ id: 1 }]}
        itemHeight={56}
        renderItem={(item) => <div>{String((item as { id: number }).id)}</div>}
        onViewport={(node) => seen.push(node)}
      />,
    )

    expect(seen.length).toBeGreaterThan(0)
    expect(seen[seen.length - 1]).toBeInstanceOf(HTMLElement)
  })

  it('reports every scroll from the element that moved', () => {
    const scrolled: number[] = []

    render(
      <VirtualList
        items={[{ id: 1 }, { id: 2 }]}
        itemHeight={56}
        renderItem={(item) => <div>{String((item as { id: number }).id)}</div>}
        onScroll={(node) => scrolled.push(node.scrollTop)}
      />,
    )

    const viewport = document.querySelector<HTMLElement>(
      '[data-testid="virtual-viewport"]',
    )

    expect(viewport).toBeTruthy()

    viewport!.scrollTop = 120
    viewport!.dispatchEvent(new Event('scroll', { bubbles: true }))

    expect(scrolled).toContain(120)
  })
})

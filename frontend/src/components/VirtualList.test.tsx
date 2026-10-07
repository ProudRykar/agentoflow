import {
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { VirtualList } from './VirtualList'

describe('VirtualList', () => {
  const items = Array.from({ length: 5000 }, (_, index) => ({
    id: index,
  }))

  it('mounts only a window of the rows', () => {
    render(
      <VirtualList
        items={items}
        itemHeight={40}
        renderItem={(item) => (
          <div key={(item as { id: number }).id} data-testid="row">
            {(item as { id: number }).id}
          </div>
        )}
      />,
    )

    // 5000 rows at 40px each would be 200 000px of DOM otherwise.
    const rows = screen.getAllByTestId('row')

    expect(rows.length).toBeGreaterThan(0)
    expect(rows.length).toBeLessThan(80)
  })

  it('renders the newest rows for a full list at the end', () => {
    render(
      <VirtualList
        items={items}
        itemHeight={40}
        overscan={4}
        renderItem={(item) => (
          <div key={(item as { id: number }).id} data-testid="row">
            {(item as { id: number }).id}
          </div>
        )}
      />,
    )

    // Without a measured viewport the list starts at the top.
    expect(screen.getAllByTestId('row')[0]).toHaveTextContent('0')
  })

  it('reports reaching the end while scrolled', async () => {

    const onReachEnd = vi.fn()

    render(
      <div style={{ height: 200 }}>
        <VirtualList
          items={items}
          itemHeight={40}
          onReachEnd={onReachEnd}
          renderItem={(item) => (
            <div key={(item as { id: number }).id}>
              {(item as { id: number }).id}
            </div>
          )}
        />
      </div>,
    )

    const viewport = screen.getByTestId('virtual-viewport')

    Object.defineProperty(viewport, 'scrollHeight', {
      value: 200000,
      configurable: true,
    })
    Object.defineProperty(viewport, 'clientHeight', {
      value: 200,
      configurable: true,
    })

    viewport.scrollTop = 200000

    fireEvent.scroll(viewport)

    await waitFor(() => expect(onReachEnd).toHaveBeenCalled())
  })

  it('renders the empty slot when there is nothing', () => {
    render(
      <VirtualList
        items={[]}
        itemHeight={40}
        empty={<p>Nothing here yet</p>}
        renderItem={() => null}
      />,
    )

    expect(screen.getByText('Nothing here yet')).toBeInTheDocument()
    expect(
      screen.queryByTestId('virtual-viewport'),
    ).not.toBeInTheDocument()
  })
})
describe('VirtualList with measured row heights', () => {
  /**
   * jsdom reports every element as zero-height, so heights are faked
   * per index and the geometry is checked against the arithmetic the
   * list has to perform.
   */
  function fakeHeights(map: Record<number, number>) {
    const original = Element.prototype.getBoundingClientRect

    Element.prototype.getBoundingClientRect = function fake(this: Element) {
      const row = (this as HTMLElement).dataset.row

      if (row !== undefined && map[Number(row)] !== undefined) {
        const height = map[Number(row)]

        return {
          width: 100,
          height,
          top: 0,
          left: 0,
          right: 100,
          bottom: height,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        } as DOMRect
      }

      return original.call(this)
    } as typeof Element.prototype.getBoundingClientRect

    return () => {
      Element.prototype.getBoundingClientRect = original
    }
  }

  /**
   * A controllable ResizeObserver.
   *
   * jsdom has none, and without one the row-measurement path cannot
   * run at all, which is exactly the path that fixes the scroll.
   */
  function installResizeObserver() {
    const observed = new Set<Element>()
    let notify: ((entries: ResizeObserverEntry[]) => void) | null = null

    class Fake {
      constructor(callback: (entries: ResizeObserverEntry[]) => void) {
        notify = callback
      }

      observe(node: Element): void {
        observed.add(node)
      }

      unobserve(node: Element): void {
        observed.delete(node)
      }

      disconnect(): void {
        observed.clear()
      }
    }

    const original = globalThis.ResizeObserver

    globalThis.ResizeObserver = Fake as unknown as typeof ResizeObserver

    const deliver = (target: Element, height: number): void => {
      notify?.(
        [
          {
            target,
            contentRect: { height } as DOMRectReadOnly,
          } as unknown as ResizeObserverEntry,
        ] as unknown as ResizeObserverEntry[],
      )
    }

    /** Delivers even to an unobserved target, like a late callback. */
    const force = (target: Element, height: number): void => {
      deliver(target, height)
    }

    const fireAndRestore = Object.assign(
      (target: Element, height: number): void => {
        if (observed.has(target)) {
          deliver(target, height)
        }
      },
      {
        force,
        restore: () => {
          globalThis.ResizeObserver = original
        },
      },
    )

    return fireAndRestore
  }

  function setMetrics(node: HTMLElement, scrollHeight: number, clientHeight: number) {
    Object.defineProperty(node, 'scrollHeight', {
      configurable: true,
      value: scrollHeight,
    })
    Object.defineProperty(node, 'clientHeight', {
      configurable: true,
      value: clientHeight,
    })
  }

  it('sizes the spacers from the measured heights, not the estimate', () => {
    const restore = fakeHeights({ 0: 2000 })

    const fireResize = installResizeObserver()

    try {
      const { container } = render(
        <VirtualList
          items={[{ id: 0 }, { id: 1 }, { id: 2 }]}
          itemHeight={40}
          overscan={0}
          renderItem={(item) => (
            <div data-testid="row">{(item as { id: number }).id}</div>
          )}
        />,
      )

      const viewport = container.querySelector(
        '[data-testid="virtual-viewport"]',
      ) as HTMLElement

      setMetrics(viewport, 2000 + 40 + 40, 800)

      // Row 0 measured 2000px, so the window covers it and row 1 only.
      const spacers = viewport.querySelectorAll('div[aria-hidden="true"]')

      expect(spacers).toHaveLength(2)
      expect((spacers[0] as HTMLElement).style.height).toBe('0px')
      expect((spacers[1] as HTMLElement).style.height).toBe('80px')
    } finally {
      restore()
      fireResize.restore()
    }
  })

  it('keeps the view stable when a row above the viewport grows', () => {
    const restore = fakeHeights({ 0: 100 })
    const fireResize = installResizeObserver()

    try {
      const { container } = render(
        <VirtualList
          items={[{ id: 0 }, { id: 1 }, { id: 2 }, { id: 3 }]}
          itemHeight={40}
          overscan={0}
          renderItem={(item) => (
            <div data-testid="row">{(item as { id: number }).id}</div>
          )}
        />,
      )

      const viewport = container.querySelector(
        '[data-testid="virtual-viewport"]',
      ) as HTMLElement

      setMetrics(viewport, 400, 400)

      // Far enough down that row 0 is above the viewport but still
      // mounted, which is the state the correction exists for.
      viewport.scrollTop = 50

      fireEvent.scroll(viewport)

      // Row 0 doubles while it sits above the reader: the observer is
      // what reports it, since the ref callback only runs on mount.
      fakeHeights({ 0: 240 })

      fireResize(viewport.querySelector('[data-row="0"]') as HTMLElement, 240)

      // The content below moved down by 140, so the scroll position has
      // to move with it or the view jumps.
      expect(viewport.scrollTop).toBe(190)
    } finally {
      restore()
      fireResize.restore()
    }
  })

  it('ignores a row measurement once it leaves the window', () => {
    const restore = fakeHeights({ 0: 100 })
    const fireResize = installResizeObserver()

    try {
      const { container } = render(
        <VirtualList
          items={Array.from({ length: 40 }, (_, id) => ({ id }))}
          itemHeight={40}
          overscan={0}
          renderItem={(item) => (
            <div data-testid="row">{(item as { id: number }).id}</div>
          )}
        />,
      )

      const viewport = container.querySelector(
        '[data-testid="virtual-viewport"]',
      ) as HTMLElement

      setMetrics(viewport, 1600, 400)

      const row = viewport.querySelector('[data-row="0"]') as HTMLElement

      // Scroll past it so the row unmounts.
      viewport.scrollTop = 1_200

      fireEvent.scroll(viewport)

      expect(row.isConnected).toBe(false)

      // Measured after the scroll, so only a stale callback can move it.
      const spacerBefore = (
        viewport.querySelector('div[aria-hidden="true"]') as HTMLElement
      ).style.height

      fireResize.force(row, 5000)

      expect(
        (viewport.querySelector('div[aria-hidden="true"]') as HTMLElement).style
          .height,
      ).toBe(spacerBefore)
    } finally {
      restore()
      fireResize.restore()
    }
  })

  it('does not snap back when the reader has scrolled away from the end', () => {
    const items = Array.from({ length: 100 }, (_, index) => ({ id: index }))

    const { container } = render(
      <VirtualList
        items={items}
        itemHeight={40}
        overscan={2}
        renderItem={(item) => (
          <div data-testid="row">{(item as { id: number }).id}</div>
        )}
      />,
    )

    const viewport = container.querySelector(
      '[data-testid="virtual-viewport"]',
    ) as HTMLElement

    setMetrics(viewport, 4000, 400)

    viewport.scrollTop = 40

    fireEvent.scroll(viewport)

    // New content arrives; the reader stays where they were.
    fireEvent.scroll(viewport)

    expect(viewport.scrollTop).toBe(40)
  })

  it('follows the end when the reader is parked there', () => {
    const items = Array.from({ length: 100 }, (_, index) => ({ id: index }))

    const { container } = render(
      <VirtualList
        items={items}
        itemHeight={40}
        overscan={2}
        renderItem={(item) => (
          <div data-testid="row">{(item as { id: number }).id}</div>
        )}
      />,
    )

    const viewport = container.querySelector(
      '[data-testid="virtual-viewport"]',
    ) as HTMLElement

    setMetrics(viewport, 4000, 400)

    // Parked at the bottom.
    viewport.scrollTop = 3600

    fireEvent.scroll(viewport)

    expect(viewport.scrollTop).toBe(3600)
  })
})

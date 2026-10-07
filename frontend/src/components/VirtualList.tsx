import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'

interface VirtualListProps {
  items: unknown[]
  /**
   * Height assumed for a row before it has been measured, and the
   * fallback when measurement is unavailable.
   */
  itemHeight: number
  /** Rows rendered above and below the window, to hide seams. */
  overscan?: number
  /** Called when the visible range reaches the last row. */
  onReachEnd?: () => void
  renderItem: (item: unknown, index: number) => ReactNode
  className?: string
  empty?: ReactNode
  /**
   * Reports the scrolling element.
   *
   * The list owns its own scroll position, so a caller that needs it
   * -- to pin to the newest row, or to jump back -- has to be told
   * which node it is rather than wrapping the list and hoping the
   * wrapper scrolls.
   */
  onViewport?: (node: HTMLDivElement | null) => void
  /** Reports each scroll, with the element that moved. */
  onScroll?: (node: HTMLDivElement) => void
  /**
   * Distance from the bottom, in pixels, within which the view counts
   * as parked at the newest content.
   */
  pinThreshold?: number
}

const DEFAULT_OVERSCAN = 4

/** Sub-pixel measurement noise is not a real height change. */
const MEASURE_EPSILON = 0.5

/**
 * Windowed list for the transcript.
 *
 * Rows here are whole runs, so they are not uniform: the main
 * conversation can be tens of thousands of pixels tall while an
 * abandoned run is one line. An earlier version assumed every row was
 * `itemHeight` tall, so its spacers disagreed with the real content,
 * the scrollbar range was wrong and wheel scrolling jumped around.
 * Heights are measured instead, and the window is located from the
 * measured offsets.
 */
export function VirtualList({
  items,
  itemHeight,
  overscan = DEFAULT_OVERSCAN,
  onReachEnd,
  renderItem,
  className,
  empty,
  onViewport,
  onScroll,
  pinThreshold = 48,
}: VirtualListProps) {
  const viewportRef = useRef<HTMLDivElement | null>(null)
  const observerRef = useRef<ResizeObserver | null>(null)
  const nodesRef = useRef(new Map<number, HTMLElement>())

  /** Canonical row heights, indexed by row. */
  const heightsRef = useRef<number[]>([])
  /** Bumped to re-render after a measurement lands in the ref. */
  const [version, setVersion] = useState(0)
  /** Whether the reader is parked at the newest content. */
  const pinnedRef = useRef(true)

  const [scrollTop, setScrollTop] = useState(0)
  const [height, setHeight] = useState(0)

  const total = items.length

  // Prefix sums: offsets[i] is where row i starts. Rebuilt each render
  // from the ref so no measurement can be read stale.
  const offsets: number[] = new Array(total + 1)
  offsets[0] = 0

  for (let index = 0; index < total; index += 1) {
    offsets[index + 1] =
      offsets[index] + (heightsRef.current[index] ?? itemHeight)
  }

  const contentHeight = offsets[total]

  /**
   * Records one row's height.
   *
   * A row changing height while it sits above the viewport pushes
   * everything below it down, so the scroll position is corrected by
   * the same amount. Without that correction, measuring a row yanks
   * the view, which reads as the wheel fighting back.
   *
   * Everything comes from refs, so this callback is stable and a ref
   * attached to a row is not re-invoked on every render.
   */
  const setRowHeight = useCallback(
    (index: number, next: number): void => {
      // A zero or non-finite height is not a measurement: a hidden
      // container, a display:none ancestor, or a test renderer all
      // report 0. Taking it at face value collapses every offset and
      // the window search then re-mounts rows forever.
      if (!Number.isFinite(next) || next <= 0) {
        return
      }

      const previous = heightsRef.current[index] ?? itemHeight

      if (Math.abs(next - previous) < MEASURE_EPSILON) {
        return
      }

      heightsRef.current[index] = next

      const node = viewportRef.current

      if (node && offsets[index] < node.scrollTop) {
        node.scrollTop += next - previous
      }

      setVersion((value) => value + 1)
    },
    // `offsets` is derived from the ref during render; reading the
    // ref here keeps the identity stable across measurements.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [itemHeight],
  )

  /** Reverse lookup, for an observer entry back to its row index. */
  const indexOfNode = useCallback((node: HTMLElement): number | undefined => {
    for (const [index, candidate] of nodesRef.current) {
      if (candidate === node) {
        return index
      }
    }

    return undefined
  }, [])

  const measureViewport = useCallback(() => {
    const node = viewportRef.current

    if (node) {
      setHeight(node.clientHeight)
    }
  }, [])

  useEffect(() => {
    onViewport?.(viewportRef.current)
  }, [onViewport])

  useEffect(() => {
    measureViewport()

    const node = viewportRef.current

    if (!node || typeof ResizeObserver === 'undefined') {
      return
    }

    /**
     * One observer, two kinds of target.
     *
     * A single callback that only handled the viewport silently
     * discarded every row measurement, so a row that changed height
     * after mount -- an expanding tool call, a re-measured table --
     * was never picked up. The target decides which it is.
     */
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.target === viewportRef.current) {
          measureViewport()
          continue
        }

        const index = indexOfNode(entry.target as HTMLElement)

        if (index === undefined) {
          continue
        }

        setRowHeight(
          index,
          entry.contentRect?.height ??
            (entry.target as HTMLElement).getBoundingClientRect().height,
        )
      }
    })

    observer.observe(node)

    // Rows are already mounted by the time this effect runs, because
    // ref callbacks fire during the commit that precedes it.
    for (const row of nodesRef.current.values()) {
      observer.observe(row)
    }

    observerRef.current = observer

    return () => {
      observer.disconnect()
      observerRef.current = null
    }
  }, [measureViewport, setRowHeight])

  const registerRow = useCallback(
    (index: number, node: HTMLElement | null): void => {
      const previous = nodesRef.current.get(index)

      if (previous) {
        observerRef.current?.unobserve(previous)
      }

      if (node) {
        nodesRef.current.set(index, node)
        observerRef.current?.observe(node)
        setRowHeight(index, node.getBoundingClientRect().height)
      } else {
        nodesRef.current.delete(index)
      }
    },
    [setRowHeight],
  )

  const distanceFromEnd = useCallback((): number => {
    const node = viewportRef.current

    if (!node) {
      return Number.POSITIVE_INFINITY
    }

    return node.scrollHeight - node.scrollTop - node.clientHeight
  }, [])

  const handleScroll = useCallback(
    (event: React.UIEvent<HTMLDivElement>) => {
      setScrollTop(event.currentTarget.scrollTop)

      onScroll?.(event.currentTarget)

      const node = event.currentTarget
      const distance =
        node.scrollHeight - node.scrollTop - node.clientHeight

      // Remember where the reader chose to be, so streaming content
      // does not pull them back down.
      pinnedRef.current = distance < pinThreshold

      if (onReachEnd && distance < pinThreshold) {
        onReachEnd()
      }
    },
    [onReachEnd, onScroll, pinThreshold],
  )

  /**
   * Stay at the bottom while new content arrives.
   *
   * The old version compared a distance expressed in rows, which for
   * rows of wildly different heights meant a reader who had scrolled
   * slightly up was snapped back anyway. It is measured in pixels now,
   * and it only moves when the reader was already parked there.
   */
  useLayoutEffect(() => {
    const node = viewportRef.current

    if (node && pinnedRef.current && distanceFromEnd() < pinThreshold) {
      node.scrollTop = node.scrollHeight

      setScrollTop(node.scrollTop)
    }
    // Re-runs whenever the content grows or a measurement lands.
  }, [total, contentHeight, version, pinThreshold, distanceFromEnd])

  if (total === 0 && empty) {
    return <div className={className}>{empty}</div>
  }

  const start = Math.max(0, findRow(offsets, scrollTop) - overscan)
  const end = Math.min(
    total,
    findRow(offsets, scrollTop + Math.max(height, 1)) + overscan + 1,
  )

  return (
    <div
      ref={viewportRef}
      className={className}
      onScroll={handleScroll}
      data-testid="virtual-viewport"
    >
      <div style={{ height: offsets[start] }} aria-hidden="true" />

      {items.slice(start, end).map((item, offset) => {
        const index = start + offset

        return (
          <div key={index} ref={(node) => registerRow(index, node)} data-row={index}>
            {renderItem(item, index)}
          </div>
        )
      })}

      <div style={{ height: contentHeight - offsets[end] }} aria-hidden="true" />
    </div>
  )
}

/**
 * The row that contains `position`.
 *
 * A binary search, because a long transcript reaches tens of thousands
 * of pixels and this runs on every scroll frame.
 */
function findRow(offsets: number[], position: number): number {
  const last = offsets.length - 2

  if (last <= 0) {
    return 0
  }

  let low = 0
  let high = last

  while (low < high) {
    const middle = Math.floor((low + high + 1) / 2)

    if (offsets[middle] <= position) {
      low = middle
    } else {
      high = middle - 1
    }
  }

  return Math.min(low, last)
}
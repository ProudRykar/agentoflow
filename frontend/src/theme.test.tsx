import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { useTheme } from './theme'

const KEY = 'agentoflow.theme'

/**
 * This jsdom build has no localStorage at all, so the tests install an
 * in-memory stand-in rather than depending on it.
 */
let store: Record<string, string>

function installStorage(): void {
  store = {}

  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: {
      getItem: (name: string): string | null => store[name] ?? null,
      setItem: (name: string, value: string): void => {
        store[name] = value
      },
      removeItem: (name: string): void => {
        delete store[name]
      },
      clear: (): void => {
        store = {}
      },
    },
  })
}

function Probe() {
  const { theme, toggle } = useTheme()

  return (
    <button type="button" onClick={toggle}>
      {theme}
    </button>
  )
}

beforeEach(() => {
  installStorage()
})

afterEach(() => {
  document.documentElement.removeAttribute('data-theme')
})

describe('useTheme', () => {
  it('defaults to dark when nothing is stored', () => {
    render(<Probe />)

    expect(screen.getByRole('button').textContent).toBe('dark')
  })

  it('reads a stored choice', () => {
    window.localStorage.setItem(KEY, 'light')

    render(<Probe />)

    expect(screen.getByRole('button').textContent).toBe('light')
  })

  it('ignores a nonsense stored value', () => {
    window.localStorage.setItem(KEY, 'neon')

    render(<Probe />)

    expect(screen.getByRole('button').textContent).toBe('dark')
  })

  it('toggles and applies the attribute on the html element', async () => {
    const user = userEvent.setup()

    render(<Probe />)

    expect(document.documentElement.dataset.theme).toBe('dark')

    await user.click(screen.getByRole('button'))

    expect(screen.getByRole('button').textContent).toBe('light')
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(window.localStorage.getItem(KEY)).toBe('light')

    await user.click(screen.getByRole('button'))

    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(window.localStorage.getItem(KEY)).toBe('dark')
  })

  it('follows the system when nothing is stored', () => {
    const listeners = new Set<() => void>()

    Object.defineProperty(window, 'matchMedia', {
      configurable: true,
      value: (query: string) => ({
        matches: query.includes('light'),
        addEventListener: (_: string, fn: () => void) => listeners.add(fn),
        removeEventListener: (_: string, fn: () => void) =>
          listeners.delete(fn),
      }),
    })

    try {
      render(<Probe />)

      expect(screen.getByRole('button').textContent).toBe('light')
      expect(listeners.size).toBe(1)
      // Nothing is remembered until the reader actually picks.
      expect(window.localStorage.getItem(KEY)).toBeNull()
    } finally {
      Reflect.deleteProperty(window, 'matchMedia')
    }
  })

  it('survives a storage write failure', async () => {
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      value: {
        getItem: () => null,
        setItem: () => {
          throw new Error('denied')
        },
        removeItem: () => undefined,
        clear: () => undefined,
      },
    })

    const user = userEvent.setup()

    render(<Probe />)

    // The theme still applies even though it cannot be remembered.
    await expect(user.click(screen.getByRole('button'))).resolves.not.toThrow()
    expect(document.documentElement.dataset.theme).toBe('light')
  })
})
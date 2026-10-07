import { useCallback, useEffect, useState } from 'react'

export type Theme = 'dark' | 'light'

const THEME_KEY = 'agentoflow.theme'

/**
 * Storage and matchMedia are both optional.
 *
 * A test jsdom without localStorage, a locked-down browser profile and
 * a private window can each fail or omit them, and the theme has to
 * degrade to a default rather than take the app down with it.
 */
function storage(): Storage | null {
  try {
    return window.localStorage ?? null
  } catch {
    return null
  }
}

function prefersLight(): boolean {
  const media = window.matchMedia?.('(prefers-color-scheme: light)')

  return Boolean(media?.matches)
}

function read(): Theme {
  try {
    const stored = storage()?.getItem(THEME_KEY)

    if (stored === 'light' || stored === 'dark') {
      return stored
    }
  } catch {
    // Fall through to the system preference.
  }

  return prefersLight() ? 'light' : 'dark'
}

/**
 * Applies the theme to the document without recording a choice.
 *
 * Persisting here would be wrong: on a first visit with a light system
 * preference this would write "light" to storage, and the follow-the-
 * system effect would then see a stored choice and stop listening. Only
 * an explicit toggle may be remembered.
 */
function apply(theme: Theme): void {
  document.documentElement.dataset.theme = theme
}

/**
 * Reads and writes the theme, keeping the attribute on <html> in sync.
 *
 * The palettes are plain custom properties in index.css, so the theme is
 * a single attribute rather than a second stylesheet, and there is no
 * flash of the wrong colours while a second file loads.
 */
export function useTheme(): { theme: Theme; toggle: () => void } {
  const [theme, setTheme] = useState<Theme>(read)

  useEffect(() => {
    apply(theme)
  }, [theme])

  // Follow the system until the reader picks something themselves.
  useEffect(() => {
    let chosen: string | null = null

    try {
      chosen = storage()?.getItem(THEME_KEY) ?? null
    } catch {
      chosen = null
    }

    if (chosen === 'light' || chosen === 'dark') {
      return
    }

    const media = window.matchMedia?.('(prefers-color-scheme: light)')

    if (!media) {
      return
    }

    const follow = (): void => setTheme(media.matches ? 'light' : 'dark')

    media.addEventListener('change', follow)

    return () => media.removeEventListener('change', follow)
  }, [])

  const toggle = useCallback((): void => {
    setTheme((current) => {
      const next: Theme = current === 'dark' ? 'light' : 'dark'

      try {
        storage()?.setItem(THEME_KEY, next)
      } catch {
        // Non-fatal: the theme still applies for this session.
      }

      return next
    })
  }, [])

  return { theme, toggle }
}
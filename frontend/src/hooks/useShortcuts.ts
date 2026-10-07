import { useEffect } from 'react'

export interface ShortcutHandlers {
  /** Cmd/Ctrl+K: open the command palette. */
  onPalette: () => void
  /** Cmd/Ctrl+F or /: focus transcript search. */
  onSearch: () => void
  /** Escape: stop the run, or close the palette. */
  onStop: () => void
  onClosePalette: () => void
  paletteOpen: boolean
  /** Escape closes the palette before it stops a run. */
  stopEnabled: boolean
}

function inField(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false
  }

  return (
    target.tagName === 'INPUT' ||
    target.tagName === 'TEXTAREA' ||
    target.tagName === 'SELECT' ||
    target.isContentEditable
  )
}

/**
 * Keyboard shortcuts.
 *
 * An agent run is watched while it happens, so the keys that control
 * it should not require the mouse.
 *
 * - Cmd/Ctrl+K opens the command palette
 * - Cmd/Ctrl+F or "/" focuses transcript search
 * - Escape stops a run, or closes the palette first
 */
export function useShortcuts({
  onPalette,
  onSearch,
  onStop,
  onClosePalette,
  paletteOpen,
  stopEnabled,
}: ShortcutHandlers) {
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const meta = event.metaKey || event.ctrlKey

      if (meta && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        onPalette()
        return
      }

      if (meta && event.key.toLowerCase() === 'f') {
        event.preventDefault()
        onSearch()
        return
      }

      if (event.key === '/' && !inField(event.target)) {
        event.preventDefault()
        onSearch()
        return
      }

      if (event.key === 'Escape') {
        if (paletteOpen) {
          event.preventDefault()
          onClosePalette()
          return
        }

        if (stopEnabled) {
          event.preventDefault()
          onStop()
        }
      }
    }

    window.addEventListener('keydown', handler)

    return () => window.removeEventListener('keydown', handler)
  }, [
    onClosePalette,
    onPalette,
    onSearch,
    onStop,
    paletteOpen,
    stopEnabled,
  ])
}
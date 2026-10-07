import { useEffect, useMemo, useRef, useState } from 'react'

export interface Command {
  id: string
  label: string
  hint?: string
  run: () => void
}

interface CommandPaletteProps {
  open: boolean
  commands: Command[]
  onClose: () => void
}

/**
 * Cmd/Ctrl+K palette.
 *
 * The app has actions that live behind buttons in panels that are
 * not always on screen; a palette keeps them reachable.
 */
export function CommandPalette({
  open,
  commands,
  onClose,
}: CommandPaletteProps) {
  const [query, setQuery] = useState('')
  const [index, setIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement | null>(null)

  const needle = query.trim().toLowerCase()

  const visible = useMemo(
    () =>
      needle
        ? commands.filter((command) =>
            command.label.toLowerCase().includes(needle),
          )
        : commands,
    [commands, needle],
  )

  useEffect(() => {
    if (!open) {
      setQuery('')
      setIndex(0)
      return
    }

    inputRef.current?.focus()
  }, [open])

  useEffect(() => {
    setIndex(0)
  }, [needle])

  if (!open) {
    return null
  }

  const choose = (command: Command | undefined) => {
    if (!command) {
      return
    }

    command.run()
    onClose()
  }

  return (
    <div
      className="palette-backdrop"
      role="presentation"
      onClick={onClose}
    >
      <div
        className="palette"
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onClick={(event) => event.stopPropagation()}
      >
        <input
          ref={inputRef}
          className="palette-input"
          value={query}
          placeholder="Run a command…"
          aria-label="Command"
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault()
              choose(visible[index])
              return
            }

            if (event.key === 'ArrowDown') {
              event.preventDefault()
              setIndex((previous) =>
                visible.length === 0
                  ? 0
                  : (previous + 1) % visible.length,
              )
              return
            }

            if (event.key === 'ArrowUp') {
              event.preventDefault()
              setIndex((previous) =>
                visible.length === 0
                  ? 0
                  : (previous - 1 + visible.length) %
                    visible.length,
              )
            }
          }}
        />

        {visible.length === 0 ? (
          <p className="palette-empty">No matching command</p>
        ) : (
          <ul className="palette-list">
            {visible.map((command, position) => (
              <li key={command.id}>
                <button
                  type="button"
                  className={`palette-item${
                    position === index ? ' is-active' : ''
                  }`}
                  onMouseEnter={() => setIndex(position)}
                  onClick={() => choose(command)}
                >
                  <span>{command.label}</span>
                  {command.hint && (
                    <span className="palette-hint">{command.hint}</span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
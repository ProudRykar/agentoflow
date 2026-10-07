import { Fragment, type ReactNode, useState } from 'react'
import { highlight } from './highlight'

interface CodeBlockProps {
  language: string
  text: string
  /** Forces the disclosure even for a short block. */
  collapsible?: boolean
}

/**
 * Below this a block fits on screen, and a chevron that hides three
 * visible lines is noise rather than a control.
 */
const COLLAPSE_THRESHOLD = 12

/**
 * A code block with a language label, a copy button and, when it is
 * long enough to be worth hiding, a disclosure.
 *
 * Shared by Markdown fences and by tool arguments, so a `python_exec`
 * call and a ```python fence look the same. An unknown language yields
 * one plain token rather than no output.
 *
 * Expanded by default: the transcript is scrolled by the reader, and
 * starting collapsed hides the result of the call they just made.
 */
export function CodeBlock({
  language,
  text,
  collapsible = false,
}: CodeBlockProps): ReactNode {
  const [collapsed, setCollapsed] = useState(false)

  const lines = text.split('\n').length
  const canCollapse = collapsible || lines > COLLAPSE_THRESHOLD
  const tokens = highlight(text, language)

  return (
    <div className="md-code-wrap" data-collapsed={collapsed || undefined}>
      <div className="md-code-bar">
        {canCollapse ? (
          <button
            type="button"
            className="md-code-toggle"
            onClick={() => setCollapsed((value) => !value)}
            aria-expanded={!collapsed}
          >
            <span className="md-code-caret" aria-hidden="true">
              {collapsed ? '\u25b8' : '\u25be'}
            </span>
            <span className="md-code-lang">{language || 'text'}</span>
            <span className="md-code-meta">
              {collapsed ? `${lines} lines \u00b7 show` : `${lines} lines`}
            </span>
          </button>
        ) : (
          <span className="md-code-lang">{language || 'text'}</span>
        )}

        <CopyButton text={text} />
      </div>

      {collapsed ? (
        <div className="md-code-collapsed">{firstLine(text)}</div>
      ) : (
        <div className="md-code-body">
          <pre className="md-code" data-language={language}>
          <code>
            {tokens.map((token, index) =>
              token.kind === 'text' ? (
                <Fragment key={index}>{token.text}</Fragment>
              ) : (
                <span key={index} className={`tok-${token.kind}`}>
                  {token.text}
                </span>
              ),
            )}
          </code>
          </pre>
        </div>
      )}
    </div>
  )
}

/** The collapsed preview: whatever the first line says. */
function firstLine(text: string): string {
  const line = text.split('\n')[0] ?? ''

  return line.trim() ? line.trim() : '(empty)'
}

function CopyButton({ text }: { text: string }): ReactNode {
  const [copied, setCopied] = useState(false)

  const mark = (): void => {
    setCopied(true)

    window.setTimeout(() => setCopied(false), 1500)
  }

  const copy = (): void => {
    // `writeText` is unavailable on a plain http origin, which is how a
    // local setup is often reached.
    if (navigator.clipboard?.writeText) {
      void navigator.clipboard.writeText(text).then(mark, () => setCopied(false))
      return
    }

    const area = document.createElement('textarea')

    area.value = text
    area.setAttribute('readonly', '')
    area.style.position = 'fixed'
    area.style.opacity = '0'

    document.body.append(area)
    area.select()

    try {
      document.execCommand('copy')
      mark()
    } catch {
      setCopied(false)
    } finally {
      area.remove()
    }
  }

  return (
    <button
      type="button"
      className="md-code-copy"
      onClick={copy}
      aria-label="Copy code"
      title="Copy code"
    >
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}
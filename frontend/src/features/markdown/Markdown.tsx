import { Fragment, type CSSProperties, type ReactNode, useMemo } from 'react'
import { CodeBlock } from './CodeBlock'
import { parseBlocks, parseInline } from './parse'
import './Markdown.css'

interface MarkdownProps {
  source: string
}

/**
 * Render the Markdown subset as React elements.
 *
 * Nothing is injected as HTML, so agent output is displayed as
 * text no matter what it contains.
 */
export function Markdown({ source }: MarkdownProps) {
  // During streaming this component re-renders on every delta, and a
  // long reply means re-parsing the whole thing each time. The result
  // is memoised on the source string, so a re-render caused by anything
  // else (a sibling, the theme) reuses the parse.
  const blocks = useMemo(() => parseBlocks(source), [source])

  return (
    <div className="markdown">
      {blocks.map((block, index) => (
        <Fragment key={index}>{renderBlock(block)}</Fragment>
      ))}
    </div>
  )
}

function renderBlock(block: ReturnType<typeof parseBlocks>[number]) {
  switch (block.kind) {
    case 'heading': {
      // Headings are shifted down two levels so agent output never
      // competes with the page's own structure.
      const Tag = (
        'h3 h4 h5 h6 h6 h6'.split(' ')[
          Math.min(block.level, 5)
        ] ?? 'h6'
      ) as 'h3'

      return (
        <Tag className="md-heading">
          <Inline source={block.text} />
        </Tag>
      )
    }

    case 'code':
      return <CodeBlock language={block.language} text={block.text} />

    case 'list': {
      const items = block.items.map((item, index) => (
        <li
          key={index}
          className="md-list-item"
          data-task={item.checked === null ? undefined : 'true'}
        >
          {item.checked !== null && (
            <span
              className="md-task"
              data-checked={item.checked ? 'true' : undefined}
              role="img"
              aria-label={item.checked ? 'done' : 'not done'}
            >
              {item.checked ? '\u2713' : '\u25cb'}
            </span>
          )}

          <span className="md-list-text">
            <Multiline source={item.text} />
          </span>

          {item.children.length > 0 && (
            <span className="md-list-children">
              {item.children.map((child, childIndex) => (
                <Fragment key={childIndex}>{renderBlock(child)}</Fragment>
              ))}
            </span>
          )}
        </li>
      ))

      return block.ordered ? (
        <ol className="md-list" start={block.start > 1 ? block.start : undefined}>
          {items}
        </ol>
      ) : (
        <ul className="md-list">{items}</ul>
      )
    }

    case 'rule':
      return <hr className="md-rule" />

    case 'quote':
      return (
        <blockquote className="md-quote">
          <Inline source={block.text} />
        </blockquote>
      )

    case 'table': {
      const align = (index: number): CSSProperties | undefined => {
        const value = block.align[index]

        if (value !== 'left' && value !== 'center' && value !== 'right') {
          return undefined
        }

        return { textAlign: value }
      }

      return (
        <div className="md-table-scroll">
          <table className="md-table">
            <thead>
              <tr>
                {block.header.map((cell, index) => (
                  <th key={index} style={align(index)}>
                    <Inline source={cell} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, index) => (
                    <td key={index} style={align(index)}>
                      <Inline source={cell} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
    }

    case 'paragraph':
    default:
      return (
        <p className="md-paragraph">
          <Multiline source={block.text} />
        </p>
      )
  }
}

/**
 * Inline content that honours single newlines.
 *
 * HTML collapses a newline inside a paragraph into a space, so
 * "first\nsecond" would read as one run-on line. A single newline is a
 * line break in Markdown and agent output relies on it, so each line is
 * parsed on its own and joined with a break.
 */
function Multiline({ source }: { source: string }): ReactNode {
  const lines = source.split('\n')

  return lines.map((line, index) => (
    <Fragment key={index}>
      {index > 0 && <br />}
      <Inline source={line} />
    </Fragment>
  ))
}

function Inline({ source }: { source: string }): ReactNode {
  const tokens = parseInline(source)

  return (
    <>
      {tokens.map((token, index) => {
        switch (token.kind) {
          case 'code':
            return (
              <code key={index} className="md-inline-code">
                {token.text}
              </code>
            )

          case 'strong':
            return <strong key={index}>{token.text}</strong>

          case 'em':
            return <em key={index}>{token.text}</em>

          case 'del':
            return <del key={index}>{token.text}</del>

          case 'autolink':
            return (
              <a
                key={index}
                className="md-link"
                href={token.href}
                target="_blank"
                rel="noopener noreferrer nofollow"
              >
                {token.href}
              </a>
            )

          case 'link':
            return (
              <a
                key={index}
                className="md-link"
                href={token.href}
                target="_blank"
                rel="noopener noreferrer nofollow"
              >
                {token.text}
              </a>
            )

          case 'text':
          default:
            return <Fragment key={index}>{token.text}</Fragment>
        }
      })}
    </>
  )
}

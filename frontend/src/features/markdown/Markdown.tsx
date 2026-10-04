import { Fragment, type ReactNode } from 'react'
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
  const blocks = parseBlocks(source)

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
      return (
        <pre className="md-code" data-language={block.language}>
          {block.text}
        </pre>
      )

    case 'list': {
      const items = block.items.map((item, index) => (
        <li key={index} className="md-list-item">
          <Inline source={item} />
        </li>
      ))

      return block.ordered ? (
        <ol className="md-list">{items}</ol>
      ) : (
        <ul className="md-list">{items}</ul>
      )
    }

    case 'quote':
      return (
        <blockquote className="md-quote">
          <Inline source={block.text} />
        </blockquote>
      )

    case 'paragraph':
    default:
      return (
        <p className="md-paragraph">
          <Inline source={block.text} />
        </p>
      )
  }
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

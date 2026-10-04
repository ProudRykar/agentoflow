/**
 * Minimal, dependency-free Markdown subset renderer.
 *
 * Agent output is Markdown, so the browser needs structure rather
 * than a wall of text. This handles the subset that actually shows
 * up in agent replies: headings, fenced and inline code, bullet and
 * numbered lists, blockquotes and emphasis.
 *
 * It deliberately produces React elements instead of an HTML
 * string: no `dangerouslySetInnerHTML`, so model output can never
 * inject markup. Anything unrecognised degrades to plain text.
 */

export type Block =
  | { kind: 'heading'; level: number; text: string }
  | { kind: 'paragraph'; text: string }
  | { kind: 'code'; language: string; text: string }
  | { kind: 'list'; ordered: boolean; items: string[] }
  | { kind: 'quote'; text: string }

const FENCE = /^```([\w+-]*)\s*$/

export function parseBlocks(source: string): Block[] {
  const lines = source.replace(/\r\n/g, '\n').split('\n')
  const blocks: Block[] = []

  let index = 0

  while (index < lines.length) {
    const line = lines[index]

    if (!line.trim()) {
      index += 1
      continue
    }

    const fence = FENCE.exec(line.trim())

    if (fence) {
      const language = fence[1] ?? ''
      const body: string[] = []

      index += 1

      while (index < lines.length) {
        if (lines[index].trim() === '```') {
          index += 1
          break
        }

        body.push(lines[index])
        index += 1
      }

      blocks.push({
        kind: 'code',
        language,
        text: body.join('\n'),
      })

      continue
    }

    const heading = /^(#{1,6})\s+(.*)$/.exec(line)

    if (heading) {
      blocks.push({
        kind: 'heading',
        level: heading[1].length,
        text: heading[2].trim(),
      })

      index += 1
      continue
    }

    const bullet = /^\s*[-*+]\s+(.*)$/.exec(line)

    if (bullet) {
      const items: string[] = []

      while (index < lines.length) {
        const match = /^\s*[-*+]\s+(.*)$/.exec(lines[index])

        if (!match) {
          break
        }

        items.push(match[1])
        index += 1
      }

      blocks.push({ kind: 'list', ordered: false, items })

      continue
    }

    const numbered = /^\s*\d+[.)]\s+(.*)$/.exec(line)

    if (numbered) {
      const items: string[] = []

      while (index < lines.length) {
        const match = /^\s*\d+[.)]\s+(.*)$/.exec(lines[index])

        if (!match) {
          break
        }

        items.push(match[1])
        index += 1
      }

      blocks.push({ kind: 'list', ordered: true, items })

      continue
    }

    const quote = /^\s*>\s?(.*)$/.exec(line)

    if (quote) {
      const body: string[] = [quote[1]]

      index += 1

      while (index < lines.length) {
        const next = /^\s*>\s?(.*)$/.exec(lines[index])

        if (!next) {
          break
        }

        body.push(next[1])
        index += 1
      }

      blocks.push({
        kind: 'quote',
        text: body.join('\n'),
      })

      continue
    }

    const paragraph: string[] = [line]

    index += 1

    while (index < lines.length) {
      const next = lines[index]

      if (
        !next.trim() ||
        FENCE.test(next.trim()) ||
        /^(#{1,6})\s+/.test(next) ||
        /^\s*[-*+]\s+/.test(next) ||
        /^\s*\d+[.)]\s+/.test(next) ||
        /^\s*>\s?/.test(next)
      ) {
        break
      }

      paragraph.push(next)
      index += 1
    }

    blocks.push({
      kind: 'paragraph',
      text: paragraph.join('\n'),
    })
  }

  return blocks
}

export type Inline =
  | { kind: 'text'; text: string }
  | { kind: 'code'; text: string }
  | { kind: 'strong'; text: string }
  | { kind: 'em'; text: string }
  | { kind: 'link'; text: string; href: string }

/**
 * Tokenise inline markup.
 *
 * Links are recognised but only http(s) targets are kept, so a
 * `javascript:` URL from model output cannot become a clickable
 * link.
 */
export function parseInline(source: string): Inline[] {
  // The link alternative allows one level of balanced parentheses
  // so URLs such as `javascript:alert(1)` are captured whole and
  // can be rejected, instead of being split mid-token.
  const target = '(?:[^()\\s]|\\([^)\\s]*\\))+'
  const link = `\\[[^\\]]+\\]\\(${target}\\)`
  const pattern = new RegExp(
    '(`[^`]+`)' +
      '|(\\*\\*[^*]+\\*\\*)' +
      '|(__[^_]+__)' +
      '|(\\*[^*\\n]+\\*)' +
      '|(_[^_\\n]+_)' +
      `|(${link})`,
    'g',
  )

  const tokens: Inline[] = []

  let last = 0
  let match: RegExpExecArray | null

  while ((match = pattern.exec(source)) !== null) {
    if (match.index > last) {
      tokens.push({
        kind: 'text',
        text: source.slice(last, match.index),
      })
    }

    const [whole, code, strong, strongAlt, em, emAlt, link] = match

    if (code) {
      tokens.push({ kind: 'code', text: code.slice(1, -1) })
    } else if (strong ?? strongAlt) {
      tokens.push({
        kind: 'strong',
        text: (strong ?? strongAlt).slice(2, -2),
      })
    } else if (em ?? emAlt) {
      tokens.push({
        kind: 'em',
        text: (em ?? emAlt).slice(1, -1),
      })
    } else if (link) {
      const split = /^\[([^\]]+)\]\((.+)\)$/.exec(link)

      if (split && /^https?:\/\//i.test(split[2])) {
        tokens.push({
          kind: 'link',
          text: split[1],
          href: split[2],
        })
      } else {
        // Unsafe or malformed target: render the raw text.
        tokens.push({ kind: 'text', text: whole })
      }
    }

    last = match.index + whole.length
  }

  if (last < source.length) {
    tokens.push({ kind: 'text', text: source.slice(last) })
  }

  return tokens
}

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

export interface ListItem {
  /** Inline source of the item, task marker already removed. */
  text: string
  /** null when the item is not a task. */
  checked: boolean | null
  /** Nested blocks, which is how a sublist is represented. */
  children: Block[]
}

export type Block =
  | { kind: 'heading'; level: number; text: string }
  | { kind: 'paragraph'; text: string }
  | { kind: 'code'; language: string; text: string }
  | { kind: 'list'; ordered: boolean; start: number; items: ListItem[] }
  | { kind: 'quote'; text: string }
  | { kind: 'table'; header: string[]; rows: string[][]; align: string[] }
  | { kind: 'rule' }

const FENCE = /^```([\w+-]*)\s*$/

/** Three or more of one rule character, spaces allowed between. */
const RULE = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/

/** `Title` underlined with `===` (level 1) or `---` (level 2). */
const SETEXT = /^ {0,3}(=+|-+)[ \t]*$/

/** A list marker plus its indentation, tab expanded to two columns. */
const LIST_ITEM = /^(\s*)([-*+]|\d+[.)])[ \t]+(.*)$/

/**
 * A GFM table row: a pipe-separated line.
 *
 * The leading and trailing pipes are optional, which is why this is
 * split on every pipe and the empties trimmed rather than matched
 * with a fixed pattern.
 */
const TABLE_ROW = /^\s*\|?(.+?)\|?\s*$/

/** The header separator: `| --- | :--: |` and its variations. */
const TABLE_DIVIDER = /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$/

function splitRow(line: string): string[] {
  const match = TABLE_ROW.exec(line)

  if (!match) {
    return []
  }

  // A pipe inside inline code is escaped as \| by the model often
  // enough that leaving it split is worse than dropping the escape.
  return match[1]
    .split('|')
    .map((cell) => cell.trim().replace(/\\\|/g, '|'))
}

function alignments(divider: string): string[] {
  return splitRow(divider).map((cell) => {
    const left = cell.startsWith(':')
    const right = cell.endsWith(':')

    if (left && right) {
      return 'center'
    }

    if (right) {
      return 'right'
    }

    if (left) {
      return 'left'
    }

    return ''
  })
}

/** Reads a table starting at a header line, or returns null. */
function readTable(
  lines: string[],
  start: number,
): { header: string[]; rows: string[][]; align: string[]; next: number } | null {
  const header = splitRow(lines[start])

  if (header.length === 0) {
    return null
  }

  const divider = lines[start + 1]

  if (divider === undefined || !TABLE_DIVIDER.test(divider)) {
    return null
  }

  const align = alignments(divider)
  const rows: string[][] = []

  let index = start + 2

  while (index < lines.length) {
    const line = lines[index]

    if (!line.trim()) {
      break
    }

    if (!line.includes('|')) {
      break
    }

    const cells = splitRow(line)

    if (cells.length === 0) {
      break
    }

    // Short rows are padded so every cell has a column.
    while (cells.length < header.length) {
      cells.push('')
    }

    rows.push(cells.slice(0, header.length))
    index += 1
  }

  return { header, rows, align, next: index }
}


/** Tabs count as two columns so a tab-indented list still nests. */
function indentWidth(indent: string): number {
  let width = 0

  for (const char of indent) {
    width += char === '\t' ? 2 : 1
  }

  return width
}

/** `- [x] done` -> checked; `- plain` -> null. */
function taskState(text: string): { checked: boolean | null; rest: string } {
  const match = /^\[([ xX])\]\s+(.*)$/.exec(text)

  if (!match) {
    return { checked: null, rest: text }
  }

  return { checked: match[1] !== ' ', rest: match[2] }
}

interface ListFrame {
  indent: number
  items: ListItem[]
}

/**
 * Reads a whole list, including sublists, from the indentation.
 *
 * A deeper marker becomes a child list on the last item of the frame
 * above it; a marker at or above the current frame's indent pops back
 * out. That is enough structure for agent output, where a sublist is
 * always indented further than its parent.
 */
function readList(
  lines: string[],
  start: number,
): { ordered: boolean; first: number; items: ListItem[]; next: number } {
  const roots: ListItem[] = []
  const stack: ListFrame[] = [{ indent: -1, items: roots }]

  let ordered = false
  let first = 1
  let index = start

  const deepest = (): ListItem | null => {
    const frame = stack[stack.length - 1]

    return frame.items[frame.items.length - 1] ?? null
  }

  while (index < lines.length) {
    const line = lines[index]

    if (!line.trim()) {
      // A blank line only continues the list when another item
      // follows; otherwise the list has ended.
      const ahead = lines[index + 1]

      if (ahead === undefined || !LIST_ITEM.test(ahead)) {
        break
      }

      index += 1
      continue
    }

    const match = LIST_ITEM.exec(line)

    if (match) {
      const indent = indentWidth(match[1])
      const marker = match[2]
      const isOrdered = /^\d/.test(marker)

      if (roots.length === 0) {
        ordered = isOrdered
        first = isOrdered ? Number.parseInt(marker, 10) || 1 : 1

        // The root level takes its indent from the first marker, not
        // from a sentinel: with a sentinel every sibling after the
        // first would look like a child of it.
        stack[0].indent = indent
      }

      while (stack.length > 1 && indent < stack[stack.length - 1].indent) {
        stack.pop()
      }

      const frame = stack[stack.length - 1]
      const task = taskState(match[3])

      if (indent > frame.indent) {
        const parent = deepest()

        if (parent) {
          const children: ListItem[] = []

          parent.children.push({
            kind: 'list',
            ordered: isOrdered,
            start: 1,
            items: children,
          })

          stack.push({ indent, items: children })
        }
      }

      // Into the innermost frame, which is the one just opened when
      // this marker was deeper than its parent.
      stack[stack.length - 1].items.push({
        text: task.rest,
        checked: task.checked,
        children: [],
      })

      index += 1
      continue
    }

    // An indented, unmarked line continues the previous item.
    const parent = deepest()

    if (parent && /^ {2,}|\t/.test(line)) {
      parent.text += `\n${line.trim()}`
      index += 1
      continue
    }

    break
  }

  return { ordered, first, items: roots, next: index }
}

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

    // `---` after a line of text underlines it as a heading; on its
    // own it is a rule. Markdown resolves this in favour of the
    // heading, and so does GitHub.
    const underline = lines[index + 1]

    if (underline !== undefined && line.trim() && SETEXT.test(underline) && !RULE.test(line)) {
      blocks.push({
        kind: 'heading',
        level: underline.trim().startsWith('=') ? 1 : 2,
        text: line.trim(),
      })

      index += 2
      continue
    }

    if (RULE.test(line)) {
      blocks.push({ kind: 'rule' })

      index += 1
      continue
    }

    if (line.includes('|')) {
      const table = readTable(lines, index)

      if (table) {
        blocks.push({
          kind: 'table',
          header: table.header,
          rows: table.rows,
          align: table.align,
        })

        index = table.next

        continue
      }
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

    if (LIST_ITEM.test(line)) {
      const list = readList(lines, index)

      blocks.push({
        kind: 'list',
        ordered: list.ordered,
        start: list.first,
        items: list.items,
      })

      index = list.next

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
        RULE.test(next) ||
        SETEXT.test(next) ||
        /^\s*[-*+]\s+/.test(next) ||
        (next.includes('|') &&
          TABLE_DIVIDER.test(lines[index + 1] ?? '')) ||
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
  | { kind: 'del'; text: string }
  | { kind: 'link'; text: string; href: string }
  | { kind: 'autolink'; href: string }

/**
 * Tokenise inline markup.
 *
 * Links are recognised but only http(s) targets are kept, so a
 * `javascript:` URL from model output cannot become a clickable
 * link.
 */
/**
 * Drops punctuation that follows a bare URL in prose.
 *
 * `see https://x.dev.` should link `https://x.dev`, not the sentence
 * ending, while a URL that genuinely ends in `)` keeps it.
 */
function trimUrl(href: string): string {
  return href.replace(/[.,;:!?]+$/, '')
}

export function parseInline(source: string): Inline[] {
  // The link alternative allows one level of balanced parentheses
  // so URLs such as `javascript:alert(1)` are captured whole and
  // can be rejected, instead of being split mid-token.
  const target = '(?:[^()\\s]|\\([^)\\s]*\\))+'
  const link = `\\[[^\\]]+\\]\\(${target}\\)`
  // A bare URL, and the autolink form <https://x>. Trailing
  // punctuation is excluded so a full stop does not end up in the
  // href, and the closing angle bracket is not swallowed either.
  const bare = 'https?://[^\\s<>`"]+'
  const angle = `<${bare}>`
  const pattern = new RegExp(
    '(`[^`]+`)' +
      '|(\\*\\*\\*[^*]+\\*\\*\\*)' +
      '|(\\*\\*[^*]+\\*\\*)' +
      '|(__[^_]+__)' +
      '|(\\*[^*\\n]+\\*)' +
      '|(_[^_\\n]+_)' +
      '|(~~[^~]+~~)' +
      `|(${link})` +
      `|(${angle})` +
      `|(${bare})`,
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

    const [
      whole,
      code,
      strongTriple,
      strong,
      strongAlt,
      em,
      emAlt,
      del,
      link,
      angle,
      bare,
    ] = match

    if (code) {
      tokens.push({ kind: 'code', text: code.slice(1, -1) })
    } else if (strongTriple) {
      // `***x***` is emphasis wrapped in strong.
      tokens.push({ kind: 'em', text: strongTriple.slice(3, -3) })
    } else if (del) {
      tokens.push({ kind: 'del', text: del.slice(2, -2) })
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
    } else if (angle ?? bare) {
      const inner = angle ? angle.slice(1, -1) : (bare as string)
      const href = trimUrl(inner)

      tokens.push({ kind: 'autolink', href })

      // `https://x.dev.` links `https://x.dev` and the full stop stays
      // in the sentence rather than vanishing into the href.
      const trailing = inner.slice(href.length)

      if (trailing) {
        tokens.push({ kind: 'text', text: trailing })
      }
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

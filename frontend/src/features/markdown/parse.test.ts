import { describe, expect, it } from 'vitest'
import { parseBlocks, parseInline } from './parse'

describe('parseBlocks', () => {
  it('parses headings', () => {
    const blocks = parseBlocks('# Title\n\n## Sub')

    expect(blocks).toEqual([
      { kind: 'heading', level: 1, text: 'Title' },
      { kind: 'heading', level: 2, text: 'Sub' },
    ])
  })

  it('parses fenced code with a language', () => {
    const blocks = parseBlocks('```python\nprint(1)\n```')

    expect(blocks).toEqual([
      { kind: 'code', language: 'python', text: 'print(1)' },
    ])
  })

  it('parses an unterminated fence to the end', () => {
    const [block] = parseBlocks('```\nraw\nlines')

    expect(block).toEqual({
      kind: 'code',
      language: '',
      text: 'raw\nlines',
    })
  })

  it('parses bullet and ordered lists', () => {
    expect(parseBlocks('- a\n- b')).toEqual([
      {
        kind: 'list',
        ordered: false,
        start: 1,
        items: [
          { text: 'a', checked: null, children: [] },
          { text: 'b', checked: null, children: [] },
        ],
      },
    ])

    expect(parseBlocks('1. a\n2. b')).toEqual([
      {
        kind: 'list',
        ordered: true,
        start: 1,
        items: [
          { text: 'a', checked: null, children: [] },
          { text: 'b', checked: null, children: [] },
        ],
      },
    ])
  })

  it('joins wrapped paragraph lines', () => {
    expect(parseBlocks('one\ntwo')).toEqual([
      { kind: 'paragraph', text: 'one\ntwo' },
    ])
  })

  it('parses blockquotes', () => {
    expect(parseBlocks('> quoted\n> lines')).toEqual([
      { kind: 'quote', text: 'quoted\nlines' },
    ])
  })

  it('ignores blank lines', () => {
    expect(parseBlocks('\n\n\n')).toEqual([])
  })

  it('separates a paragraph from a following list', () => {
    const blocks = parseBlocks('intro\n\n- a')

    expect(blocks.map((b) => b.kind)).toEqual([
      'paragraph',
      'list',
    ])
  })

  it('normalises CRLF', () => {
    expect(parseBlocks('a\r\n\r\nb')).toEqual([
      { kind: 'paragraph', text: 'a' },
      { kind: 'paragraph', text: 'b' },
    ])
  })
})

describe('parseInline', () => {
  it('keeps plain text', () => {
    expect(parseInline('hello world')).toEqual([
      { kind: 'text', text: 'hello world' },
    ])
  })

  it('parses inline code', () => {
    expect(parseInline('use `run()` here')).toEqual([
      { kind: 'text', text: 'use ' },
      { kind: 'code', text: 'run()' },
      { kind: 'text', text: ' here' },
    ])
  })

  it('parses strong and emphasis', () => {
    expect(parseInline('**bold** and *soft*')).toEqual([
      { kind: 'strong', text: 'bold' },
      { kind: 'text', text: ' and ' },
      { kind: 'em', text: 'soft' },
    ])
  })

  it('keeps safe links', () => {
    expect(parseInline('[docs](https://example.com)')).toEqual([
      { kind: 'link', text: 'docs', href: 'https://example.com' },
    ])
  })

  it('neutralises javascript: links', () => {
    const tokens = parseInline(
      '[click](javascript:alert(1))',
    )

    expect(tokens).toEqual([
      { kind: 'text', text: '[click](javascript:alert(1))' },
    ])
  })

  it('neutralises data: links', () => {
    const tokens = parseInline('[x](data:text/html;base64,PHNjcmlwdD4=)')

    expect(tokens[0].kind).toBe('text')
  })

  it('preserves unmatched markup as text', () => {
    expect(parseInline('a ** b')).toEqual([
      { kind: 'text', text: 'a ** b' },
    ])
  })

  it('handles empty input', () => {
    expect(parseInline('')).toEqual([])
  })
})

describe('tables', () => {
  const table = [
    '| Категория | Близкие теги |',
    '| :--- | :---: |',
    '| Позы | Cowgirl Position |',
    '| Анальная тематика | Anal Penetration |',
  ].join('\n')

  it('parses a GFM table with alignment', () => {
    const blocks = parseBlocks(table)

    expect(blocks).toHaveLength(1)
    expect(blocks[0]).toEqual({
      kind: 'table',
      header: ['Категория', 'Близкие теги'],
      rows: [
        ['Позы', 'Cowgirl Position'],
        ['Анальная тематика', 'Anal Penetration'],
      ],
      align: ['left', 'center'],
    })
  })

  it('parses a table without the outer pipes', () => {
    const blocks = parseBlocks('a | b\n--- | ---\n1 | 2')

    expect(blocks[0]).toMatchObject({
      kind: 'table',
      header: ['a', 'b'],
      rows: [['1', '2']],
    })
  })

  it('reads right and centre alignment', () => {
    const blocks = parseBlocks('| a | b | c |\n| :-- | :-: | --: |\n| 1 | 2 | 3 |')

    expect(blocks[0]).toMatchObject({ align: ['left', 'center', 'right'] })
  })

  it('pads a short row and drops extra cells', () => {
    const blocks = parseBlocks('| a | b |\n| --- | --- |\n| 1 |\n| 1 | 2 | 3 |')

    expect(blocks[0]).toMatchObject({
      rows: [['1', ''], ['1', '2']],
    })
  })

  it('stops at a blank line', () => {
    const blocks = parseBlocks('| a |\n| --- |\n| 1 |\n\nafter')

    expect(blocks).toHaveLength(2)
    expect(blocks[1]).toEqual({ kind: 'paragraph', text: 'after' })
  })

  it('keeps inline markdown inside cells', () => {
    const blocks = parseBlocks('| a |\n| --- |\n| `x` and **y** |')

    expect(blocks[0]).toMatchObject({ rows: [['`x` and **y**']] })
  })

  it('treats a lone pipe line as a paragraph', () => {
    const blocks = parseBlocks('just | a pipe')

    expect(blocks[0]).toMatchObject({ kind: 'paragraph' })
  })

  it('does not swallow a paragraph that precedes a table', () => {
    const blocks = parseBlocks('intro:\n\n| a |\n| --- |\n| 1 |')

    expect(blocks.map((block) => block.kind)).toEqual(['paragraph', 'table'])
  })

  it('handles a table directly after a paragraph line', () => {
    const blocks = parseBlocks('intro\n| a |\n| --- |\n| 1 |')

    expect(blocks.map((block) => block.kind)).toEqual(['paragraph', 'table'])
    expect(blocks[0]).toMatchObject({ text: 'intro' })
  })

  it('accepts a three dash divider', () => {
    const blocks = parseBlocks('| a |\n| --- |\n| 1 |')

    expect(blocks[0]).toMatchObject({ kind: 'table' })
  })

  it('reads a setext underline as a heading, as GitHub does', () => {
    const blocks = parseBlocks('Title\n---')

    expect(blocks).toEqual([{ kind: 'heading', level: 2, text: 'Title' }])

    expect(parseBlocks('Title\n===')).toEqual([
      { kind: 'heading', level: 1, text: 'Title' },
    ])
  })

  it('reads a rule on its own', () => {
    expect(parseBlocks('---')).toEqual([{ kind: 'rule' }])
    expect(parseBlocks('***')).toEqual([{ kind: 'rule' }])
    expect(parseBlocks('___')).toEqual([{ kind: 'rule' }])
    expect(parseBlocks('- - -')).toEqual([{ kind: 'rule' }])
  })

  it('does not read bold text as a rule', () => {
    expect(parseBlocks('***bold***')).toEqual([
      { kind: 'paragraph', text: '***bold***' },
    ])
  })

  it('reads a nested list', () => {
    const blocks = parseBlocks('- a\n  - b\n  - c\n- d')

    expect(blocks[0]).toMatchObject({ kind: 'list' })
    const items = (blocks[0] as { items: { text: string; children: unknown[] }[] }).items

    expect(items.map((item) => item.text)).toEqual(['a', 'd'])
    expect(items[0].children).toHaveLength(1)
    expect(items[0].children[0]).toMatchObject({
      kind: 'list',
      items: [{ text: 'b' }, { text: 'c' }],
    })
    expect(items[1].children).toEqual([])
  })

  it('reads three levels of nesting', () => {
    const blocks = parseBlocks('- a\n  - b\n    - c')
    const level1 = (blocks[0] as { items: { children: { items: unknown[] }[] }[] }).items

    expect(level1[0].children[0].items).toHaveLength(1)
  })

  it('pops back out when the indent shrinks', () => {
    const blocks = parseBlocks('- a\n  - b\n- c\n  - d')
    const items = (blocks[0] as { items: { text: string; children: unknown[] }[] }).items

    expect(items.map((item) => item.text)).toEqual(['a', 'c'])
    expect(items[1].children[0]).toMatchObject({ items: [{ text: 'd' }] })
  })

  it('reads a task list', () => {
    const blocks = parseBlocks('- [ ] todo\n- [x] done\n- plain')
    const items = (blocks[0] as { items: { text: string; checked: boolean | null }[] }).items

    expect(items).toEqual([
      { text: 'todo', checked: false, children: [] },
      { text: 'done', checked: true, children: [] },
      { text: 'plain', checked: null, children: [] },
    ])
  })

  it('keeps the first number of an ordered list', () => {
    const blocks = parseBlocks('3. a\n4. b')

    expect(blocks[0]).toMatchObject({ ordered: true, start: 3 })
  })

  it('treats a tab indented item as a child', () => {
    const blocks = parseBlocks('- a\n\t- b')
    const items = (blocks[0] as { items: { children: unknown[] }[] }).items

    expect(items[0].children).toHaveLength(1)
  })

  it('folds an indented continuation into the item', () => {
    const blocks = parseBlocks('- a\n    continued\n- b')
    const items = (blocks[0] as { items: { text: string }[] }).items

    expect(items[0].text).toBe('a\ncontinued')
  })

  it('ends the list at a plain line', () => {
    const blocks = parseBlocks('- a\nplain')

    expect(blocks.map((block) => block.kind)).toEqual(['list', 'paragraph'])
  })
})

describe('paragraph line breaks', () => {
  it('keeps a single newline inside one paragraph', () => {
    const blocks = parseBlocks('first\nsecond')

    expect(blocks).toHaveLength(1)
    expect(blocks[0]).toMatchObject({ kind: 'paragraph', text: 'first\nsecond' })
  })

  it('joins wrapped prose without a trailing break', () => {
    expect(parseBlocks('a\nb\nc')).toHaveLength(1)
  })
})

describe('inline additions', () => {
  it('reads strikethrough', () => {
    expect(parseInline('~~gone~~')).toEqual([{ kind: 'del', text: 'gone' }])
  })

  it('reads triple emphasis as strong then em', () => {
    // `***x***` is rendered as emphasis inside the strong span; the
    // parser emits the emphasis marker and the renderer nests it.
    expect(parseInline('***both***')).toEqual([{ kind: 'em', text: 'both' }])
  })

  it('links a bare url', () => {
    expect(parseInline('see https://stash.example/graphiql now')).toEqual([
      { kind: 'text', text: 'see ' },
      { kind: 'autolink', href: 'https://stash.example/graphiql' },
      { kind: 'text', text: ' now' },
    ])
  })

  it('drops a sentence ending from the href', () => {
    expect(parseInline('read https://x.dev.')).toEqual([
      { kind: 'text', text: 'read ' },
      { kind: 'autolink', href: 'https://x.dev' },
      { kind: 'text', text: '.' },
    ])
  })

  it('links the angle bracket form', () => {
    expect(parseInline('<https://x.dev>')).toEqual([
      { kind: 'autolink', href: 'https://x.dev' },
    ])
  })

  it('does not link a non http scheme', () => {
    expect(parseInline('javascript:alert(1)')).toEqual([
      { kind: 'text', text: 'javascript:alert(1)' },
    ])
  })

  it('does not autolink inside inline code', () => {
    expect(parseInline('`https://x.dev`')).toEqual([
      { kind: 'code', text: 'https://x.dev' },
    ])
  })

  it('still rejects a javascript link target', () => {
    expect(parseInline('[x](javascript:alert(1))')).toEqual([
      { kind: 'text', text: '[x](javascript:alert(1))' },
    ])
  })
})

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
      { kind: 'list', ordered: false, items: ['a', 'b'] },
    ])

    expect(parseBlocks('1. a\n2. b')).toEqual([
      { kind: 'list', ordered: true, items: ['a', 'b'] },
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

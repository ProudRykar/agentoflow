import { describe, expect, it } from 'vitest'
import { highlight, resolveLanguage } from './highlight'

function kinds(code: string, language: string): Map<string, string> {
  const found = new Map<string, string>()

  for (const token of highlight(code, language)) {
    found.set(token.kind, (found.get(token.kind) ?? '') + token.text)
  }

  return found
}

describe('highlight', () => {
  it('returns a single plain token for an unknown language', () => {
    const tokens = highlight('anything at all', 'brainfuck')

    expect(tokens).toEqual([{ kind: 'text', text: 'anything at all' }])
  })

  it('returns a single plain token for a missing language', () => {
    expect(highlight('x = 1', '')).toEqual([{ kind: 'text', text: 'x = 1' }])
  })

  it('never loses or reorders a character', () => {
    const source = 'def f(x):\n    return {**x, "a": 1}  # note\n'
    const joined = highlight(source, 'python')
      .map((token) => token.text)
      .join('')

    expect(joined).toBe(source)
  })

  it('marks python keywords, strings and comments', () => {
    const found = kinds('import re  # grab it', 'python')

    expect(found.get('keyword')).toBe('import')
    expect(found.get('comment')).toBe('# grab it')
  })

  it('does not read a keyword inside a python string', () => {
    const tokens = highlight('print("import class")', 'python')
    const strings = tokens
      .filter((token) => token.kind === 'string')
      .map((token) => token.text)
      .join('')

    expect(strings).toBe('"import class"')
    // The word inside the quotes stays string, not keyword.
    expect(tokens.some((token) => token.kind === 'keyword')).toBe(false)
  })

  it('does not read a comment marker inside a string', () => {
    const tokens = highlight('url = "http://x#y"', 'python')

    expect(tokens.some((token) => token.kind === 'comment')).toBe(false)
  })

  it('keeps a triple quoted python string whole', () => {
    const tokens = highlight('"""\nimport not_a_keyword\n"""', 'python')

    expect(tokens).toHaveLength(1)
    expect(tokens[0].kind).toBe('string')
  })

  it('marks a def target and a call as functions', () => {
    expect(kinds('def preprocess(tag):', 'python').get('function')).toBe(
      'preprocess',
    )
    expect(kinds('preprocess(tag)', 'python').get('function')).toBe(
      'preprocess',
    )
  })

  it('leaves digits inside an identifier alone', () => {
    const found = kinds('utf8 = f2', 'python')

    expect(found.get('number')).toBeUndefined()
  })

  it('marks a python decorator as a property', () => {
    expect(kinds('@cache', 'python').get('property')).toBe('@cache')
  })

  it('distinguishes a json key from a json string value', () => {
    const found = kinds('{"name": "Anal"}', 'json')

    expect(found.get('property')).toBe('"name"')
    expect(found.get('string')).toBe('"Anal"')
  })

  it('marks a graphql variable and operation', () => {
    const tokens = highlight(
      'query Tags($q: String!) {\n  findTags(name: $q) {\n    id\n  }\n}',
      'graphql',
    )

    expect(
      tokens.filter((token) => token.text.startsWith('$')).map((t) => t.text),
    ).toEqual(['$q', '$q'])
    expect(
      tokens.filter((token) => token.kind === 'keyword').map((t) => t.text),
    ).toContain('query')
  })

  it('marks sql keywords without regard to case', () => {
    const wordsFor = (code: string): string[] =>
      highlight(code, 'sql')
        .filter((token) => token.kind === 'keyword')
        .map((token) => token.text)

    expect(wordsFor('select id from tags')).toEqual(['select', 'from'])
    expect(wordsFor('SELECT id FROM tags')).toEqual(['SELECT', 'FROM'])
  })

  it('does not read a shell comment marker inside quotes', () => {
    const tokens = highlight('echo "a # b"', 'shell')

    expect(tokens.some((token) => token.kind === 'comment')).toBe(false)
  })

  it('marks a shell variable inside double quotes', () => {
    expect(kinds('echo "$HOME"', 'shell').get('property')).toBe('$HOME')
  })

  it('does not interpolate a variable inside single quotes', () => {
    const tokens = highlight("echo '$HOME'", 'shell')

    expect(tokens.some((token) => token.kind === 'property')).toBe(false)
    expect(tokens.some((token) => token.kind === 'string')).toBe(true)
  })

  it('does not paint every shell word as a function', () => {
    // A regression guard: a rule that matched any bare word made shell
    // output unreadable.
    const tokens = highlight('for f in a b c; do', 'shell')

    expect(tokens.some((token) => token.kind === 'function')).toBe(false)
  })

  it('marks added and removed diff lines', () => {
    const found = kinds('@@ -1 +1 @@\n+added\n-removed', 'diff')

    expect(found.get('added')).toBe('+added')
    expect(found.get('removed')).toBe('-removed')
    expect(found.get('keyword')).toBe('@@ -1 +1 @@')
  })

  it('resolves fence aliases and case', () => {
    expect(resolveLanguage('Python')).toBe(resolveLanguage('py'))
    expect(resolveLanguage('BASH')).toBe(resolveLanguage('shell'))
    expect(resolveLanguage('nonsense')).toBeUndefined()
  })

  it('preserves trailing newlines and empty input', () => {
    expect(highlight('', 'python')).toEqual([])
    expect(
      highlight('a\n', 'python')
        .map((token) => token.text)
        .join(''),
    ).toBe('a\n')
  })
})
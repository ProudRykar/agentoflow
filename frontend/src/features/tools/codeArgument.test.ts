import { describe, expect, it } from 'vitest'
import { codeArgument } from './codeArgument'

describe('codeArgument', () => {
  const script = 'import re\n\n\ndef f():\n    return re.sub(r"x", "y", "z")\n'

  it('treats a python_exec script as python', () => {
    expect(codeArgument('python_exec', 'code', script)).toBe('python')
  })

  it('treats a graphql query as graphql', () => {
    expect(
      codeArgument('stash_graphql', 'query', 'query Tags {\n  findTags {\n    id\n  }\n}'),
    ).toBe('graphql')
  })

  it('treats a long single line query as code', () => {
    const long = `{ findTags(filter: { per_page: 250 }) { ${'id name '.repeat(30)} } }`

    expect(codeArgument('stash_graphql', 'query', long)).toBe('graphql')
  })

  it('keeps a short single line query inline', () => {
    expect(
      codeArgument('stash_graphql', 'query', '{ findTags { id } }'),
    ).toBeUndefined()
  })

  it('keeps a short scalar argument inline', () => {
    expect(codeArgument('python_exec', 'timeout', '30')).toBeUndefined()
  })

  it('keeps a short prose argument inline', () => {
    expect(codeArgument('web_research', 'prompt', 'find similar tags')).toBeUndefined()
  })

  it('preserves indentation for an unknown multi line argument', () => {
    expect(codeArgument('some_tool', 'whatever', 'one\ntwo')).toBe('')
  })

  it('ignores non string values', () => {
    expect(codeArgument('python_exec', 'code', { a: 1 })).toBeUndefined()
    expect(codeArgument('python_exec', 'code', null)).toBeUndefined()
    expect(codeArgument('python_exec', 'code', 42)).toBeUndefined()
  })

  it('prefers the tool specific language over the argument name', () => {
    // `code` in python_exec is Python; the argument name alone says
    // python too, but `query` on graphql.execute must not fall back to
    // prose.
    expect(codeArgument('graphql.execute', 'query', 'a\nb')).toBe('graphql')
  })
})
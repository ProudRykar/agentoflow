/**
 * Which tool arguments deserve a code block rather than a text run.
 *
 * The reason this exists: a multi-line string inside an inline element
 * loses every newline, so a `python_exec` script renders as one
 * unreadable paragraph. A `graphql.execute` query has the same problem.
 * Anything known to carry source gets the full treatment; an
 * unrecognised multi-line string still gets its indentation preserved.
 */

/**
 * Argument name to fence language.
 *
 * The tool name wins when it is specific enough, because `query` is
 * GraphQL for `stash_graphql` but plain prose for something else, and
 * `code` is Python for `python_exec`.
 */
const BY_ARGUMENT: Record<string, string> = {
  code: 'python',
  script: 'python',
  source: '',
  snippet: '',
  program: 'python',
  query: 'graphql',
  gql: 'graphql',
  mutation: 'graphql',
  command: 'shell',
  cmd: 'shell',
  script_path: '',
  sql: 'sql',
  json: 'json',
  payload: 'json',
  body: '',
}

const BY_TOOL: Record<string, Record<string, string>> = {
  python_exec: { code: 'python' },
  graphql: { query: 'graphql' },
  web_research: { prompt: '' },
}

/** A single line this long is still code, not prose. */
const LONG_LINE = 200

/**
 * Returns the fence language for an argument, or undefined when the
 * value should stay a plain inline string.
 */
export function codeArgument(
  tool: string,
  argument: string,
  value: unknown,
): string | undefined {
  if (typeof value !== 'string') {
    return undefined
  }

  const perTool = BY_TOOL[tool]?.[argument]
  const language = perTool ?? BY_ARGUMENT[argument]

  if (language === undefined) {
    // Unknown argument: preserve indentation but do not claim a
    // language we cannot vouch for.
    return value.includes('\n') ? '' : undefined
  }

  if (value.includes('\n') || value.length >= LONG_LINE) {
    return language
  }

  return undefined
}
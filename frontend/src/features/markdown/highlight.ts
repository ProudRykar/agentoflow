/**
 * A small, dependency-free syntax highlighter for the code that shows
 * up in agent transcripts.
 *
 * Adding a full highlighter (Prism, Shiki) would cost hundreds of
 * kilobytes for output that is overwhelmingly Python, JSON, shell,
 * SQL and GraphQL. This scans with ordered sticky regexes instead: at
 * each position the patterns are tried in order and the first match
 * wins, so a comment can never be re-matched as a keyword and a
 * string's contents can never be read as code.
 *
 * It is deliberately forgiving. An unknown language, or a construct
 * it does not know, comes back as plain text rather than throwing, so
 * a bad guess costs highlighting but never content.
 */

export type TokenKind =
  | 'text'
  | 'keyword'
  | 'string'
  | 'number'
  | 'comment'
  | 'function'
  | 'constant'
  | 'property'
  | 'operator'
  | 'added'
  | 'removed'

export interface Token {
  text: string
  kind: TokenKind
}

type Rule = [TokenKind, RegExp]

/** Joins words into one alternation, longest first so `elseif` beats `else`. */
function words(source: string): string {
  return [...new Set(source.split(/\s+/).filter(Boolean))]
    .sort((a, b) => b.length - a.length)
    .join('|')
}

const PYTHON_KEYWORDS = words(`
  and as assert async await break class continue def del elif else except
  finally for from global if import in is lambda match nonlocal not or pass
  raise return try while with yield case
`)

const PYTHON_CONSTANTS = words(`True False None Ellipsis NotImplemented self cls`)

const C_KEYWORDS = words(`
  abstract as async await break case catch class const continue debugger
  default delete do else enum export extends final finally for from function
  get goto if implements import in instanceof interface let namespace new of
  package private protected public return set static super switch this throw
  try type typeof var void while with yield
  bool byte char double float int long short signed unsigned void
  fn mut struct trait impl pub use mod match loop move ref where unsafe dyn
`)

const C_CONSTANTS = words(`true false null undefined nil None NaN Infinity`)

const SHELL_KEYWORDS = words(`
  if then else elif fi case esac for while until do done in function select
  time coproc return break continue local export readonly declare source alias
  unset shift trap set echo exit eval exec
`)

const SQL_KEYWORDS = words(`
  select from where group by having order limit offset insert into values
  update set delete create drop alter table index view join inner left right
  full outer cross on as and or not null is in like ilike between exists
  union all distinct case when then else end with recursive returning asc desc
  primary key foreign references unique default check constraint cascade
  begin commit rollback transaction grant revoke
`)

const GRAPHQL_KEYWORDS = words(`
  query mutation subscription fragment on type input enum interface union
  scalar schema directive implements extend
`)

const YAML_KEYWORDS = words(`true false null yes no on off`)

const TOML_KEYWORDS = words(`true false`)

interface Language {
  rules: Rule[]
  /** Case-insensitive matching for the keyword rule only. */
  ignoreCase?: boolean
}

function sticky(body: string, flags = ''): RegExp {
  return new RegExp(body, `${flags}y`)
}

const PYTHON: Language = {
  rules: [
    ['comment', sticky('#[^\\n]*')],
    [
      'string',
      sticky(
        '(?:[furbFURB]{0,2})(?:"""[\\s\\S]*?"""|\'\'\'[\\s\\S]*?\'\'\')',
      ),
    ],
    [
      'string',
      sticky(
        '(?:[furbFURB]{0,2})(?:"(?:\\\\.|[^"\\\\\\n])*"|\'(?:\\\\.|[^\'\\\\\\n])*\')',
      ),
    ],
    ['property', sticky('@[A-Za-z_][\\w.]*')],
    ['number', sticky('0[xXbBoO][0-9a-fA-F_]+')],
    [
      'number',
      sticky('(?<![\\w.])\\d[\\d_]*(?:\\.[\\d_]+)?(?:[eE][+-]?\\d+)?[jJ]?'),
    ],
    ['keyword', sticky(`\\b(?:${PYTHON_KEYWORDS})\\b`)],
    ['constant', sticky(`\\b(?:${PYTHON_CONSTANTS})\\b`)],
    // `def name`, `class Name` and a call `name(` all read as functions.
    ['function', sticky('(?<=\\bdef\\s)\\w+')],
    ['function', sticky('(?<=\\bclass\\s)\\w+')],
    ['function', sticky('[A-Za-z_]\\w*(?=\\s*\\()')],
    ['operator', sticky('[-+*/%=<>!&|^~]+')],
  ],
}

const C_LIKE: Language = {
  rules: [
    ['comment', sticky('//[^\\n]*')],
    ['comment', sticky('/\\*[\\s\\S]*?\\*/')],
    ['string', sticky('`(?:\\\\.|[^`\\\\])*`')],
    ['string', sticky('"(?:\\\\.|[^"\\\\\\n])*"')],
    ['string', sticky("'(?:\\\\.|[^'\\\\\\n])*'")],
    ['number', sticky('0[xXbBoO][0-9a-fA-F_]+n?')],
    ['number', sticky('\\d[\\d_]*(?:\\.[\\d_]+)?(?:[eE][+-]?\\d+)?n?')],
    ['keyword', sticky(`\\b(?:${C_KEYWORDS})\\b`)],
    ['constant', sticky(`\\b(?:${C_CONSTANTS})\\b`)],
    ['function', sticky('[A-Za-z_$][\\w$]*(?=\\s*\\()')],
    ['property', sticky('\\b[A-Za-z_$][\\w$]*(?=\\s*:)')],
    ['operator', sticky('[-+*/%=<>!&|^~?:]+')],
  ],
}

const JSON_LIKE: Language = {
  rules: [
    ['property', sticky('"(?:\\\\.|[^"\\\\])*"(?=\\s*:)')],
    ['string', sticky('"(?:\\\\.|[^"\\\\])*"')],
    ['string', sticky("'(?:\\\\.|[^'\\\\])*'")],
    ['number', sticky('-?\\d[\\d_]*(?:\\.[\\d_]+)?(?:[eE][+-]?\\d+)?')],
    ['keyword', sticky('\\b(?:true|false|null)\\b')],
  ],
}

const SHELL: Language = {
  rules: [
    ['comment', sticky('#[^\\n]*')],
    // A variable is matched before the strings so it still lights up
    // inside double quotes, where it is interpolated. Single quotes
    // suppress interpolation, so they win by coming first.
    ['string', sticky("'[^']*'")],
    ['property', sticky('\\$(?:\\{[^}\\n]*\\}|[A-Za-z_]\\w*|[0-9@*#?$!])')],
    // The body excludes `$` so an interpolated variable is not swallowed.
    ['string', sticky('"(?:\\\\.|[^"\\\\$])*"')],
    ['number', sticky('\\b\\d+\\b')],
    ['keyword', sticky(`\\b(?:${SHELL_KEYWORDS})\\b`)],
    ['operator', sticky('[-+*/%=<>!&|^~]+')],
  ],
}

const SQL: Language = {
  ignoreCase: true,
  rules: [
    ['comment', sticky('--[^\\n]*')],
    ['comment', sticky('/\\*[\\s\\S]*?\\*/')],
    ['string', sticky("'(?:''|[^'])*'")],
    ['number', sticky('\\b\\d+(?:\\.\\d+)?\\b')],
    ['keyword', sticky(`\\b(?:${SQL_KEYWORDS})\\b`, 'i')],
    ['function', sticky('\\b[a-z_]\\w*(?=\\s*\\()', 'i')],
    ['operator', sticky('[-+*/%=<>!|]+')],
  ],
}

const GRAPHQL: Language = {
  rules: [
    ['comment', sticky('#[^\\n]*')],
    ['string', sticky('"""[\\s\\S]*?"""')],
    ['string', sticky('"(?:\\\\.|[^"\\\\])*"')],
    ['property', sticky('\\$[A-Za-z_]\\w*')],
    ['number', sticky('-?\\d+\\.?\\d*')],
    ['keyword', sticky(`\\b(?:${GRAPHQL_KEYWORDS})\\b`)],
    ['property', sticky('[A-Za-z_]\\w*(?=\\s*:)')],
    ['operator', sticky('[-+*/%=<>!&|{}()\\[\\]:=]+')],
  ],
}

const YAML: Language = {
  rules: [
    ['comment', sticky('#[^\\n]*')],
    ['string', sticky('"(?:\\\\.|[^"\\\\])*"')],
    ['string', sticky("'[^']*'")],
    ['property', sticky('^\\s*(?=[\\w.$-]+\\s*:)', 'm')],
    ['keyword', sticky(`\\b(?:${YAML_KEYWORDS})\\b`, 'i')],
    ['number', sticky('-?\\d+(?:\\.\\d+)?')],
  ],
}

const TOML: Language = {
  rules: [
    ['comment', sticky('#[^\\n]*')],
    ['string', sticky('"""[\\s\\S]*?"""')],
    ['string', sticky('"(?:\\\\.|[^"\\\\])*"')],
    ['string', sticky("'[^']*'")],
    ['property', sticky('^\\s*(?=[\\w.-]+\\s*=)', 'm')],
    ['keyword', sticky(`\\b(?:${TOML_KEYWORDS})\\b`)],
    ['number', sticky('-?\\d+(?:\\.\\d+)?')],
  ],
}

const DIFF: Language = {
  rules: [
    ['keyword', sticky('@@[^\\n]*@@')],
    ['added', sticky('\\+[^\\n]*')],
    ['removed', sticky('-[^-][^\\n]*')],
    ['comment', sticky('^[ ][^\\n]*', 'm')],
  ],
}

const LANGUAGES: Record<string, Language> = {
  py: PYTHON,
  python: PYTHON,
  python3: PYTHON,
  json: JSON_LIKE,
  jsonc: JSON_LIKE,
  javascript: C_LIKE,
  js: C_LIKE,
  jsx: C_LIKE,
  ts: C_LIKE,
  tsx: C_LIKE,
  typescript: C_LIKE,
  go: C_LIKE,
  rust: C_LIKE,
  rs: C_LIKE,
  c: C_LIKE,
  cpp: C_LIKE,
  'c++': C_LIKE,
  java: C_LIKE,
  kotlin: C_LIKE,
  swift: C_LIKE,
  css: C_LIKE,
  scss: C_LIKE,
  sh: SHELL,
  shell: SHELL,
  bash: SHELL,
  zsh: SHELL,
  console: SHELL,
  sql: SQL,
  graphql: GRAPHQL,
  gql: GRAPHQL,
  yaml: YAML,
  yml: YAML,
  toml: TOML,
  ini: TOML,
  diff: DIFF,
  patch: DIFF,
}

/** Normalises a fence label so ```Python and ``` py both resolve. */
export function resolveLanguage(language: string): Language | undefined {
  const key = language.trim().toLowerCase()

  return LANGUAGES[key]
}

/**
 * Splits source into styled tokens.
 *
 * Unknown languages come back as a single plain token, so callers can
 * always render the result without special-casing.
 */
export function highlight(source: string, language: string): Token[] {
  const definition = resolveLanguage(language)

  if (!definition) {
    return [{ text: source, kind: 'text' }]
  }

  const tokens: Token[] = []
  let cursor = 0

  const push = (text: string, kind: TokenKind): void => {
    const last = tokens[tokens.length - 1]

    if (last && last.kind === kind) {
      last.text += text
      return
    }

    tokens.push({ text, kind })
  }

  while (cursor < source.length) {
    let matched = false

    for (const [kind, pattern] of definition.rules) {
      pattern.lastIndex = cursor

      const match = pattern.exec(source)

      if (match && match[0]) {
        push(match[0], kind)
        cursor += match[0].length
        matched = true
        break
      }
    }

    if (!matched) {
      // One character, never to the end of the line: consuming a run
      // here would skip past the start of whatever token comes next
      // and colour the rest of the line as plain text. Adjacent plain
      // characters merge back into one token in `push`.
      push(source[cursor] ?? '', 'text')
      cursor += 1
    }
  }

  return tokens
}
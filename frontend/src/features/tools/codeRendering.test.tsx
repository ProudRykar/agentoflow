import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { Markdown } from '../markdown/Markdown'
import { ToolCard } from './ToolCard'
import type { ToolEntry } from '../../stores/transcript'

const BASE: ToolEntry = {
  kind: 'tool',
  id: '1',
  callId: 'call-1',
  name: 'python_exec',
  arguments: {},
  status: 'success',
  output: null,
  errorCode: null,
  errorMessage: null,
  durationSeconds: 0.069,
  expanded: true,
  runId: '',
}

function entry(overrides: Partial<ToolEntry>): ToolEntry {
  return { ...BASE, ...overrides }
}

const SCRIPT = [
  'import re',
  '',
  'def preprocess(tag):',
  '    """Cleans the name."""',
  "    return re.sub(r'[()]', '', tag.lower())",
].join('\n')

describe('ToolCard code arguments', () => {
  it('renders a python_exec script with its line breaks and indent intact', () => {
    const { container } = render(
      <ToolCard
        entry={entry({ arguments: { code: SCRIPT } })}
        onToggle={() => {}}
      />,
    )

    const pre = container.querySelector('pre')

    // The original bug: a multi-line string inside an inline element
    // collapsed into one wall of text.
    expect(pre?.textContent).toBe(SCRIPT)
    // Five lines and a docstring, not one run-on line.
    expect(pre?.textContent?.split('\n')).toHaveLength(SCRIPT.split('\n').length)
    expect(container.textContent).not.toContain('import re def preprocess')
  })

  it('labels the block as python and highlights it', () => {
    const { container } = render(
      <ToolCard
        entry={entry({ arguments: { code: SCRIPT } })}
        onToggle={() => {}}
      />,
    )

    expect(container.querySelector('.md-code-lang')?.textContent).toBe('python')
    expect(container.querySelectorAll('.tok-keyword').length).toBeGreaterThan(0)
    expect(container.querySelector('.tok-string')?.textContent).toBe(
      '"""Cleans the name."""',
    )
  })

  it('keeps a scalar argument as inline text', () => {
    const { container } = render(
      <ToolCard
        entry={entry({ arguments: { timeout: 30 } })}
        onToggle={() => {}}
      />,
    )

    expect(container.querySelector('pre')).toBeNull()
    expect(screen.getByText('30')).toBeTruthy()
  })

  it('does not highlight a non string code argument', () => {
    const { container } = render(
      <ToolCard entry={entry({ arguments: { code: 12 } })} onToggle={() => {}} />,
    )

    expect(container.querySelector('pre')).toBeNull()
  })

  it('renders a graphql query as graphql', () => {
    const { container } = render(
      <ToolCard
        entry={entry({
          name: 'stash_graphql',
          arguments: { query: 'query Tags {\n  findTags {\n    id\n  }\n}' },
        })}
        onToggle={() => {}}
      />,
    )

    expect(container.querySelector('.md-code-lang')?.textContent).toBe('graphql')
  })
})

describe('Markdown tables', () => {
  const source = [
    '### Примеры',
    '',
    '| Категория | Близкие теги |',
    '| :--- | :---: |',
    '| Позы | Cowgirl Position |',
    '| Анальная тематика | Anal Penetration |',
  ].join('\n')

  it('renders a real table element instead of pipes', () => {
    const { container } = render(<Markdown source={source} />)

    expect(container.querySelectorAll('table')).toHaveLength(1)
    expect(container.textContent).not.toContain('| :--- |')
    expect(container.textContent).not.toContain('| :---')

    const header = Array.from(container.querySelectorAll('th')).map(
      (cell) => cell.textContent,
    )

    expect(header).toEqual(['Категория', 'Близкие теги'])

    const rows = container.querySelectorAll('tbody tr')

    expect(rows).toHaveLength(2)
    expect(rows[0].textContent).toBe('ПозыCowgirl Position')
  })

  it('applies the declared alignment', () => {
    const { container } = render(<Markdown source={source} />)
    const cells = container.querySelectorAll('th')

    expect(cells[0].style.textAlign).toBe('left')
    expect(cells[1].style.textAlign).toBe('center')
  })

  it('renders inline formatting inside a cell', () => {
    const { container } = render(
      <Markdown source={'| a |\n| --- |\n| `Anal` and **bold** |'} />,
    )

    expect(container.querySelector('td code')?.textContent).toBe('Anal')
    expect(container.querySelector('td strong')?.textContent).toBe('bold')
  })

  it('keeps a heading above the table', () => {
    const { container } = render(<Markdown source={source} />)

    // Headings shift down two levels, so `###` becomes an h6.
    expect(container.querySelector('h6')?.textContent).toBe('Примеры')
  })

  it('highlights a fenced python block', () => {
    const { container } = render(
      <Markdown source={'```python\ndef f():\n    return 1\n```'} />,
    )

    expect(container.querySelector('.md-code-lang')?.textContent).toBe('python')
    expect(container.querySelector('pre')?.textContent).toBe(
      'def f():\n    return 1',
    )
    expect(
      container.querySelector('.tok-function')?.textContent,
    ).toBe('f')
  })

  it('leaves an unknown fence language as plain text', () => {
    const { container } = render(
      <Markdown source={'```brainfuck\n+++.\n```'} />,
    )

    expect(container.querySelector('pre')?.textContent).toBe('+++.')
    expect(container.querySelector('.tok-keyword')).toBeNull()
  })

  it('offers a copy button', () => {
    render(<Markdown source={'```python\nx = 1\n```'} />)

    expect(screen.getByLabelText('Copy code')).toBeTruthy()
  })

  it('never injects markup from a cell', () => {
    const { container } = render(
      <Markdown source={'| a |\n| --- |\n| <img src=x onerror=alert(1)> |'} />,
    )

    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('td')?.textContent).toBe(
      '<img src=x onerror=alert(1)>',
    )
  })
})
describe('collapsible code blocks', () => {
  const LONG = Array.from({ length: 30 }, (_, i) => `line_${i + 1}`).join('\n')

  it('starts expanded so the result of a call is visible', () => {
    const { container } = render(<Markdown source={`\`\`\`python\n${LONG}\n\`\`\``} />)

    expect(container.querySelector('pre')?.textContent).toContain('line_30')
    expect(container.querySelector('.md-code-wrap')?.hasAttribute('data-collapsed')).toBe(
      false,
    )
  })

  it('collapses and expands on click', async () => {
    const user = userEvent.setup()
    const { container } = render(<Markdown source={`\`\`\`python\n${LONG}\n\`\`\``} />)

    const toggle = container.querySelector('.md-code-toggle') as HTMLButtonElement

    expect(toggle.getAttribute('aria-expanded')).toBe('true')

    await user.click(toggle)

    expect(container.querySelector('pre')).toBeNull()
    expect(container.querySelector('.md-code-collapsed')?.textContent).toBe('line_1')
    expect(toggle.getAttribute('aria-expanded')).toBe('false')

    await user.click(toggle)

    expect(container.querySelector('pre')?.textContent).toContain('line_30')
  })

  it('still offers copy while collapsed', async () => {
    const user = userEvent.setup()
    const { container } = render(<Markdown source={`\`\`\`python\n${LONG}\n\`\`\``} />)

    await user.click(container.querySelector('.md-code-toggle') as HTMLButtonElement)

    expect(screen.getByLabelText('Copy code')).toBeTruthy()
  })

  it('shows no chevron for a short block', () => {
    const { container } = render(<Markdown source={'```python\nx = 1\n```'} />)

    expect(container.querySelector('.md-code-toggle')).toBeNull()
  })

  it('reports the line count', () => {
    const { container } = render(<Markdown source={`\`\`\`python\n${LONG}\n\`\`\``} />)

    expect(container.querySelector('.md-code-meta')?.textContent).toBe('30 lines')
  })

  it('collapses a long python_exec argument too', async () => {
    const user = userEvent.setup()
    const { container } = render(
      <ToolCard
        entry={entry({
          arguments: { code: Array.from({ length: 40 }, (_, i) => `x_${i}`).join('\n') },
        })}
        onToggle={() => {}}
      />,
    )

    expect(container.querySelector('.md-code-toggle')).not.toBeNull()

    await user.click(container.querySelector('.md-code-toggle') as HTMLButtonElement)

    expect(container.querySelector('pre')).toBeNull()
  })
})

describe('markdown line breaks', () => {
  it('renders a single newline as a line break', () => {
    const { container } = render(<Markdown source={'first\nsecond'} />)

    expect(container.querySelector('br')).not.toBeNull()
    expect(container.querySelectorAll('br')).toHaveLength(1)
  })

  it('does not add a break for a blank-line separated paragraph', () => {
    const { container } = render(<Markdown source={'first\n\nsecond'} />)

    expect(container.querySelectorAll('p')).toHaveLength(2)
    expect(container.querySelector('br')).toBeNull()
  })

  it('keeps formatting on both lines', () => {
    const { container } = render(<Markdown source={'**a**\n`b`'} />)

    expect(container.querySelector('strong')?.textContent).toBe('a')
    expect(container.querySelector('code')?.textContent).toBe('b')
  })
})

describe('markdown block additions', () => {
  it('renders a rule', () => {
    const { container } = render(<Markdown source={'above\n\n---\n\nbelow'} />)

    expect(container.querySelectorAll('hr')).toHaveLength(1)
  })

  it('renders a nested list', () => {
    const { container } = render(<Markdown source={'- a\n  - b\n- c'} />)

    expect(container.querySelectorAll('ul')).toHaveLength(2)

    const nested = container.querySelectorAll('ul')[1]

    expect(nested?.querySelector('li')?.textContent).toBe('b')
    // The sublist lives inside the parent item, not beside it.
    expect(nested?.closest('li')?.textContent).toBe('ab')
  })

  it('renders a task list with markers', () => {
    const { container } = render(<Markdown source={'- [ ] todo\n- [x] done'} />)
    const items = container.querySelectorAll('li')

    expect(items[0].getAttribute('data-task')).toBe('true')
    expect(container.querySelector('.md-task[data-checked]')?.textContent).toBe('✓')
    expect(container.querySelectorAll('.md-task')).toHaveLength(2)
    expect(container.querySelector('.md-task')?.textContent).toBe('○')
  })

  it('does not mark a plain item as a task', () => {
    const { container } = render(<Markdown source={'- plain'} />)

    expect(container.querySelector('li')?.hasAttribute('data-task')).toBe(false)
  })

  it('renders strikethrough', () => {
    const { container } = render(<Markdown source={'keep ~~drop~~'} />)

    expect(container.querySelector('del')?.textContent).toBe('drop')
  })

  it('renders triple emphasis', () => {
    const { container } = render(<Markdown source={'***both***'} />)

    expect(container.querySelector('em')?.textContent).toBe('both')
  })

  it('renders a bare url as a safe link', () => {
    const { container } = render(<Markdown source={'see https://x.dev/graphiql'} />)
    const link = container.querySelector('a') as HTMLAnchorElement

    expect(link.getAttribute('href')).toBe('https://x.dev/graphiql')
    expect(link.getAttribute('rel')).toContain('noopener')
    expect(link.textContent).toBe('https://x.dev/graphiql')
  })

  it('keeps a trailing full stop out of the link', () => {
    const { container } = render(<Markdown source={'read https://x.dev.'} />)

    expect(container.querySelector('a')?.getAttribute('href')).toBe('https://x.dev')
    expect(container.querySelector('.md-paragraph')?.textContent).toBe(
      'read https://x.dev.',
    )
  })

  it('starts an ordered list at its first number', () => {
    const { container } = render(<Markdown source={'3. a\n4. b'} />)

    expect(container.querySelector('ol')?.getAttribute('start')).toBe('3')
  })

  it('renders a setext heading', () => {
    const { container } = render(<Markdown source={'Title\n==='} />)

    // Headings shift down two levels, so a level 1 becomes an h4.
    expect(container.querySelector('h4')?.textContent).toBe('Title')

    const { container: second } = render(<Markdown source={'Title\n---'} />)

    expect(second.querySelector('h5')?.textContent).toBe('Title')
  })
})

describe('code block display', () => {
  const LONG = Array.from({ length: 20 }, (_, i) => `line_${i + 1}`).join('\n')

  it('offers no wrap toggle, because code always wraps now', () => {
    // Wrapping is the only mode: a nowrap mode can only scroll
    // horizontally, and that scroll container took the wheel away
    // from the chat.
    const { container } = render(<Markdown source={`\`\`\`python\n${LONG}\n\`\`\``} />)

    expect(container.querySelector('.md-code-wrap')?.hasAttribute('data-wrap')).toBe(
      false,
    )
    expect(screen.queryByRole('button', { name: /wrap long lines/i })).toBeNull()
  })

  it('has no line-number gutter, which cannot align with wrapped lines', () => {
    const { container } = render(<Markdown source={`\`\`\`python\n${LONG}\n\`\`\``} />)

    expect(container.querySelector('.md-code-gutter')).toBeNull()
  })

  it('keeps the code text intact', () => {
    const { container } = render(<Markdown source={`\`\`\`python\n${LONG}\n\`\`\``} />)

    expect(container.querySelector('pre')?.textContent).toBe(LONG)
  })
})



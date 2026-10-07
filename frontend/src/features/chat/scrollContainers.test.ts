import { describe, expect, it } from 'vitest'

/**
 * Only the chat may scroll.
 *
 * This is a source-level guard on purpose. A nested `overflow: auto`
 * cannot be detected from the DOM here, because jsdom never applies
 * the stylesheet, and the bug it causes is invisible in tests: the
 * wheel is spent on the inner element, which looks like the chat
 * scroll being broken rather than like a scroll container existing at
 * all.
 *
 * The subtlety worth recording: `overflow-x: auto` with the default
 * `overflow-y: visible` computes overflow-y to `auto`. A code block
 * asking only for horizontal scrolling still becomes a two-axis scroll
 * container and swallows vertical wheel events.
 */

/*
 * Read through node:fs rather than a `?raw` import: vitest replaces CSS
 * modules with an empty stub, so `?raw` yields "" and the assertions
 * below would pass against nothing.
 *
 * This project has no Node types, so the import and the path helper are
 * declared locally rather than pulling in @types/node for one test.
 */
interface FileSystem {
  readFileSync: (path: string, encoding: 'utf8') => string
}

interface MetaWithDirname {
  dirname?: string
}

const { readFileSync } = (await import(
  // @ts-expect-error - node:fs has no types in this DOM-typed project
  'node:fs'
)) as unknown as FileSystem

// src/features/chat -> src
const src = ((import.meta as MetaWithDirname).dirname ?? 'src/features/chat').replace(
  /features\/chat$/,
  '',
)

const stylesheets = ['App.css', 'features/markdown/Markdown.css', 'components/primitives.css'].map(
  (name) => ({ name, css: readFileSync(`${src}/${name}`, 'utf8') }),
)

/** Rules that legitimately scroll. */
const allowed = new Set([
  // The chat itself.
  '.transcript-viewport',
  '.chat-main',
  '.chat-side',
  // Command palette overlay, not chat content.
  '.palette-list',
])

function rulesOf(css: string): { selector: string; body: string }[] {
  const found: { selector: string; body: string }[] = []

  for (const match of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    found.push({
      selector: match[1].split('\n').pop()?.trim() ?? '',
      body: match[2] ?? '',
    })
  }

  return found
}

describe('no nested scroll containers in the transcript', () => {
  it('reads the stylesheets it guards', () => {
    // Guards against the checks silently passing on an empty string.
    expect(stylesheets.length).toBe(3)

    for (const sheet of stylesheets) {
      expect(sheet.css.length, `${sheet.name} was empty`).toBeGreaterThan(500)
    }
  })

  for (const sheet of stylesheets) {
    it(`${sheet.name} declares no scroll container`, () => {
      const offenders: string[] = []

      for (const rule of rulesOf(sheet.css)) {
        const scrolls =
          /(?:^|[;\s])overflow(?:-[xy])?\s*:\s*(?:auto|scroll)/.test(rule.body)

        if (!scrolls) {
          continue
        }

        for (const selector of rule.selector.split(',').map((part) => part.trim())) {
          if (!allowed.has(selector)) {
            offenders.push(`${selector} { ${rule.body.trim()} }`)
          }
        }
      }

      expect(offenders).toEqual([])
    })
  }

  it('lets code blocks wrap instead of scrolling', () => {
    const css = stylesheets[1].css

    expect(css).toMatch(/\.md-code \{[^}]*white-space: pre-wrap/)
  })

  it('does not clip the code block body with overflow: hidden', () => {
    const pre = /\.md-code \{([^}]*)\}/.exec(stylesheets[1].css)?.[1] ?? ''

    expect(pre).not.toMatch(/overflow\s*:\s*hidden/)
  })
})

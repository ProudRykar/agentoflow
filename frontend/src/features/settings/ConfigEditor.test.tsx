import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { SettingsFileInfo } from '../../api/types'
import { ConfigEditor } from './ConfigEditor'

const CONFIG: SettingsFileInfo = {
  name: 'config.toml',
  path: '/tmp/config.toml',
  exists: true,
  data: {
    agent: { max_iterations: 10 },
    llm: { provider: 'ollama', model: 'gemma4:12b', timeout: 600 },
  },
  text: '[agent]\nmax_iterations = 10\n',
}

function stubFetch(
  overrides: Partial<Record<string, unknown>> = {},
) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)

    const body = url.includes('/settings/')
      ? { ...CONFIG, ...overrides }
      : {}

    return {
      ok: true,
      status: 200,
      json: async () => body,
    } as Response
  })
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('ConfigEditor', () => {
  it('renders the form instead of a loading placeholder', async () => {
    // Regression: load() used to run only in raw mode while the
    // loading guard covered the whole component, so the default
    // view was stuck on "Loading configuration…" forever.
    vi.stubGlobal('fetch', stubFetch())

    render(
      <ConfigEditor onSaved={() => {}} onError={() => {}} />,
    )

    expect(
      await screen.findByRole('button', { name: 'Fields' }),
    ).toBeInTheDocument()

    // The form replaces the placeholder once the document arrives;
    // it must not stay stuck on the loading state.
    expect(await screen.findByText('Agent')).toBeInTheDocument()

    expect(
      screen.queryByText('Loading configuration…'),
    ).not.toBeInTheDocument()
  })

  it('populates fields from the loaded document', async () => {
    vi.stubGlobal('fetch', stubFetch())

    render(
      <ConfigEditor onSaved={() => {}} onError={() => {}} />,
    )

    const iterations = await screen.findByDisplayValue(
      '10',
    )

    expect(iterations).toBeInTheDocument()

    expect(screen.getByDisplayValue('gemma4:12b')).toBeInTheDocument()
    expect(screen.getByDisplayValue('ollama')).toBeInTheDocument()
  })

  it('shows the raw editor only in raw mode', async () => {
    vi.stubGlobal('fetch', stubFetch())

    render(
      <ConfigEditor onSaved={() => {}} onError={() => {}} />,
    )

    const raw = await screen.findByRole('button', {
      name: 'Raw TOML',
    })

    expect(
      document.querySelector('.settings-textarea'),
    ).toBeNull()

    raw.click()

    await waitFor(() =>
      expect(
        document.querySelector('.settings-textarea'),
      ).not.toBeNull(),
    )
  })

  it('surfaces a load failure instead of spinning', async () => {
    const messages: string[] = []

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('network down')
      }),
    )

    render(
      <ConfigEditor
        onSaved={() => {}}
        onError={(cause) =>
          messages.push(
            cause instanceof Error ? cause.message : 'failed',
          )
        }
      />,
    )

    await waitFor(() =>
      expect(
        screen.getByText('Could not load configuration.'),
      ).toBeInTheDocument(),
    )

    expect(messages[0]).toBe('network down')

    // Must not spin forever on failure.
    expect(screen.queryByText('Loading…')).not.toBeInTheDocument()
    expect(
      screen.queryByText('Loading configuration…'),
    ).not.toBeInTheDocument()

    // Exactly one attempt: an unstable onError must not re-trigger
    // the load effect.
    expect(messages).toHaveLength(1)
  })

  it('offers a retry that recovers', async () => {
    const attempts = { count: 0 }
    const messages: string[] = []

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        attempts.count += 1

        if (attempts.count === 1) {
          throw new Error('network down')
        }

        return {
          ok: true,
          status: 200,
          json: async () => CONFIG,
        } as Response
      }),
    )

    render(
      <ConfigEditor
        onSaved={() => {}}
        onError={(cause) =>
          messages.push(
            cause instanceof Error ? cause.message : 'failed',
          )
        }
      />,
    )

    const retry = await screen.findByRole('button', {
      name: 'Retry',
    })

    retry.click()

    expect(await screen.findByText('Agent')).toBeInTheDocument()
    expect(attempts.count).toBe(2)
  })
})

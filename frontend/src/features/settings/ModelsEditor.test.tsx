import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ModelsEditor } from './ModelsEditor'

const CATALOG = [
  {
    name: 'gemma4:12b',
    description: 'local model',
    requirements: { context_size: 4096, thinking: true },
    capabilities: { code: 4, vision: 2 },
  },
]

function stubFetch() {
  return vi.fn(
    async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)

    if (url.includes('/settings/models/list')) {
      return {
        ok: true,
        status: 200,
        json: async () => CATALOG,
      } as Response
    }

    if (url.includes('/settings/models/entry') && init?.method === 'PUT') {
      return {
        ok: true,
        status: 200,
        json: async () => ({}),
      } as Response
    }

      return {
        ok: true,
        status: 200,
        json: async () => ({}),
      } as Response
    },
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('ModelsEditor', () => {
  it('edits an existing capability name', async () => {
    // Regression: the name input was readOnly because the name was
    // the object key, so no word could ever be typed.
    vi.stubGlobal('fetch', stubFetch())

    const user = userEvent.setup()

    render(<ModelsEditor onSaved={() => {}} onError={() => {}} />)

    await screen.findByText('gemma4:12b')

    // Capability rows live inside the draft editor.
    await user.click(
      screen.getByRole('button', { name: 'Edit gemma4:12b' }),
    )

    const name = screen.getByDisplayValue('code')

    expect(name).not.toHaveAttribute('readonly')

    await user.clear(name)
    await user.type(name, 'reasoning')

    expect(
      (name as HTMLInputElement).value,
    ).toBe('reasoning')
  })

  it('keeps focus while typing a capability name', async () => {
    vi.stubGlobal('fetch', stubFetch())

    const user = userEvent.setup()

    render(<ModelsEditor onSaved={() => {}} onError={() => {}} />)

    await screen.findByText('gemma4:12b')

    // Capability rows live inside the draft editor.
    await user.click(
      screen.getByRole('button', { name: 'Edit gemma4:12b' }),
    )

    const name = screen.getByDisplayValue('code')

    await user.clear(name)
    await user.type(name, 'abc')

    // Every keystroke changes the row key; focus must survive.
    expect(name).toHaveFocus()
    expect((name as HTMLInputElement).value).toBe('abc')
  })

  it('adds a new capability row and lets the name be typed', async () => {
    vi.stubGlobal('fetch', stubFetch())

    const user = userEvent.setup()

    render(<ModelsEditor onSaved={() => {}} onError={() => {}} />)

    await screen.findByText('gemma4:12b')

    // Capability rows live inside the draft editor.
    await user.click(
      screen.getByRole('button', { name: 'Edit gemma4:12b' }),
    )

    await user.click(
      screen.getByRole('button', { name: 'Add capability' }),
    )

    const names = screen.getAllByLabelText('Capability name')

    // 2 existing + 1 new
    expect(names).toHaveLength(3)

    const fresh = names[names.length - 1] as HTMLInputElement

    await user.type(fresh, 'tools')

    expect(fresh.value).toBe('tools')
  })

  it('adds two rows instead of collapsing them', async () => {
    // Regression: rows were keyed by name, so a second empty row
    // overwrote the first.
    vi.stubGlobal('fetch', stubFetch())

    const user = userEvent.setup()

    render(<ModelsEditor onSaved={() => {}} onError={() => {}} />)

    await screen.findByText('gemma4:12b')

    // Capability rows live inside the draft editor.
    await user.click(
      screen.getByRole('button', { name: 'Edit gemma4:12b' }),
    )

    const add = screen.getByRole('button', {
      name: 'Add capability',
    })

    await user.click(add)
    await user.click(add)

    expect(
      screen.getAllByLabelText('Capability name'),
    ).toHaveLength(4)
  })

  it('removes a capability row', async () => {
    vi.stubGlobal('fetch', stubFetch())

    const user = userEvent.setup()

    render(<ModelsEditor onSaved={() => {}} onError={() => {}} />)

    await screen.findByText('gemma4:12b')

    // Capability rows live inside the draft editor.
    await user.click(
      screen.getByRole('button', { name: 'Edit gemma4:12b' }),
    )

    await user.click(screen.getByRole('button', { name: 'Remove code' }))

    await waitFor(() =>
      expect(
        screen.getAllByLabelText('Capability name'),
      ).toHaveLength(1),
    )
  })

  it('sends capability names and scores on save', async () => {
    const fetchMock = stubFetch()

    vi.stubGlobal('fetch', fetchMock)

    const user = userEvent.setup()

    render(<ModelsEditor onSaved={() => {}} onError={() => {}} />)

    await screen.findByText('gemma4:12b')

    // Capability rows live inside the draft editor.
    await user.click(
      screen.getByRole('button', { name: 'Edit gemma4:12b' }),
    )

    const name = screen.getByDisplayValue('vision')

    await user.clear(name)
    await user.type(name, 'tools')

    await user.click(
      screen.getByRole('button', { name: 'Save model' }),
    )

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([input, init]) =>
          String(input).includes('/settings/models/entry') &&
          init?.method === 'PUT',
      )

      expect(call).toBeTruthy()

      const body = JSON.parse(String(call?.[1]?.body))

      expect(body.capabilities).toMatchObject({
        code: 4,
        tools: 2,
      })
    })
  })
})
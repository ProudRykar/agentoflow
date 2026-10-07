import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ToolsPage } from './ToolsPage'

const EMPTY_MCP = {
  enabled: false,
  servers: [],
  permissions: [],
  total_tools: 0,
  connected_servers: 0,
}

const ONE_SERVER = {
  enabled: true,
  servers: [
    {
      name: 'stash',
      transport: 'stdio',
      command: 'npx',
      url: '',
      args: ['-y', 'stash-mcp'],
      enabled: true,
      state: 'connected',
      error: null,
      server_version: '1.0.0',
      protocol_version: '2025-06-18',
      instructions: null,
      tools: [],
      connected_at: null,
      startup_seconds: 0.1,
    },
  ],
  permissions: ['mcp.execute'],
  total_tools: 3,
  connected_servers: 1,
}

interface StubTool {
  name: string
  description: string
  source: string
  enabled?: boolean
  permissions: string[]
  missing_permissions: string[]
  requires_approval: boolean
  timeout: number
  max_output_size: number
  parameters: unknown[]
  required_parameters: string[]
  mcp_server: string | null
  stats: Record<string, unknown>
}

interface Stub {
  mcp: unknown
  calls: [string, string | undefined, string | undefined][]
  tools: {
    tools: StubTool[]
    summary?: Record<string, unknown>
    disabled?: string[]
  }
}

function stubTool(name: string, enabled?: boolean): StubTool {
  return {
    name,
    description: `the ${name} tool`,
    source: 'builtin',
    ...(enabled === undefined ? {} : { enabled }),
    permissions: [],
    missing_permissions: [],
    requires_approval: false,
    timeout: 1,
    max_output_size: 10,
    parameters: [],
    required_parameters: [],
    mcp_server: null,
    stats: {
      calls: 0,
      errors: 0,
      running: 0,
      average_duration: null,
      last_used: null,
      error_rate: 0,
    },
  }
}

function stubFetch(
  initial: unknown = EMPTY_MCP,
  failSave = false,
): Stub {
  const state: Stub = { mcp: initial, calls: [], tools: { tools: [] } }

  vi.stubGlobal(
    'fetch',
    vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        const method = init?.method ?? 'GET'

        state.calls.push([
          url,
          method,
          typeof init?.body === 'string' ? init.body : undefined,
        ])

        if (url.includes('/mcp/') && url.endsWith('/servers')) {
          if (failSave) {
            return {
              ok: false,
              status: 422,
              statusText: 'Unprocessable',
              json: async () => ({
                detail: 'A stdio server needs a command',
              }),
            } as Response
          }

          state.mcp = ONE_SERVER

          return {
            ok: true,
            status: 200,
            json: async () => state.mcp,
          } as Response
        }

        if (url.includes('/tools')) {
          state.tools = {
            summary: {
              total_calls: 0,
              total_successes: 0,
              total_failures: 0,
              error_rate: 0,
              registered_tools: state.tools.tools.length,
              tools_used: 0,
              total_errors: 0,
              approvals_requested: 0,
              approvals_denied: 0,
            },
            tools: state.tools.tools,
            disabled: state.tools.tools
              .filter((tool) => tool.enabled === false)
              .map((tool) => tool.name),
          }

          return {
            ok: true,
            status: 200,
            json: async () => state.tools,
          } as Response
        }

        if (url.includes('/skills')) {
          return {
            ok: true,
            status: 200,
            json: async () => [],
          } as Response
        }

        if (url.includes('/plugins')) {
          return {
            ok: true,
            status: 200,
            json: async () => [],
          } as Response
        }

        if (url.includes('/mcp/') && method === 'GET') {
          return {
            ok: true,
            status: 200,
            json: async () => state.mcp,
          } as Response
        }

        return {
          ok: true,
          status: 200,
          json: async () => ({}),
        } as Response
      },
    ),
  )

  return state
}

async function openMcpTab() {
  render(
    <ToolsPage sessionId="session-1" refreshKey={0} />,
  )

  await screen.findByRole('button', { name: /MCP/ })

  await userEvent.click(screen.getByRole('button', { name: /MCP/ }))
}

/** Tools is the default tab, so no click is needed. */
async function renderToolsTab(): Promise<void> {
  render(
    <ToolsPage sessionId="session-1" refreshKey={0} />,
  )

  await screen.findByRole('button', { name: /Tools/ })
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('ToolsPage MCP tab', () => {
  it('offers the add form while MCP is disabled', async () => {
    // Regression: the panel used to render only the instruction
    // "add an [mcp] section by hand", so the form that creates that
    // section was unreachable.
    stubFetch(EMPTY_MCP)

    await openMcpTab()

    expect(
      await screen.findByRole('button', { name: 'Add server' }),
    ).toBeInTheDocument()

    expect(
      screen.queryByText(/MCP is disabled/),
    ).not.toBeInTheDocument()
  })

  it('shows an enable action while MCP is off', async () => {
    stubFetch(EMPTY_MCP)

    await openMcpTab()

    expect(
      await screen.findByRole('button', { name: 'Enable MCP' }),
    ).toBeInTheDocument()
  })

  it('creates a stdio server from the form', async () => {
    const stub = stubFetch(EMPTY_MCP)

    await openMcpTab()

    await userEvent.click(
      await screen.findByRole('button', { name: 'Add server' }),
    )

    await userEvent.type(
      screen.getByLabelText('Server name'),
      'stash',
    )

    await userEvent.type(screen.getByLabelText('Command'), 'npx')

    await userEvent.type(
      screen.getByLabelText('Arguments'),
      '-y stash-mcp',
    )

    await userEvent.type(
      screen.getByLabelText('Environment'),
      'STASH_URL=http://127.0.0.1:9999',
    )

    await userEvent.click(
      screen.getByRole('button', { name: 'Save server' }),
    )

    await waitFor(() => {
      const call = stub.calls.find(
        ([url, method]) =>
          url.endsWith('/servers') && method === 'POST',
      )

      expect(call).toBeTruthy()
    })

    expect(await screen.findByText('stash')).toBeInTheDocument()
  })

  it('switches to a URL field for http transport', async () => {
    stubFetch(EMPTY_MCP)

    await openMcpTab()

    await userEvent.click(
      await screen.findByRole('button', { name: 'Add server' }),
    )

    await userEvent.selectOptions(
      screen.getByLabelText('Transport'),
      'http',
    )

    expect(screen.getByLabelText('URL')).toBeInTheDocument()
    expect(
      screen.queryByLabelText('Command'),
    ).not.toBeInTheDocument()
  })

  it('blocks save until the required fields are filled', async () => {
    stubFetch(EMPTY_MCP)

    await openMcpTab()

    await userEvent.click(
      await screen.findByRole('button', { name: 'Add server' }),
    )

    const save = screen.getByRole('button', {
      name: 'Save server',
    })

    expect(save).toBeDisabled()

    await userEvent.type(screen.getByLabelText('Server name'), 'x')

    expect(save).toBeDisabled()

    await userEvent.type(screen.getByLabelText('Command'), 'npx')

    expect(save).toBeEnabled()
  })

  it('surfaces a rejected server instead of closing the form', async () => {
    stubFetch(EMPTY_MCP, true)

    await openMcpTab()

    await userEvent.click(
      await screen.findByRole('button', { name: 'Add server' }),
    )

    await userEvent.type(screen.getByLabelText('Server name'), 'x')
    await userEvent.type(screen.getByLabelText('Command'), 'npx')

    await userEvent.click(
      screen.getByRole('button', { name: 'Save server' }),
    )

    expect(
      await screen.findByText(
        'A stdio server needs a command',
      ),
    ).toBeInTheDocument()

    // The form stays open so the user can correct the entry.
    expect(screen.getByLabelText('Command')).toBeInTheDocument()
  })

  it('lists configured servers with a remove action', async () => {
    stubFetch(ONE_SERVER)

    await openMcpTab()

    expect(await screen.findByText('stash')).toBeInTheDocument()

    expect(
      screen.getByRole('button', { name: 'Remove' }),
    ).toBeInTheDocument()
  })

  it('shows a configuration warning from the backend', async () => {
    stubFetch({
      ...ONE_SERVER,
      servers: [
        {
          ...ONE_SERVER.servers[0],
          warnings: [
            "Server 'podman' is a container runtime, but its " +
              'arguments pass no -e/--env flag, so ' +
              'STASH_ENDPOINT will not reach the container.',
          ],
        },
      ],
    })

    await openMcpTab()

    expect(
      await screen.findByText(/will not reach the container/),
    ).toBeInTheDocument()
  })

  it('renders a server whose response predates warnings', async () => {
    // An older backend omits the field entirely; the page must still
    // work instead of throwing on undefined.map.
    const legacy = {
      ...ONE_SERVER,
      servers: [{ ...ONE_SERVER.servers[0], warnings: undefined }],
    }

    stubFetch(legacy)

    await openMcpTab()

    expect(await screen.findByText('stash')).toBeInTheDocument()
  })

  it('warns when saving over an existing server', async () => {
    stubFetch(ONE_SERVER)

    await openMcpTab()

    await userEvent.click(
      await screen.findByRole('button', { name: 'Add server' }),
    )

    await userEvent.type(screen.getByLabelText('Server name'), 'stash')

    expect(
      await screen.findByText(/already exists/),
    ).toBeInTheDocument()
  })
})
describe('ToolsPage MCP edit', () => {
  it('opens a prefilled form from the Edit button', async () => {
    stubFetch(ONE_SERVER)

    await openMcpTab()

    await screen.findByText('stash')

    await userEvent.click(screen.getByRole('button', { name: 'Edit' }))

    expect(
      await screen.findByText('Edit MCP server'),
    ).toBeInTheDocument()

    expect(screen.getByLabelText('Command')).toHaveValue('npx')
    expect(screen.getByLabelText('Arguments')).toHaveValue(
      '-y stash-mcp',
    )
  })

  it('locks the name and transport while editing', async () => {
    stubFetch(ONE_SERVER)

    await openMcpTab()

    await screen.findByText('stash')

    await userEvent.click(screen.getByRole('button', { name: 'Edit' }))

    const name = await screen.findByLabelText('Server name')

    expect(name).toHaveValue('stash')
    expect(name).toHaveAttribute('readonly')
    expect(screen.getByLabelText('Transport')).toBeDisabled()
  })

  it('saves changed arguments', async () => {
    const stub = stubFetch(ONE_SERVER)

    await openMcpTab()

    await screen.findByText('stash')

    await userEvent.click(screen.getByRole('button', { name: 'Edit' }))

    const args = await screen.findByLabelText('Arguments')

    await userEvent.clear(args)
    await userEvent.type(args, 'run --rm stash-mcp')

    await userEvent.click(
      screen.getByRole('button', { name: 'Save changes' }),
    )

    await waitFor(() => {
      const call = stub.calls.find(
        ([url, method]) =>
          url.endsWith('/servers') && method === 'POST',
      )

      expect(call).toBeTruthy()
      expect(JSON.parse(call?.[2] ?? '{}')).toMatchObject({
        name: 'stash',
        command: 'npx',
        args: ['run', '--rm', 'stash-mcp'],
      })
    })
  })

  it('omits env when untouched so stored values survive', async () => {
    const stub = stubFetch(ONE_SERVER)

    await openMcpTab()

    await screen.findByText('stash')

    await userEvent.click(screen.getByRole('button', { name: 'Edit' }))

    await screen.findByLabelText('Command')

    await userEvent.click(
      screen.getByRole('button', { name: 'Save changes' }),
    )

    await waitFor(() => {
      const call = stub.calls.find(
        ([url, method]) =>
          url.endsWith('/servers') && method === 'POST',
      )

      expect(call).toBeTruthy()

      const body = JSON.parse(call?.[2] ?? '{}')

      // Absent, not {}: the backend preserves stored secrets.
      expect('env' in body).toBe(false)
    })
  })

  it('sends env when the user edits it', async () => {
    const stub = stubFetch(ONE_SERVER)

    await openMcpTab()

    await screen.findByText('stash')

    await userEvent.click(screen.getByRole('button', { name: 'Edit' }))

    await userEvent.type(
      await screen.findByLabelText('Environment'),
      'STASH_URL=http://127.0.0.1:9999',
    )

    await userEvent.click(
      screen.getByRole('button', { name: 'Save changes' }),
    )

    await waitFor(() => {
      const call = stub.calls.find(
        ([url, method]) =>
          url.endsWith('/servers') && method === 'POST',
      )

      expect(JSON.parse(call?.[2] ?? '{}').env).toEqual({
        STASH_URL: 'http://127.0.0.1:9999',
      })
    })
  })

  it('warns that env is hidden while editing', async () => {
    stubFetch(ONE_SERVER)

    await openMcpTab()

    await screen.findByText('stash')

    await userEvent.click(screen.getByRole('button', { name: 'Edit' }))

    expect(
      await screen.findByText(/never sent to the browser/),
    ).toBeInTheDocument()
  })

  it('closes the form on cancel', async () => {
    stubFetch(ONE_SERVER)

    await openMcpTab()

    await screen.findByText('stash')

    await userEvent.click(screen.getByRole('button', { name: 'Edit' }))

    await screen.findByText('Edit MCP server')

    await userEvent.click(
      screen.getByRole('button', { name: 'Cancel' }),
    )

    expect(
      screen.queryByText('Edit MCP server'),
    ).not.toBeInTheDocument()

    expect(
      screen.getByRole('button', { name: 'Add server' }),
    ).toBeInTheDocument()
  })
})

describe('ToolsPage tool switching', () => {
  it('shows a disabled tool as off and unchecked', async () => {
    const stub = stubFetch()

    stub.tools.tools = [stubTool('alpha', false)]

    await renderToolsTab()

    await screen.findByText('alpha')

    const toggle = screen.getByLabelText('Enable alpha')

    expect(toggle).not.toBeChecked()

    // The chip marks the row; the switch label repeats the word, so
    // the assertion is scoped rather than a bare text match.
    expect(
      document.querySelector('.chip-off')?.textContent,
    ).toBe('off')

    expect(screen.getByText(/1 tool off/i)).toBeTruthy()
  })

  it('treats a missing enabled field as on', async () => {
    // An older backend omits the field; absent must mean enabled
    // rather than throwing on undefined.length.
    const stub = stubFetch()

    stub.tools.tools = [stubTool('legacy')]

    await renderToolsTab()

    await screen.findByText('legacy')

    expect(screen.getByLabelText('Enable legacy')).toBeChecked()
  })

  it('posts the toggle with an encoded tool name', async () => {
    const stub = stubFetch()

    stub.tools.tools = [stubTool('mcp_stash_get_tags', true)]

    await renderToolsTab()

    await screen.findByText('mcp_stash_get_tags')

    await userEvent.click(
      screen.getByLabelText('Enable mcp_stash_get_tags'),
    )

    const post = stub.calls.find(
      ([, method]) => method === 'POST',
    )

    expect(post).toBeDefined()
    expect(post?.[0]).toContain(
      `/tools/${'mcp_stash_get_tags'}`,
    )
    expect(post?.[2]).toContain('"enabled":false')
  })
})

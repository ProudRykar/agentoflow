import { useCallback, useEffect, useState } from 'react'
import {
  deleteMcpServer,
  listMcp,
  listPlugins,
  listSkills,
  listTools,
  mcpReloadAll,
  mcpServerAction,
  saveMcpServer,
  setMcpEnabled,
  type MCPServerAction,
  type MCPServerInput,
} from '../../api/client'
import type {
  MCPServerInfo,
  MCPResponse,
  PluginInfo,
  SkillInfo,
  ToolInfo,
  ToolsResponse,
} from '../../api/types'
import { SkillEditor } from '../skills/SkillEditor'
import './ToolsPage.css'
import '../skills/SkillEditor.css'

type Tab = 'tools' | 'skills' | 'mcp'

interface ToolsPageProps {
  sessionId: string | null
  /** Bumped by the caller so stats refresh after each run. */
  refreshKey: unknown
}

export function ToolsPage({ sessionId, refreshKey }: ToolsPageProps) {
  const [tab, setTab] = useState<Tab>('tools')

  const [tools, setTools] = useState<ToolsResponse | null>(null)
  const [skills, setSkills] = useState<SkillInfo[]>([])
  const [plugins, setPlugins] = useState<PluginInfo[]>([])
  const [mcp, setMcp] = useState<MCPResponse | null>(null)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [busyServer, setBusyServer] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  const load = useCallback(async () => {
    if (!sessionId) {
      return
    }

    setLoading(true)
    setError(null)

    try {
      const [nextTools, nextSkills, nextPlugins, nextMcp] =
        await Promise.all([
          listTools(sessionId),
          listSkills(sessionId),
          listPlugins(sessionId),
          listMcp(sessionId),
        ])

      setTools(nextTools)
      setSkills(nextSkills)
      setPlugins(nextPlugins)
      setMcp(nextMcp)
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : 'load failed',
      )
    } finally {
      setLoading(false)
    }
  }, [sessionId])

  useEffect(() => {
    void load()
  }, [load, refreshKey])

  const handleServerAction = useCallback(
    async (server: string, action: MCPServerAction) => {
      if (!sessionId) {
        return
      }

      setBusyServer(server)
      setError(null)

      try {
        await mcpServerAction(sessionId, server, action)
        await load()
      } catch (cause) {
        setError(
          cause instanceof Error
            ? cause.message
            : 'action failed',
        )
      } finally {
        setBusyServer(null)
      }
    },
    [load, sessionId],
  )

  const handleSaveServer = useCallback(
    async (payload: MCPServerInput): Promise<boolean> => {
      if (!sessionId) {
        return false
      }

      setBusyServer('*')
      setError(null)

      try {
        await saveMcpServer(sessionId, payload)
        await load()

        return true
      } catch (cause) {
        setError(
          cause instanceof Error
            ? cause.message
            : 'could not save the server',
        )

        // The form keeps the entry when this is false, so a
        // rejected server never costs the user their input.
        return false
      } finally {
        setBusyServer(null)
      }
    },
    [load, sessionId],
  )

  const handleDeleteServer = useCallback(
    async (name: string) => {
      if (!sessionId) {
        return
      }

      setBusyServer(name)
      setError(null)

      try {
        await deleteMcpServer(sessionId, name)
        await load()
      } catch (cause) {
        setError(
          cause instanceof Error
            ? cause.message
            : 'could not remove the server',
        )
      } finally {
        setBusyServer(null)
      }
    },
    [load, sessionId],
  )

  const handleSetEnabled = useCallback(
    async (enabled: boolean) => {
      if (!sessionId) {
        return
      }

      setBusyServer('*')
      setError(null)

      try {
        await setMcpEnabled(sessionId, enabled)
        await load()
      } catch (cause) {
        setError(
          cause instanceof Error
            ? cause.message
            : 'could not change the MCP state',
        )
      } finally {
        setBusyServer(null)
      }
    },
    [load, sessionId],
  )

  const handleReloadAll = useCallback(async () => {
    if (!sessionId) {
      return
    }

    setBusyServer('*')
    setError(null)

    try {
      await mcpReloadAll(sessionId)
      await load()
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : 'reload failed',
      )
    } finally {
      setBusyServer(null)
    }
  }, [load, sessionId])

  if (!sessionId) {
    return (
      <div className="tools-empty">
        Select a session to inspect its tools, skills and MCP servers.
      </div>
    )
  }

  return (
    <div className="tools-page">
      <header className="tools-head">
        <nav className="tools-tabs" aria-label="Tool sections">
          <TabButton
            active={tab === 'tools'}
            onClick={() => setTab('tools')}
          >
            Tools
            {tools && (
              <span className="tab-count">
                {tools.tools.length}
              </span>
            )}
          </TabButton>

          <TabButton
            active={tab === 'skills'}
            onClick={() => setTab('skills')}
          >
            Skills
            <span className="tab-count">{skills.length}</span>
          </TabButton>

          <TabButton
            active={tab === 'mcp'}
            onClick={() => setTab('mcp')}
          >
            MCP
            {mcp && (
              <span className="tab-count">{mcp.total_tools}</span>
            )}
          </TabButton>
        </nav>

        <div className="tools-actions">
          <input
            className="tools-search"
            value={query}
            placeholder="Filter…"
            onChange={(event) => setQuery(event.target.value)}
          />

          <button
            type="button"
            className="btn"
            onClick={() => void load()}
            disabled={loading}
          >
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </header>

      {error && <div className="banner banner-error">{error}</div>}

      {notice && (
        <div
          className="banner banner-ok"
          onClick={() => setNotice(null)}
          role="status"
        >
          {notice}
        </div>
      )}

      {tab === 'tools' && (
        <ToolsTab
          data={tools}
          query={query}
          loading={loading}
        />
      )}

      {tab === 'skills' && (
        <SkillsTab
          skills={skills}
          plugins={plugins}
          query={query}
          onError={setError}
          onRefresh={load}
          onNotice={setNotice}
        />
      )}

      {tab === 'mcp' && (
        <McpTab
          data={mcp}
          query={query}
          busyServer={busyServer}
          onAct={handleServerAction}
          onReloadAll={handleReloadAll}
          onSave={handleSaveServer}
          onDelete={handleDeleteServer}
          onSetEnabled={handleSetEnabled}
        />
      )}
    </div>
  )
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      className={`tools-tab${active ? ' is-active' : ''}`}
      onClick={onClick}
    >
      {children}
    </button>
  )
}

// ======================================================================
// Tools
// ======================================================================


function ToolsTab({
  data,
  query,
  loading,
}: {
  data: ToolsResponse | null
  query: string
  loading: boolean
}) {
  if (!data) {
    return <p className="muted">{loading ? 'Loading…' : 'No data.'}</p>
  }

  const needle = query.trim().toLowerCase()

  const filtered = data.tools.filter((tool) =>
    needle
      ? tool.name.toLowerCase().includes(needle) ||
        tool.description.toLowerCase().includes(needle)
      : true,
  )

  const { summary } = data

  return (
    <div className="tools-body">
      <div className="stat-grid">
        <Stat label="Registered" value={summary.registered_tools} />
        <Stat label="Used" value={summary.tools_used} />
        <Stat label="Calls" value={summary.total_calls} />
        <Stat
          label="Errors"
          value={summary.total_errors}
          tone={summary.total_errors > 0 ? 'bad' : undefined}
        />
        <Stat
          label="Error rate"
          value={`${(summary.error_rate * 100).toFixed(1)}%`}
          tone={summary.error_rate > 0 ? 'warn' : undefined}
        />
        <Stat label="Approvals" value={summary.approvals_requested} />
        <Stat
          label="Denied"
          value={summary.approvals_denied}
          tone={summary.approvals_denied > 0 ? 'warn' : undefined}
        />
      </div>

      {filtered.length === 0 ? (
        <p className="muted">
          {data.tools.length === 0
            ? 'This session has no tools registered.'
            : 'Nothing matches the filter.'}
        </p>
      ) : (
        <table className="tool-table">
          <thead>
            <tr>
              <th>Tool</th>
              <th>Source</th>
              <th className="num">Calls</th>
              <th className="num">Errors</th>
              <th className="num">Avg</th>
              <th>Permissions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((tool) => (
              <ToolRow key={tool.name} tool={tool} />
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

function ToolRow({ tool }: { tool: ToolInfo }) {
  const [open, setOpen] = useState(false)

  const duration =
    tool.stats.average_duration === null
      ? '—'
      : tool.stats.average_duration < 1
        ? `${Math.round(tool.stats.average_duration * 1000)}ms`
        : `${tool.stats.average_duration.toFixed(2)}s`

  return (
    <>
      <tr
        className={`tool-row${
          tool.stats.errors > 0 ? ' has-error' : ''
        }`}
        onClick={() => setOpen(!open)}
      >
        <td>
          <div className="tool-cell-name">
            <span className="twisty" aria-hidden="true">
              {open ? '▾' : '▸'}
            </span>
            <code>{tool.name}</code>
            {tool.requires_approval && (
              <span
                className="chip chip-approval"
                title="Requires approval before running"
              >
                approval
              </span>
            )}
            {tool.stats.running > 0 && (
              <span className="chip chip-running">running</span>
            )}
          </div>
          <div className="tool-cell-desc">
            {tool.description || '—'}
          </div>
        </td>
        <td>
          <span className={`chip chip-${tool.source}`}>
            {tool.source}
          </span>
          {tool.mcp_server && (
            <div className="muted small">
              {tool.mcp_server}
            </div>
          )}
        </td>
        <td className="num">{tool.stats.calls}</td>
        <td
          className={`num${
            tool.stats.errors > 0 ? ' bad' : ''
          }`}
        >
          {tool.stats.errors}
          {tool.stats.last_error_code && (
            <div className="muted small">
              {tool.stats.last_error_code}
            </div>
          )}
        </td>
        <td className="num">{duration}</td>
        <td>
          {tool.permissions.length === 0 ? (
            <span className="muted">—</span>
          ) : (
            <div className="perm-list">
              {tool.permissions.map((permission) => (
                <span
                  key={permission}
                  className={`chip chip-perm${
                    tool.missing_permissions.includes(
                      permission,
                    )
                      ? ' is-missing'
                      : ''
                  }`}
                  title={
                    tool.missing_permissions.includes(
                      permission,
                    )
                      ? 'Not granted yet; approval required'
                      : permission
                  }
                >
                  {permission}
                </span>
              ))}
            </div>
          )}
        </td>
      </tr>

      {open && (
        <tr className="tool-detail-row">
          <td colSpan={6}>
            <ToolDetail tool={tool} />
          </td>
        </tr>
      )}
    </>
  )
}

function ToolDetail({ tool }: { tool: ToolInfo }) {
  return (
    <div className="tool-detail">
      <div className="detail-col">
        <h4>Parameters</h4>

        {tool.parameters.length === 0 ? (
          <p className="muted">Takes no arguments.</p>
        ) : (
          <ul className="param-list">
            {tool.parameters.map((parameter) => (
              <li key={parameter.name}>
                <code>{parameter.name}</code>
                <span className="muted">
                  {' '}
                  : {parameter.type}
                </span>
                {parameter.required && (
                  <span className="chip chip-required">
                    required
                  </span>
                )}
                {parameter.description && (
                  <div className="muted small">
                    {parameter.description}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="detail-col">
        <h4>Policy</h4>
        <dl className="detail-list">
          <dt>Timeout</dt>
          <dd>{tool.timeout}s</dd>
          <dt>Max output</dt>
          <dd>{tool.max_output_size} chars</dd>
          <dt>Approval</dt>
          <dd>
            {tool.requires_approval ? 'required' : 'not required'}
          </dd>
          <dt>Missing perms</dt>
          <dd>
            {tool.missing_permissions.length === 0
              ? '—'
              : tool.missing_permissions.join(', ')}
          </dd>
        </dl>

        {Object.keys(tool.stats.error_codes).length > 0 && (
          <>
            <h4>Errors</h4>
            <ul className="param-list">
              {Object.entries(tool.stats.error_codes).map(
                ([code, count]) => (
                  <li key={code}>
                    <code>{code}</code>
                    <span className="muted"> ×{count}</span>
                  </li>
                ),
              )}
            </ul>
          </>
        )}
      </div>
    </div>
  )
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string
  value: number | string
  tone?: 'bad' | 'warn'
}) {
  return (
    <div className={`stat${tone ? ` stat-${tone}` : ''}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

// ======================================================================
// Skills
// ======================================================================


function SkillsTab({
  skills,
  plugins,
  query,
  onError,
  onRefresh,
  onNotice,
}: {
  skills: SkillInfo[]
  plugins: PluginInfo[]
  query: string
  onError: (message: string | null) => void
  onRefresh: () => Promise<void>
  onNotice: (message: string) => void
}) {
  const needle = query.trim().toLowerCase()

  const filtered = skills.filter((skill) =>
    needle
      ? skill.name.toLowerCase().includes(needle) ||
        skill.description.toLowerCase().includes(needle)
      : true,
  )

  const used = skills.filter((skill) => skill.used)
  const active = skills.filter((skill) => skill.active)

  return (
    <div className="tools-body">
      <div className="stat-grid">
        <Stat label="Available" value={skills.length} />
        <Stat label="Active" value={active.length} />
        <Stat label="Used" value={used.length} />
        <Stat label="Plugins" value={plugins.length} />
      </div>

      {filtered.length === 0 ? (
        <p className="muted">
          {skills.length === 0
            ? 'No skills available for this session.'
            : 'Nothing matches the filter.'}
        </p>
      ) : (
        <ul className="card-list">
          {filtered.map((skill) => (
            <li key={skill.name} className="card">
              <div className="card-head">
                <code>{skill.name}</code>
                <span className="muted small">
                  v{skill.version}
                </span>
                {skill.used && (
                  <span className="chip chip-on">used</span>
                )}
                {skill.active && !skill.used && (
                  <span className="chip chip-on">active</span>
                )}
                {skill.loaded && !skill.active && (
                  <span className="chip">loaded</span>
                )}
              </div>
              <p className="card-desc">
                {skill.description || '—'}
              </p>
            </li>
          ))}
        </ul>
      )}

      <div className="section-split">
        <h3 className="section-title">Skill library</h3>
        <SkillEditor
          onError={(cause) =>
            onError(
              cause instanceof Error
                ? cause.message
                : 'skill operation failed',
            )
          }
          onSaved={(message) => {
            onError(null)
            onNotice(message)
            void onRefresh()
          }}
        />
      </div>

      {plugins.length > 0 && (
        <>
          <h3 className="section-title">Plugins</h3>
          <ul className="card-list">
            {plugins.map((plugin) => (
              <li key={plugin.name} className="card">
                <div className="card-head">
                  <code>{plugin.name}</code>
                  <span className="muted small">
                    v{plugin.version}
                  </span>
                  <span
                    className={`chip ${
                      plugin.error
                        ? 'chip-bad'
                        : 'chip-on'
                    }`}
                  >
                    {plugin.status}
                  </span>
                </div>
                {plugin.description && (
                  <p className="card-desc">
                    {plugin.description}
                  </p>
                )}
                {plugin.error && (
                  <p className="card-error">{plugin.error}</p>
                )}
                <div className="card-counts">
                  <span>{plugin.tools.length} tools</span>
                  <span>{plugin.skills.length} skills</span>
                  {plugin.capabilities.length > 0 && (
                    <span>
                      {plugin.capabilities.length} capabilities
                    </span>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  )
}

// ======================================================================
// MCP
// ======================================================================


function McpTab({
  data,
  query,
  busyServer,
  onAct,
  onReloadAll,
  onSave,
  onDelete,
  onSetEnabled,
}: {
  data: MCPResponse | null
  query: string
  busyServer: string | null
  onAct: (
    server: string,
    action: MCPServerAction,
  ) => Promise<void>
  onReloadAll: () => Promise<void>
  onSave: (payload: MCPServerInput) => Promise<boolean>
  onDelete: (name: string) => Promise<void>
  onSetEnabled: (enabled: boolean) => Promise<void>
}) {
  const needle = query.trim().toLowerCase()
  const [editing, setEditing] = useState<MCPServerInfo | null>(null)

  if (!data) {
    return <p className="muted">Loading…</p>
  }

  // The panel must render even when MCP is off: this is the UI that
  // creates the [mcp] section, so gating it on that section existing
  // left no way to add the first server without editing TOML.
  const disabled = !data.enabled

  const servers = data.servers.filter((server) =>
    needle
      ? server.name.toLowerCase().includes(needle) ||
        server.command.toLowerCase().includes(needle)
      : true,
  )

  return (
    <div className="tools-body">
      {disabled && (
        <div className="mcp-banner">
          <p>
            MCP is off. Add a server below, or turn it on to use the
            servers already in your config.
          </p>

          <button
            type="button"
            className="btn"
            onClick={() => void onSetEnabled(true)}
            disabled={busyServer !== null}
          >
            {busyServer === '*' ? 'Enabling…' : 'Enable MCP'}
          </button>
        </div>
      )}

      <div className="stat-grid">
        <Stat label="Servers" value={data.servers.length} />
        <Stat label="Connected" value={data.connected_servers} />
        <Stat
          label="Failed"
          value={
            data.servers.length - data.connected_servers
          }
          tone={
            data.servers.length > data.connected_servers
              ? 'bad'
              : undefined
          }
        />
        <Stat label="MCP tools" value={data.total_tools} />
      </div>

      <p className="muted small">
        Permissions granted to MCP tools:{' '}
        {data.permissions.join(', ') || '\u2014'}
      </p>

      <ServerForm
        busy={busyServer !== null}
        existing={data.servers.map((server) => server.name)}
        editing={editing}
        onSave={onSave}
        onCancel={() => setEditing(null)}
      />

      {data.enabled && (
        <div className="mcp-toolbar">
          <button
            type="button"
            className="btn"
            onClick={() => void onReloadAll()}
            disabled={busyServer !== null}
          >
            {busyServer === '*' ? 'Reloading\u2026' : 'Reload all'}
          </button>

          <button
            type="button"
            className="btn"
            onClick={() => void onSetEnabled(false)}
            disabled={busyServer !== null}
          >
            Disable MCP
          </button>

          <span className="muted small">
            Tools added at runtime become visible to the agent on
            its next run.
          </span>
        </div>
      )}

      {servers.length === 0 ? (
        <p className="muted">
          {data.servers.length === 0
            ? 'No servers configured yet.'
            : 'No servers match the filter.'}
        </p>
      ) : (
        <ul className="card-list">
          {servers.map((server) => (
            <ServerCard
              key={server.name}
              server={server}
              busy={busyServer === server.name}
              disabled={busyServer !== null}
              onAct={onAct}
              onDelete={onDelete}
              onEdit={(server) => setEditing(server)}
            />
          ))}
        </ul>
      )}
    </div>
  )
}

function ServerForm({
  busy,
  existing,
  editing,
  onSave,
  onCancel,
}: {
  busy: boolean
  existing: string[]
  editing: MCPServerInfo | null
  onSave: (payload: MCPServerInput) => Promise<boolean>
  onCancel: () => void
}) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const [transport, setTransport] = useState<'stdio' | 'http'>('stdio')
  const [command, setCommand] = useState('')
  const [args, setArgs] = useState('')
  const [url, setUrl] = useState('')
  const [env, setEnv] = useState('')
  // Stored env is never sent to the browser, so an untouched field
  // means "keep it" and only an explicit edit replaces it.
  const [envTouched, setEnvTouched] = useState(false)

  // Entering edit mode seeds the form from the stored definition.
  useEffect(() => {
    if (!editing) {
      return
    }

    setOpen(true)
    setName(editing.name)
    setTransport(
      editing.transport === 'http' ? 'http' : 'stdio',
    )
    setCommand(editing.command)
    setArgs(editing.args.join(' '))
    setUrl(editing.url)
    setEnv('')
    setEnvTouched(false)
  }, [editing])

  const reset = useCallback(() => {
    setOpen(false)
    setName('')
    setCommand('')
    setArgs('')
    setUrl('')
    setEnv('')
    setEnvTouched(false)
  }, [])

  const canSave =
    name.trim().length > 0 &&
    (transport === 'stdio'
      ? command.trim().length > 0
      : url.trim().length > 0)

  const submit = useCallback(async () => {
    if (!canSave) {
      return
    }

    const payload: MCPServerInput = {
      name: name.trim(),
      transport,
    }

    if (transport === 'stdio') {
      payload.command = command.trim()
      payload.args = args
        .split(' ')
        .map((item) => item.trim())
        .filter(Boolean)
    } else {
      payload.url = url.trim()
    }

    if (envTouched) {
      const pairs = env
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
          const index = line.indexOf('=')

          return index === -1
            ? null
            : ([
                line.slice(0, index).trim(),
                line.slice(index + 1).trim(),
              ] as const)
        })
        .filter((pair): pair is readonly [string, string] =>
          pair !== null,
        )

      // An emptied box clears the stored values on purpose.
      payload.env = Object.fromEntries(pairs)
    }

    const saved = await onSave(payload)

    if (!saved) {
      return
    }

    reset()
    onCancel()
  }, [
    args,
    canSave,
    command,
    env,
    envTouched,
    name,
    onCancel,
    onSave,
    reset,
    transport,
    url,
  ])

  if (!open) {
    return (
      <div className="mcp-toolbar">
        <button
          type="button"
          className="btn btn-send"
          onClick={() => setOpen(true)}
          disabled={busy}
        >
          Add server
        </button>
      </div>
    )
  }

  const editingExisting = editing !== null

  return (
    <fieldset className="fieldset">
      <legend>
        {editingExisting ? 'Edit MCP server' : 'New MCP server'}
      </legend>

      <label className="field">
        <span className="field-label">Name</span>
        <input
          value={name}
          placeholder="stash"
          aria-label="Server name"
          // The name is the server's identity: renaming here would
          // silently create a second server instead of editing this
          // one, so it stays fixed while editing.
          readOnly={editingExisting}
          title={
            editingExisting
              ? 'The name identifies the server and cannot be changed'
              : undefined
          }
          onChange={(event) => setName(event.target.value)}
        />
      </label>

      <label className="field">
        <span className="field-label">Transport</span>
        <select
          value={transport}
          aria-label="Transport"
          disabled={editingExisting}
          onChange={(event) =>
            setTransport(
              event.target.value === 'http' ? 'http' : 'stdio',
            )
          }
        >
          <option value="stdio">stdio</option>
          <option value="http">http</option>
        </select>
      </label>

      {transport === 'stdio' ? (
        <>
          <label className="field">
            <span className="field-label">Command</span>
            <input
              value={command}
              placeholder="npx"
              aria-label="Command"
              onChange={(event) => setCommand(event.target.value)}
            />
          </label>

          <label className="field">
            <span className="field-label">Arguments</span>
            <input
              value={args}
              placeholder="-y package-name"
              aria-label="Arguments"
              onChange={(event) => setArgs(event.target.value)}
            />
          </label>
        </>
      ) : (
        <label className="field">
          <span className="field-label">URL</span>
          <input
            value={url}
            placeholder="http://127.0.0.1:8931/mcp"
            aria-label="URL"
            onChange={(event) => setUrl(event.target.value)}
          />
        </label>
      )}

      <label className="field">
        <span className="field-label">
          Environment (one KEY=value per line)
        </span>
        <textarea
          className="settings-textarea"
          value={env}
          aria-label="Environment"
          placeholder={
            editingExisting
              ? 'unchanged — type to replace, clear to remove'
              : 'KEY=value'
          }
          onChange={(event) => {
            setEnv(event.target.value)
            setEnvTouched(true)
          }}
        />
      </label>

      {editingExisting ? (
        <p className="muted small">
          Environment values are stored on the server and never sent
          to the browser. Leave the field empty to keep them.
        </p>
      ) : existing.includes(name.trim()) ? (
        <p className="muted small">
          A server named &ldquo;{name.trim()}&rdquo; already exists.
          Saving replaces its definition.
        </p>
      ) : null}

      <div className="settings-actions settings-actions-end">
        <button
          type="button"
          className="btn"
          onClick={() => {
            reset()
            onCancel()
          }}
          disabled={busy}
        >
          Cancel
        </button>

        <button
          type="button"
          className="btn btn-send"
          onClick={() => void submit()}
          disabled={busy || !canSave}
        >
          {busy
            ? 'Saving…'
            : editingExisting
              ? 'Save changes'
              : 'Save server'}
        </button>
      </div>
    </fieldset>
  )
}

function ServerCard({
  server,
  busy,
  disabled,
  onAct,
  onDelete,
  onEdit,
}: {
  server: MCPServerInfo
  busy: boolean
  disabled: boolean
  onAct: (
    server: string,
    action: MCPServerAction,
  ) => Promise<void>
  onDelete: (name: string) => Promise<void>
  onEdit: (server: MCPServerInfo) => void
}) {
  const connected = server.state === 'connected'

  return (
    <li className="card">
      <div className="card-head">
        <code>{server.name}</code>
        <span
          className={`chip ${
            server.state === 'connected'
              ? 'chip-on'
              : server.state === 'failed'
                ? 'chip-bad'
                : ''
          }`}
        >
          {server.state}
        </span>
        {server.enabled === false && (
          <span className="chip">disabled</span>
        )}
        <span className="chip">{server.transport}</span>
        {server.server_version && (
          <span className="muted small">
            v{server.server_version}
          </span>
        )}
        {server.protocol_version && (
          <span className="muted small">
            {server.protocol_version}
          </span>
        )}
      </div>

      <div className="card-desc">
        <code className="cmd">
          {server.transport === 'http'
            ? server.url
            : `${server.command}${
                server.args.length > 0
                  ? ` ${server.args.join(' ')}`
                  : ''
              }`}
        </code>
      </div>

      {server.error && (
        <p className="card-error">{server.error}</p>
      )}

      <div className="server-actions">
        {connected ? (
          <>
            <button
              type="button"
              className="btn btn-cancel"
              onClick={() =>
                void onAct(server.name, 'disconnect')
              }
              disabled={disabled || busy}
            >
              Disconnect
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => void onAct(server.name, 'reload')}
              disabled={disabled || busy}
            >
              Reload
            </button>
          </>
        ) : (
          <button
            type="button"
            className="btn btn-send"
            onClick={() => void onAct(server.name, 'connect')}
            disabled={disabled || busy || !server.enabled}
          >
            {busy ? 'Connecting…' : 'Connect'}
          </button>
        )}

        <button
          type="button"
          className="btn"
          onClick={() => onEdit(server)}
          disabled={disabled || busy}
        >
          Edit
        </button>

        <button
          type="button"
          className="btn btn-cancel"
          onClick={() => void onDelete(server.name)}
          disabled={disabled || busy}
        >
          Remove
        </button>
      </div>

      {server.instructions && (
        <p className="card-note">{server.instructions}</p>
      )}

      {server.tools.length > 0 && (
        <ul className="mcp-tool-list">
          {server.tools.map((tool) => (
            <li key={tool.qualified_name}>
              <code>{tool.qualified_name}</code>
              <span className="muted small">
                {' '}
                ({tool.remote_name})
              </span>
              {tool.description && (
                <div className="card-desc">
                  {tool.description}
                </div>
              )}
              <div className="perm-list">
                {tool.parameters.map((parameter) => (
                  <span
                    key={parameter}
                    className={`chip chip-perm${
                      tool.required.includes(parameter)
                        ? ' is-required'
                        : ''
                    }`}
                  >
                    {parameter}
                  </span>
                ))}
                {tool.requires_approval && (
                  <span className="chip chip-approval">
                    approval
                  </span>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </li>
  )
}

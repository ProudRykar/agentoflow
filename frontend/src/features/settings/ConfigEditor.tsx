import { useCallback, useEffect, useRef, useState } from 'react'
import { readSettings, writeSettings } from '../../api/client'
import { Field as SharedField } from '../../components/Field'

const CONFIG_FILE = 'config.toml'

interface ConfigEditorProps {
  onSaved: (message: string) => void
  onError: (cause: unknown) => void
}

type Mode = 'form' | 'raw'

export function ConfigEditor({
  onSaved,
  onError,
}: ConfigEditorProps) {
  const [mode, setMode] = useState<Mode>('form')
  const [text, setText] = useState('')
  const [dirty, setDirty] = useState(false)
  const [document, setDocument] = useState<
    Record<string, unknown> | null
  >(null)
  const [loading, setLoading] = useState(true)
  const [failed, setFailed] = useState(false)
  const [saving, setSaving] = useState(false)

  // Hold the callbacks in refs so the load effect runs once. If
  // load() depended on them directly, a parent that passes a new
  // function each render would re-fetch in a loop.
  const errorRef = useRef(onError)
  const savedRef = useRef(onSaved)

  useEffect(() => {
    errorRef.current = onError
    savedRef.current = onSaved
  }, [onError, onSaved])

  const load = useCallback(async () => {
    setLoading(true)

    try {
      const loaded = await readSettings(CONFIG_FILE)

      setText(loaded.text)
      setDocument(loaded.data)
      setDirty(false)
      setFailed(false)
    } catch (cause) {
      setFailed(true)
      errorRef.current(cause)
    } finally {
      setLoading(false)
    }
  }, [])

  // Load once on mount so both modes have the current document.
  // Previously this only ran in raw mode, which left the default
  // form view stuck on its loading guard forever.
  useEffect(() => {
    void load()
  }, [load])

  const saveRaw = useCallback(async () => {
    setSaving(true)

    try {
      const document = await writeSettings(CONFIG_FILE, { text })

      setText(document.text)
      setDirty(false)
      savedRef.current('Configuration saved')
    } catch (cause) {
      errorRef.current(cause)
    } finally {
      setSaving(false)
    }
  }, [text])

  return (
    <div className="settings-panel">
      <div className="settings-toolbar">
        <div className="settings-modes">
          <button
            type="button"
            className={mode === 'form' ? 'is-active' : ''}
            onClick={() => setMode('form')}
          >
            Fields
          </button>
          <button
            type="button"
            className={mode === 'raw' ? 'is-active' : ''}
            onClick={() => setMode('raw')}
          >
            Raw TOML
          </button>
        </div>

        {mode === 'raw' && (
          <div className="settings-actions">
            <button
              type="button"
              className="btn"
              onClick={() => void load()}
              disabled={saving}
            >
              Reload
            </button>
            <button
              type="button"
              className="btn btn-send"
              onClick={() => void saveRaw()}
              disabled={saving || !dirty}
            >
              {saving ? 'Saving…' : 'Save'}
            </button>
          </div>
        )}
      </div>

      {mode === 'form' ? (
        failed && !document ? (
          <div className="settings-panel">
            <p className="error">Could not load configuration.</p>

            <button
              type="button"
              className="btn"
              onClick={() => void load()}
            >
              Retry
            </button>
          </div>
        ) : !document ? (
          <p className="muted">Loading configuration…</p>
        ) : (
          <ConfigFields
            document={document}
            onChange={setDocument}
            onSaved={onSaved}
            onError={onError}
          />
        )
      ) : loading ? (
        <p className="muted">Loading configuration…</p>
      ) : (
        <textarea
          className="settings-textarea"
          value={text}
          spellCheck={false}
          onChange={(event) => {
            setText(event.target.value)
            setDirty(true)
          }}
        />
      )}
    </div>
  )
}

interface FieldProps {
  label: string
  hint?: string
  /** Shown instead of the hint when the value is out of range. */
  warning?: string
  children: React.ReactNode
}

function Field({ label, hint, warning, children }: FieldProps) {
  return (
    <SharedField label={label} hint={hint} warning={warning}>
      {children}
    </SharedField>
  )
}

function ConfigFields({
  document,
  onChange,
  onSaved,
  onError,
}: {
  document: Record<string, unknown>
  onChange: (
    update:
      | Record<string, unknown>
      | ((
          previous: Record<string, unknown> | null,
        ) => Record<string, unknown>),
  ) => void
  onSaved: (message: string) => void
  onError: (cause: unknown) => void
}) {
  const [saving, setSaving] = useState(false)

  const errorRef = useRef(onError)
  const savedRef = useRef(onSaved)

  useEffect(() => {
    errorRef.current = onError
    savedRef.current = onSaved
  }, [onError, onSaved])

  const set = useCallback(
    (table: string, key: string, value: unknown) => {
      onChange((previous) => {
        const next = { ...(previous ?? {}) }

        const section = {
          ...((next[table] as Record<string, unknown>) ?? {}),
        }

        section[key] = value
        next[table] = section

        return next
      })
    },
    [onChange],
  )

  const save = useCallback(async () => {
    setSaving(true)

    try {
      const updated = await writeSettings(CONFIG_FILE, { data: document })

      // Keep the parent (and raw view) in sync with what was
      // actually persisted.
      onChange(updated.data)
      savedRef.current('Configuration saved')
    } catch (cause) {
      errorRef.current(cause)
    } finally {
      setSaving(false)
    }
  }, [document, onChange])

  const agent = (document.agent ?? {}) as Record<string, unknown>
  const llm = (document.llm ?? {}) as Record<string, unknown>
  const context = (document.context ?? {}) as Record<string, unknown>
  const memory = (document.memory ?? {}) as Record<string, unknown>
  const models = (document.models ?? {}) as Record<string, unknown>
  const subagent =
    (document.subagent ?? {}) as Record<string, unknown>

  const mcpEnabled = Boolean(document.mcp !== undefined)

  return (
    <div className="settings-panel">
      <fieldset className="fieldset">
        <legend>Agent</legend>

        <Field label="Max iterations">
          <input
            type="number"
            min={1}
            value={String(agent.max_iterations ?? 10)}
            onChange={(event) =>
              set(
                'agent',
                'max_iterations',
                Number(event.target.value),
              )
            }
          />
        </Field>
      </fieldset>

      <fieldset className="fieldset">
        <legend>LLM</legend>

        <Field label="Provider">
          <input
            value={String(llm.provider ?? 'ollama')}
            onChange={(event) =>
              set('llm', 'provider', event.target.value)
            }
          />
        </Field>

        <Field label="Model">
          <input
            value={String(llm.model ?? '')}
            onChange={(event) =>
              set('llm', 'model', event.target.value)
            }
          />
        </Field>

        <Field label="Timeout (seconds)">
          <input
            type="number"
            min={1}
            value={String(llm.timeout ?? 600)}
            onChange={(event) =>
              set('llm', 'timeout', Number(event.target.value))
            }
          />
        </Field>
      </fieldset>

      <fieldset className="fieldset">
        <legend>Context</legend>

        <Field
          label="Max messages"
          hint="Counts chat message objects, not turns: one assistant reply plus its tool results counts as several. Leave empty for no limit; the token budget still applies."
        >
          <input
            type="number"
            min={1}
            value={
              context.max_messages === null ||
              context.max_messages === undefined
                ? ''
                : String(context.max_messages)
            }
            onChange={(event) =>
              set(
                'context',
                'max_messages',
                event.target.value === ''
                  ? null
                  : Number(event.target.value),
              )
            }
          />
        </Field>

        <Field
          label="Token estimation divisor"
          hint="Characters per token. 4 fits prose, 2-3 fits code and JSON; lower means the agent trims earlier."
          warning={
            Number(context.token_estimation_divisor ?? 4) > 4
              ? 'Above 4 undercounts code and JSON, so the budget will overrun.'
              : undefined
          }
        >
          <input
            type="number"
            min={1}
            value={String(
              context.token_estimation_divisor ?? 4,
            )}
            onChange={(event) =>
              set(
                'context',
                'token_estimation_divisor',
                Number(event.target.value),
              )
            }
          />
        </Field>
      </fieldset>

      <fieldset className="fieldset">
        <legend>Memory</legend>

        <Field label="Database file">
          <input
            value={String(memory.database ?? 'memory.db')}
            onChange={(event) =>
              set('memory', 'database', event.target.value)
            }
          />
        </Field>
      </fieldset>

      <fieldset className="fieldset">
        <legend>Models &amp; subagents</legend>

        <Field label="Catalog file">
          <input
            value={String(models.catalog ?? 'models.toml')}
            onChange={(event) =>
              set('models', 'catalog', event.target.value)
            }
          />
        </Field>

        <Field label="VRAM budget (GB)" hint="Leave empty for auto">
          <input
            type="number"
            min={0}
            step="0.5"
            value={
              models.vram_budget_gb === undefined ||
              models.vram_budget_gb === null
                ? ''
                : String(models.vram_budget_gb)
            }
            onChange={(event) =>
              set(
                'models',
                'vram_budget_gb',
                event.target.value === ''
                  ? null
                  : Number(event.target.value),
              )
            }
          />
        </Field>

        <Field label="Single model mode">
          <input
            type="checkbox"
            checked={Boolean(models.single_model_mode)}
            onChange={(event) =>
              set(
                'models',
                'single_model_mode',
                event.target.checked,
              )
            }
          />
        </Field>

        <Field label="Subagent max iterations">
          <input
            type="number"
            min={1}
            value={String(subagent.max_iterations ?? 25)}
            onChange={(event) =>
              set(
                'subagent',
                'max_iterations',
                Number(event.target.value),
              )
            }
          />
        </Field>

        <Field label="Subagent escalation">
          <input
            type="checkbox"
            checked={Boolean(subagent.escalation)}
            onChange={(event) =>
              set(
                'subagent',
                'escalation',
                event.target.checked,
              )
            }
          />
        </Field>
      </fieldset>

      <fieldset className="fieldset">
        <legend>MCP</legend>

        {mcpEnabled ? (
          <>
            <Field label="Enabled">
              <input
                type="checkbox"
                checked={Boolean(
                  (document.mcp as Record<string, unknown>)
                    ?.enabled,
                )}
                onChange={(event) =>
                  set('mcp', 'enabled', event.target.checked)
                }
              />
            </Field>

            <Field
              label="Require approval"
              hint="Recommended: MCP tools run third-party code"
            >
              <input
                type="checkbox"
                checked={Boolean(
                  (document.mcp as Record<string, unknown>)
                    ?.require_approval,
                )}
                onChange={(event) =>
                  set(
                    'mcp',
                    'require_approval',
                    event.target.checked,
                  )
                }
              />
            </Field>

            <p className="muted small">
              Servers are listed on the Tools → MCP tab.
            </p>
          </>
        ) : (
          <p className="muted small">
            No <code>[mcp]</code> section yet. Add one in Raw TOML,
            or connect a server from the Tools tab.
          </p>
        )}
      </fieldset>

      <div className="settings-actions settings-actions-end">
        <button
          type="button"
          className="btn btn-send"
          onClick={() => void save()}
          disabled={saving}
        >
          {saving ? 'Saving…' : 'Save configuration'}
        </button>
      </div>
    </div>
  )
}

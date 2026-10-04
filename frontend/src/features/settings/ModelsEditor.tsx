import { useCallback, useEffect, useRef, useState } from 'react'
import {
  deleteModel,
  listModels,
  saveModel,
} from '../../api/client'
import type { ModelProfileInfo } from '../../api/types'

interface ModelsEditorProps {
  onSaved: (message: string) => void
  onError: (cause: unknown) => void
}

// A capability row keeps a stable id so editing the name does not
// remount the input and steal focus on every keystroke.
interface CapabilityRow {
  id: string
  name: string
  score: string
}

let capabilitySeq = 0

function newCapability(name = '', score = '3'): CapabilityRow {
  capabilitySeq += 1

  return { id: `capability-${capabilitySeq}`, name, score }
}

const EMPTY = {
  name: '',
  description: '',
  contextSize: '',
  vramGb: '',
  thinking: false,
  capabilities: [] as CapabilityRow[],
}

type Draft = typeof EMPTY

function toDraft(model: ModelProfileInfo): Draft {
  const requirements = model.requirements ?? {}
  const capabilities = Object.entries(model.capabilities ?? {}).map(
    ([key, value]) => newCapability(key, String(value)),
  )

  return {
    name: model.name,
    description: model.description,
    contextSize:
      requirements.context_size === undefined ||
      requirements.context_size === null
        ? ''
        : String(requirements.context_size),
    vramGb:
      requirements.vram_gb === undefined ||
      requirements.vram_gb === null
        ? ''
        : String(requirements.vram_gb),
    thinking: Boolean(requirements.thinking),
    capabilities,
  }
}

export function ModelsEditor({
  onSaved,
  onError,
}: ModelsEditorProps) {
  const [models, setModels] = useState<ModelProfileInfo[]>([])
  const [draft, setDraft] = useState<Draft | null>(null)
  const [saving, setSaving] = useState(false)
  const [loading, setLoading] = useState(true)
  const [focusId, setFocusId] = useState<string | null>(null)

  // Callbacks in refs: depending on them directly would re-fetch the
  // catalog whenever the parent passes new function identities.
  const errorRef = useRef(onError)
  const savedRef = useRef(onSaved)

  useEffect(() => {
    errorRef.current = onError
    savedRef.current = onSaved
  }, [onError, onSaved])

  const load = useCallback(async () => {
    setLoading(true)

    try {
      setModels(await listModels())
    } catch (cause) {
      errorRef.current(cause)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const submit = useCallback(async () => {
    if (!draft || !draft.name.trim()) {
      return
    }

    setSaving(true)

    const capabilities: Record<string, number> = {}

    for (const row of draft.capabilities) {
      const name = row.name.trim()
      const value = Number(row.score)

      if (name && Number.isFinite(value)) {
        capabilities[name] = value
      }
    }

    const requirements: Record<string, unknown> = {
      thinking: draft.thinking,
    }

    if (draft.contextSize.trim()) {
      requirements.context_size = Number(draft.contextSize)
    }

    if (draft.vramGb.trim()) {
      requirements.vram_gb = Number(draft.vramGb)
    }

    try {
      await saveModel({
        name: draft.name.trim(),
        description: draft.description,
        capabilities,
        requirements,
      })

      setDraft(null)
      await load()

      savedRef.current(`Model "${draft.name.trim()}" saved`)
    } catch (cause) {
      errorRef.current(cause)
    } finally {
      setSaving(false)
    }
  }, [draft, load])

  const remove = useCallback(
    async (name: string) => {
      try {
        await deleteModel(name)

        await load()

        savedRef.current(`Model "${name}" removed`)
      } catch (cause) {
        errorRef.current(cause)
      }
    },
    [load],
  )

  const updateCapability = useCallback(
    (id: string, patch: Partial<CapabilityRow>) => {
      setDraft((previous) =>
        previous
          ? {
              ...previous,
              capabilities: previous.capabilities.map((row) =>
                row.id === id ? { ...row, ...patch } : row,
              ),
            }
          : previous,
      )
    },
    [],
  )

  const addCapability = useCallback(() => {
    const row = newCapability()

    setDraft((previous) =>
      previous
        ? { ...previous, capabilities: [...previous.capabilities, row] }
        : previous,
    )

    setFocusId(row.id)
  }, [])

  if (loading) {
    return <p className="muted">Loading catalog…</p>
  }

  return (
    <div className="settings-panel">
      {models.length === 0 && !draft && (
        <p className="muted">
          The catalog is empty. Add a model below so subagents can
          be routed to it.
        </p>
      )}

      <ul className="model-list">
        {models.map((model) => (
          <li key={model.name} className="model-row">
            <div className="model-info">
              <code>{model.name}</code>
              {model.description && (
                <span className="muted small">
                  {model.description}
                </span>
              )}
              <div className="model-meta">
                {Object.entries(model.capabilities ?? {}).map(
                  ([key, value]) => (
                    <span key={key} className="chip">
                      {key} {value}
                    </span>
                  ),
                )}
                {model.requirements?.context_size !== undefined &&
                  model.requirements?.context_size !== null && (
                    <span className="chip">
                      ctx {String(model.requirements.context_size)}
                    </span>
                  )}
                {model.requirements?.thinking === true && (
                  <span className="chip chip-on">thinking</span>
                )}
              </div>
            </div>

            <div className="model-actions">
              <button
                type="button"
                className="icon-button"
                onClick={() => setDraft(toDraft(model))}
                aria-label={`Edit ${model.name}`}
              >
                ✎
              </button>
              <button
                type="button"
                className="icon-button"
                onClick={() => void remove(model.name)}
                aria-label={`Delete ${model.name}`}
              >
                ×
              </button>
            </div>
          </li>
        ))}
      </ul>

      {draft ? (
        <fieldset className="fieldset">
          <legend>
            {models.some((m) => m.name === draft.name)
              ? 'Edit model'
              : 'New model'}
          </legend>

          <label className="field">
            <span className="field-label">Name</span>
            <input
              value={draft.name}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  name: event.target.value,
                })
              }
            />
          </label>

          <label className="field">
            <span className="field-label">Description</span>
            <input
              value={draft.description}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  description: event.target.value,
                })
              }
            />
          </label>

          <div className="field-row">
            <label className="field">
              <span className="field-label">Context size</span>
              <input
                type="number"
                min={1}
                value={draft.contextSize}
                onChange={(event) =>
                  setDraft({
                    ...draft,
                    contextSize: event.target.value,
                  })
                }
              />
            </label>

            <label className="field">
              <span className="field-label">VRAM (GB)</span>
              <input
                type="number"
                min={0}
                step="0.5"
                value={draft.vramGb}
                onChange={(event) =>
                  setDraft({
                    ...draft,
                    vramGb: event.target.value,
                  })
                }
              />
            </label>
          </div>

          <label className="field">
            <span className="field-label">Thinking</span>
            <input
              type="checkbox"
              checked={draft.thinking}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  thinking: event.target.checked,
                })
              }
            />
          </label>

          <div className="field">
            <span className="field-label">
              Capabilities (name → score 0-5)
            </span>

            {draft.capabilities.map((row) => (
              <div className="capability-row" key={row.id}>
                <input
                  value={row.name}
                  placeholder="capability"
                  aria-label="Capability name"
                  autoFocus={row.id === focusId}
                  onChange={(event) =>
                    updateCapability(row.id, {
                      name: event.target.value,
                    })
                  }
                />
                <input
                  type="number"
                  min={0}
                  max={5}
                  value={row.score}
                  aria-label={`Score for ${row.name || 'capability'}`}
                  onChange={(event) =>
                    updateCapability(row.id, {
                      score: event.target.value,
                    })
                  }
                />
                <button
                  type="button"
                  className="icon-button"
                  onClick={() =>
                    setDraft((previous) =>
                      previous
                        ? {
                            ...previous,
                            capabilities:
                              previous.capabilities.filter(
                                (item) => item.id !== row.id,
                              ),
                          }
                        : previous,
                    )
                  }
                  aria-label={`Remove ${row.name || 'capability'}`}
                >
                  ×
                </button>
              </div>
            ))}

            <button
              type="button"
              className="btn"
              onClick={addCapability}
            >
              Add capability
            </button>
          </div>

          <div className="settings-actions settings-actions-end">
            <button
              type="button"
              className="btn"
              onClick={() => setDraft(null)}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn btn-send"
              onClick={() => void submit()}
              disabled={saving || !draft.name.trim()}
            >
              {saving ? 'Saving…' : 'Save model'}
            </button>
          </div>
        </fieldset>
      ) : (
        <button
          type="button"
          className="btn btn-send"
          onClick={() => setDraft({ ...EMPTY })}
        >
          Add model
        </button>
      )}
    </div>
  )
}

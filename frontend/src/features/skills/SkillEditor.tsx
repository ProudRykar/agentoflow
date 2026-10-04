import { useCallback, useEffect, useState } from 'react'
import {
  deleteLibrarySkill,
  listLibrarySkills,
  readLibrarySkill,
  saveLibrarySkill,
} from '../../api/client'
import type { SkillDetail } from '../../api/types'
import './SkillEditor.css'

interface SkillEditorProps {
  onError: (cause: unknown) => void
  onSaved: (message: string) => void
}

interface Draft {
  originalName: string | null
  name: string
  description: string
  version: string
  instructions: string
}

const EMPTY: Draft = {
  originalName: null,
  name: '',
  description: '',
  version: '1.0.0',
  instructions: '',
}

export function SkillEditor({
  onError,
  onSaved,
}: SkillEditorProps) {
  const [skills, setSkills] = useState<SkillDetail[]>([])
  const [draft, setDraft] = useState<Draft | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [confirmName, setConfirmName] = useState<string | null>(
    null,
  )

  const load = useCallback(async () => {
    setLoading(true)

    try {
      setSkills(await listLibrarySkills())
    } catch (cause) {
      onError(cause)
    } finally {
      setLoading(false)
    }
  }, [onError])

  useEffect(() => {
    void load()
  }, [load])

  const edit = useCallback(
    async (name: string) => {
      setConfirmName(null)

      try {
        const skill = await readLibrarySkill(name)

        setDraft({
          originalName: skill.name,
          name: skill.name,
          description: skill.description,
          version: skill.version,
          instructions: skill.instructions,
        })
      } catch (cause) {
        onError(cause)
      }
    },
    [onError],
  )

  const save = useCallback(async () => {
    if (!draft || !draft.name.trim() || !draft.description.trim()) {
      return
    }

    setSaving(true)

    try {
      await saveLibrarySkill(
        {
          name: draft.name.trim(),
          description: draft.description.trim(),
          instructions: draft.instructions,
          version: draft.version || '1.0.0',
        },
        draft.originalName ?? undefined,
      )

      setDraft(null)
      await load()

      onSaved(`Skill "${draft.name.trim()}" saved`)
    } catch (cause) {
      onError(cause)
    } finally {
      setSaving(false)
    }
  }, [draft, load, onError, onSaved])

  const remove = useCallback(
    async (name: string) => {
      try {
        await deleteLibrarySkill(name)

        if (draft?.originalName === name) {
          setDraft(null)
        }

        await load()

        onSaved(`Skill "${name}" deleted`)
      } catch (cause) {
        onError(cause)
      } finally {
        setConfirmName(null)
      }
    },
    [draft, load, onError, onSaved],
  )

  if (loading) {
    return <p className="muted">Loading skills library…</p>
  }

  return (
    <div className="skill-editor">
      <p className="muted small">
        Skills are Markdown files with YAML frontmatter stored in{' '}
        <code>~/.agentoflow/skills/&lt;name&gt;/SKILL.md</code>. A
        skill written here is loadable by the agent immediately.
      </p>

      {skills.length === 0 ? (
        <p className="muted">No skills on disk yet.</p>
      ) : (
        <ul className="skill-editor-list">
          {skills.map((skill) => (
            <li key={skill.name} className="skill-editor-row">
              <div className="skill-editor-info">
                <code>{skill.name}</code>
                <span className="muted small">
                  v{skill.version}
                </span>
                <span className="skill-editor-desc">
                  {skill.description}
                </span>
              </div>

              <div className="skill-editor-actions">
                {confirmName === skill.name ? (
                  <>
                    <button
                      type="button"
                      className="btn btn-cancel"
                      onClick={() => void remove(skill.name)}
                    >
                      Confirm
                    </button>
                    <button
                      type="button"
                      className="btn"
                      onClick={() => setConfirmName(null)}
                    >
                      Cancel
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      type="button"
                      className="icon-button"
                      onClick={() => void edit(skill.name)}
                      aria-label={`Edit ${skill.name}`}
                      title="Edit"
                    >
                      ✎
                    </button>
                    <button
                      type="button"
                      className="icon-button"
                      onClick={() => setConfirmName(skill.name)}
                      aria-label={`Delete ${skill.name}`}
                      title="Delete"
                    >
                      ×
                    </button>
                  </>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}

      {draft ? (
        <div className="skill-form">
          <div className="field-row">
            <label className="field">
              <span className="field-label">Name</span>
              <input
                value={draft.name}
                placeholder="my-skill"
                onChange={(event) =>
                  setDraft({ ...draft, name: event.target.value })
                }
              />
              <span className="field-hint">
                Lowercase letters, digits, '-' and '_'
              </span>
            </label>

            <label className="field">
              <span className="field-label">Version</span>
              <input
                value={draft.version}
                onChange={(event) =>
                  setDraft({
                    ...draft,
                    version: event.target.value,
                  })
                }
              />
            </label>
          </div>

          <label className="field">
            <span className="field-label">Description</span>
            <input
              value={draft.description}
              placeholder="What this skill is for"
              onChange={(event) =>
                setDraft({
                  ...draft,
                  description: event.target.value,
                })
              }
            />
            <span className="field-hint">
              One line; the model reads this to decide when to load
              the skill.
            </span>
          </label>

          <label className="field">
            <span className="field-label">
              Instructions (Markdown)
            </span>
            <textarea
              className="skill-textarea"
              value={draft.instructions}
              spellCheck={false}
              placeholder={'# Title\n\n1. First step\n2. Second step'}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  instructions: event.target.value,
                })
              }
            />
          </label>

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
              onClick={() => void save()}
              disabled={
                saving ||
                !draft.name.trim() ||
                !draft.description.trim() ||
                !draft.instructions.trim()
              }
            >
              {saving ? 'Saving…' : 'Save skill'}
            </button>
          </div>
        </div>
      ) : (
        <button
          type="button"
          className="btn btn-send"
          onClick={() => setDraft({ ...EMPTY })}
        >
          New skill
        </button>
      )}
    </div>
  )
}

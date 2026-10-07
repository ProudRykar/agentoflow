/**
 * Form primitives.
 *
 * These styles used to live in SettingsModal.css and were borrowed by
 * ToolsPage, so a feature only rendered correctly because the other
 * feature happened to be in the same bundle. The primitives are now
 * shared on purpose.
 */

import type { ReactNode } from 'react'
import './primitives.css'

interface FieldProps {
  label: string
  /** Explains the unit or the consequence, not just the label. */
  hint?: string
  /** Shown when the value is out of the recommended range. */
  warning?: string
  children: ReactNode
  htmlFor?: string
}

/** A labelled control with room for a hint and a warning. */
export function Field({
  label,
  hint,
  warning,
  children,
  htmlFor,
}: FieldProps) {
  return (
    <div className={`field${warning ? ' has-warning' : ''}`}>
      <label className="field-label" htmlFor={htmlFor}>
        {label}
      </label>

      {children}

      {warning ? (
        <p className="field-warning">{warning}</p>
      ) : hint ? (
        <p className="field-hint">{hint}</p>
      ) : null}
    </div>
  )
}

interface SelectProps {
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
  ariaLabel?: string
  disabled?: boolean
  id?: string
}

export function Select({
  value,
  onChange,
  options,
  ariaLabel,
  disabled,
  id,
}: SelectProps) {
  return (
    <select
      id={id}
      className="field-control"
      value={value}
      disabled={disabled}
      aria-label={ariaLabel}
      onChange={(event) => onChange(event.target.value)}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  )
}

interface TextareaProps {
  value: string
  onChange: (value: string) => void
  ariaLabel?: string
  placeholder?: string
  rows?: number
}

export function Textarea({
  value,
  onChange,
  ariaLabel,
  placeholder,
  rows = 4,
}: TextareaProps) {
  return (
    <textarea
      className="field-control field-textarea"
      value={value}
      rows={rows}
      spellCheck={false}
      placeholder={placeholder}
      aria-label={ariaLabel}
      onChange={(event) => onChange(event.target.value)}
    />
  )
}
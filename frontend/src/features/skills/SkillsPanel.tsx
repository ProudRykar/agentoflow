import type { SkillInfo } from '../../api/types'

interface SkillsPanelProps {
  skills: SkillInfo[]
}

export function SkillsPanel({ skills }: SkillsPanelProps) {
  const available = skills.length

  const used = skills.filter((skill) => skill.used)
  const active = skills.filter((skill) => skill.active)

  return (
    <section className="panel">
      <h3>Skills</h3>

      {available === 0 ? (
        <p className="muted">No skills available</p>
      ) : (
        <>
          {used.length > 0 && (
            <ul className="panel-list">
              {used.map((skill) => (
                <li key={skill.name} className="panel-item panel-item-on">
                  <span className="panel-mark">✓</span>
                  <span className="panel-name">{skill.name}</span>
                  <span className="panel-tag">used</span>
                </li>
              ))}
            </ul>
          )}

          {active.length > 0 && (
            <ul className="panel-list">
              {active
                .filter((skill) => !skill.used)
                .map((skill) => (
                  <li
                    key={skill.name}
                    className="panel-item panel-item-on"
                  >
                    <span className="panel-mark">✓</span>
                    <span className="panel-name">{skill.name}</span>
                    <span className="panel-tag">active</span>
                  </li>
                ))}
            </ul>
          )}

          <p className="muted">
            {available} available
            {skills.some((skill) => skill.loaded) && ' · some loaded'}
          </p>
        </>
      )}
    </section>
  )
}

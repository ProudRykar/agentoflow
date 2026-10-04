import { useState } from 'react'
import type { PluginInfo } from '../../api/types'

interface PluginsPanelProps {
  plugins: PluginInfo[]
}

export function PluginsPanel({ plugins }: PluginsPanelProps) {
  const [open, setOpen] = useState<string | null>(null)

  if (plugins.length === 0) {
    return (
      <section className="panel">
        <h3>Plugins</h3>
        <p className="muted">No plugins installed</p>
      </section>
    )
  }

  return (
    <section className="panel">
      <h3>Plugins</h3>

      <ul className="panel-list">
        {plugins.map((plugin) => (
          <li key={plugin.name} className="plugin">
            <button
              type="button"
              className="plugin-head"
              onClick={() =>
                setOpen(open === plugin.name ? null : plugin.name)
              }
            >
              <span className="panel-mark">
                {plugin.error ? '✗' : '✓'}
              </span>
              <span className="panel-name">{plugin.name}</span>
              <span className="panel-tag">{plugin.status}</span>
            </button>

            {open === plugin.name && (
              <div className="plugin-detail">
                <div className="muted">v{plugin.version}</div>

                {plugin.description && (
                  <p>{plugin.description}</p>
                )}

                {plugin.error && (
                  <div className="plugin-error">{plugin.error}</div>
                )}

                {plugin.capabilities.length > 0 && (
                  <div>
                    <div className="panel-subhead">Capabilities</div>
                    <ul className="panel-bullets">
                      {plugin.capabilities.map((capability) => (
                        <li key={capability}>{capability}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {plugin.tools.length > 0 && (
                  <div>
                    <div className="panel-subhead">
                      Tools ({plugin.tools.length})
                    </div>
                    <ul className="panel-bullets">
                      {plugin.tools.map((tool) => (
                        <li key={tool}>{tool}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {plugin.skills.length > 0 && (
                  <div>
                    <div className="panel-subhead">
                      Skills ({plugin.skills.length})
                    </div>
                    <ul className="panel-bullets">
                      {plugin.skills.map((skill) => (
                        <li key={skill}>{skill}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </li>
        ))}
      </ul>
    </section>
  )
}

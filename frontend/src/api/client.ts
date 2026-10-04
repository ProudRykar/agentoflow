import type {
  MCPResponse,
  ModelProfileInfo,
  PluginInfo,
  SessionInfo,
  SettingsFileInfo,
  SkillDetail,
  SkillInfo,
  SystemInfo,
  ToolsResponse,
} from './types'

const BASE = '/api'

/**
 * Token used for both REST calls and the WebSocket handshake.
 *
 * Kept in memory only: the backend reads it from the environment,
 * so it is never stored in the bundle or in localStorage.
 */
let apiToken = ''

export function setApiToken(token: string): void {
  apiToken = token.trim()
}

export function hasApiToken(): boolean {
  return apiToken.length > 0
}

function headers(extra?: Record<string, string>): Record<string, string> {
  const base: Record<string, string> = {
    'Content-Type': 'application/json',
    ...extra,
  }

  if (apiToken) {
    base['x-agentoflow-token'] = apiToken
  }

  return base
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: headers(init?.headers as Record<string, string>),
  })

  if (!response.ok) {
    let detail = response.statusText

    try {
      const body = await response.json()
      detail = body.detail ?? detail
    } catch {
      // Non-JSON error body; keep the status text.
    }

    throw new ApiError(response.status, detail)
  }

  if (response.status === 204) {
    return undefined as T
  }

  return (await response.json()) as T
}

export function createSession(
  metadata: Record<string, unknown> = {},
): Promise<{ session_id: string; state: string }> {
  return request('/sessions', {
    method: 'POST',
    body: JSON.stringify({ metadata }),
  })
}

export function listSessions(): Promise<SessionInfo[]> {
  return request('/sessions')
}

export function getSession(sessionId: string): Promise<SessionInfo> {
  return request(`/sessions/${sessionId}`)
}

export function closeSession(sessionId: string): Promise<unknown> {
  return request(`/sessions/${sessionId}`, { method: 'DELETE' })
}

export function startRun(
  sessionId: string,
  prompt: string,
): Promise<{ session_id: string; status: string; state: string }> {
  return request(`/sessions/${sessionId}/run`, {
    method: 'POST',
    body: JSON.stringify({ prompt }),
  })
}

export function continueRun(
  sessionId: string,
  prompt: string,
): Promise<{ session_id: string; status: string; state: string }> {
  return request(`/sessions/${sessionId}/continue`, {
    method: 'POST',
    body: JSON.stringify({ prompt }),
  })
}

/**
 * Submit a user turn.
 *
 * `mode: 'auto'` lets the session decide between opening a new
 * conversation and continuing the current one, so the UI does not
 * duplicate that rule.
 */
export function sendMessage(
  sessionId: string,
  content: string,
  mode: 'auto' | 'run' | 'continue' | 'resume' = 'auto',
): Promise<{
  session_id: string
  status: string
  state: string
  mode: string | null
}> {
  return request(`/sessions/${sessionId}/messages`, {
    method: 'POST',
    body: JSON.stringify({ content, mode }),
  })
}

export function resumeRun(sessionId: string): Promise<unknown> {
  return request(`/sessions/${sessionId}/resume`, { method: 'POST' })
}

export function cancelRun(
  sessionId: string,
): Promise<{ session_id: string; cancelled: boolean; state: string }> {
  return request(`/sessions/${sessionId}/cancel`, { method: 'POST' })
}

export function getPendingApproval(
  sessionId: string,
): Promise<PendingApprovalInfo | null> {
  return request(`/sessions/${sessionId}/approval`)
}

export interface PendingApprovalInfo {
  approval_id: string
  tool_name: string
  permission: string
  reason: string
  arguments: Record<string, unknown>
}

export function allowApproval(
  sessionId: string,
  approvalId: string,
): Promise<unknown> {
  return request(
    `/sessions/${sessionId}/approvals/${approvalId}/allow`,
    { method: 'POST' },
  )
}

export function denyApproval(
  sessionId: string,
  approvalId: string,
): Promise<unknown> {
  return request(
    `/sessions/${sessionId}/approvals/${approvalId}/deny`,
    { method: 'POST' },
  )
}

export function listSkills(sessionId: string): Promise<SkillInfo[]> {
  return request(`/skills/${sessionId}`)
}

export function listPlugins(sessionId: string): Promise<PluginInfo[]> {
  return request(`/plugins/${sessionId}`)
}

export function listTools(sessionId: string): Promise<ToolsResponse> {
  return request(`/tools/${sessionId}`)
}

export function listMcp(sessionId: string): Promise<MCPResponse> {
  return request(`/mcp/${sessionId}`)
}

export type MCPServerAction = 'connect' | 'disconnect' | 'reload'

export function mcpServerAction(
  sessionId: string,
  server: string,
  action: MCPServerAction,
): Promise<{
  session_id: string
  server: string
  action: string
  state: string
  total_tools: number
}> {
  return request(
    `/mcp/${sessionId}/servers/${server}/${action}`,
    { method: 'POST' },
  )
}

export function mcpReloadAll(sessionId: string): Promise<unknown> {
  return request(`/mcp/${sessionId}/reload`, { method: 'POST' })
}

export interface MCPServerInput {
  name: string
  transport: 'stdio' | 'http'
  command?: string
  args?: string[]
  url?: string
  env?: Record<string, string>
  headers?: Record<string, string>
  enabled?: boolean
}

export function saveMcpServer(
  sessionId: string,
  payload: MCPServerInput,
): Promise<MCPResponse> {
  return request(`/mcp/${sessionId}/servers`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function deleteMcpServer(
  sessionId: string,
  server: string,
): Promise<MCPResponse> {
  return request(`/mcp/${sessionId}/servers/${server}`, {
    method: 'DELETE',
  })
}

export function setMcpEnabled(
  sessionId: string,
  enabled: boolean,
): Promise<MCPResponse> {
  return request(`/mcp/${sessionId}/enabled`, {
    method: 'POST',
    body: JSON.stringify({ enabled }),
  })
}

export function getSystemInfo(sessionId: string): Promise<SystemInfo> {
  return request(`/system/${sessionId}`)
}

/** WebSocket URL for a session, carrying the cursor to resume from. */
export function socketUrl(sessionId: string, since: number): string {
  const protocol =
    window.location.protocol === 'https:' ? 'wss:' : 'ws:'

  const query = new URLSearchParams()

  if (since >= 0) {
    query.set('since', String(since))
  }

  if (apiToken) {
    query.set('token', apiToken)
  }

  const suffix = query.toString()

  return (
    `${protocol}//${window.location.host}` +
    `/ws/sessions/${sessionId}` +
    (suffix ? `?${suffix}` : '')
  )
}


// ======================================================================
// Settings
// ======================================================================

export function listSettingsFiles(): Promise<string[]> {
  return request('/settings')
}

export function readSettings(name: string): Promise<SettingsFileInfo> {
  return request(`/settings/${name}`)
}

export function writeSettings(
  name: string,
  body: { data?: unknown; text?: string },
): Promise<SettingsFileInfo> {
  return request(`/settings/${name}`, {
    method: 'PUT',
    body: JSON.stringify(body),
  })
}

// ======================================================================
// Model catalog
// ======================================================================

export function listModels(): Promise<ModelProfileInfo[]> {
  return request('/settings/models/list')
}

export function saveModel(model: {
  name: string
  description?: string
  capabilities?: Record<string, number>
  attributes?: Record<string, unknown>
  requirements?: Record<string, unknown>
}): Promise<SettingsFileInfo> {
  return request('/settings/models/entry', {
    method: 'PUT',
    body: JSON.stringify(model),
  })
}

export function deleteModel(name: string): Promise<unknown> {
  return request(`/settings/models/entry/${name}`, {
    method: 'DELETE',
  })
}

// ======================================================================
// Skills library
// ======================================================================

export function listLibrarySkills(): Promise<SkillDetail[]> {
  return request('/skills/library')
}

export function readLibrarySkill(
  name: string,
): Promise<SkillDetail> {
  return request(`/skills/library/${name}`)
}

export function saveLibrarySkill(
  skill: {
    name: string
    description: string
    instructions: string
    version?: string
    metadata?: Record<string, unknown>
  },
  originalName?: string,
): Promise<SkillDetail> {
  if (originalName) {
    return request(`/skills/library/${originalName}`, {
      method: 'PUT',
      body: JSON.stringify(skill),
    })
  }

  return request('/skills/library', {
    method: 'POST',
    body: JSON.stringify(skill),
  })
}

export function deleteLibrarySkill(name: string): Promise<unknown> {
  return request(`/skills/library/${name}`, {
    method: 'DELETE',
  })
}

// ======================================================================
// Session naming
// ======================================================================

export function renameSession(
  sessionId: string,
  title: string,
): Promise<{ session_id: string; title: string }> {
  return request(`/sessions/${sessionId}/title`, {
    method: 'PATCH',
    body: JSON.stringify({ title }),
  })
}

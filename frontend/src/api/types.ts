/**
 * Wire contract shared with the backend.
 *
 * These types mirror `agent_workflow.web.serialization`. They are
 * the only place the frontend is allowed to know the wire shape;
 * everything else works with the view model built from them.
 */

export interface WireEnvelope<T = Record<string, unknown>> {
  type: string
  session_id: string
  seq: number
  timestamp: string
  run_id?: string
  parent_run_id?: string
  data: T
}

export interface SessionSnapshotPayload {
  session_id: string
  state: string
  created_at: number
  metadata: Record<string, unknown>
  model: string
  working_directory: string
  phase: string | null
  iteration: number | null
  running: boolean
  last_error: string
  approval: PendingApproval | null
}

export interface SnapshotEnvelope
  extends WireEnvelope<SessionSnapshotPayload> {
  type: 'session.snapshot'
  oldest_seq: number
  replayed: number
}

export interface PendingApproval {
  approval_id: string
  tool_name: string
  permission: string
  reason: string
  arguments: Record<string, unknown>
}

export interface AgentStartedData {
  prompt: string
  run_id: string
  parent_run_id: string | null
  agent_id: string
  role: string
  model: string
}

export interface PhaseChangedData {
  previous_phase: string
  phase: string
  reason: string
  iteration: number
  run_id: string
  parent_run_id: string | null
}

export interface LLMRequestedData {
  iteration: number
  message_count: number
  tool_count: number
  run_id: string
  parent_run_id: string | null
  estimated_tokens: number | null
}

export interface ChunkData {
  iteration: number
  content: string
  run_id: string
  parent_run_id: string | null
}

export interface LLMRespondedData {
  iteration: number
  content: string | null
  thinking: string | null
  tool_call_count: number
  run_id: string
  parent_run_id: string | null
}

export interface ToolStartedData {
  iteration: number
  tool_call_id: string
  tool_name: string
  arguments: Record<string, unknown>
  run_id: string
  parent_run_id: string | null
}

export interface ToolFinishedData {
  iteration: number
  tool_call_id: string
  tool_name: string
  output: string | null
  error_code: string | null
  error_message: string | null
  run_id: string
  parent_run_id: string | null
  duration_seconds: number | null
}

export interface AgentFinishedData {
  result: string
  run_id: string
  parent_run_id: string | null
  agent_id: string
  role: string
  model: string
}

export interface ApprovalRequestedData {
  approval_id: string
  tool_name: string
  arguments: Record<string, unknown>
  permission: string
  reason: string
}

export interface ApprovalResolvedData {
  approval_id: string
  approved: boolean
}

export interface RunFailedData {
  session_id: string
  error: string
}

export type EventType =
  | 'session.snapshot'
  | 'agent.started'
  | 'agent.phase_changed'
  | 'agent.finished'
  | 'llm.requested'
  | 'llm.thinking_chunk'
  | 'llm.content_chunk'
  | 'llm.responded'
  | 'tool.started'
  | 'tool.finished'
  | 'approval.requested'
  | 'approval.resolved'
  | 'run.failed'
  | 'heartbeat'
  | 'ping'
  | 'pong'

export interface SkillInfo {
  name: string
  description: string
  version: string
  active: boolean
  loaded: boolean
  used: boolean
}

export interface PluginInfo {
  name: string
  version: string
  status: string
  description: string | null
  author: string | null
  capabilities: string[]
  tools: string[]
  skills: string[]
  error: string | null
}

export interface SessionInfo {
  session_id: string
  created_at: number
  title: string
  title_source: string
  metadata: Record<string, unknown>
  model: string
  working_directory: string
  subscriber_count: number
  latest_seq: number
  session: {
    state: string
    phase: string | null
    iteration: number | null
    running: boolean
    last_error: string
  }
}

export interface SystemInfo {
  version: string
  model: string
  working_directory: string
  max_iterations: number | null
  agent_phase: string | null
}


export interface ToolParameter {
  name: string
  type: string
  required: boolean
  description: string
}

export interface ToolUsageStats {
  calls: number
  running: number
  successes: number
  errors: number
  error_rate: number
  average_duration: number | null
  last_error_code: string | null
  error_codes: Record<string, number>
}

export interface ToolInfo {
  name: string
  description: string
  source: 'builtin' | 'plugin' | 'mcp' | string
  permissions: string[]
  missing_permissions: string[]
  requires_approval: boolean
  timeout: number
  max_output_size: number
  parameters: ToolParameter[]
  required_parameters: string[]
  mcp_server: string | null
  stats: ToolUsageStats
}

export interface ToolsSummary {
  registered_tools: number
  tools_used: number
  total_calls: number
  total_successes: number
  total_errors: number
  error_rate: number
  approvals_requested: number
  approvals_allowed: number
  approvals_denied: number
  approval_pending: boolean
}

export interface ToolsResponse {
  tools: ToolInfo[]
  summary: ToolsSummary
}

export interface MCPToolInfo {
  qualified_name: string
  remote_name: string
  description: string
  parameters: string[]
  required: string[]
  requires_approval: boolean
  permissions: string[]
}

export interface MCPServerInfo {
  name: string
  transport: string
  command: string
  url: string
  args: string[]
  enabled: boolean
  state: string
  error: string
  server_version: string
  protocol_version: string
  instructions: string
  tools: MCPToolInfo[]
  connected_at: number
  startup_seconds: number
}

export interface MCPResponse {
  enabled: boolean
  servers: MCPServerInfo[]
  permissions: string[]
  total_tools: number
  connected_servers: number
}


export interface MCPActionResult {
  session_id: string
  server: string
  action: string
  state: string
  total_tools: number
}


// ======================================================================
// Settings
// ======================================================================

export interface SettingsFileInfo {
  name: string
  path: string
  exists: boolean
  data: Record<string, unknown>
  text: string
}

export interface ModelProfileInfo {
  name: string
  description: string
  capabilities: Record<string, number>
  attributes: Record<string, unknown>
  requirements: Record<string, unknown>
}

// ======================================================================
// Skills library
// ======================================================================

export interface SkillDetail {
  name: string
  description: string
  version: string
  instructions: string
  metadata: Record<string, unknown>
  path: string
  builtin: boolean
}

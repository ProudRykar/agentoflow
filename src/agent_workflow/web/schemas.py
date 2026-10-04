from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    metadata: dict = Field(default_factory=dict)

    # Sandbox root for this session's agent. It becomes the
    # ToolContext working directory and the PathPolicy allow-list,
    # so it is validated before use.
    working_directory: str | None = None


class MessageRequest(BaseModel):
    """A user turn without deciding how the run is started.

    ``mode`` defaults to ``auto``, which starts a new conversation
    on the first turn and continues the existing one afterwards.
    """

    content: str = Field(min_length=1)
    mode: Literal["auto", "run", "continue", "resume"] = "auto"


class SessionStateInfo(BaseModel):
    state: str
    phase: str | None = None
    iteration: int | None = None
    running: bool = False
    last_error: str = ""


class SessionInfo(BaseModel):
    session_id: str
    created_at: float
    metadata: dict = Field(default_factory=dict)
    title: str = ""
    title_source: str = "auto"
    model: str = ""
    working_directory: str = ""
    subscriber_count: int = 0
    latest_seq: int = 0
    session: SessionStateInfo


class SessionCreateResponse(BaseModel):
    session_id: str
    state: str


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1)


class RunResponse(BaseModel):
    session_id: str
    status: str
    state: str
    mode: Literal["run", "continue"] | None = None


class ResumeResponse(BaseModel):
    session_id: str
    status: str
    state: str


class CancelResponse(BaseModel):
    session_id: str
    cancelled: bool
    state: str


class ApprovalDecisionResponse(BaseModel):
    session_id: str
    approval_id: str
    accepted: bool


class ApprovalStateInfo(BaseModel):
    approval_id: str
    tool_name: str
    permission: str
    reason: str
    arguments: dict = Field(default_factory=dict)


class SkillInfo(BaseModel):
    name: str
    description: str
    version: str
    active: bool = False
    loaded: bool = False
    used: bool = False


class PluginInfo(BaseModel):
    name: str
    version: str
    status: str
    description: str | None = None
    author: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    error: str | None = None


class SystemInfo(BaseModel):
    version: str
    model: str
    working_directory: str
    max_iterations: int | None = None
    agent_phase: str | None = None


# ======================================================================
# Tool inventory and statistics
# ======================================================================


class ToolParameter(BaseModel):
    name: str
    type: str
    required: bool = False
    description: str = ""


class ToolUsageStats(BaseModel):
    calls: int = 0
    running: int = 0
    successes: int = 0
    errors: int = 0
    error_rate: float = 0.0
    average_duration: float | None = None
    last_error_code: str | None = None
    error_codes: dict[str, int] = Field(default_factory=dict)


class ToolInfo(BaseModel):
    name: str
    description: str
    source: str
    permissions: list[str] = Field(default_factory=list)
    missing_permissions: list[str] = Field(default_factory=list)
    requires_approval: bool = False
    timeout: float = 0.0
    max_output_size: int = 0
    parameters: list[ToolParameter] = Field(default_factory=list)
    required_parameters: list[str] = Field(default_factory=list)
    mcp_server: str | None = None
    stats: ToolUsageStats = Field(default_factory=ToolUsageStats)


class ToolsSummary(BaseModel):
    registered_tools: int = 0
    tools_used: int = 0
    total_calls: int = 0
    total_successes: int = 0
    total_errors: int = 0
    error_rate: float = 0.0
    approvals_requested: int = 0
    approvals_allowed: int = 0
    approvals_denied: int = 0
    approval_pending: bool = False


class ToolsResponse(BaseModel):
    tools: list[ToolInfo] = Field(default_factory=list)
    summary: ToolsSummary = Field(default_factory=ToolsSummary)


# ======================================================================
# MCP
# ======================================================================


class MCPToolInfo(BaseModel):
    qualified_name: str
    remote_name: str
    description: str
    parameters: list[str] = Field(default_factory=list)
    required: list[str] = Field(default_factory=list)
    requires_approval: bool = True
    permissions: list[str] = Field(default_factory=list)


class MCPServerInfo(BaseModel):
    name: str
    transport: str = "stdio"
    command: str = ""
    url: str = ""
    args: list[str] = Field(default_factory=list)
    enabled: bool = True
    state: str
    error: str = ""
    server_version: str = ""
    protocol_version: str = ""
    instructions: str = ""
    tools: list[MCPToolInfo] = Field(default_factory=list)
    connected_at: float = 0.0
    startup_seconds: float = 0.0


class MCPResponse(BaseModel):
    enabled: bool = False
    servers: list[MCPServerInfo] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    total_tools: int = 0
    connected_servers: int = 0


class GlobalToolUsage(BaseModel):
    tool_name: str
    calls: int
    successes: int
    errors: int
    average_duration: float | None = None
    error_rate: float = 0.0
    last_error_code: str | None = None


class GlobalStatsResponse(BaseModel):
    total_calls: int = 0
    distinct_tools: int = 0
    top_tools: list[GlobalToolUsage] = Field(default_factory=list)


# ======================================================================
# Settings
# ======================================================================


class SettingsFileInfo(BaseModel):
    name: str
    path: str
    exists: bool
    data: dict = Field(default_factory=dict)
    text: str = ""


class SettingsWriteRequest(BaseModel):
    data: dict | None = None
    text: str | None = None
    keep_secrets: dict = Field(default_factory=dict)


class ModelProfileInfo(BaseModel):
    name: str
    description: str = ""
    capabilities: dict[str, int] = Field(default_factory=dict)
    attributes: dict[str, object] = Field(default_factory=dict)
    requirements: dict[str, object] = Field(default_factory=dict)


class ModelWriteRequest(BaseModel):
    name: str
    description: str = ""
    capabilities: dict[str, int] = Field(default_factory=dict)
    attributes: dict[str, object] = Field(default_factory=dict)
    requirements: dict[str, object] = Field(default_factory=dict)


# ======================================================================
# Skills authoring
# ======================================================================


class SkillWriteRequest(BaseModel):
    name: str
    description: str
    instructions: str
    version: str = "1.0.0"
    metadata: dict = Field(default_factory=dict)


class SkillDetail(BaseModel):
    name: str
    description: str
    version: str
    instructions: str
    metadata: dict = Field(default_factory=dict)
    path: str = ""
    builtin: bool = False


# ======================================================================
# Session naming
# ======================================================================


class SessionRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)

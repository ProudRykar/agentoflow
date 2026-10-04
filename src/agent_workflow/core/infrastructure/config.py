from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tomllib

# A count of chat message objects, not turns: one assistant message
# plus its tool results is already several entries. The previous
# default of 4 left room for roughly a single user turn plus one tool
# group, so the model routinely lost the thread mid-task.
DEFAULT_MAX_MESSAGES = 24


@dataclass(slots=True, frozen=True)
class AgentConfig:
    max_iterations: int = 10


@dataclass(slots=True, frozen=True)
class LLMConfig:
    provider: str = "ollama"
    model: str = "gemma4:12b"
    timeout: float = 600.0


@dataclass(slots=True, frozen=True)
class SubagentConfig:
    max_iterations: int = 5
    max_tool_calls: int = 10
    timeout: float = 600.0
    escalation: bool = False


@dataclass(slots=True, frozen=True)
class ModelsConfig:
    catalog: str = "models.toml"
    vram_budget_gb: float | None = None
    single_model_mode: bool = False


@dataclass(slots=True, frozen=True)
class ContextConfig:
    max_messages: int | None = DEFAULT_MAX_MESSAGES
    token_estimation_divisor: int = 4


@dataclass(slots=True, frozen=True)
class MemoryConfig:
    database: str = "memory.db"


@dataclass(slots=True, frozen=True)
class MCPServerConfigEntry:
    name: str
    # stdio (default) needs a command; http needs a url.
    transport: str = "stdio"
    command: str = ""
    url: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    startup_timeout: float = 20.0
    request_timeout: float = 60.0
    prefix: str = ""


@dataclass(slots=True, frozen=True)
class MCPConfig:
    """Model Context Protocol servers exposed as tools."""

    enabled: bool = False
    servers: tuple[MCPServerConfigEntry, ...] = ()
    default_permissions: tuple[str, ...] = ("mcp.execute",)
    require_approval: bool = True


@dataclass(slots=True, frozen=True)
class Config:
    agent: AgentConfig
    llm: LLMConfig
    subagent: SubagentConfig
    context: ContextConfig
    memory: MemoryConfig
    models: ModelsConfig
    mcp: MCPConfig = MCPConfig()


class ConfigError(ValueError):
    """A configuration value has the wrong type or is out of range."""


def _int(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"{field} must be an integer, got {type(value).__name__}"
        )

    if value < minimum:
        raise ConfigError(f"{field} must be >= {minimum}")

    return value


def _float(
    value: object,
    field: str,
    *,
    minimum: float = 0.0,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value, int | float
    ):
        raise ConfigError(
            f"{field} must be a number, got {type(value).__name__}"
        )

    number = float(value)

    if number < minimum:
        raise ConfigError(f"{field} must be >= {minimum}")

    return number


def _bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(
            f"{field} must be true or false, "
            f"got {type(value).__name__}"
        )

    return value


def _optional_float(
    value: object,
    field: str,
) -> float | None:
    if value is None:
        return None

    return _float(value, field)


def _optional_int(
    value: object,
    field: str,
) -> int | None:
    if value is None:
        return None

    return _int(value, field)


def _table(data: object, name: str) -> dict:
    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigError(f"[{name}] must be a table")

    return data


class ConfigLoader:
    def load(
        self,
        path: Path,
    ) -> Config:
        with path.open("rb") as file:
            data = tomllib.load(file)

        agent_data = _table(data.get("agent"), "agent")
        llm_data = _table(data.get("llm"), "llm")
        subagent_data = _table(data.get("subagent"), "subagent")
        context_data = _table(data.get("context"), "context")
        memory_data = _table(data.get("memory"), "memory")
        models_data = _table(data.get("models"), "models")
        mcp_data = _table(data.get("mcp"), "mcp")

        return Config(
            agent=AgentConfig(
                max_iterations=_int(
                    agent_data.get("max_iterations", 10),
                    "agent.max_iterations",
                ),
            ),
            llm=LLMConfig(
                provider=llm_data.get(
                    "provider",
                    "ollama",
                ),
                model=llm_data.get(
                    "model",
                    "gemma4:12b",
                ),
                timeout=_float(
                    llm_data.get("timeout", 600.0),
                    "llm.timeout",
                ),
            ),
            subagent=SubagentConfig(
                max_iterations=_int(
                    subagent_data.get("max_iterations", 25),
                    "subagent.max_iterations",
                ),
                max_tool_calls=_int(
                    subagent_data.get("max_tool_calls", 10),
                    "subagent.max_tool_calls",
                ),
                timeout=_float(
                    subagent_data.get("timeout", 600.0),
                    "subagent.timeout",
                ),
                escalation=_bool(
                    subagent_data.get("escalation", False),
                    "subagent.escalation",
                ),
            ),
            context=ContextConfig(
                max_messages=_optional_int(
                    context_data.get(
                        "max_messages",
                        DEFAULT_MAX_MESSAGES,
                    ),
                    "context.max_messages",
                ),
                token_estimation_divisor=_int(
                    context_data.get(
                        "token_estimation_divisor",
                        4,
                    ),
                    "context.token_estimation_divisor",
                ),
            ),
            memory=MemoryConfig(
                database=memory_data.get(
                    "database",
                    "memory.db",
                ),
            ),
            models=ModelsConfig(
                catalog=str(
                    models_data.get("catalog", "models.toml")
                ),
                vram_budget_gb=_optional_float(
                    models_data.get("vram_budget_gb"),
                    "models.vram_budget_gb",
                ),
                single_model_mode=_bool(
                    models_data.get("single_model_mode", False),
                    "models.single_model_mode",
                ),
            ),
            mcp=_load_mcp(mcp_data),
        )


def _load_mcp(data: dict) -> MCPConfig:
    """Parse the ``[mcp]`` table.

    Two TOML shapes are accepted, because both are natural to
    write by hand::

        [mcp.servers.files]        # keyed form
        command = "npx"

        [[mcp.servers]]            # array form
        name = "files"
        command = "npx"

    A malformed entry is skipped rather than aborting startup: a
    broken MCP block must not make the whole agent unusable.
    """

    servers: list[MCPServerConfigEntry] = []

    for name, entry in _iter_server_entries(data.get("servers")):
        if not isinstance(entry, dict):
            continue

        transport = str(entry.get("transport", "stdio")).lower()

        command = entry.get("command", "")
        url = entry.get("url", "")

        command = command.strip() if isinstance(command, str) else ""
        url = url.strip() if isinstance(url, str) else ""

        # Each transport needs its own required field.
        if transport == "http":
            if not url:
                continue
        elif not command:
            continue

        args_raw = entry.get("args", [])

        args = (
            tuple(str(item) for item in args_raw)
            if isinstance(args_raw, list)
            else ()
        )

        env_raw = entry.get("env", {})

        env = (
            {
                str(key): str(value)
                for key, value in env_raw.items()
            }
            if isinstance(env_raw, dict)
            else {}
        )

        headers_raw = entry.get("headers", {})

        headers = (
            {
                str(key): str(value)
                for key, value in headers_raw.items()
            }
            if isinstance(headers_raw, dict)
            else {}
        )

        servers.append(
            MCPServerConfigEntry(
                name=name,
                transport=transport,
                command=command,
                url=url,
                args=args,
                env=env,
                headers=headers,
                enabled=bool(entry.get("enabled", True)),
                startup_timeout=float(
                    entry.get("startup_timeout", 20.0)
                ),
                request_timeout=float(
                    entry.get("request_timeout", 60.0)
                ),
                prefix=str(entry.get("prefix", "")),
            )
        )

    permissions = data.get(
        "default_permissions",
        ["mcp.execute"],
    )

    if not isinstance(permissions, list):
        raise ConfigError(
            "[mcp] default_permissions must be an array"
        )

    return MCPConfig(
        enabled=_bool(
            data.get("enabled", bool(servers)),
            "mcp.enabled",
        ),
        servers=tuple(servers),
        default_permissions=(
            tuple(str(item) for item in permissions)
            if isinstance(permissions, list)
            else ("mcp.execute",)
        ),
        require_approval=_bool(
            data.get("require_approval", True),
            "mcp.require_approval",
        ),
    )


def _iter_server_entries(raw: object):
    """Yield (name, table) pairs from either TOML shape."""

    if isinstance(raw, dict):
        for name, entry in raw.items():
            yield str(name), entry

    elif isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue

            name = entry.get("name")

            if not isinstance(name, str) or not name.strip():
                continue

            yield name.strip(), entry
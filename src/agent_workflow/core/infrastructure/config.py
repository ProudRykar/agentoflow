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

    # USD per million tokens, as (prompt, completion). None by default:
    # the stock provider is a local Ollama, where a price would be a
    # fiction, and a wrong number is worse than no number because it
    # reads as a bill.
    price_per_million: tuple[float, float] | None = None

    # Hard ceiling on a run's measured tokens. None means no ceiling,
    # which is only safe while iteration is capped -- see
    # AgentConfig.max_iterations for the other half of that.
    max_prompt_tokens: int | None = None


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

    # Compaction replaces a window the budget is about to drop with a
    # handover note, instead of forgetting it. On by default: a long
    # run that reaches its limit and then behaves as if it had just
    # started is the failure this prevents.
    compaction: bool = True

    # How long that note should be. Modest, because it is paid for on
    # every request after the window rolls.
    compaction_tokens: int = 1_200


@dataclass(slots=True, frozen=True)
class PythonConfig:
    """Sandbox for the ``python_exec`` tool.

    The limits below are enforced. Filesystem and network
    confinement are NOT: Python code runs with the privileges of the
    user running the agent, so it can read any file that user can
    read. Treat this as a guardrail against runaway and accidents,
    not as a security boundary.
    """

    enabled: bool = True
    # Empty means "the interpreter Agentoflow itself runs under",
    # which is what you want for venv work.
    interpreter: str = ""
    sandbox_dir: str = "agentoflow-python"
    timeout: float = 30.0
    # Address-space cap in MiB. 0 disables it.
    memory_mb: int = 1024
    # CPU seconds. 0 disables it.
    cpu_seconds: float = 60.0
    max_output: int = 20_000
    # Wall-clock seconds the caller may request per call.
    max_timeout: float = 120.0
    # Static denylist. This is a guardrail against a careless script,
    # NOT a sandbox: getattr(__import__("os"), "system") passes it.
    blocked_modules: tuple[str, ...] = (
        "ctypes",
        "multiprocessing",
        "pty",
        "shutil",
        "socket",
        "ssl",
        "subprocess",
        "urllib",
        "webbrowser",
    )


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
    batch_concurrency: int = 4
    """How many calls one batch tool runs at once.

    Bounded because the server is usually the constraint and not the
    client: an unbounded batch against a single-threaded server just
    queues, and against a database it turns one slow write into a
    connection storm. Four is enough to overlap a round trip without
    asking a local Stash to serve everything at once.
    """

    batch_tools: tuple[str, ...] = ()
    """Remote tool names to also offer in a list form.

    Empty by default, and the default matters: offering a batch twin of
    every tool doubles the tool list, and a model choosing from a long
    list picks worse. Batch each tool only where doing one item at a
    time is genuinely painful -- bulk writes, mostly, where the cost is
    an approval prompt per item rather than a round trip.

    "*" enables it for every tool on every server.
    """

    def __post_init__(self) -> None:
        if self.batch_concurrency < 1:
            raise ValueError(
                "mcp.batch_concurrency must be >= 1"
            )

        for name in self.batch_tools:
            if not name.strip():
                raise ValueError(
                    "mcp.batch_tools must not contain blanks"
                )

    def batches(self, remote_name: str) -> bool:
        """Whether this tool should also be offered as a batch."""

        if not self.batch_tools:
            return False

        return (
            "*" in self.batch_tools
            or remote_name in self.batch_tools
        )


@dataclass(slots=True, frozen=True)
class ToolsConfig:
    """Which tools the model is offered.

    A narrow model picks badly when every option is on display, and
    two ways to do the same thing is worse than one: ``get_tags`` and
    ``stash_graphql`` both list tags, so which one gets used depends
    on the mood of the model rather than on the task.
    """

    disabled: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in self.disabled:
            if not name.strip():
                raise ValueError(
                    "tools.disabled must not contain blanks"
                )

    def blocks(self, name: str) -> bool:
        return name in self.disabled


@dataclass(slots=True, frozen=True)
class WebConfig:
    """How web search reaches an engine.

    Every keyless search front-end reachable over the open internet
    answers a scripted client with a JavaScript challenge rather than
    results, and changing the User-Agent does not change that: the
    gate is on the address, not the request. A self-hosted SearXNG
    instance is the way out, because its rate limits and formats are
    yours rather than someone else's anti-bot policy.
    """

    # ``searxng`` uses the instance below; ``duckduckgo`` is the
    # keyless scraper and is expected to be challenged.
    backend: str = "duckduckgo"

    searxng_endpoint: str = "http://localhost:8080"
    """Base URL of the instance, without a trailing slash."""

    searxng_timeout: float = 20.0

    searxng_categories: str = "general"
    """Instance category filter, e.g. ``general`` or ``general,social``."""

    def __post_init__(self) -> None:
        if self.backend not in ("searxng", "duckduckgo"):
            raise ValueError(
                "web.backend must be 'searxng' or 'duckduckgo', "
                f"not {self.backend!r}"
            )

        if self.searxng_timeout <= 0:
            raise ValueError("web.searxng_timeout must be > 0")

        if self.backend == "searxng" and not self.searxng_endpoint.strip():
            raise ValueError(
                "web.searxng_endpoint is required when "
                "web.backend is 'searxng'"
            )


@dataclass(slots=True, frozen=True)
class ApprovalConfig:
    """How granted permissions behave for the rest of a session.

    ``remember`` decides whether one grant covers later calls of the
    same tool. Without it, a bulk task asks once per item, which
    trains the operator to approve without reading.
    """

    remember: bool = True


@dataclass(slots=True, frozen=True)
class GraphQLConfig:
    """Read-only GraphQL access to a Stash server.

    Off unless an endpoint is configured. The tool can read the whole
    library, so enabling it is an explicit act rather than a default.
    """

    enabled: bool = False

    endpoint: str = ""

    # Empty means read it from env_file, which avoids keeping a second
    # copy of a credential in a file readable by anything that can
    # read the home directory.
    api_key: str = ""

    env_file: str = ""

    timeout: float = 30.0

    max_query_chars: int = 8_000

    max_response_chars: int = 60_000

    max_rows: int = 250

    # Ceiling on client-side ordering: ranking by a count has to read
    # every match, so this is what stops a wide filter from turning
    # into an unbounded crawl.
    max_scan_rows: int = 5_000

    allow_introspection: bool = True

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError("graphql.timeout must be > 0")

        if self.max_query_chars < 1:
            raise ValueError(
                "graphql.max_query_chars must be >= 1"
            )

        if self.max_response_chars < 1:
            raise ValueError(
                "graphql.max_response_chars must be >= 1"
            )

        if self.max_rows < 1:
            raise ValueError(
                "graphql.max_rows must be >= 1"
            )


@dataclass(slots=True, frozen=True)
class Config:
    agent: AgentConfig
    llm: LLMConfig
    subagent: SubagentConfig
    context: ContextConfig
    memory: MemoryConfig
    models: ModelsConfig
    python: PythonConfig = PythonConfig()
    mcp: MCPConfig = MCPConfig()
    graphql: GraphQLConfig = GraphQLConfig()
    approval: ApprovalConfig = ApprovalConfig()
    tools: ToolsConfig = ToolsConfig()
    web: WebConfig = WebConfig()


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


def _price_per_million(
    section: Mapping[str, Any],
) -> tuple[float, float] | None:
    """Read (prompt, completion) USD per million tokens, or nothing.

    Only accepted when both are present and non-negative. Half a price
    would silently price one half of every request at zero, which is
    the one reading that is always wrong and never obvious.
    """

    prompt_price = section.get("price_prompt_per_million")
    completion_price = section.get(
        "price_completion_per_million"
    )

    if prompt_price is None or completion_price is None:
        return None

    try:
        prompt = float(prompt_price)  # type: ignore[arg-type]
        completion = float(completion_price)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    if prompt < 0 or completion < 0:
        return None

    return (prompt, completion)


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


def _strings(value: object, field: str) -> tuple[str, ...]:
    """A list of non-empty strings, or the empty tuple."""

    if value is None:
        return ()

    if not isinstance(value, list):
        raise ConfigError(
            f"{field} must be an array"
        )

    items: list[str] = []

    for entry in value:
        if not isinstance(entry, str):
            raise ConfigError(
                f"{field} must contain only strings"
            )

        text = entry.strip()

        if not text:
            raise ConfigError(
                f"{field} must not contain blanks"
            )

        items.append(text)

    return tuple(items)


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
        python_data = _table(data.get("python"), "python")
        graphql_data = _table(data.get("graphql"), "graphql")
        approval_data = _table(data.get("approval"), "approval")
        tools_data = _table(data.get("tools"), "tools")
        mcp_data = _table(data.get("mcp"), "mcp")

        return Config(
            agent=AgentConfig(
                max_iterations=_int(
                    agent_data.get("max_iterations", 10),
                    "agent.max_iterations",
                ),
                price_per_million=_price_per_million(
                    agent_data
                ),
                max_prompt_tokens=(
                    None
                    if agent_data.get("max_prompt_tokens") is None
                    else _int(
                        agent_data["max_prompt_tokens"],
                        "agent.max_prompt_tokens",
                    )
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
                compaction=_bool(
                    context_data.get("compaction", True),
                    "context.compaction",
                ),
                compaction_tokens=_int(
                    context_data.get(
                        "compaction_tokens",
                        1_200,
                    ),
                    "context.compaction_tokens",
                    minimum=100,
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
            python=_load_python(python_data),
            graphql=_load_graphql(graphql_data),
            approval=_load_approval(approval_data),
            tools=_load_tools(tools_data),
            web=_load_web(_table(data.get("web"), "web")),
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


def _load_python(data: dict) -> PythonConfig:
    """Parse the ``[python]`` table."""

    raw_blocked = data.get(
        "blocked_modules",
        list(PythonConfig().blocked_modules),
    )

    if not isinstance(raw_blocked, list):
        raise ConfigError(
            "python.blocked_modules must be an array",
        )

    return PythonConfig(
        enabled=_bool(data.get("enabled", True), "python.enabled"),
        interpreter=str(data.get("interpreter", "") or ""),
        sandbox_dir=str(
            data.get("sandbox_dir", "agentoflow-python")
        ),
        timeout=_float(
            data.get("timeout", 30.0),
            "python.timeout",
            minimum=1.0,
        ),
        memory_mb=_int(
            data.get("memory_mb", 1024),
            "python.memory_mb",
            minimum=0,
        ),
        cpu_seconds=_float(
            data.get("cpu_seconds", 60.0),
            "python.cpu_seconds",
            minimum=0.0,
        ),
        max_output=_int(
            data.get("max_output", 20_000),
            "python.max_output",
            minimum=1_000,
        ),
        max_timeout=_float(
            data.get("max_timeout", 120.0),
            "python.max_timeout",
            minimum=1.0,
        ),
        blocked_modules=tuple(
            str(item) for item in raw_blocked
        ),
    )


# Container runtimes do not pass the parent environment into the
# container unless told to. Without -e / --env, every variable set on
# a podman or docker entry lands in the runtime's own environment and
# is never seen by the MCP server, so it silently falls back to its
# defaults.
_CONTAINER_RUNTIMES = frozenset({"podman", "docker"})
_ENV_FLAGS = frozenset({"-e", "--env"})


def server_configuration_warnings(
    command: str,
    args: tuple[str, ...],
    env: dict[str, str],
) -> tuple[str, ...]:
    """Flag configuration that cannot work as written.

    A warning rather than an error: the variable may legitimately be
    meant for the runtime process rather than the container, and
    refusing to start would be the wrong call for that case.
    """

    if not env:
        return ()

    executable = Path(command).name

    if executable not in _CONTAINER_RUNTIMES:
        return ()

    if "run" not in args:
        return ()

    for index, argument in enumerate(args):
        if argument in _ENV_FLAGS:
            # "-e VAR" forwards from the environment; "--env VAR=x"
            # carries its own value. Both mean the operator is aware.
            following = args[index + 1] if index + 1 < len(args) else ""

            if argument == "-e" and not following.startswith("="):
                return ()

            if argument == "--env" and "=" in following:
                return ()

    forwarded = ", ".join(sorted(env))

    return (
        f"Server '{command}' is a container runtime, but its "
        f"arguments pass no -e/--env flag, so {forwarded} will not "
        "reach the container. The server will use its own defaults "
        "instead. Add '-e VAR' for each variable to forward it.",
    )


def _load_web(data: dict) -> WebConfig:
    """Parse the ``[web]`` table."""

    backend = str(data.get("backend", "duckduckgo"))

    timeout = data.get("searxng_timeout", 20.0)

    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
        raise ConfigError(
            "web.searxng_timeout must be a number"
        )

    try:
        return WebConfig(
            backend=backend,
            searxng_endpoint=str(
                data.get("searxng_endpoint", "http://localhost:8080")
            ).strip().rstrip("/"),
            searxng_timeout=float(timeout),
            searxng_categories=str(
                data.get("searxng_categories", "general")
            ).strip(),
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _load_tools(data: dict) -> ToolsConfig:
    """Parse the ``[tools]`` table."""

    disabled = data.get("disabled", [])

    if not isinstance(disabled, list):
        raise ConfigError(
            "tools.disabled must be an array of tool names"
        )

    return ToolsConfig(
        disabled=tuple(
            str(name) for name in disabled
        ),
    )


def _load_approval(data: dict) -> ApprovalConfig:
    """Parse the ``[approval]`` table."""

    return ApprovalConfig(
        remember=_bool(
            data.get("remember", True),
            "approval.remember",
        ),
    )


def _load_graphql(data: dict) -> GraphQLConfig:
    """Parse the ``[graphql]`` table."""

    return GraphQLConfig(
        enabled=_bool(
            data.get("enabled", False),
            "graphql.enabled",
        ),
        endpoint=str(data.get("endpoint", "") or ""),
        api_key=str(data.get("api_key", "") or ""),
        env_file=str(data.get("env_file", "") or ""),
        timeout=_float(
            data.get("timeout", 30.0),
            "graphql.timeout",
            minimum=1.0,
        ),
        max_query_chars=_int(
            data.get("max_query_chars", 8_000),
            "graphql.max_query_chars",
            minimum=100,
        ),
        max_response_chars=_int(
            data.get("max_response_chars", 60_000),
            "graphql.max_response_chars",
            minimum=1_000,
        ),
        max_rows=_int(
            data.get("max_rows", 250),
            "graphql.max_rows",
            minimum=1,
        ),
        max_scan_rows=_int(
            data.get("max_scan_rows", 5_000),
            "graphql.max_scan_rows",
            minimum=100,
        ),
        allow_introspection=_bool(
            data.get("allow_introspection", True),
            "graphql.allow_introspection",
        ),
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
        batch_concurrency=_int(
            data.get("batch_concurrency", 4),
            "mcp.batch_concurrency",
        ),
        batch_tools=_strings(
            # A list default, not a tuple: the helper is strict about
            # the type, and a tuple default made every config without
            # the key fail to load.
            data.get("batch_tools", []),
            "mcp.batch_tools",
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
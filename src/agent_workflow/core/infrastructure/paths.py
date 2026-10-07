from pathlib import Path


DEFAULT_CONFIG = """\
[agent]
max_iterations = 10

[llm]
provider = "ollama"
model = "gemma4:12b"
timeout = 600

[context]
max_messages = 24
# Replace a window the budget is about to drop with a handover note
# instead of forgetting it.
compaction = true
compaction_tokens = 1200

[memory]
database = "memory.db"

[python]
enabled = true
timeout = 30
memory_mb = 1024
cpu_seconds = 60
max_output = 20000
max_timeout = 120
blocked_modules = [
    "ctypes",
    "multiprocessing",
    "pty",
    "shutil",
    "socket",
    "ssl",
    "subprocess",
    "urllib",
    "webbrowser",
]

# Search backend. Every keyless search engine on the open internet
# answers a scripted client with a JavaScript challenge, so the
# default scraper usually returns nothing. Point this at an instance you
# run; see the SearXNG section of README.md for how to start one.
[web]
backend = "duckduckgo"
searxng_endpoint = "http://localhost:8080"
searxng_timeout = 20
searxng_categories = "general"

[approval]
# One grant covers later calls of the same tool for the rest of the
# session. Set false to be asked every single time.
remember = true

# Off until an endpoint is set: the tool can read the whole library.
[graphql]
enabled = false
endpoint = ""
timeout = 30
max_rows = 250
allow_introspection = true
"""


class AgentWorkflowPaths:
    def __init__(self, home: Path | None = None) -> None:
        self.home = home or Path.home() / ".agentoflow"

    @property
    def config(self) -> Path:
        return self.home / "config.toml"

    @property
    def models(self) -> Path:
        return self.home / "models.toml"

    @property
    def memory_database(self) -> Path:
        return self.home / "memory.db"

    @property
    def logs(self) -> Path:
        return self.home / "logs"

    @property
    def traces(self) -> Path:
        return self.home / "traces"

    @property
    def history(self) -> Path:
        return self.home / "history"

    @property
    def sessions(self) -> Path:
        """Where the web layer keeps its session registry."""

        return self.home / "sessions.json"

    def resolve(self, path: str | Path) -> Path:
        path = Path(path)

        if path.is_absolute():
            return path

        return self.home / path

    def ensure(self) -> None:
        self.home.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.logs.mkdir(exist_ok=True)
        self.traces.mkdir(exist_ok=True)

        if not self.config.exists():
            self.config.write_text(
                DEFAULT_CONFIG,
                encoding="utf-8",
            )
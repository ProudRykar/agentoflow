from pathlib import Path


DEFAULT_CONFIG = """\
[agent]
max_iterations = 10

[llm]
provider = "ollama"
model = "gemma4:12b"
timeout = 600

[context]
max_messages = 4

[memory]
database = "memory.db"
"""


class AgentWorkflowPaths:
    def __init__(self, home: Path | None = None) -> None:
        self.home = home or Path.home() / ".agentoflow"

    @property
    def config(self) -> Path:
        return self.home / "config.toml"

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
from __future__ import annotations

import ast
import asyncio
import os
import resource
import signal
from dataclasses import dataclass
from pathlib import Path

from agent_workflow.core.entities.models.builtin.execute_shell import (
    build_safe_environment,
)
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.infrastructure.config import PythonConfig


# ==========================================================================
# Static guardrail
# ==========================================================================

# Attribute calls that are never legitimate for a computation and are
# almost always a mistake when they appear.
BLOCKED_CALLS = frozenset({
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "input",
    "open_code",
    "quit",
})

BLOCKED_ATTRIBUTES = frozenset({
    ("os", "execv"),
    ("os", "execve"),
    ("os", "fork"),
    ("os", "kill"),
    ("os", "popen"),
    ("os", "spawnv"),
    ("os", "startfile"),
    ("os", "system"),
    ("shutil", "rmtree"),
})


class BlockedConstructError(Exception):
    """The script uses a construct the sandbox refuses to run.

    This is a guardrail against a careless script, not a sandbox.
    Python offers no way to forbid a call from inside the language,
    and ``getattr(__import__("os"), "system")("...")`` passes every
    static check. The real boundaries are the resource limits and
    the fact that the process runs as the agent's own user.
    """


def _dotted(node: ast.AST) -> str:
    """Render ``a.b.c`` from an attribute or name chain."""

    parts: list[str] = []

    current = node

    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value

    if isinstance(current, ast.Name):
        parts.append(current.id)

    return ".".join(reversed(parts))


def _module_of(node: ast.AST) -> str:
    """The root module name an import statement refers to."""

    if isinstance(node, ast.Import):
        return node.names[0].name.split(".")[0]

    if isinstance(node, ast.ImportFrom):
        if node.level:
            # A relative import cannot reach stdlib.
            return ""

        return (node.module or "").split(".")[0]

    return ""


def check_script(
    code: str,
    blocked_modules: frozenset[str],
) -> None:
    """Reject obvious dangerous constructs before running anything.

    Parsing first means a syntax error is reported as such instead of
    being swallowed by the interpreter.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise BlockedConstructError(
            f"Syntax error on line {exc.lineno}: {exc.msg}"
        ) from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            module = _module_of(node)

            if module and module in blocked_modules:
                raise BlockedConstructError(
                    f"Import of '{module}' is not allowed by the "
                    "python sandbox. Adjust "
                    "python.blocked_modules in config.toml if you "
                    "genuinely need it."
                )

            continue

        if isinstance(node, ast.Call):
            callee = node.func

            if isinstance(callee, ast.Name):
                if callee.id in BLOCKED_CALLS:
                    raise BlockedConstructError(
                        f"Call to '{callee.id}()' is not allowed"
                    )

            pair = tuple(
                _dotted(callee).split(".")
            )[-2:]

            if len(pair) == 2 and pair in BLOCKED_ATTRIBUTES:
                raise BlockedConstructError(
                    f"Call to '{'.'.join(pair)}()' is not allowed"
                )


# ==========================================================================
# Execution
# ==========================================================================


@dataclass(slots=True, frozen=True)
class PythonExecInput:
    code: str
    """Python source to run."""

    filename: str = ""
    """Optional script name, so tracebacks are readable."""

    timeout_seconds: float = 0.0
    """Override the default timeout, clamped to python.max_timeout."""


class PythonExecError(Exception):
    """The script exited non-zero or violated a limit."""

    def __init__(
        self,
        message: str,
        *,
        exit_code: int,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

        parts = [message]

        if stderr:
            parts.append(f"[stderr]\n{stderr}")

        if stdout:
            parts.append(f"[stdout]\n{stdout}")

        super().__init__("\n".join(parts))


def sandbox_root(
    config: PythonConfig,
    context: ToolContext,
) -> Path:
    """Where scripts run and write.

    Absolute paths in ``sandbox_dir`` are honoured; a relative value
    is resolved against the session's working directory, which keeps
    the sandbox inside the project by default and moves it to /tmp
    only when configured that way.
    """

    configured = Path(config.sandbox_dir).expanduser()

    if not configured.is_absolute():
        configured = context.working_directory / configured

    root = configured.resolve()
    root.mkdir(parents=True, exist_ok=True)

    return root


def _limits(
    memory_mb: int,
    cpu_seconds: float,
) -> object:
    """Build the preexec hook that applies rlimits.

    RLIMIT_AS caps the address space, which is what catches a runaway
    allocation. RLIMIT_CPU is a backstop for a tight loop that ignores
    the wall-clock timeout. RLIMIT_FSIZE keeps a script from filling
    the disk with one huge file.
    """

    def apply() -> None:
        os.setsid()

        if memory_mb > 0:
            limit = memory_mb * 1024 * 1024

            resource.setrlimit(
                resource.RLIMIT_AS,
                (limit, limit),
            )

        if cpu_seconds > 0:
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (int(cpu_seconds), int(cpu_seconds) + 1),
            )

        resource.setrlimit(
            resource.RLIMIT_FSIZE,
            (64 * 1024 * 1024, 64 * 1024 * 1024),
        )

        resource.setrlimit(
            resource.RLIMIT_CORE,
            (0, 0),
        )

    return apply


async def _read_limited(
    stream: asyncio.StreamReader,
    limit: int,
) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    total = 0
    truncated = False

    while True:
        chunk = await stream.read(4_096)

        if not chunk:
            break

        remaining = limit - total

        if remaining <= 0:
            truncated = True
            break

        if len(chunk) > remaining:
            chunks.append(chunk[:remaining])
            truncated = True
            break

        chunks.append(chunk)
        total += len(chunk)

    return b"".join(chunks), truncated


def _terminate_group(
    process: asyncio.subprocess.Process,
) -> None:
    if process.returncode is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def _safe_name(name: str) -> str:
    cleaned = "".join(
        character
        for character in name
        if character.isalnum() or character in ("_", "-", ".")
    )

    if not cleaned or cleaned.startswith("."):
        cleaned = f"script-{cleaned}" if cleaned else "script.py"

    if not cleaned.endswith(".py"):
        cleaned = f"{cleaned}.py"

    return cleaned


def _environment(
    context: ToolContext,
    root: Path,
) -> dict[str, str]:
    environment = build_safe_environment(context)

    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["TMPDIR"] = str(root)

    return environment


async def python_exec(
    arguments: PythonExecInput,
    context: ToolContext,
    config: PythonConfig,
) -> str:
    """Run a Python snippet inside the configured sandbox."""

    if not config.enabled:
        raise PythonExecError(
            "python_exec is disabled in the configuration",
            exit_code=-1,
        )

    if not arguments.code.strip():
        raise PythonExecError(
            "No code supplied",
            exit_code=-1,
        )

    check_script(
        arguments.code,
        frozenset(config.blocked_modules),
    )

    root = sandbox_root(config, context)

    filename = _safe_name(arguments.filename or "scratch.py")
    script = root / filename

    script.write_text(
        arguments.code,
        encoding="utf-8",
    )

    interpreter = config.interpreter or os.sys.executable

    timeout = config.timeout

    if arguments.timeout_seconds > 0:
        timeout = min(
            arguments.timeout_seconds,
            config.max_timeout,
        )

    process = await asyncio.create_subprocess_exec(
        interpreter,
        "-I",  # isolated: ignore PYTHON* and the user site directory
        str(script),
        cwd=str(root),
        env=_environment(context, root),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        preexec_fn=_limits(
            config.memory_mb,
            config.cpu_seconds,
        ),
    )

    assert process.stdout is not None
    assert process.stderr is not None

    stdout_task = asyncio.create_task(
        _read_limited(process.stdout, config.max_output)
    )
    stderr_task = asyncio.create_task(
        _read_limited(process.stderr, config.max_output)
    )

    timed_out = False

    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True

        _terminate_group(process)

        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass

    (stdout_bytes, stdout_cut), (stderr_bytes, stderr_cut) = (
        await asyncio.gather(
            stdout_task,
            stderr_task,
        )
    )

    stdout = stdout_bytes.decode("utf-8", errors="replace").rstrip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").rstrip()

    if stdout_cut:
        stdout = f"{stdout}\n[stdout truncated]"

    if stderr_cut:
        stderr = f"{stderr}\n[stderr truncated]"

    exit_code = process.returncode if process.returncode is not None else -1

    if timed_out:
        raise PythonExecError(
            f"Script timed out after {timeout:g}s and was killed. "
            f"It ran in {root}.",
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        )

    if exit_code != 0:
        detail = ""

        if exit_code < 0:
            detail = " (killed by a resource limit)"

        raise PythonExecError(
            f"Script exited with code {exit_code}{detail}",
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        )

    if not stdout and not stderr:
        return f"exit_code: 0\n(no output)\nscript: {script}"

    parts = ["exit_code: 0"]

    if stdout:
        parts.append(f"[stdout]\n{stdout}")

    if stderr:
        parts.append(f"[stderr]\n{stderr}")

    parts.append(f"script: {script}")

    return "\n".join(parts)
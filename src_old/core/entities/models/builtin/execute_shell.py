from __future__ import annotations

import asyncio
import os
import signal
from dataclasses import dataclass

from core.entities.models.path_policy import PathPolicy
from core.entities.models.shell_policy import ShellPolicy
from core.entities.models.tool import ToolContext


COMMAND_TIMEOUT = 10.0
MAX_OUTPUT_SIZE = 20_000
READ_CHUNK_SIZE = 4_096

SAFE_ENV_KEYS = frozenset({
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "USER",
})

SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"


@dataclass(slots=True, frozen=True)
class ExecuteShellInput:
    command: str


class ShellCommandError(Exception):
    """Raised when a shell command exits with a non-zero status."""

    def __init__(
        self,
        *,
        command: str,
        exit_code: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.command = command
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

        message_parts = [
            f"Command exited with code {exit_code}",
        ]

        if stderr:
            message_parts.append(
                f"stderr:\n{stderr}",
            )

        if stdout:
            message_parts.append(
                f"stdout:\n{stdout}",
            )

        super().__init__(
            "\n".join(message_parts),
        )


def build_safe_environment(
    context: ToolContext,
) -> dict[str, str]:
    """
    Build a minimal environment for the subprocess.

    We intentionally do not inherit the complete host environment.
    In particular, credentials, tokens, SSH configuration and unrelated
    process variables are not passed to the command.
    """

    environment: dict[str, str] = {
        "PATH": SAFE_PATH,
        "HOME": str(context.working_directory),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LC_CTYPE": "C.UTF-8",
        "PAGER": "cat",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
    }

    for key in SAFE_ENV_KEYS:
        value = context.environment.get(key)

        if value is not None:
            environment[key] = value

    # Never allow the caller's PATH to override our restricted PATH.
    environment["PATH"] = SAFE_PATH

    return environment


async def _terminate_process_group(
    process: asyncio.subprocess.Process,
) -> None:
    """
    Kill the whole process group.

    This matters because a command can spawn child processes. Killing only
    the direct child may leave descendants running after timeout.
    """

    if process.returncode is not None:
        return

    try:
        os.killpg(
            process.pid,
            signal.SIGKILL,
        )
    except ProcessLookupError:
        return

    try:
        await process.wait()
    except ProcessLookupError:
        return


async def _read_limited(
    stream: asyncio.StreamReader,
    limit: int,
) -> tuple[bytes, bool]:
    """
    Read at most `limit` bytes from a subprocess stream.

    Returns:
        (data, truncated)
    """

    chunks: list[bytes] = []
    total = 0
    truncated = False

    while True:
        chunk = await stream.read(READ_CHUNK_SIZE)

        if not chunk:
            break

        remaining = limit - total

        if remaining <= 0:
            truncated = True
            break

        if len(chunk) > remaining:
            chunks.append(chunk[:remaining])
            total += remaining
            truncated = True
            break

        chunks.append(chunk)
        total += len(chunk)

        if total >= limit:
            # There may still be unread data. We deliberately stop here.
            truncated = True
            break

    return b"".join(chunks), truncated


async def _execute_argv(
    argv: list[str],
    context: ToolContext,
) -> tuple[int, str, str, bool, bool]:
    """
    Execute an argv vector without invoking a shell.

    Returns:
        exit_code,
        stdout,
        stderr,
        stdout_truncated,
        stderr_truncated
    """

    environment = build_safe_environment(context)

    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=context.working_directory,
        env=environment,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    assert process.stdout is not None
    assert process.stderr is not None

    stdout_task = asyncio.create_task(
        _read_limited(
            process.stdout,
            MAX_OUTPUT_SIZE,
        ),
    )

    stderr_task = asyncio.create_task(
        _read_limited(
            process.stderr,
            MAX_OUTPUT_SIZE,
        ),
    )

    try:
        await asyncio.wait_for(
            process.wait(),
            timeout=COMMAND_TIMEOUT,
        )

        (
            stdout_bytes,
            stdout_truncated,
        ), (
            stderr_bytes,
            stderr_truncated,
        ) = await asyncio.gather(
            stdout_task,
            stderr_task,
        )

    except asyncio.TimeoutError:
        stdout_task.cancel()
        stderr_task.cancel()

        await asyncio.gather(
            stdout_task,
            stderr_task,
            return_exceptions=True,
        )

        await _terminate_process_group(
            process,
        )

        raise TimeoutError(
            f"Shell command timed out after "
            f"{COMMAND_TIMEOUT} seconds",
        ) from None

    return (
        process.returncode or 0,
        stdout_bytes.decode(
            "utf-8",
            errors="replace",
        ),
        stderr_bytes.decode(
            "utf-8",
            errors="replace",
        ),
        stdout_truncated,
        stderr_truncated,
    )


def _format_output(
    *,
    exit_code: int,
    stdout: str,
    stderr: str,
    stdout_truncated: bool,
    stderr_truncated: bool,
) -> str:
    parts = [
        f"exit_code: {exit_code}",
    ]

    if stdout:
        stdout_suffix = (
            "\n[stdout truncated]"
            if stdout_truncated
            else ""
        )

        parts.append(
            f"[stdout]\n{stdout}{stdout_suffix}",
        )

    if stderr:
        stderr_suffix = (
            "\n[stderr truncated]"
            if stderr_truncated
            else ""
        )

        parts.append(
            f"[stderr]\n{stderr}{stderr_suffix}",
        )

    if len(parts) == 1:
        parts.append("(no output)")

    return "\n".join(parts)


async def execute_shell(
    arguments: ExecuteShellInput,
    context: ToolContext,
) -> str:
    """
    Execute a policy-checked command without invoking a shell.

    The command is parsed and validated by ShellPolicy before execution.
    Filesystem paths are checked against the same PathPolicy used by the
    dedicated filesystem tools.
    """

    path_policy = PathPolicy(
        context.allowed_path,
    )

    policy = ShellPolicy(
        path_policy=path_policy,
    )

    argv = policy.validate(
        command=arguments.command,
        context=context,
    )

    (
        exit_code,
        stdout,
        stderr,
        stdout_truncated,
        stderr_truncated,
    ) = await _execute_argv(
        argv,
        context,
    )

    stdout = stdout.rstrip()
    stderr = stderr.rstrip()

    if exit_code != 0:
        raise ShellCommandError(
            command=arguments.command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        )

    return _format_output(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )
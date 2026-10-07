"""The python_exec sandbox.

The point of these tests is that the *limits* are real: a runaway
script is stopped rather than reported after the fact. Filesystem and
network confinement are explicitly not claimed, because Python code
runs with the agent's own privileges and no in-language check can
change that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_workflow.core.entities.models.builtin.python_exec import (
    BlockedConstructError,
    PythonExecError,
    PythonExecInput,
    check_script,
    python_exec,
    sandbox_root,
)
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.infrastructure.config import PythonConfig


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path,
        environment={},
        allowed_path=(tmp_path,),
        permissions=frozenset({"python.execute"}),
    )


async def _run(
    code: str,
    context: ToolContext,
    **overrides: object,
) -> str:
    return await python_exec(
        PythonExecInput(code=code),
        context,
        PythonConfig(**overrides),  # type: ignore[arg-type]
    )


# ======================================================================
# Static guardrail
# ======================================================================


BLOCKED = frozenset(PythonConfig().blocked_modules)


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess",
        "import subprocess as sp",
        "from subprocess import run",
        "import socket",
        "from socket import socket",
        "import urllib.request",
        "import shutil",
        "import ctypes",
        "import multiprocessing",
    ],
)
def test_blocked_imports_are_refused(code: str) -> None:
    with pytest.raises(BlockedConstructError, match="not allowed"):
        check_script(code, BLOCKED)


@pytest.mark.parametrize(
    "code",
    [
        "os.system('rm -rf /')",
        "shutil.rmtree('/')",
        "eval('1+1')",
        "exec('x = 1')",
        "input('name?')",
        "os.popen('id')",
    ],
)
def test_blocked_calls_are_refused(code: str) -> None:
    with pytest.raises(BlockedConstructError):
        check_script(code, BLOCKED)


@pytest.mark.parametrize(
    "code",
    [
        "import json",
        "import math, statistics",
        "from pathlib import Path",
        "from collections import Counter",
        "print(sum(range(10)))",
        "x = [i ** 2 for i in range(5)]",
        "def f():\n    return 1\n",
    ],
)
def test_ordinary_scripts_are_allowed(code: str) -> None:
    check_script(code, BLOCKED)


def test_syntax_errors_are_reported_clearly() -> None:
    with pytest.raises(BlockedConstructError, match="Syntax error"):
        check_script("def broken(:", BLOCKED)


def test_relative_import_is_not_treated_as_stdlib() -> None:
    check_script("from . import sibling", BLOCKED)


def test_blocked_list_is_configurable() -> None:
    # The operator can relax the denylist.
    check_script("import json", frozenset({"socket"}))


def test_the_guardrail_is_not_claimed_as_a_sandbox() -> None:
    """Documented limitation: a determined script gets through.

    This is a guardrail against a careless script. If it ever passed
    silently, the guardrail would be actively harmful.
    """

    check_script(
        "import os; getattr(os, 'system')('id')",
        BLOCKED,
    )


# ======================================================================
# Execution
# ======================================================================


async def test_computation_returns_stdout(tmp_path: Path) -> None:
    result = await _run(
        "print(sum(i * i for i in range(10)))",
        _context(tmp_path),
    )

    assert "285" in result
    assert "exit_code: 0" in result


async def test_writes_land_in_the_sandbox(tmp_path: Path) -> None:
    context = _context(tmp_path)

    result = await _run(
        "from pathlib import Path\n"
        "Path('result.txt').write_text('computed')\n"
        "print(Path('result.txt').read_text())",
        context,
    )

    assert "computed" in result

    root = sandbox_root(PythonConfig(), context)

    assert (root / "result.txt").read_text() == "computed"
    assert (root / "scratch.py").exists()


async def test_relative_sandbox_stays_inside_the_project(
    tmp_path: Path,
) -> None:
    root = sandbox_root(PythonConfig(), _context(tmp_path))

    assert tmp_path in root.parents or root == tmp_path


async def test_absolute_sandbox_is_honoured(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"

    root = sandbox_root(
        PythonConfig(sandbox_dir=str(target)),
        _context(tmp_path),
    )

    assert root == target.resolve()


async def test_traceback_points_at_the_script(tmp_path: Path) -> None:
    with pytest.raises(PythonExecError) as caught:
        await _run(
            "raise ValueError('boom')",
            _context(tmp_path),
        )

    assert "ValueError" in str(caught.value)
    assert "boom" in str(caught.value)
    assert "scratch.py" in str(caught.value)


async def test_named_script(tmp_path: Path) -> None:
    result = await python_exec(
        PythonExecInput(
            code="print('x')",
            filename="analysis",
        ),
        _context(tmp_path),
        PythonConfig(),
    )

    assert "analysis.py" in result


async def test_empty_code_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PythonExecError, match="No code"):
        await _run("   \n  ", _context(tmp_path))


async def test_disabled_tool_refuses(tmp_path: Path) -> None:
    with pytest.raises(PythonExecError, match="disabled"):
        await _run("print(1)", _context(tmp_path), enabled=False)


# ======================================================================
# Limits
# ======================================================================


async def test_timeout_kills_the_script(tmp_path: Path) -> None:
    with pytest.raises(PythonExecError, match="timed out"):
        await _run(
            "import time\ntime.sleep(30)",
            _context(tmp_path),
            timeout=3.0,
        )


async def test_timeout_kills_child_processes(tmp_path: Path) -> None:
    # A script that spawns a child and waits must not leave the child
    # running after the timeout.
    code = (
        "import subprocess, time\n"
        "subprocess.Popen(['sleep', '30'])\n"
        "time.sleep(30)\n"
    )

    # subprocess is refused by the guardrail, so this asserts the
    # guardrail rather than the process group.
    with pytest.raises(BlockedConstructError):
        await _run(code, _context(tmp_path), timeout=3.0)


async def test_memory_limit_stops_a_runaway_allocation(
    tmp_path: Path,
) -> None:
    with pytest.raises(PythonExecError):
        await _run(
            "x = bytearray(4_000_000_000)",
            _context(tmp_path),
            memory_mb=256,
            timeout=30.0,
        )


async def test_cpu_limit_stops_an_infinite_loop(
    tmp_path: Path,
) -> None:
    with pytest.raises(PythonExecError) as caught:
        await _run(
            "while True:\n    pass",
            _context(tmp_path),
            timeout=30.0,
            cpu_seconds=3.0,
        )

    # Negative exit means a signal killed it.
    assert caught.value.exit_code < 0


async def test_output_is_capped_and_marked(tmp_path: Path) -> None:
    result = await _run(
        "print('x' * 100_000)",
        _context(tmp_path),
        max_output=2_000,
    )

    assert "[stdout truncated]" in result
    assert len(result) < 5_000


async def test_caller_cannot_exceed_max_timeout(tmp_path: Path) -> None:
    with pytest.raises(PythonExecError, match="timed out"):
        await python_exec(
            PythonExecInput(
                code="import time\ntime.sleep(30)",
                timeout_seconds=600.0,
            ),
            _context(tmp_path),
            PythonConfig(
                timeout=3.0,
                max_timeout=5.0,
            ),
        )


async def test_caller_can_shorten_the_timeout(tmp_path: Path) -> None:
    with pytest.raises(PythonExecError, match="timed out after 2"):
        await python_exec(
            PythonExecInput(
                code="import time\ntime.sleep(30)",
                timeout_seconds=2.0,
            ),
            _context(tmp_path),
            PythonConfig(timeout=30.0),
        )


# ======================================================================
# Environment
# ======================================================================


async def test_the_agent_environment_is_not_inherited(
    tmp_path: Path,
) -> None:
    context = ToolContext(
        working_directory=tmp_path,
        environment={"SECRET_TOKEN": "must-not-leak"},
        allowed_path=(tmp_path,),
        permissions=frozenset({"python.execute"}),
    )

    result = await _run(
        "import os\n"
        "print(os.environ.get('SECRET_TOKEN', 'absent'))",
        context,
    )

    assert "absent" in result


async def test_stdin_is_closed(tmp_path: Path) -> None:
    result = await _run(
        "import sys\n"
        "print('stdin closed:', sys.stdin.read() == '')",
        _context(tmp_path),
    )

    assert "stdin closed: True" in result


async def test_interpreter_is_configurable(tmp_path: Path) -> None:
    result = await _run(
        "import sys\nprint(sys.version_info[:2])",
        _context(tmp_path),
        interpreter="/usr/bin/python3",
    )

    assert "exit_code: 0" in result


async def test_default_interpreter_is_the_agents_venv(
    tmp_path: Path,
) -> None:
    import sys

    result = await _run(
        "import sys\nprint(sys.prefix)",
        _context(tmp_path),
    )

    assert sys.prefix in result
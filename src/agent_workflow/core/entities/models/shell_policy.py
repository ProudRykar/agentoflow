from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from agent_workflow.core.entities.models.tool import ToolContext


@dataclass(slots=True, frozen=True)
class ShellCommandClassification:
    executable: str
    category: str
    permission: str | None


class ShellPolicyError(Exception):
    """Raised when a shell command violates the execution policy."""


# ---------------------------------------------------------------------------
# Shell command categories
# ---------------------------------------------------------------------------

SHELL_NETWORK_COMMANDS: Final = frozenset({
    "aria2c",
    "curl",
    "ftp",
    "http",
    "httpie",
    "nc",
    "netcat",
    "ncat",
    "nslookup",
    "rsync",
    "scp",
    "sftp",
    "ssh",
    "telnet",
    "tftp",
    "wget",
})


SHELL_PACKAGE_MANAGERS: Final = frozenset({
    "apk",
    "apt",
    "apt-cache",
    "apt-get",
    "aptitude",
    "brew",
    "cargo",
    "conda",
    "dnf",
    "emerge",
    "flatpak",
    "gem",
    "go",
    "gradle",
    "mamba",
    "micromamba",
    "mix",
    "npm",
    "npx",
    "pacman",
    "paru",
    "pip",
    "pip3",
    "pipx",
    "pnpm",
    "poetry",
    "port",
    "rpm",
    "snap",
    "uv",
    "yay",
    "yarn",
})


SHELL_ADMIN_COMMANDS: Final = frozenset({
    "chattr",
    "chgrp",
    "chmod",
    "chown",
    "chroot",
    "crontab",
    "doas",
    "insmod",
    "iptables",
    "journalctl",
    "kexec",
    "mount",
    "modprobe",
    "passwd",
    "pkexec",
    "poweroff",
    "reboot",
    "rmmod",
    "runuser",
    "service",
    "shutdown",
    "su",
    "sudo",
    "systemctl",
    "sysctl",
    "umount",
    "useradd",
    "userdel",
    "usermod",
})


SHELL_DESTRUCTIVE_COMMANDS: Final = frozenset({
    "dd",
    "fdisk",
    "kill",
    "killall",
    "mkfs",
    "mkswap",
    "parted",
    "pkill",
    "poweroff",
    "reboot",
    "rm",
    "rmdir",
    "shred",
    "shutdown",
    "truncate",
    "unlink",
    "wipefs",
})


SHELL_WRITE_COMMANDS: Final = frozenset({
    "cp",
    "install",
    "ln",
    "mkdir",
    "mv",
    "rename",
    "sed",
    "tee",
    "touch",
})


SHELL_INTERPRETERS: Final = frozenset({
    "ash",
    "bash",
    "csh",
    "dash",
    "fish",
    "ksh",
    "node",
    "nodejs",
    "perl",
    "php",
    "python",
    "python2",
    "python3",
    "ruby",
    "sh",
    "tcsh",
    "zsh",
})


SHELL_COMMAND_WRAPPERS: Final = frozenset({
    "busybox",
    "command",
    "env",
    "exec",
    "find",
    "nice",
    "nohup",
    "setsid",
    "timeout",
    "watch",
    "xargs",
})


# ---------------------------------------------------------------------------
# Read-only commands
# ---------------------------------------------------------------------------

SHELL_READ_ONLY_COMMANDS: Final = frozenset({
    "basename",
    "cat",
    "cut",
    "dirname",
    "du",
    "echo",
    "file",
    "git",
    "grep",
    "head",
    "ls",
    "printf",
    "pwd",
    "readlink",
    "realpath",
    "rg",
    "sort",
    "stat",
    "tail",
    "tree",
    "tr",
    "uniq",
    "wc",
})


# ---------------------------------------------------------------------------
# Shell syntax
# ---------------------------------------------------------------------------

FORBIDDEN_SHELL_OPERATORS: Final = frozenset({
    ";",
    "&&",
    "||",
    "|",
    ">",
    ">>",
    "<",
    "<<",
    "&",
})


FORBIDDEN_SHELL_SUBSTITUTION_MARKERS: Final = frozenset({
    "`",
})


# ---------------------------------------------------------------------------
# Command-specific restrictions
# ---------------------------------------------------------------------------

FORBIDDEN_ARGUMENTS: Final[dict[str, frozenset[str]]] = {
    "find": frozenset({
        "-delete",
        "-exec",
        "-execdir",
        "-ok",
        "-okdir",
    }),
    "git": frozenset({
        "--exec-path",
    }),
}


GIT_MUTATING_COMMANDS: Final = frozenset({
    "add",
    "am",
    "apply",
    "branch",
    "checkout",
    "cherry-pick",
    "clean",
    "clone",
    "commit",
    "config",
    "fetch",
    "init",
    "merge",
    "pull",
    "push",
    "rebase",
    "remote",
    "reset",
    "restore",
    "revert",
    "stash",
    "switch",
    "tag",
})


GIT_READ_ONLY_COMMANDS: Final = frozenset({
    "blame",
    "diff",
    "describe",
    "log",
    "ls-files",
    "rev-parse",
    "shortlog",
    "show",
    "status",
})


# ---------------------------------------------------------------------------
# Permissions that can and cannot be approved
# ---------------------------------------------------------------------------

APPROVABLE_PERMISSIONS: Final = frozenset({
    "shell.network",
    "shell.write",
    "shell.destructive",
})


NEVER_APPROVABLE_PERMISSIONS: Final = frozenset({
    "shell.package_install",
    "shell.admin",
    "shell.interpreter",
    "shell.wrapper",
})


class ShellPolicy:
    """
    Central policy for shell command classification and validation.

    Security model:

    - shell execution itself is controlled by ToolPolicy("shell.execute")
    - read-only commands require no additional capability
    - network/write/destructive operations require explicit approval
    - package managers, admin commands, interpreters and command wrappers
      are permanently blocked
    - shell syntax such as pipes, redirection and command chaining is blocked
    - all filesystem paths are checked through PathPolicy
    - unknown commands are denied by default
    """

    def __init__(
        self,
        path_policy,
    ) -> None:
        self._path_policy = path_policy

    def validate(
        self,
        command: str,
        context: ToolContext,
    ) -> list[str]:
        self._validate_command_text(command)

        try:
            argv = shlex.split(
                command,
                posix=True,
            )
        except ValueError as exc:
            raise ShellPolicyError(
                f"Invalid shell command syntax: {exc}",
            ) from exc

        if not argv:
            raise ShellPolicyError(
                "Shell command must not be empty",
            )

        self._validate_argv_syntax(argv)

        executable = self._command_name(argv[0])

        effective_permissions = (
            context.permissions
            | context.approved_permissions
        )

        self._validate_executable(
            executable=executable,
            permissions=effective_permissions,
        )

        self._validate_arguments(
            executable=executable,
            arguments=argv[1:],
        )

        self._validate_paths(
            executable=executable,
            arguments=argv[1:],
            context=context,
        )

        return argv

    # -----------------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------------

    @staticmethod
    def classify(
        executable: str,
    ) -> ShellCommandClassification:
        executable = Path(executable).name

        if executable in SHELL_DESTRUCTIVE_COMMANDS:
            return ShellCommandClassification(
                executable=executable,
                category="destructive",
                permission="shell.destructive",
            )

        if executable in SHELL_ADMIN_COMMANDS:
            return ShellCommandClassification(
                executable=executable,
                category="admin",
                permission="shell.admin",
            )

        if executable in SHELL_NETWORK_COMMANDS:
            return ShellCommandClassification(
                executable=executable,
                category="network",
                permission="shell.network",
            )

        if executable in SHELL_PACKAGE_MANAGERS:
            return ShellCommandClassification(
                executable=executable,
                category="package_manager",
                permission="shell.package_install",
            )

        if executable in SHELL_WRITE_COMMANDS:
            return ShellCommandClassification(
                executable=executable,
                category="write",
                permission="shell.write",
            )

        if executable in SHELL_INTERPRETERS:
            return ShellCommandClassification(
                executable=executable,
                category="interpreter",
                permission="shell.interpreter",
            )

        if executable in SHELL_COMMAND_WRAPPERS:
            return ShellCommandClassification(
                executable=executable,
                category="wrapper",
                permission="shell.wrapper",
            )

        if executable in SHELL_READ_ONLY_COMMANDS:
            return ShellCommandClassification(
                executable=executable,
                category="read_only",
                permission=None,
            )

        return ShellCommandClassification(
            executable=executable,
            category="unknown",
            permission=None,
        )

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    @staticmethod
    def _validate_command_text(
        command: str,
    ) -> None:
        if not command.strip():
            raise ShellPolicyError(
                "Shell command must not be empty",
            )

        if "\x00" in command:
            raise ShellPolicyError(
                "NUL bytes are not allowed in shell commands",
            )

        if "\n" in command or "\r" in command:
            raise ShellPolicyError(
                "Newlines are not allowed in shell commands",
            )

        for marker in FORBIDDEN_SHELL_SUBSTITUTION_MARKERS:
            if marker in command:
                raise ShellPolicyError(
                    f"Shell syntax '{marker}' is not allowed",
                )

    @staticmethod
    def _validate_argv_syntax(
        argv: list[str],
    ) -> None:
        for token in argv:
            if token in FORBIDDEN_SHELL_OPERATORS:
                raise ShellPolicyError(
                    f"Shell operator '{token}' is not allowed",
                )

            if any(
                operator in token
                for operator in (
                    "&&",
                    "||",
                    ">>",
                    "<<",
                )
            ):
                raise ShellPolicyError(
                    "Shell command chaining and redirection "
                    "are not allowed",
                )

            if "$(" in token:
                raise ShellPolicyError(
                    "Command substitution is not allowed",
                )

            if "${" in token:
                raise ShellPolicyError(
                    "Shell variable expansion is not allowed",
                )

            if "\x00" in token:
                raise ShellPolicyError(
                    "NUL bytes are not allowed",
                )

    @staticmethod
    def _command_name(
        executable: str,
    ) -> str:
        return Path(executable).name

    def _validate_executable(
        self,
        *,
        executable: str,
        permissions: frozenset[str],
    ) -> ShellCommandClassification:
        classification = self.classify(
            executable,
        )

        if classification.category == "read_only":
            return classification

        permission = classification.permission

        if (
            classification.category in {
                "package_manager",
                "admin",
                "interpreter",
                "wrapper",
            }
        ):
            if permission is None:
                raise ShellPolicyError(
                    f"Command '{executable}' is permanently blocked",
                )

            raise ShellPolicyError(
                f"Command '{executable}' is permanently blocked; "
                f"permission '{permission}' cannot be granted",
            )

        if permission in permissions:
            return classification

        if classification.category == "network":
            raise ShellPolicyError(
                "Network access requires permission "
                "'shell.network'",
            )

        if classification.category == "write":
            raise ShellPolicyError(
                "Filesystem modification requires permission "
                "'shell.write'",
            )

        if classification.category == "destructive":
            raise ShellPolicyError(
                "Destructive command requires permission "
                "'shell.destructive'",
            )

        raise ShellPolicyError(
            f"Command '{classification.executable}' "
            "is not in the shell allow-list",
        )

    def _validate_arguments(
        self,
        *,
        executable: str,
        arguments: list[str],
    ) -> None:
        forbidden = FORBIDDEN_ARGUMENTS.get(
            executable,
            frozenset(),
        )

        for argument in arguments:
            if argument in forbidden:
                raise ShellPolicyError(
                    f"Argument '{argument}' is blocked for "
                    f"command '{executable}'",
                )

        if executable == "git":
            self._validate_git(arguments)

    @staticmethod
    def _validate_git(
        arguments: list[str],
    ) -> None:
        if not arguments:
            raise ShellPolicyError(
                "Bare 'git' command is not allowed",
            )

        command = next(
            (
                argument
                for argument in arguments
                if not argument.startswith("-")
            ),
            None,
        )

        if command is None:
            raise ShellPolicyError(
                "Git command must specify a read-only operation",
            )

        if command in GIT_MUTATING_COMMANDS:
            raise ShellPolicyError(
                f"Git operation '{command}' is blocked",
            )

        if command not in GIT_READ_ONLY_COMMANDS:
            raise ShellPolicyError(
                f"Git operation '{command}' is not allowed",
            )

    def _validate_paths(
        self,
        *,
        executable: str,
        arguments: list[str],
        context: ToolContext,
    ) -> None:
        for index in self._path_argument_indexes(
            executable,
            arguments,
        ):
            argument = arguments[index]

            if not argument or argument.startswith("-"):
                continue

            path = Path(argument)

            if path.is_absolute():
                raise ShellPolicyError(
                    f"Absolute path '{argument}' is not allowed",
                )

            try:
                self._path_policy.resolve(
                    context.working_directory / path,
                )
            except Exception as exc:
                raise ShellPolicyError(
                    f"Path '{argument}' is outside allowed paths",
                ) from exc

    @staticmethod
    def _path_argument_indexes(
        executable: str,
        arguments: list[str],
    ) -> tuple[int, ...]:
        if executable in {
            "cat",
            "du",
            "file",
            "head",
            "ls",
            "readlink",
            "realpath",
            "stat",
            "tail",
        }:
            return tuple(
                index
                for index, argument in enumerate(arguments)
                if not argument.startswith("-")
            )

        if executable in {"grep", "rg"}:
            non_options = [
                index
                for index, argument in enumerate(arguments)
                if not argument.startswith("-")
            ]

            if len(non_options) <= 1:
                return ()

            return tuple(non_options[1:])

        if executable == "git":
            return ShellPolicy._git_path_argument_indexes(
                arguments,
            )

        return ()

    @staticmethod
    def _git_path_argument_indexes(
        arguments: list[str],
    ) -> tuple[int, ...]:
        if not arguments:
            return ()

        command_index: int | None = None

        for index, argument in enumerate(arguments):
            if not argument.startswith("-"):
                command_index = index
                break

        if command_index is None:
            return ()

        command = arguments[command_index]

        if command not in GIT_READ_ONLY_COMMANDS:
            return ()

        return tuple(
            index
            for index, argument in enumerate(arguments)
            if index > command_index
            and not argument.startswith("-")
        )
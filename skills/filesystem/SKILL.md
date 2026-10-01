---
name: filesystem-cleanup
description: Instructions for cleaning up and organizing filesystem.
version: 1.0.0
tags: [filesystem, cleanup, organization, maintenance]
---

# Filesystem Cleanup

## Purpose

Organize, clean up, and maintain filesystem structure.

## Workflow

1. Use `filesystem.list_directory` to inspect directory structure.
2. Use `filesystem.find_files` to locate files by pattern.
3. Identify temporary, backup, or duplicate files.
4. Use `filesystem.read` to verify file contents before deletion.
5. Use `filesystem.edit_file` or `execute_shell` to remove/move files.
6. Verify cleanup results.

## Rules

- Never delete files without verifying contents.
- Preserve git-tracked files unless explicitly requested.
- Ask for confirmation before destructive operations.
- Prefer moving to trash over permanent deletion.
- Maintain directory structure logic.

## Tools

- `filesystem.list_directory` - List directory contents
- `filesystem.find_files` - Find files by pattern
- `filesystem.read` - Read file contents
- `filesystem.write_file` - Write files
- `execute_shell` - Run cleanup commands
---
name: python-testing
description: Instructions for writing and running Python tests.
version: 1.0.0
tags: [python, testing, pytest, quality]
---

# Python Testing

## Purpose

Write, run, and debug Python tests using pytest and related tools.

## Workflow

1. Use `filesystem.read` to examine existing test files.
2. Use `filesystem.search_files` to find test patterns.
3. Write tests following project conventions (pytest, unittest).
4. Use `execute_shell` to run tests with `pytest`.
5. Analyze test output and fix failures.
6. Ensure coverage targets are met.

## Rules

- Follow existing test naming conventions.
- Use fixtures for common setup.
- Prefer parametrized tests for multiple cases.
- Mock external dependencies.
- Run tests in isolation when possible.

## Tools

- `filesystem.read` - Read test files
- `filesystem.write_file` - Write test files
- `filesystem.edit_file` - Modify test files
- `execute_shell` - Run pytest
- `filesystem.search_files` - Find test patterns
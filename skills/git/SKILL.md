---
name: git-code-review
description: Instructions for reviewing Git changes and commits.
version: 1.0.0
tags: [git, review, code-quality]
---

# Git Code Review

## Purpose

Safely inspect and review Git repository changes, commits, and history.

## Workflow

1. Use `git.log` to inspect recent commits.
2. Use `git.diff` to review changes in working directory or staged.
3. Use `git.show` to examine specific commits.
4. Analyze commit messages for clarity and completeness.
5. Check for sensitive data, large files, or breaking changes.
6. Provide structured feedback.

## Rules

- Never rewrite history without explicit user approval.
- Prefer `git.diff` over `git.show` for working directory changes.
- Always verify the branch context before reviewing.
- Report potential issues: large files, secrets, merge conflicts.

## Tools

- `git.log` - List commits
- `git.diff` - Show changes
- `git.show` - Show commit details
- `git.status` - Repository status
- `git.branch` - Branch information
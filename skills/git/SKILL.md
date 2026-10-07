---
name: git-code-review
description: Inspect and review Git changes and commits before they are pushed.
version: 1.1.0
tags: [git, review, code-quality]
---

# Git Code Review

## Purpose

Inspect and review Git repository changes and history without
changing anything.

## Tools

There is no `git.*` tool. Git is run through `execute_shell`, which
means every command is visible to the operator and may be subject to
approval.

- `execute_shell` - run `git`, `diff`, `log`, `show`, `status`
- `read_file` - read a file the diff points at
- `search_files` - find related code

## Read-only commands

Use these to gather evidence. None of them modify the repository.

```bash
git status                      # what is changed and staged
git diff                        # unstaged changes
git diff --staged               # staged changes
git log --oneline -10           # recent commits
git show <sha>                  # one commit in full
git diff --stat                 # which files, how large
git diff -- <path>              # one file only
git log -p -- <path>            # history of one file
```

`git diff` with no arguments is the one to reach for: it shows working
tree changes, which is what is under review most of the time.
`git show` is for a commit that is already committed.

## Workflow

1. `git status` to learn the branch and what is staged.
2. `git diff --stat` for the shape of the change.
3. `git diff` to read it. Narrow with `-- <path>` once the change is
   large.
4. For each suspicious hunk, `read_file` the surrounding code. A diff
   alone hides whether a change is correct in context.
5. Report what you found, separated from what you merely noticed.

## What to look for

- Secrets: API keys, tokens, passwords, private keys, `.env` contents.
- Debug leftovers: `print`, `console.log`, `breakpoint`, commented-out
  code, stray `.only(`.
- Unrelated changes riding along in the same commit.
- Generated or vendored files committed by accident.
- Tests missing for changed behaviour.
- Renames that look like deletions, which `git diff -M` reveals.
- Trailing whitespace and merge markers.

## Rules

- **Never** rewrite history, amend, force-push, reset, or clean
  without explicit instruction in the current conversation.
- Never commit or push unless asked in that same message.
- Stage deliberately. Never `git add .` when only some files belong to
  the change.
- Report findings with `file:line` so they can be navigated to.
- Say plainly when something looks wrong. Do not soften a real defect
  into a suggestion, and do not invent findings to seem thorough.
- If the working tree is dirty in a way that complicates the review,
  say so before reviewing.

## Reporting

For each finding: the location, what is wrong, and why it matters.
Keep correctness problems separate from style preferences, and label
style preferences as such.
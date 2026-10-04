# agentoflow

A local LLM agent harness with two interchangeable front ends: a
Textual TUI and a React web UI. Both are clients of the same agent
core.

## Architecture

```
                    ┌──────────────────────────────┐
                    │        React Web UI          │
                    │                              │
                    │ chat / tools / approvals     │
                    │ skills / plugins / runs      │
                    └──────────────┬───────────────┘
                                   │
                          WebSocket + HTTP
                                   │
                    ┌──────────────▼───────────────┐
                    │       Web/API layer          │
                    │                              │
                    │ FastAPI routes + WS adapter  │
                    └──────────────┬───────────────┘
                                   │
                          SessionManager
                                   │
                    ┌──────────────▼───────────────┐
                    │       AgentSession           │
                    │                              │
                    │ EventBus + ApprovalController│
                    └──────────────┬───────────────┘
                                   │
                             AgentRuntime
                                   │
                    ┌──────────────▼───────────────┐
                    │         Agent Core           │
                    │                              │
                    │ Agent / ToolRegistry         │
                    │ ToolContext / ToolExecutor    │
                    │ Skills / Plugins / Subagents │
                    │ Memory / LLM                  │
                    └──────────────────────────────┘
```

The original Textual TUI plugs into the same `AgentRuntime` and
`ApprovalController`:

```
Agent Core → AgentRuntime → ┬ → SessionManager → Web API → React
                            └ → ApprovalController → Textual TUI
```

### Why an EventBus

`Agent.run()` already accepts a single `on_event` callback and
emits the full `AgentEvent` union. Rather than replace that API,
the session keeps a stable reference to its own `emit` method and
passes it as `on_event`. `EventBus` adapts that one callback into a
fan-out with sequence numbers and a bounded replay buffer.

This is what lets several WebSocket clients observe the same
session, and lets a reconnecting client ask for exactly the events
it missed instead of polling for state.

## Layout

```
src/agent_workflow/
├── core/                      # no web, no TUI imports
│   ├── application/
│   │   ├── runtime.py         # create_runtime() -> AgentRuntime
│   │   ├── session.py         # AgentSession, SessionManager
│   │   ├── events.py          # EventBus (fan-out + replay)
│   │   ├── session_store.py   # durable session registry (JSON)
│   │   ├── tool_stats.py      # usage statistics from the event bus
│   │   ├── tool_stats_store.py # durable counters (SQLite)
│   │   ├── settings_service.py # config.toml / models.toml editing
│   │   ├── skills_service.py  # skill library CRUD
│   │   └── titles.py          # model-generated session names
│   ├── context/
│   ├── entities/models/       # Agent, tools, skills, plugins
│   └── infrastructure/
├── cli/                       # Textual TUI (unchanged behaviour)
│   ├── approval.py            # re-export of the core controller
│   └── ui/
└── web/                       # FastAPI layer
    ├── app.py
    ├── auth.py
    ├── schemas.py             # Pydantic, web boundary only
    ├── serialization.py       # AgentEvent -> JSON envelope
    ├── api/                   # sessions, settings, skills, tools, mcp, …
    └── websocket/agent.py

frontend/
├── src/
│   ├── api/                   # typed HTTP client + wire types
│   ├── websocket/             # reconnecting SessionSocket
│   ├── stores/transcript.ts   # pure event -> view reducer
│   ├── components/
│   └── features/{chat,tools,approval,skills,plugins,sessions}
```

## Context management

One LLM request is assembled from fixed and evictable blocks.

**Non-evictable** (always sent, budgeted by reservation): the harness
prompt, `[TASK ANCHOR]`, `[TASK STATE]`, `[EXECUTION]`, `[RESEARCH]`,
`[CHECKPOINT]`, catalogs and active skills, plus
`[CURRENT INSTRUCTION]`.

**Evictable** (trimmed to fit): the dialogue, `[EVIDENCE]`,
`[MEMORY]`, `[HISTORY]`. Allocation order is the reverse of eviction
priority — dialogue first, then evidence, memory, history — because
losing the conversation strands the model mid-task, while evidence
and memory can be re-fetched.

```python
budget = ContextBudget.for_model(context_size)   # None -> defaults
budget.available      # maximum_tokens - reserved_output
budget.fixed_reserved # reserved_system + reserved_task
budget.evictable      # what the evictable blocks share
```

`for_model` scales the reservations to the window: a model declaring
`context_size = 8000` cannot afford the default 2k + 2k + 4k, so they
shrink instead of raising. The value comes from the active model's
`requirements.context_size` in `models.toml` and is also sent to
Ollama as `num_ctx`, so the local budget and the provider agree. An
unknown model falls back to a 32k budget rather than a guess.

`build()` returns a request within `budget.available` except when the
non-evictable blocks alone exceed it. That case keeps the newest
dialogue message, truncated and marked, because dropping the
conversation as well would leave the model with nothing.

### Rollover

When a request is over budget the controller checkpoints and opens a
new window:

1. save the checkpoint (a failure skips the rollover; nothing is
   cleared without a checkpoint)
2. read it back and compare
3. select a bounded history tail — user messages, assistant messages
   **and tool results**, since dropping tool traffic lost everything
   the model had actually read
4. clear the window, keeping the most recent **user instruction**

Step 4 searches for the instruction instead of requiring it to be the
last message: rollover runs at the top of an iteration, where the
window normally ends with a tool result, so the old check discarded
the original prompt every time.

### Bounds

Nothing in the context path grows without a limit:

| Structure | Bound |
| --- | --- |
| dialogue | `context.max_messages` |
| evidence | 200 items, pinned by checkpoint, hard ceiling 2x |
| history | 2000 items, oldest evicted |
| plan steps in `[TASK STATE]` | active step + 12 newest |
| listed URLs | 20 fetched, 20 failed, 10 suggested |
| evidence excerpt | 4000 chars |
| history item | 4000 chars |
| tool result in a request | truncated to fit, visibly marked |

A subagent gets its own controller and history. Sharing the parent's
meant a subagent rollover advanced the parent's window counter and
wrote checkpoints into the parent's store under a foreign task id.

### Token counting

`counter_for_model()` returns a real tokenizer when `tiktoken` is
installed and its BPE table is reachable, and the length estimate
otherwise. Measured ratios show why this matters:

| Text | chars/token | divisor-4 count | exact |
| --- | --- | --- | --- |
| `hello world` | 5.5 | 2 | 2 |
| `def f(x): return x + 1` | 2.4 | 5 | 9 |
| `{"a": 1, "b": [2,3]}` | 1.4 | 5 | 14 |

The encoding is chosen from the model name (`gpt-4` → `o200k_base`,
Gemma/Llama/Mistral/Qwen → `cl100k_base`).

Two honest limits: the encoding is a **proxy** — `cl100k_base` is
OpenAI's, so a Gemma model is counted with the wrong vocabulary, just
far closer than a divisor — and the BPE table needs a ~1.7 MB download
once. Without it the counter degrades to `ApproximateTokenCounter`
rather than failing a run, and `TiktokenCounter.exact` says which you
got. Counts are memoised because the assembler counts the same fixed
blocks every iteration.

The value decides what to trim and is reported as `estimated_tokens`.
It is never sent to the provider.

### Conversation window

`context.max_messages` counts chat message objects, not turns: one
assistant message plus its tool results is already several entries.
The default is 24, up from 4, which left room for roughly a single
user turn plus one tool group — enough for the model to lose the
thread mid-task. Leave it empty for no limit; the token budget still
applies.

## Running the backend

```bash
uv sync
uv run agentoflow-web
```

Defaults to `127.0.0.1:8000`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AGENTOFLOW_HOST` | `127.0.0.1` | Bind address |
| `AGENTOFLOW_PORT` | `8000` | Bind port |
| `AGENTOFLOW_CORS_ORIGINS` | `localhost:5173,127.0.0.1:5173` | Comma-separated origins |
| `AGENTOFLOW_API_TOKEN` | *(unset)* | Enables bearer-token auth |

## Running the frontend

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173, proxies /api and /ws
npm run build    # type-check and bundle to frontend/dist
npm test         # reducer unit tests
```

When `frontend/dist` exists the backend serves the SPA at `/`, so a
production build needs only one process.

## Remote access

This service is a **remote control plane for an agent** that can
run shell commands, read and write files, and reach the network.
The existing `ToolPolicy`, `PathPolicy`, `ShellPolicy` and
guardrails remain the authority — the web layer never bypasses them
and never decides permissions itself.

Recommended setup over a VPN:

```bash
# On the agent host, reachable only from the VPN interface
AGENTOFLOW_HOST=10.8.0.5 \
AGENTOFLOW_API_TOKEN="$(python -m agent_workflow.web.auth)" \
AGENTOFLOW_CORS_ORIGINS="http://10.8.0.5:5173" \
uv run agentoflow-web
```

- Keep the bind address on the VPN interface or `127.0.0.1` with an
  SSH/tunnel forward. Never bind `0.0.0.0` on a public interface.
- Set `AGENTOFLOW_API_TOKEN` for anything beyond localhost. Tokens
  are read from the environment and never committed.
- CORS defaults to the Vite dev origins; set it explicitly when the
  UI is served from another origin.
- Each session is isolated: its own `Agent`, `ToolRegistry`,
  `ContextManager`, memory handle and event stream.
- `working_directory` per session narrows the filesystem sandbox.
- MCP servers execute third-party code. They are gated by the
  approval flow (`require_approval = true` by default) and carry
  only the permissions declared in `[mcp] default_permissions`.
  Only enable servers you trust.

## MCP (Model Context Protocol)

MCP servers are exposed to the agent as ordinary tools. They are
registered in the same `ToolRegistry` as builtins, so the same
permissions, approval flow, guardrails, events and statistics
apply — there is no second execution path.

### Adding a server from the UI

**Tools → MCP → Add server.** The form writes the `[mcp]` section for
you, so no hand-editing is needed. Fill in name, transport and the
transport-specific fields:

| Transport | Fields |
| --- | --- |
| `stdio` | Command, Arguments (space separated) |
| `http` | URL |

Optional **Environment** takes one `KEY=value` per line.

Saving persists the definition to `~/.agentoflow/config.toml`,
applies it to the running session and connects the server right
away — the panel then reports its real state, including the
server's own error output if it refuses to start.

Each server card has **Edit**, **Connect/Disconnect/Reload** and
**Remove**; **Enable/Disable MCP** toggles the section for future
sessions.

**Edit** seeds the form from the stored definition. Two details are
deliberate:

- The **name is the server's identity**, so it stays read-only and
  the transport is fixed while editing. Renaming would otherwise
  create a second server instead of editing the first.
- **Environment values are never sent to the browser.** An untouched
  environment field means *keep what is stored*; typing lines
  replaces them, and clearing the box removes them. `env` and
  `headers` are `null` by default in the write payload for exactly
  this reason — omitting them is not the same as sending `{}`.

MCP does not need to be enabled before you add a server: adding the
first one enables the section.

### Configuration by hand

Servers may also be declared directly in
`~/.agentoflow/config.toml`. Servers use
`[[mcp.servers]]` (with an explicit `name`) or the keyed
`[mcp.servers.<name>]` form:

```toml
[mcp]
enabled = true
# Remote tools are third-party code, so approval is on by default.
require_approval = true
default_permissions = ["mcp.execute"]

[[mcp.servers]]
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/srv/work"]
env = { API_KEY = "..." }
startup_timeout = 20.0
request_timeout = 60.0
# Optional; defaults to the server name.
prefix = "fs"
```

A malformed server entry is skipped rather than aborting startup,
and a server that fails to start is reported without preventing
other servers (or the agent) from working.

### Transports

Two transports are supported.

**stdio** (default) launches a subprocess and speaks
newline-delimited JSON-RPC over its pipes:

```toml
[[mcp.servers]]
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/srv/work"]
env = { API_KEY = "..." }
```

**http** posts JSON-RPC to a single Streamable HTTP endpoint. Both
`application/json` and `text/event-stream` replies are understood,
and `Mcp-Session-Id` is tracked automatically:

```toml
[[mcp.servers]]
name = "remote"
transport = "http"
url = "https://mcp.example.com/mcp"
headers = { Authorization = "Bearer ..." }
```

Use an `https` URL, or keep the endpoint on the VPN. A token in
`headers` is a credential: keep it in the environment or an
included file rather than committing it.

### Naming and permissions

Tool names are namespaced to avoid collisions between servers:

```
mcp_<prefix>_<tool>        # mcp_filesystem_read_file
```

Each MCP tool declares `mcp.execute` (configurable) and sets
`requires_approval` from `require_approval`. The approval hook
therefore gates remote tools exactly like local ones.

### What is implemented

- Newline-delimited JSON-RPC 2.0 over stdio
- Streamable HTTP transport (JSON and SSE replies, session id)
- `initialize` / `notifications/initialized` handshake
- `tools/list` and `tools/call`
- JSON Schema → dataclass conversion, so the **existing**
  `SchemaGenerator` derives the schema the model sees and the
  **existing** `ArgumentDecoder` validates arguments. Unknown or
  malformed arguments are rejected before a request is sent.
- Server-reported failures are returned to the model as data
  (`Error: ...`) rather than raised, so it can adapt
- Bounded stderr capture, surfaced in `/api/mcp` error messages
- Sequential request/response correlation, so concurrent tool
  calls cannot desynchronise the pipe

Not implemented: the legacy HTTP+SSE split transport, resources and
prompts, sampling, and elicitation. Only tools are exposed.

### Inspecting MCP

`GET /api/mcp/{id}` reports each server's state (`connected`,
`failed`, `disabled`), negotiated protocol version, server version,
startup time and contributed tools. The **Tools → MCP** tab in the
web UI renders the same data.

A minimal reference server used by the tests lives at
`examples/mcp_stdio_server.py` and `examples/mcp_http_server.py`:

```bash
uv run python examples/mcp_stdio_server.py
uv run python examples/mcp_http_server.py 8931
```

## Runtime MCP control

Servers can be connected, disconnected and reloaded while the server
is running, without a restart:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/mcp/{id}/servers/{name}/connect` | Connect one server |
| `POST` | `/api/mcp/{id}/servers/{name}/disconnect` | Disconnect and unregister its tools |
| `POST` | `/api/mcp/{id}/servers/{name}/reload` | Reconnect to pick up tool changes |
| `POST` | `/api/mcp/{id}/reload` | Reconnect every server |

Connecting registers the server's tools in the live `ToolRegistry`;
disconnecting removes them, so a stale entry can never let the model
call a server that is gone. The session's `ToolContext.permissions`
is re-derived on every change, so the agent never keeps a grant the
runtime no longer offers.

Two behaviours worth knowing:

- `Agent._build_tools()` runs once before the iteration loop, so a
  tool added mid-run becomes visible on the **next** run.
- A server that refuses to start is recorded as `failed` with its
  own diagnostics; the request still returns `200` with that state
  rather than a generic gateway error.

A remote tool whose JSON Schema declares no recognisable `type`
becomes a `typing.Any` field, which `SchemaGenerator` describes as
an empty (permissive) schema. Without that, a single
typeless parameter would answer `GET /api/tools` with a `500` and
take out the whole Tools tab; such parameters show up in the UI as
type `any`. Unknown *annotations* still raise, so a genuine typo in
a local tool is not silently accepted.

The **Tools → MCP** tab exposes these actions. This is the one place
where the web layer changes runtime state, so it is deliberately
narrow: server lifecycle only, never tool permissions.

| Action | Endpoint | Effect |
| --- | --- | --- |
| Add / edit server | `POST /api/mcp/{id}/servers` | Writes `[mcp.servers.<name>]`, applies, connects |
| Remove server | `DELETE /api/mcp/{id}/servers/{name}` | Drops the entry and disconnects |
| Enable / disable | `POST /api/mcp/{id}/enabled` | Toggles `[mcp] enabled` |
| Connect / disconnect / reload | `POST /api/mcp/{id}/servers/{name}/{action}` | Lifecycle only |
| Reload all | `POST /api/mcp/{id}/reload` | Reconnects every server |

Writing goes through `SettingsService`, so it validates, keeps a
backup and preserves secrets the browser never saw. Changes reach
the **running** session via `MCPManager.apply_config()`, which
disconnects removed servers and connects new ones; other config
sections still apply to new sessions only.

## Tool statistics

Counters are collected by an observer on the session's event bus,
so they reflect what actually ran. They are flushed to
`~/.agentoflow/tool-usage.db` when a run ends and when a session
closes, and aggregated history is available at:

```
GET /api/tools/history/aggregate?limit=10
```

Live per-session counters do not survive a restart; the SQLite
totals do.

## Settings

`config.toml` and `models.toml` can be edited from the web UI
(**Settings**, ⚙ at the bottom of the sidebar).

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/settings` | Editable file names |
| `GET` | `/api/settings/{name}` | Parsed document plus raw text |
| `PUT` | `/api/settings/{name}` | Save `data` (form) or `text` (raw TOML) |
| `GET` | `/api/settings/models/list` | Model catalog |
| `PUT` | `/api/settings/models/entry` | Add or replace a model |
| `DELETE` | `/api/settings/models/entry/{name}` | Remove a model |

Only those two files are reachable; any other name is refused.

**Writes are validated before they replace anything.** A document
is rendered, parsed back, and loaded through the same
`ConfigLoader` / `ModelCatalogLoader` the harness uses, so a typo
is rejected with `422` instead of breaking the next session. The
previous file is kept as `config.toml.bak.<pid>` and the new one is
written atomically.

`ConfigLoader` now type-checks every scalar it reads, so
`timeout = "soon"` or `max_iterations = "many"` are errors rather
than values that fail later at runtime.

### Secrets

Values whose key looks like a credential (`token`, `api_key`,
`authorization`, `clientSecret`, anything ending in `_secret`) are
replaced with `***` on the way out and **never sent to the
browser**. When a redacted document is saved, the stored values are
read back from disk and restored, so editing the form or the raw
text never deletes a token.

Keep credentials in an `env` file referenced from the config where
possible; anything in `config.toml` is readable by anyone who can
read your home directory.

## Session naming

- A title is derived from the first message, then replaced by a
  short model-generated title in the background so the conversation
  never waits on it.
- If the model is unavailable the derived title stands.
- Double-click a session, or use ✎, to rename it. A manual title is
  final: later generated titles will not overwrite it.

| Method | Path | Purpose |
| --- | --- | --- |
| `PATCH` | `/api/sessions/{id}/title` | Rename |

## Skills

Skills are Markdown files with YAML frontmatter:

```
~/.agentoflow/skills/<name>/SKILL.md
```

The web UI (**Tools → Skills → Skill library**) can create, edit
and delete them. Writes use the same format and loader as the
built-in library, so a new skill is available to the agent with no
restart.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/skills/library` | Skills on disk |
| `GET` | `/api/skills/library/{name}` | One skill with instructions |
| `POST` | `/api/skills/library` | Create |
| `PUT` | `/api/skills/library/{name}` | Update |
| `DELETE` | `/api/skills/library/{name}` | Delete |

Names are restricted to lowercase letters, digits, `-` and `_`, and
every path is re-checked against the skills root, so a skill name
cannot escape the directory.

The repository also carries a `skills/` folder with the skills
shipped from source control. That is a separate copy: the runtime
reads `~/.agentoflow/skills`, so copy a skill there (or create it
in the UI) to make it available.

## Session persistence

Sessions and chats survive a restart. Two stores are involved:

| File | Contents |
| --- | --- |
| `~/.agentoflow/sessions.json` | id, sandbox root, creation time, metadata, title |
| `~/.agentoflow/session-events.db` | the event log the transcript is rebuilt from |
| `~/.agentoflow/session-state.db` | anchor, checkpoint, task state, evidence |

**Why events and not messages.** The frontend renders a chat purely
from the event stream, and the WebSocket replays that stream on
reconnect. Persisting the events therefore restores the transcript
with no protocol change and no second read path: opening a session
re-seeds its `EventBus` with what was recorded, and the client
replays from its cursor exactly as it does after a network drop.

Details worth knowing:

- `EventBus.restore()` runs before anyone subscribes, so restored
  events are replayed to the client rather than delivered twice.
- Sequence numbers survive, so a client's cursor still means
  something and a reconnect resumes instead of restarting.
- Writes are batched (every 25 events, at the end of a run and on
  close) so streaming does not hit SQLite per chunk. A hard kill
  can lose the last few events.
- The conversation window is rebuilt from `AgentStarted` prompts and
  `AgentFinished` answers only. Tool traffic belongs to a run that
  is over, and replaying it would present stale results as current.
- `GET /api/sessions` lists recorded sessions too, so the sidebar
  does not empty on restart. Listing does **not** rehydrate: an
  Agent, registry and plugin set are built per session when it is
  opened, not when it is listed.
- A restored session counts as having a first turn, so the next
  message continues the conversation instead of clearing it.
- Each session keeps its last 2 000 events.

**What still does not survive:** the live Agent, its tool registry,
MCP clients and plugin state. Those are rebuilt on demand.

Closing the server deletes nothing. `DELETE /api/sessions/{id}`
removes the record *and* the transcript.

## API

### Sessions

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/sessions` | Create a session |
| `GET` | `/api/sessions` | List sessions |
| `GET` | `/api/sessions/{id}` | Session state and runtime info |
| `DELETE` | `/api/sessions/{id}` | Cancel and close a session |

### Execution

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/sessions/{id}/messages` | Submit a user turn (see below) |
| `POST` | `/api/sessions/{id}/run` | Start a conversation (`{"prompt": "..."}`) |
| `POST` | `/api/sessions/{id}/continue` | New task, preserved conversation |
| `POST` | `/api/sessions/{id}/resume` | Resume the paused task |
| `POST` | `/api/sessions/{id}/cancel` | Cooperative cancellation |

`POST /messages` is what the chat UI uses. Body:

```json
{ "content": "hello", "mode": "auto" }
```

`mode` is `auto` (default), `run`, `continue` or `resume`. With
`auto` the **session** decides: the first turn opens a conversation,
later turns continue it. The response echoes the chosen `mode`, so
the UI never has to duplicate that rule.

`run` and `messages` return as soon as the task is scheduled;
progress arrives over the WebSocket.

### Per-session sandbox

```json
POST /api/sessions
{ "working_directory": "/srv/project", "metadata": { "project": "demo" } }
```

The directory becomes both the agent's working directory and the
`PathPolicy` allow-list. It must exist and be a directory, otherwise
the request fails with `400`.

### Approvals

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/sessions/{id}/approval` | Pending request, if any |
| `POST` | `/api/sessions/{id}/approvals/{aid}/allow` | Allow |
| `POST` | `/api/sessions/{id}/approvals/{aid}/deny` | Deny |

The Python `Future` stays on the server. The browser sends a
command; `ApprovalController` resolves the future and the agent
continues. A stale or mismatched `approval_id` is rejected with
`409`.

### Observability

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/tools/{id}` | Tool inventory, parameters, policy, usage stats |
| `GET` | `/api/skills/{id}` | Available / loaded / active / used |
| `GET` | `/api/plugins/{id}` | Name, status, tools, skills, capabilities |
| `GET` | `/api/mcp/{id}` | MCP servers, connection state, contributed tools |
| `GET` | `/api/system/{id}` | Model, working directory, phase |

`/api/tools` returns, per tool: `source` (`builtin` / `plugin` /
`mcp`), description, JSON-schema parameters with required flags,
declared and missing permissions, `requires_approval`, timeout,
output cap, and live counters (calls, successes, errors, error rate,
average duration, error codes). The summary block reports
`registered_tools` vs `tools_used` plus approval counts.

Statistics are derived from the session's own event bus, so they
reflect what actually ran rather than what was merely available.

These project the existing `SkillManager` and `PluginManager`; no
second manager is created for the web.

## WebSocket

```
ws://host/ws/sessions/{session_id}?since={seq}
```

Envelope:

```json
{
  "type": "tool.started",
  "session_id": "5f3c...",
  "seq": 12,
  "timestamp": "2026-01-01T12:00:00.123456+00:00",
  "run_id": "child-1",
  "parent_run_id": "parent-1",
  "data": {
    "iteration": 1,
    "tool_call_id": "call-1",
    "tool_name": "read_file",
    "arguments": { "path": "src/main.py" },
    "run_id": "child-1",
    "parent_run_id": "parent-1"
  }
}
```

`run_id` / `parent_run_id` appear at the top level only when
non-empty, so child runs can be grouped by the client.

### Event types

| Type | Meaning |
| --- | --- |
| `session.snapshot` | Greeting: current state plus `oldest_seq`, `replayed` |
| `agent.started` | Run began; carries the prompt |
| `agent.phase_changed` | Phase transition with reason |
| `agent.finished` | Run completed with the final result |
| `llm.requested` | Request is being assembled |
| `llm.thinking_chunk` | Streaming reasoning fragment |
| `llm.content_chunk` | Streaming answer fragment |
| `llm.responded` | Full response, tool call count |
| `tool.started` | Tool call with arguments |
| `tool.finished` | Tool result, error, duration |
| `approval.requested` | Execution is blocked on a decision |
| `approval.resolved` | Decision applied |
| `run.failed` | Session-level error |
| `heartbeat` | Idle keepalive (20s) |

A client disconnect is not an error: the handler releases its
subscription immediately and closes quietly. The socket loop races
incoming events against the peer, so a dropped connection is noticed
at once rather than after the heartbeat interval.

### Reconnect

The client tracks the highest `seq` it processed and sends it as
`since` on every connect. The server replays everything newer, then
sends `session.snapshot`. Duplicates are dropped client-side by
comparing `seq`. Reconnecting never resets the session or the agent.

If the requested cursor has already been evicted from the bounded
history, the server replays what it still retains and reports
`oldest_seq` so the client knows a gap occurred.

## TUI

Unchanged:

```bash
uv run agentoflow chat
uv run agentoflow plugins list
```

`ApprovalController` moved to `core.entities.models.approval`;
`cli/approval.py` re-exports it, so the TUI imports keep working.

## Development

```bash
uv run pytest                       # 821 tests
cd frontend && npm test             # 53 tests
cd frontend && npm run build        # tsc + vite
```

### Layout of tests

| File | Covers |
| --- | --- |
| `tests/events_test.py` | EventBus fan-out, sequencing, replay, eviction |
| `tests/session_test.py` | Session isolation and independence |
| `tests/session_store_test.py` | Durable registry and rehydration |
| `tests/approval_test.py` | Approval lifecycle, stale ids, hook integration |
| `tests/event_serialization_test.py` | `AgentEvent` → JSON, enums, nested values |
| `tests/api_test.py` | REST contract |
| `tests/websocket_test.py` | Handshake, delivery, reconnect, approval, disconnect |
| `tests/auth_test.py` | Token gate on REST and WebSocket |
| `tests/mcp_test.py` | Protocol, schema mapping, client, manager, config |
| `tests/mcp_http_test.py` | Streamable HTTP transport against a live server |
| `tests/toml_writer_test.py` | TOML emitter round-trips |
| `tests/settings_service_test.py` | Read/write, redaction, validation |
| `tests/settings_api_test.py` | Settings, models, skills, renaming endpoints |
| `tests/skills_service_test.py` | Skill library CRUD and path safety |
| `tests/titles_test.py` | Title generation and fallbacks |
| `tests/tool_stats_test.py` | Usage counters and approval stats |
| `tests/tool_stats_store_test.py` | SQLite counters and aggregates |
| `frontend/src/stores/transcript.test.ts` | Event → view reducer |
| `frontend/src/features/markdown/parse.test.ts` | Markdown subset and URL safety |

## Tests and isolation

The suite sets `HOME` to a temporary directory for the whole run, so
tests never create `config.toml`, `memory.db` or statistics files in
your real profile. Each test also gets its own session registry and
statistics database under `tmp_path`.

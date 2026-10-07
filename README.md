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

### Compaction

Rollover checkpoints and clears the window, keeping only the trailing
instruction. On its own that means a long run reaches its limit and
then behaves as if it had just started.

Compaction writes a handover note first, while the messages it
describes still exist, and seeds the fresh window with it. The note is
produced by **the same model that was doing the work**: no second
model to route, and the summary is already in its own terms. It is
asked for concrete values rather than impressions, because
identifiers are the one thing that cannot be reconstructed:

```
[compacted context]
40 earlier messages were compacted into this note when the context
window filled. Identifiers are verbatim; anything not here was not
kept.

Handover: user asked for 25 tags without descriptions sorted by
scene_count DESC. Done so far: 10 written (ids 166, 649, 337, ...).
Failed: sort by scene_count in GraphQL -> use order_by. Next: the
remaining 15 via mcp_stash_update_tag_description.
```

The block says it is compacted context so the model treats the note
as a record of its own past rather than as something the user said.

```toml
[context]
compaction = true
compaction_tokens = 1200
```

Three deliberate choices:

- **A failure to summarise stops the run** rather than degrading.
  Carrying on without a note is the amnesia this replaces, and a
  silent fallback would hide it. An empty note is refused for the
  same reason.
- **The note is trimmed by measurement, not by a
  characters-per-token guess**, so the cap holds in any language.
- **It is written before the window is cleared**, not after, because
  afterwards there is nothing left to summarise.

Compaction is a summary, so it loses detail. What survives is the
task, what was done with concrete identifiers, what failed, and the
next step. For a run where an exact earlier payload matters, put it
in a file or let a tool re-read it rather than relying on the note.

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

### Where the context window comes from

The budget is sized from the model, but a model has two numbers and
they can disagree:

- `models.toml` **declares** `context_size` per model. It also sets
  Ollama's `num_ctx`, so it is a real ceiling, not a comment.
- Ollama **reports** the model's real window through `/api/show`.

The smaller of the two wins, and a mismatch is logged rather than
absorbed. A catalog below the model's real window leaves capacity
unused; one above it asks for more than the model will serve:

```
models.toml declares a 128000 token context for 'gemma4:e4b-it-qat'
but the model reports 131072; using the smaller, since num_ctx is set
from the catalog. Update the catalog to use the full window.
```

Round numbers in a catalog are usually guesses, and they are
routinely a little low, which shows up as the meter filling before
the model's window is actually reached. Query the real value with:

```bash
curl -s localhost:11434/api/show -d '{"model":"<name>"}' \
  | python3 -c 'import json,sys; i=json.load(sys.stdin)["model_info"];
print(next(v for k,v in i.items() if k.endswith(".context_length")))'
```

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

### Running a server in a container

A stdio server is usually a command. When that command is a container
runtime, the environment stops being inherited the way you expect:

```toml
[mcp.servers.Stash]
command = "podman"
args = ["run", "-i", "--rm", "--env-file", ".../stash_mcp.env", "stash-mcp:local"]

[mcp.servers.Stash.env]
STASH_ENDPOINT = "http://host.containers.internal:9999"
```

`STASH_ENDPOINT` above reaches the `podman` process but **not** the
container. Both `podman run` and `docker run` start from the image's
own environment plus whatever `--env-file` provides; they forward the
parent environment only when told to. The server therefore silently
falls back to the value in its env file, and the symptom appears far
from the cause:

```
Stash connection not available. Endpoint: 'http://localhost:9999'
```

Inside the container `localhost` is the container, so `localhost:9999`
cannot reach a server on the host even when that server is running and
healthy.

Forward each variable explicitly. `-e VAR` with no value takes the
value from the runtime's own environment, which is what makes the
`[mcp.servers.X.env]` table above authoritative:

```toml
args = ["run", "-i", "--rm", "--env-file", ".../stash_mcp.env",
        "-e", "STASH_ENDPOINT", "-e", "STASH_API_KEY", "stash-mcp:local"]
```

`--env VAR=value` works too and carries the value in the argument
itself, at the cost of the secret appearing in the process list and in
`config.toml`.

To reach the host from the container use the runtime's host alias --
`host.containers.internal` on Podman, `host.docker.internal` on Docker
-- rather than `127.0.0.1`, and publish the port on the host.

Agentoflow detects this shape at load time and reports it on the
server card rather than leaving the failure to be discovered through a
tool result:

> Server 'podman' is a container runtime, but its arguments pass no
> -e/--env flag, so STASH_ENDPOINT will not reach the container.

It is a warning, not an error, because a variable may legitimately be
meant for the runtime process instead.

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

## Read-only GraphQL against Stash

`stash_graphql` runs GraphQL queries directly against the Stash
database. It exists because the MCP tools expose a fixed set of
queries, and anything they cannot express -- filter on an absent
field, combine criteria, sort by something other than name -- is
otherwise unreachable.

```json
stash_graphql {
  "query": "query Undocumented { findTags(tag_filter: { is_missing: \"description\" }, filter: { per_page: 10, sort: \"name\", direction: ASC }) { count tags { id name scene_count } } }"
}
```

### What it teaches the model

The tool description carries the facts a model cannot infer from the
schema, all verified against a live server:

**The filter is split in two arguments.** Entity criteria go in
`tag_filter`, paging and sorting go in `filter`. Putting one where the
other belongs is a validation error:

```graphql
findTags(tag_filter: TagFilterType, filter: FindFilterType, ids: [ID])
```

**Absent fields** use `is_missing`, which is a plain string rather
than a filter object, and lives in `tag_filter`. This is the query
for "tags with no description":

```graphql
{ findTags(tag_filter: { is_missing: "description" }) { count tags { id name } } }
```

**Comparisons** take a nested object with `value` and `modifier`,
combined with `AND`, `OR` and `NOT`:

```graphql
tag_filter: { description: { value: "", modifier: NOT_EQUALS } }
tag_filter: { AND: [{ favorite: true }, { scene_count: { value: 0, modifier: GTE } }] }
```

**Sorting** is `filter: { sort: "...", direction: ASC }` where
direction is `SortDirectionEnum`. The accepted `sort` values are
`name`, `created_at`, `updated_at` and `id`. `scene_count`,
`description` and `favorite` are rejected at runtime with
`invalid sort`, and the schema does not advertise which ones are
valid, so the tool names the rejected values too.

Anything else is discoverable by introspection:

```graphql
{ __type(name: "TagFilterType") { inputFields { name } } }
```

### Read-only is enforced in two layers

The first is local: the document is parsed and any `mutation` or
`subscription` operation is refused **before the request is built**.
Operation type is read from the parsed AST, so an operation merely
*named* `mutation` cannot slip through, and neither can a document
carrying several operations.

The second is the server. A bare brace block is syntactically a
query, so no local check can tell that a mutation field is hidden
inside one; GraphQL forbids querying a mutation field through a query
operation, and Stash enforces it:

```
The server rejected the query:
- Cannot query field "tagDestroy" on type "Query"
```

Neither layer alone is sufficient, and together they leave no way to
write. Writes go through the Stash MCP tools instead.

### Credentials and limits

The API key is attached from the configuration and is never an
argument, so it cannot reach the model's context or the transcript.
Prefer pointing `env_file` at a file you already keep rather than
copying the secret into `config.toml`, which anything able to read
your home directory can read. A field named `api_key` is also
redacted automatically in the settings API.

```toml
[graphql]
enabled = true
endpoint = "http://localhost:9999/graphql"
env_file = "/path/to/stash_mcp.env"
timeout = 30
max_rows = 250
max_query_chars = 8000
max_response_chars = 60000
allow_introspection = true
```

The tool is absent from the registry unless `enabled` is true and an
endpoint is set, so an unconfigured install does not show a tool that
cannot work. `graphql.execute` is granted by default alongside
`web.fetch`.

### Ranking by a count

Stash sorts only by `name`, `created_at`, `updated_at` and `id`.
`filter: { sort: "scene_count" }` is refused with `invalid sort`, and
no query shape gets around it.

`order_by` does the ranking in the tool:

```json
{ "query": "{ findTags(tag_filter: { is_missing: \"description\" }, filter: { per_page: 10 }) { count tags { id name scene_count } } }",
  "order_by": "scene_count",
  "order": "DESC" }
```

The distinction matters more than it looks. `filter.per_page` returns
one page ordered by name, so a page of matching tags is not a page of
the highest counts. Asking for ten tags and ordering that page by eye
returns a plausible, wrong answer: of 715 tags with no description, an
alphabetical page starts at "Birthday" and never reaches "Outdoor",
which has 41 scenes to Brown Eyes' 25. `order_by` reads every match
before it ranks, so it reports what it read:

```
[note] Ordered by scene_count DESC in the tool, over 715 matching
row(s) read across 3 request(s); Stash itself sorts only by name,
created_at, updated_at and id.
```

It reads pages of 250 until `count` is covered, which is why it costs
several requests instead of one. `graphql.max_scan_rows` (5000) caps
the crawl, and a filter wider than that is refused with a suggestion
to narrow it rather than being silently truncated. Missing values sort
to the bottom of a descending order rather than the top.

The document must start with its opening brace. A selection set is
shorthand for a query, and dropping the brace is the usual way to get
`Syntax Error: Unexpected Name 'findTags'`. The tool repairs exactly
that case -- a document that fails to parse and becomes valid when
wrapped -- and refuses nothing else, so a genuinely broken query still
reports the line.

`max_rows` caps every list in the response and trims from the end, so
a `DESC` query still shows its largest values first and the ordering
you asked for survives. The cut is reported rather than silent:

```
[note] 245 further item(s) were omitted by the max_rows limit.
Narrow the query or raise graphql.max_rows.
```

## Listing every Stash tag

`stash_list_tags` returns the whole library as an id and a name, one
tag per line:

```
Stash tags in the library: 981 shown of 981 total.
One tag per line, id then name.

1678	3rd Person Narrative
1184	4k
975	60 FPS
```

The other tag tools return full records: aliases, descriptions, scene
counts, parents and children. That is the wrong shape for the common
bulk task, where all that is needed is the id to hand to
`mcp_stash_update_tag_description` and the name to decide which tags to
act on. The compact form is roughly half the tokens of the JSON
equivalent.

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `search` | string | empty | Case-insensitive substring of the name |
| `page` | integer | 1 | 1-based, and only meaningful with `per_page` |
| `per_page` | integer | 0 | 0 means everything that fits |
| `ascending` | boolean | true | Names A-Z, or Z-A when false |

The header always states the total, so a result that was cut short
cannot be mistaken for the whole library:

```
[note] 431 more not shown. Call again with page=2, or narrow with search.
```

`search` is a substring, not a pattern. The server only offers
`MATCHES_REGEX` for a name, so the tool matches case-insensitively and
escapes regex metacharacters and GraphQL string escapes -- a name
containing `big ass`, `(male)` or a quote is searched for literally.
Searching `anal` finds `Anal`, because Stash capitalises names and a
case-sensitive match would make the first thing anyone tries look
broken.

Requires `graphql.execute`, and shares the endpoint and credential
with `stash_graphql`: one place reads the API key.

## Making web search actually work

`web_search` cannot work out of the box from most addresses, and the
reason is worth stating precisely, because it is not obvious from the
symptom. Every keyless search front-end on the open internet answers a
scripted client with a JavaScript challenge rather than with results:

| Engine | What it returns |
| --- | --- |
| DuckDuckGo `html/` and `lite/` | HTTP 202, an `anomaly.js` challenge, no results |
| Mojeek | HTTP 200 with a captcha page |
| Brave | HTTP 200 with an empty app shell, results fetched by script |
| Startpage | HTTP 303 |

A better `User-Agent` does **not** help. A plain `curl/8.5.0` and a
full Chrome 120 string both receive the identical page, because the
gate is on the address rather than on the request headers. Rephrasing
the query does not help either, which is why the tool says so instead
of suggesting a retry.

The way through is to run a SearXNG instance. Its rate limits and
output formats are yours, and enabling JSON turns scraping into an
API call, which is the only kind that keeps working.

### Start an instance

```bash
mkdir -p searxng
cat > searxng/settings.yml <<'YAML'
use_default_settings: true
server:
  secret_key: "change-me"
  bind_address: "0.0.0.0"
  port: 8080
search:
  formats:
    - html
    - json
  safe_search: 0
YAML

podman run -d --name searxng \
  -p 8080:8080 \
  -v "$PWD/searxng/settings.yml:/etc/searxng/settings.yml:ro" \
  searxng/searxng:latest
```

`json` under `search.formats` is the part that matters. Without it the
instance answers with HTML, which looks identical to an empty result
set; the tool detects this and tells you to enable it rather than
reporting no matches.

### Point the tools at it

```toml
[web]
backend = "searxng"
searxng_endpoint = "http://localhost:8080"
searxng_timeout = 30
```

`web_research` and `web_fetch_many` take the same backend, so nothing
else changes. `searxng_categories` accepts anything the instance
supports, e.g. `general,social` to include social platforms in a
query.

Two notes on what to expect. Upstream engines can still time out, and
the response says which ones, so a thin result set is sometimes the
engines rather than the query. And SearXNG aggregates other people's
rate limits: a public instance may work for a while and then start
refusing you, which is reported as rate limiting rather than as a
challenge, and waiting does help in that case.

## Web research

Five tools cover outbound HTTP. The three added together let the agent
answer a question from the internet in one call instead of driving a
search-and-read loop by hand.

| Tool | Permission | What it does |
| --- | --- | --- |
| `web_search` | `web.search` | Numbered results with titles, snippets, URLs |
| `web_fetch_many` | `web.fetch` | Downloads N pages **concurrently** |
| `web_research` | `web.search`, `web.fetch` | Search, then read the top results |
| `web_fetch` | `web.fetch` | One page, fully parsed with its links |
| `web_crawl` | `web.fetch` | Bounded breadth-first crawl from one URL |

### Choosing between them

`web_search` when you want to see what exists. `web_fetch_many` when
you already have the URLs. `web_research` when you have a question and
want the reading done for you. `web_crawl` when a single site has to be
traversed rather than sampled.

### web_research

```
web_research {"query": "python asyncio tutorial", "max_pages": 3}
```

It searches, keeps the top N results, downloads them concurrently and
returns the full result list alongside the extracted text of every page
it managed to read, so the model can see what was missed when a fetch
fails. Four pages took 2.3 s against 0.76 s for the same pages fetched
in parallel directly.

The pages come back as **text, with no answer pre-computed**. The model
reads them and reasons itself, which keeps a claim traceable to a URL
and avoids a second hidden LLM call inside a tool.

### Parallelism and partial failure

`WebBatchFetcher` caps concurrency (`max_concurrency`, default 4) and
total pages so a batch cannot hammer one site or exhaust the agent's
sockets. Each URL is independent: one dead link is recorded as a failed
page and the rest are still returned.

```
Pages: 2 fetched, 1 failed

--- [1] asyncio — Asynchronous I/O — Python documentation
    ...

Failed:

  https://realpython.com/async-io-python/
    HTTP 403 for https://realpython.com/async-io-python/
```

Sites that block bots answer 403. That is normal and not a bug in the
tool; the batch reports it and carries on with the others.

### The search backend

DuckDuckGo's HTML endpoint, no API key. It is scraped, not an API, so
it can break without notice and rate-limit under load. When that
happens the tool raises a distinct error naming the cause instead of
returning an empty list, because an empty list reads to the model as
"nothing matched" and it would retry forever:

> The search endpoint returned an anti-bot page instead of results.
> Search is rate limited; wait a moment or use web_fetch on a known URL
> instead.

Swapping in Brave or Tavily means replacing `DuckDuckGoSearch` and
returning the same `list[SearchResult]`; nothing else changes.

### Network policy

`WebPolicy` runs on **every** request hop and refuses non-HTTP schemes,
credentials in URLs, any port other than 80/443, and any host that
resolves to a private, loopback, link-local, reserved or multicast
address.

Redirects are the part that is easy to get wrong, so they are checked
explicitly. Validating only the entry URL is not enough: any reachable
page can answer with a 302 pointing at `169.254.169.254`, and a fetcher
that follows it will happily read cloud instance metadata or a service
on the agent's own machine. Every hop is therefore validated on its own,
and a refused hop fails that page rather than the batch.

What the policy does **not** do is stop a public host from returning
hostile content. Everything fetched is untrusted input that the model
reads, so treat page text as data rather than as instructions.

### Configuration

Limits live in code as trusted runtime policy rather than as tool
arguments, so the model cannot raise its own budget:

| Policy | Default | Meaning |
| --- | --- | --- |
| `WebSearchPolicy.max_results` | 8 | Results per search |
| `WebSearchPolicy.timeout` | 20 s | Search request timeout |
| `WebBatchPolicy.max_pages` | 8 | Pages per batch |
| `WebBatchPolicy.max_concurrency` | 4 | Simultaneous requests |
| `WebBatchPolicy.max_chars_per_page` | 6000 | Text kept per page |
| `WebBatchPolicy.max_total_chars` | 40000 | Text per batch |

## The python_exec sandbox

`python_exec` runs a snippet in a separate interpreter and returns its
stdout, stderr and exit code. It exists because asking a model to do
arithmetic in prose is unreliable, and because "write a script, run it,
look at the output" is a loop the agent should not have to improvise.

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `code` | string | required | Source to run |
| `filename` | string | `scratch.py` | Name used in tracebacks |
| `timeout_seconds` | number | `0` | Shorten the timeout, clamped to `max_timeout` |

```json
{ "code": "print(sum(i * i for i in range(10)))" }
```

```
exit_code: 0
[stdout]
285
script: /srv/project/agentoflow-python/scratch.py
```

The script is written to `sandbox_dir` and run with that directory as
its working directory, so relative reads and writes stay in one
predictable place. Imports available to the script are those of the
agent's own interpreter, which means the project's virtualenv is on
the path without any extra configuration.

### What is actually enforced

| Limit | Config key | Default | Effect |
| --- | --- | --- | --- |
| Wall clock | `python.timeout` | 30 s | Process group is killed |
| Caller override cap | `python.max_timeout` | 120 s | Bounds `timeout_seconds` |
| Address space | `python.memory_mb` | 1024 | `RLIMIT_AS`, so a runaway allocation raises `MemoryError` |
| CPU time | `python.cpu_seconds` | 60 s | `RLIMIT_CPU` backstop for a tight loop |
| Output | `python.max_output` | 20000 | Per stream, then marked `[stdout truncated]` |
| Disk write | fixed | 64 MB | `RLIMIT_FSIZE` |
| Environment | inherited from `ToolContext` | -- | The agent's own environment is **not** inherited |
| `stdin` | closed | -- | A script cannot block waiting for input |

Timeout enforcement kills the whole process group, not just the direct
child, so a script that spawned helpers does not leave them behind.

### What is not enforced

The static check is a **guardrail, not a sandbox**. It parses the
source and refuses obvious constructs:

- imports of the modules in `python.blocked_modules` -- by default
  `subprocess`, `socket`, `ssl`, `shutil`, `urllib`, `ctypes`,
  `multiprocessing`, `pty`, `webbrowser`
- calls to `eval`, `exec`, `input`, `os.system`, `os.popen`,
  `shutil.rmtree` and similar

A determined script gets past it. `getattr(__import__("os"), "system")("...")`
is not a syntax error and Python offers no way to forbid a call from
inside the language. Two further consequences:

- The script runs as the agent's own user, so it can read any file
  that user can read. `PathPolicy` governs tools that go through it;
  it cannot intercept a file opened by a child process.
- Network reach is a property of that user, not of this tool.

Real isolation means a container or a user namespace, which this tool
does not set up. Treat `blocked_modules` as protection against a
careless or confused model, which is the actual failure mode it
addresses -- and note that the guardrail is worthless as a boundary,
so do not rely on it as one.

### Configuration

```toml
[python]
enabled = true
interpreter = ""            # empty means the agent's own interpreter
sandbox_dir = "agentoflow-python"
timeout = 30
memory_mb = 1024
cpu_seconds = 60
max_output = 20000
max_timeout = 120
blocked_modules = ["subprocess", "socket", "ssl", "shutil", "urllib",
                   "ctypes", "multiprocessing", "pty", "webbrowser"]
```

A relative `sandbox_dir` resolves against the session's working
directory, which keeps the sandbox inside the project. Point it at an
absolute path such as `/tmp/agentoflow-python` to move it out.

`enabled = false` removes the tool from the registry entirely, which is
the setting to use when the environment makes the trade-off
unacceptable.

The tool requires the `python.execute` permission, which the runtime
grants by default alongside `shell.execute`. It does **not** prompt for
approval, for the same reason `execute_shell` does not: the agent needs
to run a script and read the result without a round trip.

## Switching tools off

Any tool can be disabled from **Tools → Tools**, with the checkbox in
the `On` column, or from `config.toml`. A disabled tool is removed
from the live registry, so the model cannot call it for the rest of
the run, and the choice is written to `[tools] disabled` so it
survives a restart.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/tools/{id}/tools/{name}` | `{"enabled": false}` to switch off |

```toml
[tools]
disabled = ["mcp_stash_get_tags", "mcp_stash_health_check"]
```

Worth doing when two tools do the same thing. `mcp_stash_get_tags`
and `stash_graphql` both list tags, so which one the model reaches
for comes down to the shape of the question rather than what the tool
can actually do; hiding the weaker one removes the choice.

A disabled tool stays in the `GET /api/tools` listing with
`"enabled": false`, dimmed, so it can be switched back on without
editing the file. `registered_tools` counts only what the model can
actually call.

Three details that are easy to get wrong:

- The filter runs **after** MCP servers connect. MCP tools do not
  exist until then, so a filter placed next to the builtin
  registration would silently do nothing for most of them.
- Reconnecting a server re-registers its tools, so the disabled set
  is re-applied on connect. Otherwise a reconnect would quietly
  undo the choice.
- A name that is not a registered tool is reported as a `404` and
  logged as a warning, not accepted silently. A typo in the config
  would otherwise look like it had worked.

## Sharing MCP processes

An MCP server used to be started per session. Two sessions and one
server is merely wasteful; a dozen sessions and four servers means
forty processes, and when the server is really
`podman run` / `docker run` that is forty containers on the machine.

Servers are now pooled by definition. A process is shared by every
session asking for the same command, arguments and environment, and
closed once the last of them lets go:

```
3 sessions      -> processes: 1  {'podman': 3}
after 1 closed  -> processes: 1  {'podman': 2}
after 2 closed  -> processes: 1  {'podman': 1}
after all closed-> processes: 0  {}
```

What stays per session is the tool registration. Each session has its
own `ToolRegistry`, so one session hiding a tool with
`[tools] disabled` cannot affect another, and disabling a server in
one session does not disturb the others using it.

Details worth knowing:

- The pool key is the server **definition**, not its name. Two names
  for one command share a process; one session renaming a server it
  changed does not collide with another's copy.
- Keys hold a digest of the environment, not the environment, so a
  pooled key never keeps an API token alive for the life of the
  process.
- A session that arrives while the first is still handshaking waits
  for it instead of starting a second process, and learns about the
  first one's failure rather than hanging.
- Reconnecting a server re-applies the disabled set, so a reconnect
  cannot undo a tool the operator turned off.
- MCP clients serialise their own calls, so sharing a process does
  not interleave two sessions' requests on one pipe.

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

A skill is worth writing when the knowledge is verified and not
derivable -- which sort fields a server accepts, which tool to reach
for, what an empty result does and does not mean. Everything in a
skill should be checked against the real thing first: the shipped
`filesystem` and `python` skills named tools with a `filesystem.`
prefix that does not exist, which teaches the model to call a tool
that cannot be called.

A skill only reaches the model once loaded. `skills.list` returns
name and description; `skills.load` activates one and its
instructions are injected into subsequent LLM requests for that run.
Activation is per run, so a follow-up question starts from scratch
unless the skill is loaded again.

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

#### One grant covers the rest of the session

A tool that declares `requires_approval` is asked about through a
synthetic permission named `tool:<name>`. Once granted, that name is
remembered on the session's `ToolContext` and later calls of the same
tool proceed without a prompt, so a bulk task -- ten descriptions, a
batch of edits -- costs one approval rather than one per item.

Grants are per tool. Approving `mcp_stash_update_tag_description` does
not approve `execute_shell`, and a denial is not remembered, so the
next attempt asks again.

```toml
[approval]
# false asks again on every single call.
remember = true
```

Set `remember = false` when each individual call should be reviewed.

The remembered grant applies to a tool's declared permission class.
`execute_shell` is the one case that is still asked per call, because
a single grant there covers every command in that class
(`shell.network` and so on), which is a wider blast radius than a
tool with fixed semantics.

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

### Reconnect and reload

The replay buffer in memory is bounded (`DEFAULT_HISTORY_LIMIT`, 500
events). Streaming publishes an event per chunk, so a long session
pushes its oldest events out of it, while the durable store keeps
everything.

A page reload reconnects with `since=-1`, which used to return only
what the buffer still held: the transcript came back truncated at the
top, and there was nothing to scroll to. When the requested cursor
points before the oldest retained event, the missing prefix is now
read from the durable store and merged ahead of the retained events,
so the sequence stays contiguous. The `oldest_seq` in the greeting
reports the durable floor rather than the buffer's, so the client
knows it has the whole transcript.

Backfill is only asked for when there is a real gap, so an ordinary
reconnect does not touch storage. A session with no durable store
still replays what the bus holds.

Durable writes are batched (`EVENT_FLUSH_THRESHOLD`) and flushed at
the end of a run and on close, so a hard kill can lose the last few
events. That is the same guarantee an append-only log gives.

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

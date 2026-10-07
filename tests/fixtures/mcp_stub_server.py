"""Stub MCP server used to measure overlap on the stdio transport.

Reads newline-delimited JSON-RPC from stdin and answers after a fixed
delay. Responses are deliberately emitted in reverse order of arrival,
so a transport that routes by arrival rather than by id will hand back
the wrong answer and the test will notice.
"""

from __future__ import annotations

import json
import sys
import threading
import time


DELAY = 0.30

# One writer at a time. Without this, two threads replying at the same
# moment interleave partial lines and the reader sees malformed JSON --
# which would look like a transport bug and is not one.
WRITE_LOCK = threading.Lock()


def respond(payload: dict) -> None:
    line = json.dumps(payload) + "\n"

    with WRITE_LOCK:
        sys.stdout.write(line)
        sys.stdout.flush()


def main() -> None:
    ready = {
        "jsonrpc": "2.0",
        "id": "init-1",
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "stub", "version": "0"},
        },
    }

    # The client does not know the id up front, so answer whatever the
    # first initialize asked for.
    for line in sys.stdin:
        text = line.strip()

        if not text:
            continue

        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            continue

        method = message.get("method")
        request_id = message.get("id")

        if method == "initialize":
            ready["id"] = request_id
            respond(ready)
        elif method == "tools/list":
            respond({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [{
                        "name": "echo",
                        "description": "echo the tag back",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"tag_id": {"type": "string"}},
                            "required": ["tag_id"],
                        },
                    }]
                },
            })
        elif method == "tools/call":
            tag = (
                message.get("params", {})
                .get("arguments", {})
                .get("tag_id", "")
            )

            # Answered on its own thread, so the server keeps reading
            # while this one is in flight. A single-threaded loop that
            # slept inline would serialise the batch itself and the
            # test would measure the stub rather than the transport.
            # Replies deliberately come back in reverse order of
            # arrival.
            def reply(
                reply_id=message.get("id"),
                tag=tag,
                raw=message.get("params", {}).get("arguments", {}),
            ) -> None:
                delay = (
                    float(tag.split("-")[-1]) if "-" in tag else DELAY
                )
                time.sleep(delay)

                # A real server rejects a call that is missing its
                # required argument. Echoing an empty value instead
                # would report a failure as a success and hide the
                # per-item error path this fixture exists to drive.
                if not isinstance(raw, dict) or "tag_id" not in raw:
                    respond({
                        "jsonrpc": "2.0",
                        "id": reply_id,
                        "error": {
                            "code": -32602,
                            "message": "tag_id is required",
                        },
                    })
                    return

                respond({
                    "jsonrpc": "2.0",
                    "id": reply_id,
                    "result": {
                        "content": [{"type": "text", "text": f"echo:{tag}"}]
                    },
                })

            threading.Thread(
                target=reply, daemon=True
            ).start()
        elif request_id is not None:
            respond({"jsonrpc": "2.0", "id": request_id, "result": {}})


if __name__ == "__main__":
    main()

"""MCP client wrapper: discovery, schema export, validated dispatch, telemetry.

The agents never see MCP types. They see `toolbox.schemas()` (an OpenAI-style
function-calling spec they can put in a prompt) and `toolbox.call(name, args)` which
always returns a `ToolResult` — never raises. Every call is timed and recorded, which
is what makes "tool call validity" and "median tool latency" reportable metrics
instead of vibes.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent import futures
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer

from ..schemas import ToolCall, ToolResult


class Toolbox:
    """Sync facade over the async MCP client, so agent nodes stay readable.

    Two connection modes:
      in-process — pass the MCPServer object. Fast, used by tests and by the CLI.
      stdio      — pass StdioServerParameters. The real deployment shape; proves the
                   agent works against a server it does not import.
    """

    def __init__(self, target: MCPServer | StdioServerParameters | str):
        self._target = target
        self._tools: list[dict[str, Any]] = []
        self.history: list[ToolResult] = []

    # ---- lifecycle -------------------------------------------------------
    # The MCP session lives inside a single long-running task on a background event
    # loop, and every call is funnelled through a queue into that same task. This is
    # not ceremony: anyio cancel scopes must be entered and exited by the same task,
    # so the naive `run_until_complete(__aenter__)` / `run_until_complete(call)`
    # pattern raises "cancel scope in a different task" on the first tool call.
    def __enter__(self) -> Toolbox:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        ready: futures.Future = futures.Future()
        self._serving = asyncio.run_coroutine_threadsafe(self._serve(ready), self._loop)
        ready.result(timeout=60)  # re-raises whatever went wrong during connect
        return self

    async def _serve(self, ready: futures.Future) -> None:
        queue: asyncio.Queue = asyncio.Queue()
        self._queue = queue
        try:
            async with Client(self._target) as client:
                self._client = client
                self._tools = await self._list_tools()
                ready.set_result(True)
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    make_coro, fut = item
                    try:
                        fut.set_result(await make_coro())
                    except Exception as exc:
                        fut.set_exception(exc)
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise

    def _submit(self, make_coro: Any, timeout: float = 120.0) -> Any:
        fut: futures.Future = futures.Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (make_coro, fut))
        return fut.result(timeout=timeout)

    def __exit__(self, *exc: Any) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        try:
            self._serving.result(timeout=30)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        self._loop.close()

    async def _list_tools(self) -> list[dict[str, Any]]:
        res = await self._client.list_tools()
        return [
            {
                "name": t.name,
                "description": (t.description or "").strip(),
                # mcp 2.x uses snake_case field names (1.x was inputSchema/isError).
                "parameters": t.input_schema,
            }
            for t in res.tools
        ]

    # ---- introspection ----
    def schemas(self) -> list[dict[str, Any]]:
        return self._tools

    def names(self) -> set[str]:
        return {t["name"] for t in self._tools}

    def prompt_block(self, only: set[str] | None = None) -> str:
        """Render the discovered tool schemas into the retriever's system prompt.

        Deliberately generated from discovery rather than hand-written: add a tool to
        the MCP server and the agent can use it with no prompt edit.
        """
        lines = []
        for t in self._tools:
            if only and t["name"] not in only:
                continue
            props = (t["parameters"] or {}).get("properties", {})
            required = set((t["parameters"] or {}).get("required", []))
            args = ", ".join(
                f"{k}: {v.get('type', 'any')}{'' if k in required else '?'}" for k, v in props.items()
            )
            lines.append(f"- {t['name']}({args})\n    {t['description'].splitlines()[0]}")
        return "\n".join(lines)

    # ---- dispatch ----
    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        arguments = arguments or {}
        t0 = time.perf_counter()
        if name not in self.names():
            res = ToolResult(name=name, ok=False, error=f"no such tool: {name}")
            self.history.append(res)
            return res
        try:
            raw = self._submit(lambda: self._client.call_tool(name, arguments))
            content = getattr(raw, "structured_content", None)
            # The SDK wraps a non-object return value as {"result": ...}; unwrap it so
            # agents see the value the tool actually returned.
            if isinstance(content, dict) and set(content) == {"result"}:
                content = content["result"]
            if content is None:
                blocks = [getattr(c, "text", "") for c in (raw.content or [])]
                joined = "\n".join(b for b in blocks if b)
                try:
                    content = json.loads(joined)
                except (json.JSONDecodeError, TypeError):
                    content = joined
            ok = not bool(getattr(raw, "is_error", False))
            res = ToolResult(
                name=name,
                ok=ok,
                content=content,
                error=None if ok else str(content),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:  # tool failure is data, not an exception path
            res = ToolResult(
                name=name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        self.history.append(res)
        return res

    def call_many(self, calls: list[ToolCall]) -> list[ToolResult]:
        return [self.call(c.name, c.arguments) for c in calls]


def stdio_toolbox(config_path: str | None = None) -> Toolbox:
    """Connect to the tool server as a separate process over stdio."""
    env = {"CURATOR_CONFIG": config_path} if config_path else None
    return Toolbox(
        StdioServerParameters(command="python", args=["-m", "curator.tools.server"], env=env)
    )

"""Audit trail for every node, LLM call, and tool call (Milestone 1: "record
prompt version, catalog date, price assumptions, tool result, latency, and
error category").

Events are kept in memory and appended to ``trace.jsonl`` in the run folder.
LLM and tool events are captured through a LangChain callback handler, so any
agent built with LangChain/LangGraph is traced without extra code.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

_MAX_TEXT = 1500


def _short(value: Any, limit: int = _MAX_TEXT) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"... [+{len(text) - limit} chars]"


class Tracer:
    def __init__(self, run_id: str, path: Path | None = None):
        self.run_id = run_id
        self.path = path
        self.events: list[dict] = []
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, name: str, status: str = "ok", **fields) -> dict:
        record = {"ts": round(time.time(), 3), "run_id": self.run_id, "kind": kind,
                  "name": name, "status": status, **fields}
        with self._lock:
            self.events.append(record)
            if self.path is not None:
                with self.path.open("a") as fh:
                    fh.write(json.dumps(record, default=str) + "\n")
        return record

    def callbacks(self, agent: str, prompt_version: str) -> list[BaseCallbackHandler]:
        return [_TraceHandler(self, agent, prompt_version)]

    def summary(self) -> dict:
        llm = [e for e in self.events if e["kind"] == "llm"]
        tools = [e for e in self.events if e["kind"] == "tool"]
        by_agent: dict[str, dict] = {}
        for e in llm:
            a = by_agent.setdefault(e.get("agent", "?"), {"llm_calls": 0, "llm_errors": 0,
                                                         "tokens_in": 0, "tokens_out": 0,
                                                         "llm_latency_ms": 0})
            a["llm_calls"] += 1
            a["llm_errors"] += e["status"] != "ok"
            a["tokens_in"] += e.get("tokens_in") or 0
            a["tokens_out"] += e.get("tokens_out") or 0
            a["llm_latency_ms"] += e.get("latency_ms") or 0
        for e in tools:
            a = by_agent.setdefault(e.get("agent", "?"), {"llm_calls": 0, "llm_errors": 0,
                                                         "tokens_in": 0, "tokens_out": 0,
                                                         "llm_latency_ms": 0})
            a["tool_calls"] = a.get("tool_calls", 0) + 1
            a["tool_errors"] = a.get("tool_errors", 0) + (e["status"] != "ok")
        return {
            "llm_calls": len(llm),
            "llm_errors": sum(e["status"] != "ok" for e in llm),
            "tokens_in": sum(e.get("tokens_in") or 0 for e in llm),
            "tokens_out": sum(e.get("tokens_out") or 0 for e in llm),
            "llm_latency_ms": sum(e.get("latency_ms") or 0 for e in llm),
            "tool_calls": len(tools),
            "tool_errors": sum(e["status"] != "ok" for e in tools),
            "fallbacks": sum(1 for e in self.events if e["kind"] == "fallback"),
            "by_agent": by_agent,
        }


class _TraceHandler(BaseCallbackHandler):
    """Times each LLM/tool call and records model, tokens, and errors."""

    def __init__(self, tracer: Tracer, agent: str, prompt_version: str):
        self.tracer = tracer
        self.agent = agent
        self.prompt_version = prompt_version
        self._start: dict[UUID, float] = {}
        self._model: dict[UUID, str] = {}
        self._tool: dict[UUID, tuple[str, str]] = {}

    # ---- LLM -------------------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start[run_id] = time.time()
        params = kwargs.get("invocation_params") or {}
        self._model[run_id] = params.get("model") or params.get("model_name") or "?"

    def on_llm_end(self, response, *, run_id, **kwargs):
        latency = int((time.time() - self._start.pop(run_id, time.time())) * 1000)
        tokens_in = tokens_out = None
        tool_calls: list[str] = []
        text = ""
        try:
            msg = response.generations[0][0].message
            usage = getattr(msg, "usage_metadata", None) or {}
            tokens_in, tokens_out = usage.get("input_tokens"), usage.get("output_tokens")
            tool_calls = [tc["name"] for tc in (getattr(msg, "tool_calls", None) or [])]
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
        except (AttributeError, IndexError, KeyError, TypeError):
            pass
        self.tracer.event("llm", "chat", agent=self.agent, model=self._model.pop(run_id, "?"),
                          prompt_version=self.prompt_version, latency_ms=latency,
                          tokens_in=tokens_in, tokens_out=tokens_out,
                          tool_calls_requested=tool_calls, output=_short(text, 600))

    def on_llm_error(self, error, *, run_id, **kwargs):
        latency = int((time.time() - self._start.pop(run_id, time.time())) * 1000)
        self.tracer.event("llm", "chat", status="error", agent=self.agent,
                          model=self._model.pop(run_id, "?"), prompt_version=self.prompt_version,
                          latency_ms=latency, error=f"{type(error).__name__}: {_short(str(error), 400)}")

    # ---- Tools -----------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._start[run_id] = time.time()
        self._tool[run_id] = ((serialized or {}).get("name", "?"), input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        latency = int((time.time() - self._start.pop(run_id, time.time())) * 1000)
        name, inp = self._tool.pop(run_id, ("?", ""))
        out = getattr(output, "content", output)
        out_text = out if isinstance(out, str) else json.dumps(out, default=str)
        status = "error" if out_text.lstrip().startswith(("ERROR", "REJECTED")) else "ok"
        self.tracer.event("tool", name, status=status, agent=self.agent, latency_ms=latency,
                          input=_short(inp, 800), output=_short(out_text, 800))

    def on_tool_error(self, error, *, run_id, **kwargs):
        latency = int((time.time() - self._start.pop(run_id, time.time())) * 1000)
        name, inp = self._tool.pop(run_id, ("?", ""))
        self.tracer.event("tool", name, status="error", agent=self.agent, latency_ms=latency,
                          input=_short(inp, 800), error=f"{type(error).__name__}: {_short(str(error), 400)}")

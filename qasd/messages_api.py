"""Anthropic Messages API (`/v1/messages`) for Claude Code, Claude Desktop, the Agent SDK
and any app built on the Anthropic SDK.

Requests go through LiteLLM's Anthropic-format interface, so the upstream can be
Anthropic itself or any other provider LiteLLM supports. The same savings layers
apply as on `/v1/chat/completions`:

* tool schemas made byte-stable (order kept when the client places its own breakpoints),
* automatic cache breakpoints for Claude models when the client sets none,
* the strict exact cache,
* opt-in routing with `model: "auto"`,
* compaction, off by default here because Claude Code compacts on its own
  (turn on with QASD_MESSAGES_COMPACTION=true or `x-qasd-compact: on`),
* the savings ledger, including Anthropic cache read/write tokens.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from typing import Any, AsyncIterator

import litellm
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import cache as rcache
from .compaction import compact, llm_summarizer, trigger_for
from .config import Settings
from .costs import baseline_cost
from .prefix import EPHEMERAL, _has_cache_control, _mark, _sort_keys, is_anthropic, stabilize_tools
from .router import Router

log = logging.getLogger("qasd")

HANDLED = {"model", "messages", "system", "tools", "stream", "max_tokens"}


# ---------- helpers ----------

def _error(status: int, message: str, etype: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"type": "error", "error": {"type": etype, "message": message}}, status_code=status)


RETRY_HEADERS = ("retry-after", "x-should-retry", "request-id")


def _upstream_error(exc: Exception) -> JSONResponse:
    """Relay the upstream error body unchanged when possible.

    Claude Code matches on the upstream's error wording to recover (for example by
    dropping a rejected capability and retrying), so the original body matters.
    """
    status = getattr(exc, "status_code", None)
    status = status if isinstance(status, int) and 400 <= status < 600 else 502
    upstream_headers = getattr(exc, "litellm_response_headers", None) or {}
    headers = {h: str(upstream_headers[h]) for h in RETRY_HEADERS if h in upstream_headers}
    text = str(getattr(exc, "message", None) or exc)
    start = text.find("{")
    if start != -1:
        try:
            body = json.loads(text[start:])
            if isinstance(body, dict) and body.get("type") == "error":
                return JSONResponse(body, status_code=status, headers=headers)
        except ValueError:
            pass
    etype = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
             404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error",
             529: "overloaded_error"}.get(status, "api_error")
    return JSONResponse({"type": "error", "error": {"type": etype, "message": text}},
                        status_code=status, headers=headers)


def _event(data: dict) -> str:
    return f"event: {data.get('type')}\ndata: {json.dumps(data, default=str)}\n\n"


def _as_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def _system_has_cache_control(system: Any) -> bool:
    return isinstance(system, list) and any(isinstance(b, dict) and "cache_control" in b for b in system)


def _count(model: str, system: Any, messages: list[dict], tools: list | None) -> int:
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    try:
        return litellm.token_counter(model=model, messages=msgs, tools=tools or None)
    except Exception:
        return len(json.dumps([system, messages, tools], default=str)) // 4


def add_breakpoints(system: Any, messages: list[dict], tools: list | None) -> tuple[Any, list[dict], list | None, int]:
    """Cache breakpoints on the last tool, the system prompt and the newest user turn."""
    if _has_cache_control(messages, tools) or _system_has_cache_control(system):
        return system, messages, tools, 0
    messages = copy.deepcopy(messages)
    tools = copy.deepcopy(tools) if tools else tools
    added = 0
    if tools:
        tools[-1]["cache_control"] = EPHEMERAL
        added += 1
    if isinstance(system, str) and system:
        system = [{"type": "text", "text": system, "cache_control": EPHEMERAL}]
        added += 1
    elif isinstance(system, list) and system and isinstance(system[-1], dict):
        system = copy.deepcopy(system)
        system[-1]["cache_control"] = EPHEMERAL
        added += 1
    if messages and messages[-1].get("role") == "user" and _mark(messages[-1]):
        added += 1
    return system, messages, tools, added


def usage_numbers(usage: dict | None) -> tuple[int, int, int, int]:
    """(total_input, output, cache_read, cache_write). Anthropic's input_tokens excludes cached tokens."""
    u = usage or {}
    cr = int(u.get("cache_read_input_tokens") or 0)
    cw = int(u.get("cache_creation_input_tokens") or 0)
    return int(u.get("input_tokens") or 0) + cr + cw, int(u.get("output_tokens") or 0), cr, cw


def message_cost(model: str, total_in: int, out: int, cr: int, cw: int) -> float:
    try:
        p, c = litellm.cost_per_token(model=model, prompt_tokens=total_in, completion_tokens=out,
                                      cache_read_input_tokens=cr, cache_creation_input_tokens=cw)
        return float(p + c)
    except Exception:
        return 0.0


def synth_stream(message: dict) -> list[str]:
    """Anthropic SSE events for a complete message (cache replays and mock mode)."""
    usage = message.get("usage") or {}
    head = {k: message.get(k) for k in ("id", "type", "role", "model")}
    events = [_event({"type": "message_start", "message": head | {
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0}}})]
    for i, block in enumerate(message.get("content") or []):
        if block.get("type") == "tool_use":
            start = {"type": "tool_use", "id": block.get("id"), "name": block.get("name"), "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block.get("input") or {})}
        elif block.get("type") == "thinking":
            start = {"type": "thinking", "thinking": ""}
            delta = {"type": "thinking_delta", "thinking": block.get("thinking", "")}
        else:
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block.get("text", "")}
        events += [
            _event({"type": "content_block_start", "index": i, "content_block": start}),
            _event({"type": "content_block_delta", "index": i, "delta": delta}),
            _event({"type": "content_block_stop", "index": i}),
        ]
    events += [
        _event({"type": "message_delta",
                "delta": {"stop_reason": message.get("stop_reason", "end_turn"), "stop_sequence": message.get("stop_sequence")},
                "usage": {"output_tokens": usage.get("output_tokens", 0)}}),
        _event({"type": "message_stop"}),
    ]
    return events


class StreamTracker:
    """Reads Anthropic SSE as it passes through: usage totals and a rebuilt message."""

    def __init__(self) -> None:
        self.buffer = ""
        self.message: dict = {}
        self.usage: dict = {}
        self.blocks: dict[int, dict] = {}
        self.stop_reason: str | None = None

    def feed(self, text: str) -> None:
        self.buffer += text.replace("\r\n", "\n")
        while "\n\n" in self.buffer:
            raw, self.buffer = self.buffer.split("\n\n", 1)
            data = "".join(line[5:].strip() for line in raw.split("\n") if line.startswith("data:"))
            if data:
                try:
                    self.handle(json.loads(data))
                except ValueError:
                    pass

    def handle(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "message_start":
            self.message = ev.get("message") or {}
            self.usage.update(self.message.get("usage") or {})
        elif kind == "content_block_start":
            self.blocks[ev.get("index", 0)] = dict(ev.get("content_block") or {})
        elif kind == "content_block_delta":
            block = self.blocks.setdefault(ev.get("index", 0), {"type": "text", "text": ""})
            delta = ev.get("delta") or {}
            if delta.get("type") == "text_delta":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif delta.get("type") == "input_json_delta":
                block["_json"] = block.get("_json", "") + delta.get("partial_json", "")
            else:
                block["_other"] = True
        elif kind == "message_delta":
            self.stop_reason = (ev.get("delta") or {}).get("stop_reason")
            self.usage.update({k: v for k, v in (ev.get("usage") or {}).items() if v is not None})

    def rebuilt(self) -> dict | None:
        """The full message, or None when it can't be rebuilt exactly (then it isn't cached)."""
        content = []
        for i in sorted(self.blocks):
            b = self.blocks[i]
            if b.get("type") != "text" or b.get("_other"):
                return None
            content.append({"type": "text", "text": b.get("text", "")})
        return {**{k: self.message.get(k) for k in ("id", "type", "role", "model")},
                "content": content, "stop_reason": self.stop_reason, "stop_sequence": None, "usage": self.usage}


# ---------- endpoint ----------

def register_messages_api(app: FastAPI, settings: Settings, authenticate) -> None:
    background: set = set()

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        authenticate(request)
        body = await request.json()
        model = body.get("model") or ""
        if Router.is_auto(model):
            model = app.state.router.baseline_model
        return {"input_tokens": _count(model, body.get("system"), body.get("messages") or [], body.get("tools"))}

    @app.post("/v1/messages")
    async def messages_endpoint(request: Request):
        tenant = authenticate(request)
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            return _error(400, "body must be JSON")
        requested = body.get("model")
        messages = body.get("messages")
        if not isinstance(requested, str) or not requested:
            return _error(400, "model: field required")
        if not isinstance(messages, list) or not messages:
            return _error(400, "messages: must be a non-empty list")

        started = time.perf_counter()
        router: Router = app.state.router
        store = app.state.store
        ledger = app.state.ledger
        header = request.headers.get
        stream = bool(body.get("stream"))
        system = body.get("system")

        # 1. Prefix stabilizer. Keep the client's tool order when it places its own breakpoints.
        tools = body.get("tools")
        if tools:
            client_breakpoints = any("cache_control" in t for t in tools)
            tools = [_sort_keys(t) for t in tools] if client_breakpoints else stabilize_tools(tools)
        phash = hashlib.sha256(
            json.dumps({"system": system, "tools": tools or []}, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        # 2. Opt-in routing
        auto = Router.is_auto(requested)
        count_model = router.baseline_model if auto else requested
        original_tokens = _count(count_model, system, messages, tools)
        model, route_reason = requested, None
        if auto:
            try:
                decision = router.route(requested, messages, tools, original_tokens)
            except ValueError as exc:
                return _error(400, str(exc))
            model, route_reason = decision.model, f"{decision.tier}:{decision.reason}"
        baseline_model = router.baseline_model if auto else requested

        out_headers = {"x-qasd-model": model}
        if route_reason:
            out_headers["x-qasd-route"] = route_reason
        common = dict(tenant=tenant, requested_model=requested, model=model, route_reason=route_reason,
                      stream=stream, prefix_hash=phash)

        # 3. Exact cache
        ok, why = rcache.eligible(body, opt_out=(header("x-qasd-cache", "").lower() == "off"))
        ok = ok and settings.cache_enabled
        cache_key = rcache.make_key(tenant, "messages:" + requested, [{"system": system}] + messages, body) if ok else None
        if cache_key:
            hit = await store.get(cache_key)
            if hit:
                out_headers["x-qasd-cache"] = "hit"
                await ledger.record(**common, gateway_cache="hit", prompt_tokens_original=hit.get("prompt_tokens", 0),
                                    cost_usd=0.0, baseline_cost_usd=hit.get("baseline_cost", 0.0),
                                    latency_ms=int((time.perf_counter() - started) * 1000))
                if stream:
                    return StreamingResponse(iter(synth_stream(hit["response"])), media_type="text/event-stream",
                                             headers=out_headers)
                return JSONResponse(hit["response"], headers=out_headers)
        out_headers["x-qasd-cache"] = "miss" if cache_key else f"skip ({why})"

        # 4. Compaction (opt-in on this endpoint)
        send = messages
        compact_info: dict = {}
        want = header("x-qasd-compact", "").lower()
        claude_code_compacting = header("x-claude-code-compaction") is not None
        if (settings.compaction_enabled and want != "off" and not claude_code_compacting
                and (settings.messages_compaction or want == "on")):
            trigger = trigger_for(model, settings.compact_trigger_tokens, settings.compact_trigger_ratio)
            summarizer = (llm_summarizer(settings.summary_model, settings.summary_max_tokens, settings.mock_response)
                          if settings.summary_model else None)
            send, compact_info = await compact(
                messages, model, store=store, trigger=trigger, target=int(trigger * settings.compact_target_ratio),
                chunk=settings.compact_chunk, summarizer=summarizer, summary_model=settings.summary_model,
            )
            if compact_info:
                out_headers["x-qasd-compacted"] = str(compact_info["removed_messages"])

        # 5. Cache breakpoints for Claude models
        breakpoints = 0
        if settings.anthropic_breakpoints and is_anthropic(model):
            system, send, tools, breakpoints = add_breakpoints(system, send, tools)

        params = {k: v for k, v in body.items() if k not in HANDLED}
        params.update(model=model, messages=send, max_tokens=body.get("max_tokens") or 4096)
        if system:
            params["system"] = system
        if tools:
            params["tools"] = tools
        # Forward every anthropic-* header as an open list (betas change with each Claude Code release)
        extra = {k: v for k, v in request.headers.items() if k.lower().startswith("anthropic-")}
        if extra:
            params["extra_headers"] = extra
        mock = settings.mock_response is not None
        if mock:
            params["mock_response"] = settings.mock_response

        sent_tokens = _count(count_model, system, send, tools) if (compact_info or mock) else original_tokens

        async def finish(usage: dict | None, response: dict | None, status: str = "ok") -> None:
            total_in, out, cr, cw = usage_numbers(usage)
            if mock and usage is not None:
                total_in, out, cr, cw = sent_tokens, out, 0, 0
            if compact_info and total_in and sent_tokens:
                orig = round(total_in * original_tokens / sent_tokens)
            else:
                orig = total_in or original_tokens
            cost = message_cost(model, total_in, out, cr, cw) if usage is not None else 0.0
            base = baseline_cost(baseline_model, orig, out) if usage is not None else 0.0
            await ledger.record(
                **common, status=status, gateway_cache="miss" if cache_key else "skip",
                prompt_tokens_original=orig, prompt_tokens_sent=total_in, completion_tokens=out,
                provider_cached_tokens=cr, compacted_messages=compact_info.get("removed_messages", 0),
                breakpoints=breakpoints, cost_usd=cost, baseline_cost_usd=base,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            if cache_key and response is not None and status == "ok":
                await store.set(cache_key, {"response": response, "prompt_tokens": orig, "baseline_cost": base},
                                ttl=settings.cache_ttl_seconds)

        async def safe_finish(*args, **kwargs) -> None:
            try:
                await finish(*args, **kwargs)
            except Exception:
                log.exception("failed to record /v1/messages request")

        # 6. Forward
        if stream:
            params["stream"] = True
        try:
            upstream = await litellm.anthropic_messages(**params)
        except Exception as exc:
            await safe_finish(None, None, status="error")
            return _upstream_error(exc)

        if not stream:
            data = _as_dict(upstream)
            await safe_finish(data.get("usage"), data)
            return JSONResponse(data, headers=out_headers)

        async def source() -> AsyncIterator[str]:
            if isinstance(upstream, dict) or hasattr(upstream, "model_dump"):
                for ev in synth_stream(_as_dict(upstream)):  # mock mode returns a whole message
                    yield ev
                return
            async for chunk in upstream:
                if isinstance(chunk, (bytes, bytearray)):
                    yield chunk.decode("utf-8", errors="replace")
                elif isinstance(chunk, str):
                    yield chunk
                else:
                    yield _event(_as_dict(chunk))

        async def relay():
            tracker = StreamTracker()
            recorded = False
            try:
                async for text in source():
                    tracker.feed(text)
                    yield text
                tracker.feed("\n\n")
                await safe_finish(tracker.usage, tracker.rebuilt())
                recorded = True
            except Exception as exc:
                await safe_finish(tracker.usage or None, None, status="error")
                recorded = True
                yield _event({"type": "error", "error": {"type": "api_error", "message": str(exc)}})
            finally:
                if not recorded:
                    import asyncio

                    task = asyncio.create_task(safe_finish(tracker.usage or None, None, status="partial"))
                    background.add(task)
                    task.add_done_callback(background.discard)

        return StreamingResponse(relay(), media_type="text/event-stream", headers=out_headers)

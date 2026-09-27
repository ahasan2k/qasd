"""Qasd gateway: an OpenAI-compatible proxy that cuts token spend safely."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import litellm
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import cache as rcache
from .compaction import compact, llm_summarizer, trigger_for
from .config import Settings
from .costs import actual_cost, baseline_cost, usage_numbers
from .ledger import Ledger
from .messages_api import register_messages_api
from .prefix import add_anthropic_breakpoints, is_anthropic, prefix_hash, stabilize_tools
from .router import Router
from .store import make_store

log = logging.getLogger("qasd")
litellm.suppress_debug_info = True

# Body fields Qasd handles itself; everything else is passed through to the provider.
HANDLED = {"model", "messages", "stream", "stream_options", "tools"}
STATIC = Path(__file__).parent / "static"


def _token_count(model: str, messages: list[dict], tools: list | None) -> int:
    try:
        return litellm.token_counter(model=model, messages=messages, tools=tools or None)
    except Exception:
        return len(json.dumps([messages, tools], default=str)) // 4


def _error(status: int, message: str, etype: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": etype}}, status_code=status)


def _upstream_error(exc: Exception) -> JSONResponse:
    status = getattr(exc, "status_code", None)
    status = status if isinstance(status, int) and 400 <= status < 600 else 502
    return _error(status, str(exc), type(exc).__name__)


def _sse(data: dict | str) -> str:
    return f"data: {data if isinstance(data, str) else json.dumps(data, default=str)}\n\n"


def _replay_stream(data: dict, include_usage: bool):
    """Turn a cached (non-stream) response back into an SSE stream."""
    base = {k: data.get(k) for k in ("id", "created", "model")} | {"object": "chat.completion.chunk"}
    for i, choice in enumerate(data.get("choices", [])):
        msg = choice.get("message") or {}
        yield _sse(base | {"choices": [{"index": i, "delta": {"role": "assistant", "content": msg.get("content")}}]})
        yield _sse(base | {"choices": [{"index": i, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}]})
    if include_usage and data.get("usage"):
        yield _sse(base | {"choices": [], "usage": data["usage"]})
    yield _sse("[DONE]")


def make_authenticator(settings: Settings):
    def authenticate(request: Request) -> str:
        """Return the tenant id for the caller: 'local', 'admin' or a key hash."""
        auth = request.headers.get("authorization", "")
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
        if settings.admin_key and key and hmac.compare_digest(key, settings.admin_key):
            return "admin"
        if not settings.api_keys:
            return "local"
        for allowed in settings.api_keys:
            if key and hmac.compare_digest(key, allowed):
                return hashlib.sha256(key.encode()).hexdigest()[:12]
        raise HTTPException(status_code=401, detail="invalid or missing gateway key")

    return authenticate


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ledger = Ledger(settings.database_url)
        await app.state.ledger.init()
        app.state.store = make_store(settings.redis_url)
        app.state.router = Router.load(settings.routing_file)
        if not settings.api_keys:
            log.warning("QASD_API_KEYS is empty: the gateway accepts any caller. Use for local testing only.")
        yield
        await app.state.store.close()
        await app.state.ledger.close()

    app = FastAPI(title="Qasd Gateway", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    background: set[asyncio.Task] = set()  # keeps late ledger writes alive

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        etype = "authentication_error" if exc.status_code == 401 else "invalid_request_error"
        return _error(exc.status_code, str(exc.detail), etype)

    authenticate = make_authenticator(settings)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/v1/models")
    async def models(request: Request):
        """Model list in a shape both OpenAI clients and Claude Code's model discovery read."""
        authenticate(request)
        router: Router = app.state.router
        entries = [("auto", "Qasd auto routing", "Routes each request to a tier by its rules")]
        entries += [(f"auto:{t}", f"Qasd {t} tier", f"Always the {t} tier: {m}") for t, m in router.tiers.items()]
        entries += [(m, m, f"{t} tier") for t, m in router.tiers.items()]
        data = [{"id": i, "object": "model", "type": "model", "owned_by": "qasd", "display_name": name,
                 "description": desc, "created_at": "2026-01-01T00:00:00Z", "created": 1767225600}
                for i, name, desc in entries]
        return {"object": "list", "data": data, "has_more": False,
                "first_id": data[0]["id"], "last_id": data[-1]["id"]}

    @app.get("/qasd/stats")
    async def stats(request: Request, days: int = 30, tenant: str | None = None):
        who = authenticate(request)
        if who not in ("admin", "local"):
            tenant = who  # a gateway key only ever sees its own numbers
        return await app.state.ledger.stats(tenant=tenant, days=max(1, min(days, 365)))

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return (STATIC / "dashboard.html").read_text()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        tenant = authenticate(request)
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            return _error(400, "body must be JSON")
        requested = body.get("model")
        messages = body.get("messages")
        if not isinstance(requested, str) or not requested:
            return _error(400, "'model' is required")
        if not isinstance(messages, list) or not messages:
            return _error(400, "'messages' must be a non-empty list")

        started = time.perf_counter()
        router: Router = app.state.router
        store = app.state.store
        ledger: Ledger = app.state.ledger
        stream = bool(body.get("stream"))
        wants_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        header = request.headers.get

        # 1. Prefix stabilizer
        tools = stabilize_tools(body.get("tools"))
        phash = prefix_hash(messages, tools)

        # 2. Opt-in routing
        auto = Router.is_auto(requested)
        count_model = router.baseline_model if auto else requested
        original_tokens = _token_count(count_model, messages, tools)
        route_reason = None
        model = requested
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

        common = dict(
            tenant=tenant, requested_model=requested, model=model, route_reason=route_reason,
            stream=stream, prefix_hash=phash,
        )

        # 3. Exact cache
        ok, why = rcache.eligible(body, opt_out=(header("x-qasd-cache", "").lower() == "off"))
        ok = ok and settings.cache_enabled
        cache_key = rcache.make_key(tenant, requested, messages, body) if ok else None
        if cache_key:
            hit = await store.get(cache_key)
            if hit:
                out_headers["x-qasd-cache"] = "hit"
                await ledger.record(
                    **common, gateway_cache="hit", prompt_tokens_original=hit.get("prompt_tokens", 0),
                    cost_usd=0.0, baseline_cost_usd=hit.get("baseline_cost", 0.0),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
                data = hit["response"]
                if stream:
                    return StreamingResponse(_replay_stream(data, wants_usage), media_type="text/event-stream",
                                             headers=out_headers)
                return JSONResponse(data, headers=out_headers)
        out_headers["x-qasd-cache"] = "miss" if cache_key else f"skip ({why})"

        # 4. Cache-aware compaction
        send = messages
        compact_info: dict = {}
        if settings.compaction_enabled and header("x-qasd-compact", "").lower() != "off":
            trigger = trigger_for(model, settings.compact_trigger_tokens, settings.compact_trigger_ratio)
            summarizer = (
                llm_summarizer(settings.summary_model, settings.summary_max_tokens, settings.mock_response)
                if settings.summary_model else None
            )
            send, compact_info = await compact(
                messages, model, store=store, trigger=trigger,
                target=int(trigger * settings.compact_target_ratio), chunk=settings.compact_chunk,
                summarizer=summarizer, summary_model=settings.summary_model,
            )
            if compact_info:
                out_headers["x-qasd-compacted"] = str(compact_info["removed_messages"])

        # 5. Provider cache breakpoints (Anthropic needs them explicitly; OpenAI caches automatically)
        breakpoints = 0
        if settings.anthropic_breakpoints and is_anthropic(model):
            send, tools, breakpoints = add_anthropic_breakpoints(send, tools)

        params = {k: v for k, v in body.items() if k not in HANDLED}
        params.update(model=model, messages=send, drop_params=True)
        if tools:
            params["tools"] = tools
        if settings.mock_response is not None:
            params["mock_response"] = settings.mock_response

        sent_tokens = _token_count(count_model, send, tools) if compact_info else original_tokens

        def scaled_original(provider_prompt: int) -> int:
            """Original prompt size in the provider's token units."""
            if not compact_info or not sent_tokens:
                return provider_prompt or original_tokens
            return round(provider_prompt * original_tokens / sent_tokens) if provider_prompt else original_tokens

        async def finish(response: Any, status: str = "ok") -> None:
            p, c, cached = usage_numbers(getattr(response, "usage", None)) if response is not None else (0, 0, 0)
            if settings.mock_response is not None and response is not None:
                # Mock usage is made up; count the real prompt so benchmarks stay meaningful.
                p = sent_tokens
                cost = baseline_cost(model, p, c)
            else:
                cost = actual_cost(response, model) if response is not None else 0.0
            orig = scaled_original(p)
            base = baseline_cost(baseline_model, orig, c) if response is not None else 0.0
            await ledger.record(
                **common, status=status, gateway_cache="miss" if cache_key else "skip",
                prompt_tokens_original=orig, prompt_tokens_sent=p, completion_tokens=c,
                provider_cached_tokens=cached, compacted_messages=compact_info.get("removed_messages", 0),
                breakpoints=breakpoints, cost_usd=cost, baseline_cost_usd=base,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            if cache_key and response is not None and status == "ok":
                await store.set(
                    cache_key,
                    {"response": response.model_dump(), "prompt_tokens": orig, "baseline_cost": base},
                    ttl=settings.cache_ttl_seconds,
                )

        # 6. Forward upstream
        if not stream:
            try:
                response = await litellm.acompletion(**params)
            except Exception as exc:
                await finish(None, status="error")
                return _upstream_error(exc)
            await finish(response)
            return JSONResponse(response.model_dump(), headers=out_headers)

        params.update(stream=True, stream_options={"include_usage": True})
        try:
            upstream = await litellm.acompletion(**params)
        except Exception as exc:
            await finish(None, status="error")
            return _upstream_error(exc)

        def build(chunks: list) -> Any:
            if not chunks:
                return None
            try:
                return litellm.stream_chunk_builder(chunks, messages=send)
            except Exception:
                return None

        async def safe_finish(built: Any, status: str) -> None:
            try:
                await finish(built, status=status if built is not None else "error")
            except Exception:
                log.exception("failed to record streamed request")

        async def relay():
            chunks: list = []
            recorded = False
            try:
                async for chunk in upstream:
                    chunks.append(chunk)
                    data = chunk.model_dump(exclude_none=True)
                    if "usage" in data and not wants_usage:
                        data.pop("usage")
                        if not any(ch.get("delta") or ch.get("finish_reason") for ch in data.get("choices", [])):
                            continue
                    yield _sse(data)
                # Record before [DONE]: clients often disconnect right after it.
                await safe_finish(build(chunks), "ok")
                recorded = True
                yield _sse("[DONE]")
            except Exception as exc:  # upstream broke mid-stream
                await safe_finish(build(chunks), "error")
                recorded = True
                yield _sse({"error": {"message": str(exc), "type": type(exc).__name__}})
            finally:
                if not recorded:  # client went away mid-stream; record without blocking cancellation
                    task = asyncio.create_task(safe_finish(build(chunks), "partial"))
                    background.add(task)
                    task.add_done_callback(background.discard)

        return StreamingResponse(relay(), media_type="text/event-stream", headers=out_headers)

    register_messages_api(app, settings, authenticate)
    return app


app = create_app()

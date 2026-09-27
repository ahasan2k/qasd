"""Prefix stabilizer.

Provider prompt caches match on the exact bytes at the start of the prompt:
tools, then system prompt, then messages. Anything that makes those bytes
drift between calls (tool order, JSON key order in schemas) turns a cache hit
into a full-price miss. This module removes that drift without changing what
the request means.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import litellm

EPHEMERAL = {"type": "ephemeral"}


def _sort_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sort_keys(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sort_keys(v) for v in value]
    return value


def _tool_name(tool: dict) -> str:
    fn = tool.get("function") or {}
    return str(fn.get("name") or tool.get("name") or "")


def stabilize_tools(tools: list[dict] | None) -> list[dict] | None:
    """Sort tools by name and sort keys inside every schema."""
    if not tools:
        return tools
    return [_sort_keys(t) for t in sorted(tools, key=_tool_name)]


def prefix_hash(messages: list[dict], tools: list[dict] | None) -> str:
    """Hash of the static prefix (tools + leading system messages).

    Logged per request so the dashboard can show how often a client's prefix
    changes, which is the main reason provider caching fails in practice.
    """
    system = []
    for m in messages:
        if m.get("role") not in ("system", "developer"):
            break
        system.append(m.get("content"))
    blob = json.dumps({"tools": tools or [], "system": system}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def is_anthropic(model: str) -> bool:
    try:
        provider = litellm.get_llm_provider(model)[1]
    except Exception:
        provider = ""
    return provider == "anthropic" or model.startswith(("claude", "anthropic/"))


def _has_cache_control(messages: list[dict], tools: list[dict] | None) -> bool:
    for t in tools or []:
        if "cache_control" in t:
            return True
    for m in messages:
        if "cache_control" in m:
            return True
        content = m.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and "cache_control" in b for b in content
        ):
            return True
    return False


def _mark(message: dict) -> bool:
    """Put a cache breakpoint on the last content block of a message."""
    content = message.get("content")
    if isinstance(content, str) and content:
        message["content"] = [{"type": "text", "text": content, "cache_control": EPHEMERAL}]
        return True
    if isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1] = {**content[-1], "cache_control": EPHEMERAL}
        return True
    return False


def add_anthropic_breakpoints(
    messages: list[dict], tools: list[dict] | None
) -> tuple[list[dict], list[dict] | None, int]:
    """Add up to 3 cache breakpoints: last tool, last system message, last message.

    Skipped entirely when the client already manages cache_control itself.
    Returns (messages, tools, breakpoints_added).
    """
    if _has_cache_control(messages, tools):
        return messages, tools, 0
    messages = copy.deepcopy(messages)
    tools = copy.deepcopy(tools) if tools else tools
    added = 0
    if tools:
        tools[-1]["cache_control"] = EPHEMERAL
        added += 1
    last_system = None
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            last_system = i
    if last_system is not None and _mark(messages[last_system]):
        added += 1
    # Rolling breakpoint on the newest message caches the conversation so far.
    if messages and last_system != len(messages) - 1:
        if messages[-1].get("role") == "user" and _mark(messages[-1]):
            added += 1
    return messages, tools, added

"""Cache-aware compaction of long conversations.

Naive sliding windows drop a few old messages on every turn, so the start of
the prompt changes every time and the provider's prompt cache never hits.
Qasd compacts differently:

* Nothing happens until the prompt passes a trigger (default 70% of the
  model's input window).
* The system prompt and the first user message (usually the task) are kept.
* Older messages are replaced by one summary, cut at a point that only moves
  in fixed steps (``compact_chunk`` messages). Between steps, every turn
  produces the same compacted prefix, so the provider cache keeps hitting.
* The cut never lands on a ``tool`` message, so a tool call is never separated
  from its result.
* Summaries are cached by the exact content they replace, so each one is
  generated once.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import litellm

from .store import Store

SUMMARY_HEADER = "[Summary of earlier conversation, condensed by the gateway]"
SUMMARY_PROMPT = (
    "Summarize the conversation below so an assistant can continue the task. "
    "Keep decisions, facts, file names, identifiers, numbers, open questions and "
    "the results of tool calls. Drop greetings and repetition. Use short bullet points."
)
DEFAULT_WINDOW = 128_000
SUMMARY_ALLOWANCE = 1_000  # tokens reserved for the summary when planning a cut


@dataclass
class CompactionPlan:
    body_start: int          # index of the first message that may be removed
    cut: int                 # messages[body_start:body_start+cut] are replaced
    tokens_before: int

    @property
    def removed_slice(self) -> slice:
        return slice(self.body_start, self.body_start + self.cut)


def is_tool_result(message: dict) -> bool:
    """True for OpenAI tool messages and Anthropic user turns carrying tool_result blocks."""
    if message.get("role") == "tool":
        return True
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def _block_text(block: dict) -> str:
    kind = block.get("type")
    if kind == "text":
        return block.get("text", "")
    if kind == "tool_use":
        return f"[calls {block.get('name')}({json.dumps(block.get('input'), default=str)[:300]})]"
    if kind == "tool_result":
        inner = block.get("content")
        if isinstance(inner, list):
            inner = " ".join(b.get("text", "") for b in inner if isinstance(b, dict))
        return f"[tool result: {str(inner or '')[:1000]}]"
    return ""


def message_tokens(model: str, message: dict) -> int:
    try:
        return litellm.token_counter(model=model, messages=[message])
    except Exception:
        return len(json.dumps(message, default=str)) // 4


def trigger_for(model: str, trigger_tokens: int, trigger_ratio: float) -> int:
    if trigger_tokens:
        return trigger_tokens
    try:
        window = litellm.get_model_info(model).get("max_input_tokens") or DEFAULT_WINDOW
    except Exception:
        window = DEFAULT_WINDOW
    return int(window * trigger_ratio)


def plan(
    messages: list[dict],
    counts: list[int],
    trigger: int,
    target: int,
    chunk: int,
) -> CompactionPlan | None:
    """Pure planning step. ``counts[i]`` is the token count of ``messages[i]``."""
    total = sum(counts)
    if total <= trigger or chunk < 1:
        return None

    head_end = 0
    while head_end < len(messages) and messages[head_end].get("role") in ("system", "developer"):
        head_end += 1
    body_start = head_end
    if body_start < len(messages) and messages[body_start].get("role") == "user":
        body_start += 1  # keep the first user message: usually the task itself

    body = messages[body_start:]
    body_counts = counts[body_start:]
    fixed = sum(counts[:body_start])
    if len(body) < 2:
        return None

    def safe(c: int) -> int:
        while c < len(body) and is_tool_result(body[c]):
            c += 1
        return c

    best = None
    c = chunk
    while c < len(body):
        cut = safe(c)
        if cut >= len(body):
            break
        best = cut
        if fixed + sum(body_counts[cut:]) + SUMMARY_ALLOWANCE <= target:
            return CompactionPlan(body_start, cut, total)
        c += chunk
    if best is None:
        # Fewer messages than one chunk: fall back to the last safe cut.
        cut = safe(len(body) - 1)
        if cut >= len(body) or cut == 0:
            return None
        best = cut
    return CompactionPlan(body_start, best, total)


def _render(messages: list[dict], limit: int = 4000) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(_block_text(b) for b in content if isinstance(b, dict))
        text = str(content or "")
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            text += f" [calls {fn.get('name')}({str(fn.get('arguments', ''))[:300]})]"
        lines.append(f"{role}: {text[:limit]}")
    return "\n".join(lines)


def deterministic_note(removed: list[dict], tokens: int) -> str:
    return (
        f"{len(removed)} earlier messages (about {tokens} tokens) were removed to fit the "
        "context window. The task and the most recent messages are kept."
    )


Summarizer = Callable[[list[dict]], Awaitable[str]]


def llm_summarizer(model: str, max_tokens: int, mock: str | None = None) -> Summarizer:
    async def summarize(removed: list[dict]) -> str:
        kwargs: dict[str, Any] = {}
        if mock is not None:
            kwargs["mock_response"] = f"- summary of {len(removed)} messages"
        resp = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": _render(removed)},
            ],
            temperature=0,
            max_tokens=max_tokens,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    return summarize


def _attach_summary(first_user: dict | None, summary: str) -> dict:
    block = f"{SUMMARY_HEADER}\n{summary}"
    if first_user is None:
        return {"role": "user", "content": block}
    msg = dict(first_user)
    content = msg.get("content")
    if isinstance(content, list):
        msg["content"] = list(content) + [{"type": "text", "text": block}]
    else:
        msg["content"] = f"{content or ''}\n\n{block}"
    return msg


async def compact(
    messages: list[dict],
    model: str,
    *,
    store: Store,
    trigger: int,
    target: int,
    chunk: int,
    summarizer: Summarizer | None,
    summary_model: str | None = None,
) -> tuple[list[dict], dict]:
    """Return (messages, info). ``info`` is empty when nothing was compacted."""
    counts = [message_tokens(model, m) for m in messages]
    p = plan(messages, counts, trigger, target, chunk)
    if p is None:
        return messages, {}

    removed = messages[p.removed_slice]
    removed_tokens = sum(counts[p.removed_slice])
    key_src = json.dumps({"m": removed, "s": summary_model}, sort_keys=True, default=str)
    key = "qasd:summary:" + hashlib.sha256(key_src.encode()).hexdigest()

    summary = await store.get(key)
    source = "cache"
    if summary is None:
        source = "deterministic"
        summary = deterministic_note(removed, removed_tokens)
        if summarizer is not None:
            try:
                summary = await summarizer(removed)
                source = "llm"
            except Exception:
                pass  # keep the deterministic note; never fail the request
        await store.set(key, summary, ttl=7 * 24 * 3600)

    head = messages[: p.body_start]
    first_user = None
    if head and head[-1].get("role") == "user" and not is_tool_result(head[-1]):
        first_user = head.pop()
    new_messages = head + [_attach_summary(first_user, summary)] + messages[p.body_start + p.cut:]

    info = {
        "removed_messages": len(removed),
        "removed_tokens": removed_tokens,
        "tokens_before": p.tokens_before,
        "summary_source": source,
    }
    return new_messages, info

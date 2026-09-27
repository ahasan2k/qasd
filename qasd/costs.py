"""Cost helpers built on LiteLLM's price table. Unknown models cost 0 rather than failing."""
from __future__ import annotations

from typing import Any

import litellm


def actual_cost(response: Any, model: str) -> float:
    try:
        return float(litellm.completion_cost(completion_response=response, model=model) or 0.0)
    except Exception:
        return 0.0


def baseline_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """What the request would have cost with no gateway: full prompt, no cache discount."""
    try:
        p, c = litellm.cost_per_token(
            model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        return float(p + c)
    except Exception:
        return 0.0


def usage_numbers(usage: Any) -> tuple[int, int, int]:
    """(prompt_tokens, completion_tokens, provider_cached_tokens) from a usage object or dict."""
    if usage is None:
        return 0, 0, 0
    get = usage.get if isinstance(usage, dict) else lambda k, d=None: getattr(usage, k, d)
    prompt = int(get("prompt_tokens", 0) or 0)
    completion = int(get("completion_tokens", 0) or 0)
    cached = int(get("cache_read_input_tokens", 0) or 0)
    details = get("prompt_tokens_details", None)
    if details is not None and not cached:
        dget = details.get if isinstance(details, dict) else lambda k, d=None: getattr(details, k, d)
        cached = int(dget("cached_tokens", 0) or 0)
    return prompt, completion, cached

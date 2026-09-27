"""Exact response cache, strict by design.

A cached answer is only correct when the same request would produce the same
answer, so the cache is used only when:

* temperature is explicitly 0,
* no tools are offered (tool calls act on changing state),
* one choice is requested (n == 1),
* the client did not opt out (``x-qasd-cache: off``).

The key covers the tenant, model and every field that can change the output,
so two different requests never share an answer and tenants never see each
other's responses.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

# Request fields that change the output and therefore belong in the key.
KEY_FIELDS = (
    "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "seed",
    "response_format", "presence_penalty", "frequency_penalty", "logit_bias",
    "reasoning_effort", "thinking", "n", "logprobs", "top_logprobs",
)


def eligible(body: dict[str, Any], opt_out: bool) -> tuple[bool, str]:
    if opt_out:
        return False, "opt-out"
    if body.get("temperature") != 0:
        return False, "temperature!=0"
    if body.get("tools") or body.get("functions"):
        return False, "tools"
    if body.get("n", 1) != 1:
        return False, "n>1"
    return True, "eligible"


def make_key(tenant: str, model: str, messages: list[dict], body: dict[str, Any]) -> str:
    material = {
        "tenant": tenant,
        "model": model,
        "messages": messages,
        "params": {k: body.get(k) for k in KEY_FIELDS if k in body},
    }
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
    return "qasd:resp:" + hashlib.sha256(blob.encode()).hexdigest()

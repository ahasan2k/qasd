"""Opt-in model router.

Routing only happens when the client asks for it with ``model: "auto"``
(or ``auto:<tier>`` to pin a tier). A model the client names explicitly is
never changed. Every decision carries a reason that is logged and returned in
the ``x-qasd-route`` response header.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

AUTO_NAMES = ("auto", "qasd/auto")
CODE_RE = re.compile(r"```|^\s{4,}\S|\bdef \w+\(|\bfunction \w+\(|\bclass \w+[:(]", re.M)

DEFAULT_CONFIG: dict[str, Any] = {
    "tiers": {
        "cheap": "claude-haiku-4-5",
        "mid": "claude-sonnet-4-6",
        "frontier": "claude-opus-4-8",
    },
    "default": "cheap",
    "baseline": "frontier",
    "rules": [
        {"tier": "frontier", "when": {"min_prompt_tokens": 30000}},
        {"tier": "frontier", "when": {"any_keywords": [
            "prove", "architecture", "architect", "design a system", "security review",
            "threat model", "root cause", "refactor", "step by step", "trade-off", "tradeoff",
        ]}},
        {"tier": "mid", "when": {"has_tools": True}},
        {"tier": "mid", "when": {"has_code": True}},
        {"tier": "mid", "when": {"min_last_user_words": 120}},
    ],
}


@dataclass
class Decision:
    model: str
    tier: str
    reason: str


@dataclass
class Router:
    tiers: dict[str, str]
    default: str = "cheap"
    baseline: str = "frontier"
    rules: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | None) -> "Router":
        cfg = dict(DEFAULT_CONFIG)
        if path and Path(path).exists():
            loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            cfg.update(loaded)
        router = cls(
            tiers=cfg["tiers"],
            default=cfg.get("default", "cheap"),
            baseline=cfg.get("baseline", "frontier"),
            rules=cfg.get("rules", []),
        )
        for name in [router.default, router.baseline] + [r["tier"] for r in router.rules]:
            if name not in router.tiers:
                raise ValueError(f"routing config refers to unknown tier '{name}'")
        return router

    @staticmethod
    def is_auto(model: str) -> bool:
        return model in AUTO_NAMES or model.startswith(("auto:", "qasd/auto:"))

    @property
    def baseline_model(self) -> str:
        return self.tiers[self.baseline]

    def route(self, model: str, messages: list[dict], tools: list | None, prompt_tokens: int) -> Decision:
        if ":" in model and self.is_auto(model):
            tier = model.rsplit(":", 1)[1]
            if tier not in self.tiers:
                raise ValueError(f"unknown tier '{tier}'; known: {', '.join(self.tiers)}")
            return Decision(self.tiers[tier], tier, f"pinned:{tier}")

        last_user = _last_user_text(messages)
        facts = {
            "prompt_tokens": prompt_tokens,
            "has_tools": bool(tools),
            "has_code": bool(CODE_RE.search(last_user)),
            "last_user_words": len(last_user.split()),
            "text": last_user.lower(),
        }
        for i, rule in enumerate(self.rules):
            matched = _match(rule.get("when", {}), facts)
            if matched:
                tier = rule["tier"]
                return Decision(self.tiers[tier], tier, f"rule{i}:{matched}")
        return Decision(self.tiers[self.default], self.default, "default")


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):
                return " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                )
            return str(content or "")
    return ""


def _match(when: dict, facts: dict) -> str | None:
    """Return a short reason if every condition in ``when`` holds, else None."""
    reasons = []
    for key, expected in when.items():
        if key == "min_prompt_tokens":
            if facts["prompt_tokens"] < expected:
                return None
            reasons.append(f"tokens>={expected}")
        elif key == "min_last_user_words":
            if facts["last_user_words"] < expected:
                return None
            reasons.append(f"words>={expected}")
        elif key == "has_tools":
            if facts["has_tools"] != bool(expected):
                return None
            reasons.append("tools")
        elif key == "has_code":
            if facts["has_code"] != bool(expected):
                return None
            reasons.append("code")
        elif key == "any_keywords":
            hit = next(
                (k for k in expected if re.search(rf"\b{re.escape(k.lower())}\b", facts["text"])),
                None,
            )
            if hit is None:
                return None
            reasons.append(f"keyword={hit}")
        else:
            raise ValueError(f"unknown routing condition '{key}'")
    return ",".join(reasons) or None

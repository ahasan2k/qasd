"""Runtime settings, read from environment variables (prefix QASD_)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(f"QASD_{name}")
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return str(_env(name, str(default))).lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # Auth: comma-separated gateway keys. Empty = open gateway (local use only).
    api_keys: list[str] = field(default_factory=list)
    admin_key: str | None = None

    database_url: str = "sqlite+aiosqlite:///./qasd.db"
    redis_url: str | None = None
    routing_file: str = "config/routing.yaml"

    # Exact cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600

    # Compaction
    compaction_enabled: bool = True
    compact_trigger_ratio: float = 0.7      # of the model's input window
    compact_trigger_tokens: int = 0         # absolute override; 0 = use ratio
    compact_target_ratio: float = 0.5       # tail kept after compaction, as share of trigger
    compact_chunk: int = 16                 # cut points move in steps of this many messages
    summary_model: str | None = None        # None = deterministic note, no extra LLM call
    summary_max_tokens: int = 800
    # /v1/messages clients such as Claude Code compact on their own; opt in per deployment
    messages_compaction: bool = False

    # Anthropic prompt-cache breakpoints
    anthropic_breakpoints: bool = True

    # Testing / benchmarking: forward nothing upstream, answer with this text
    mock_response: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        keys = [k.strip() for k in (_env("API_KEYS", "") or "").split(",") if k.strip()]
        return cls(
            api_keys=keys,
            admin_key=_env("ADMIN_KEY"),
            database_url=_env("DATABASE_URL", cls.database_url),
            redis_url=_env("REDIS_URL"),
            routing_file=_env("ROUTING_FILE", cls.routing_file),
            cache_enabled=_env_bool("CACHE_ENABLED", True),
            cache_ttl_seconds=_env_int("CACHE_TTL_SECONDS", 3600),
            compaction_enabled=_env_bool("COMPACTION_ENABLED", True),
            compact_trigger_ratio=_env_float("COMPACT_TRIGGER_RATIO", 0.7),
            compact_trigger_tokens=_env_int("COMPACT_TRIGGER_TOKENS", 0),
            compact_target_ratio=_env_float("COMPACT_TARGET_RATIO", 0.5),
            compact_chunk=_env_int("COMPACT_CHUNK", 16),
            summary_model=_env("SUMMARY_MODEL"),
            summary_max_tokens=_env_int("SUMMARY_MAX_TOKENS", 800),
            messages_compaction=_env_bool("MESSAGES_COMPACTION", False),
            anthropic_breakpoints=_env_bool("ANTHROPIC_BREAKPOINTS", True),
            mock_response=_env("MOCK_RESPONSE"),
        )

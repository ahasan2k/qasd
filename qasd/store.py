"""Key-value store for the exact cache and compaction summaries: Redis or in-memory."""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any, Protocol


class Store(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...
    async def close(self) -> None: ...


class MemoryStore:
    """Bounded LRU with per-entry TTL. Good for a single process."""

    def __init__(self, max_entries: int = 5000):
        self._data: OrderedDict[str, tuple[float | None, Any]] = OrderedDict()
        self._max = max_entries

    async def get(self, key: str) -> Any | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if expires is not None and expires < time.time():
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        expires = time.time() + ttl if ttl else None
        self._data[key] = (expires, value)
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    async def close(self) -> None:
        self._data.clear()


class RedisStore:
    def __init__(self, url: str):
        import redis.asyncio as redis

        self._r = redis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> Any | None:
        raw = await self._r.get(key)
        return json.loads(raw) if raw is not None else None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self._r.set(key, json.dumps(value), ex=ttl)

    async def close(self) -> None:
        await self._r.aclose()


def make_store(redis_url: str | None) -> Store:
    return RedisStore(redis_url) if redis_url else MemoryStore()

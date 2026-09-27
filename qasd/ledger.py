"""Savings ledger: one row per request, aggregated for the dashboard."""
from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RequestRow(Base):
    __tablename__ = "qasd_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    tenant: Mapped[str] = mapped_column(String(32), index=True)
    requested_model: Mapped[str] = mapped_column(String(128))
    model: Mapped[str] = mapped_column(String(128))
    route_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    stream: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16), default="ok")

    prompt_tokens_original: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens_sent: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    provider_cached_tokens: Mapped[int] = mapped_column(Integer, default=0)

    gateway_cache: Mapped[str] = mapped_column(String(16), default="skip")  # hit|miss|skip
    compacted_messages: Mapped[int] = mapped_column(Integer, default=0)
    breakpoints: Mapped[int] = mapped_column(Integer, default=0)
    prefix_hash: Mapped[str | None] = mapped_column(String(16), nullable=True)

    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    baseline_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)


class Ledger:
    def __init__(self, url: str):
        self.engine = create_async_engine(url, pool_pre_ping=True)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def record(self, **fields: Any) -> None:
        fields.setdefault("ts", dt.datetime.now(dt.timezone.utc))
        async with self.sessions() as s:
            s.add(RequestRow(**fields))
            await s.commit()

    async def stats(self, tenant: str | None = None, days: int = 30) -> dict[str, Any]:
        since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
        r = RequestRow
        where = [r.ts >= since]
        if tenant:
            where.append(r.tenant == tenant)
        day = func.substr(func.cast(r.ts, String), 1, 10)
        cols = (
            func.count(r.id),
            func.coalesce(func.sum(r.prompt_tokens_original), 0),
            func.coalesce(func.sum(r.prompt_tokens_sent), 0),
            func.coalesce(func.sum(r.completion_tokens), 0),
            func.coalesce(func.sum(r.provider_cached_tokens), 0),
            func.coalesce(func.sum(r.cost_usd), 0.0),
            func.coalesce(func.sum(r.baseline_cost_usd), 0.0),
        )
        async with self.sessions() as s:
            total = (await s.execute(select(*cols).where(*where))).one()
            by_day = (await s.execute(select(day, *cols).where(*where).group_by(day).order_by(day))).all()
            hits = (await s.execute(
                select(r.gateway_cache, func.count(r.id)).where(*where).group_by(r.gateway_cache)
            )).all()
            models = (await s.execute(
                select(r.model, func.count(r.id), func.coalesce(func.sum(r.cost_usd), 0.0))
                .where(*where).group_by(r.model).order_by(func.count(r.id).desc())
            )).all()
            prefixes = (await s.execute(
                select(r.tenant, func.count(func.distinct(r.prefix_hash))).where(*where).group_by(r.tenant)
            )).all()
            compacted = (await s.execute(
                select(func.count(r.id)).where(*where, r.compacted_messages > 0)
            )).scalar_one()

        def row(values) -> dict[str, Any]:
            n, p_orig, p_sent, comp, cached, cost, base = values
            return {
                "requests": int(n),
                "prompt_tokens_original": int(p_orig),
                "prompt_tokens_sent": int(p_sent),
                "completion_tokens": int(comp),
                "provider_cached_tokens": int(cached),
                "cost_usd": round(float(cost), 6),
                "baseline_cost_usd": round(float(base), 6),
                "saved_usd": round(max(0.0, float(base) - float(cost)), 6),
                "saved_pct": round(100 * (1 - float(cost) / float(base)), 1) if base else 0.0,
            }

        return {
            "days": days,
            "tenant": tenant,
            "totals": row(total),
            "by_day": [{"day": d[0], **row(d[1:])} for d in by_day],
            "gateway_cache": {k: int(v) for k, v in hits},
            "compacted_requests": int(compacted),
            "models": [{"model": m, "requests": int(n), "cost_usd": round(float(c), 6)} for m, n, c in models],
            "distinct_prefixes": {t: int(n) for t, n in prefixes},
        }

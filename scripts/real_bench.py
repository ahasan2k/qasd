"""Measured A/B benchmark against the real Anthropic API.

Arm A sends an agent-style conversation straight to Anthropic, the way a typical
client does (no cache breakpoints). Arm B sends the exact same requests through Qasd.
Both arms send identical messages turn by turn, so the only difference is Qasd.
Costs are computed from the usage Anthropic returns on every response.

Part 1, agent session: a long system prompt plus N turns of tool-output-sized messages.
Part 2, repeated lookups: the same deterministic question asked several times.

Run on the machine where Qasd is installed, with its Python:
  C:\\ProgramData\\Qasd\\venv\\Scripts\\python.exe scripts\\real_bench.py

Keys are read from C:\\ProgramData\\Qasd\\.env (or QASD_ENV_FILE), so run it from an
elevated terminal. Estimated cost with the defaults on Claude Haiku 4.5: about $1-2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

DEFAULT_ENV = r"C:\ProgramData\Qasd\.env"

# Anthropic list prices (USD per million tokens) used when LiteLLM's table is unavailable.
FALLBACK_PRICES = {"claude-haiku-4-5": {"in": 1.0, "out": 5.0, "cache_read": 0.10, "cache_write": 1.25}}


def load_env(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return values
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                values[k.strip()] = v.strip()
    return values


def prices(model: str) -> dict[str, float]:
    try:
        import litellm

        info = litellm.model_cost[model]
        return {
            "in": info["input_cost_per_token"] * 1e6,
            "out": info["output_cost_per_token"] * 1e6,
            "cache_read": info.get("cache_read_input_token_cost", info["input_cost_per_token"] * 0.1) * 1e6,
            "cache_write": info.get("cache_creation_input_token_cost", info["input_cost_per_token"] * 1.25) * 1e6,
        }
    except Exception:
        return FALLBACK_PRICES.get(model, FALLBACK_PRICES["claude-haiku-4-5"])


def cost(usage: dict, p: dict[str, float]) -> float:
    return (usage.get("input_tokens", 0) * p["in"]
            + usage.get("output_tokens", 0) * p["out"]
            + (usage.get("cache_read_input_tokens") or 0) * p["cache_read"]
            + (usage.get("cache_creation_input_tokens") or 0) * p["cache_write"]) / 1e6


# ---------- workload ----------

RULES = [
    "Read the relevant files before editing them and keep changes minimal.",
    "Follow the repository's existing naming, formatting and error-handling conventions.",
    "Never commit secrets; configuration comes from environment variables.",
    "Every database change needs a reversible migration and a test.",
    "Public functions get type hints and a one-line docstring.",
    "Prefer small pure functions; isolate I/O at the edges.",
    "Log with structured fields, never with string concatenation.",
    "Return errors to the caller with enough context to act on them.",
]


def system_prompt() -> str:
    # About 6,000 tokens: a realistic coding-agent system prompt with tool docs and rules.
    parts = ["You are a senior engineer working in a Python/FastAPI billing service. Be concise."]
    for i in range(1, 61):
        parts.append(f"Tool {i}: `tool_{i}(path: str, query: str) -> str` reads or searches part {i} of the "
                     f"repository and returns matching lines with file names and line numbers. "
                     f"Use it when the task touches module {i}; it never modifies files.")
    for r in RULES * 12:
        parts.append(f"Rule: {r}")
    return "\n".join(parts)


def tool_output(i: int) -> str:
    lines = [f"billing/module_{i % 9}.py:{100 + j}: def handle_invoice_{i}_{j}(invoice, customer): "
             f"total = sum(line.amount for line in invoice.lines)  # step {j}" for j in range(25)]
    return f"Output of step {i}:\n" + "\n".join(lines)


def assistant_turn(i: int) -> str:
    return (f"Step {i}: I read module_{i % 9}.py. The invoice handler recomputes totals without VAT rounding; "
            f"I will update it to use Decimal quantize and add a test for step {i}.")


# ---------- runner ----------

class Arm:
    def __init__(self, name: str, url: str, headers: dict[str, str]):
        self.name = name
        self.client = httpx.Client(base_url=url, headers=headers, timeout=180)
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}
        self.cost = 0.0
        self.requests = 0
        self.gateway_hits = 0
        self.seconds = 0.0

    def send(self, body: dict, p: dict[str, float]) -> None:
        start = time.perf_counter()
        for attempt in range(5):
            r = self.client.post("/v1/messages", json=body)
            if r.status_code in (429, 529) or r.status_code >= 500:
                time.sleep(int(r.headers.get("retry-after", 5 * (attempt + 1))))
                continue
            break
        self.seconds += time.perf_counter() - start
        if r.status_code != 200:
            raise SystemExit(f"{self.name}: HTTP {r.status_code}: {r.text[:500]}")
        self.requests += 1
        if r.headers.get("x-qasd-cache") == "hit":
            self.gateway_hits += 1
            return  # served by Qasd, nothing billed by Anthropic
        u = r.json().get("usage", {})
        for k in self.usage:
            self.usage[k] += int(u.get(k) or 0)
        self.cost += cost(u, p)


def run_part(arms: list[Arm], bodies: list[dict], p: dict[str, float], label: str) -> dict:
    before = {a.name: (a.cost, dict(a.usage), a.requests, a.gateway_hits) for a in arms}
    for i, body in enumerate(bodies, 1):
        for a in arms:
            a.send(body, p)
        print(f"\r  {label}: {i}/{len(bodies)}", end="", flush=True)
    print()
    out = {}
    for a in arms:
        c0, u0, n0, h0 = before[a.name]
        out[a.name] = {"cost_usd": round(a.cost - c0, 5), "requests": a.requests - n0,
                       "gateway_cache_hits": a.gateway_hits - h0,
                       **{k: a.usage[k] - u0[k] for k in a.usage}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--lookups", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--qasd-url", default="http://localhost:8787")
    ap.add_argument("--anthropic-url", default="https://api.anthropic.com")
    ap.add_argument("--env-file", default=os.getenv("QASD_ENV_FILE", DEFAULT_ENV))
    ap.add_argument("--out", default="qasd-benchmark.json")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = ap.parse_args()

    env = load_env(args.env_file)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_API_KEY")
    qasd_key = os.getenv("QASD_APP_KEY") or (env.get("QASD_API_KEYS", "").split(",")[0].strip())
    if not anthropic_key:
        raise SystemExit(f"No ANTHROPIC_API_KEY found (looked in the environment and {args.env_file}).")

    p = prices(args.model)
    system = system_prompt()
    per_turn = (len(assistant_turn(0)) + len(tool_output(0))) / 3.6
    arm_a_tokens = sum(len(system) / 3.6 + i * per_turn for i in range(args.turns))
    # Arm A pays full input price; arm B mostly pays cache reads, plus one cache write per turn.
    est = arm_a_tokens * p["in"] / 1e6 * 1.3
    print(f"Model {args.model}; {args.turns} agent turns + {args.lookups} repeated lookups per arm.")
    print(f"Estimated total cost for both arms: about ${est:.2f} (arm A pays full price, so it's most of it).")
    if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
        return 1

    common = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
    arm_a = Arm("direct", args.anthropic_url, {**common, "x-api-key": anthropic_key})
    arm_b = Arm("qasd", args.qasd_url, {**common, "x-api-key": qasd_key or "local"})
    arms = [arm_a, arm_b]

    # Part 1: agent session. Identical requests on both arms, turn by turn.
    messages: list[dict] = [{"role": "user", "content": "TASK: fix VAT rounding across the billing module."}]
    session = []
    for i in range(args.turns):
        session.append({"model": args.model, "max_tokens": args.max_tokens, "system": system,
                        "messages": [dict(m) for m in messages]})
        messages += [{"role": "assistant", "content": assistant_turn(i)},
                     {"role": "user", "content": tool_output(i)}]
    part1 = run_part(arms, session, p, "agent session")

    # Part 2: the same deterministic lookup, asked repeatedly.
    lookup = {"model": args.model, "max_tokens": args.max_tokens, "temperature": 0,
              "messages": [{"role": "user", "content": "In one sentence: what is the UAE standard VAT rate?"}]}
    part2 = run_part(arms, [lookup] * args.lookups, p, "repeated lookups")

    def pct(a: float, b: float) -> float:
        return round(100 * (1 - b / a), 1) if a else 0.0

    total_a = part1["direct"]["cost_usd"] + part2["direct"]["cost_usd"]
    total_b = part1["qasd"]["cost_usd"] + part2["qasd"]["cost_usd"]
    result = {
        "model": args.model, "turns": args.turns, "lookups": args.lookups,
        "date": time.strftime("%Y-%m-%d"), "prices_per_million": p,
        "agent_session": part1, "repeated_lookups": part2,
        "summary": {
            "direct_cost_usd": round(total_a, 4), "qasd_cost_usd": round(total_b, 4),
            "saved_pct_total": pct(total_a, total_b),
            "saved_pct_agent_session": pct(part1["direct"]["cost_usd"], part1["qasd"]["cost_usd"]),
            "saved_pct_repeated_lookups": pct(part2["direct"]["cost_usd"], part2["qasd"]["cost_usd"]),
            "seconds_direct": round(arm_a.seconds, 1), "seconds_qasd": round(arm_b.seconds, 1),
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    s = result["summary"]
    print()
    print(f"                        Direct       Through Qasd   Saved")
    print(f"  Agent session       ${part1['direct']['cost_usd']:>9.4f}   ${part1['qasd']['cost_usd']:>9.4f}     {s['saved_pct_agent_session']}%")
    print(f"  Repeated lookups    ${part2['direct']['cost_usd']:>9.4f}   ${part2['qasd']['cost_usd']:>9.4f}     {s['saved_pct_repeated_lookups']}%")
    print(f"  Total               ${total_a:>9.4f}   ${total_b:>9.4f}     {s['saved_pct_total']}%")
    print(f"  Time                {s['seconds_direct']:>9.1f}s   {s['seconds_qasd']:>9.1f}s")
    print(f"\nCached input tokens read through Qasd: {part1['qasd']['cache_read_input_tokens']:,}")
    print(f"Full results saved to {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

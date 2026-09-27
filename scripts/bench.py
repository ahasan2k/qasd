"""Send traffic through a running Qasd gateway and print what it saved.

Examples:
  # Synthetic agent loop (works with QASD_MOCK_RESPONSE set, no provider keys needed)
  python scripts/bench.py --synthetic --turns 60 --model gpt-4o-mini

  # Replay real requests: one OpenAI-style request body per line
  python scripts/bench.py --replay requests.jsonl

Point it at the gateway with --base-url and --key (defaults: localhost:8787, no key).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def synthetic(turns: int, model: str):
    system = "You are a coding agent. Follow the repository conventions. " * 40
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "TASK: migrate the billing module to the new invoice API."}]
    for i in range(turns):
        messages = messages + [
            {"role": "assistant", "content": f"Step {i}: reading files and applying changes. " + "context " * 150},
            {"role": "user", "content": f"Output of step {i}: " + "log line " * 120},
        ]
        yield {"model": model, "messages": messages, "temperature": 0.2}
    # A few repeated deterministic lookups, which the exact cache should absorb
    for _ in range(10):
        yield {"model": model, "temperature": 0,
               "messages": [{"role": "user", "content": "What is our refund policy for annual plans?"}]}


def replay(path: str):
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8787")
    ap.add_argument("--key", default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--replay")
    args = ap.parse_args()
    if not args.synthetic and not args.replay:
        ap.error("choose --synthetic or --replay FILE")

    headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}
    source = synthetic(args.turns, args.model) if args.synthetic else replay(args.replay)
    sent = failed = 0
    started = time.time()
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=120) as client:
        for body in source:
            body.pop("stream", None)
            r = client.post("/v1/chat/completions", json=body)
            sent += 1
            if r.status_code != 200:
                failed += 1
                print(f"request {sent}: HTTP {r.status_code} {r.text[:200]}", file=sys.stderr)
        stats = client.get("/qasd/stats", params={"days": 1}).json()

    t = stats["totals"]
    removed = 100 * (1 - t["prompt_tokens_sent"] / t["prompt_tokens_original"]) if t["prompt_tokens_original"] else 0
    print(f"Sent {sent} requests ({failed} failed) in {time.time() - started:.1f}s")
    print("Totals for the last day (whole gateway, not only this run):")
    print(f"  prompt tokens      {t['prompt_tokens_original']:>12,} original -> {t['prompt_tokens_sent']:,} sent ({removed:.1f}% removed)")
    print(f"  provider cached    {t['provider_cached_tokens']:>12,} tokens")
    print(f"  gateway cache hits {stats['gateway_cache'].get('hit', 0):>12,}")
    print(f"  cost               ${t['cost_usd']:.4f} vs ${t['baseline_cost_usd']:.4f} without Qasd "
          f"(saved ${t['saved_usd']:.4f}, {t['saved_pct']}%)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

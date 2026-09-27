# Qasd Gateway

Cut your LLM bill without breaking prompt caching, and see exactly what you saved.

Qasd is an open-source gateway that speaks both the OpenAI API (`/v1/chat/completions`) and the Anthropic Messages API (`/v1/messages`). Point any OpenAI or Anthropic client at it and keep using OpenAI, Anthropic, Gemini, Ollama or any other provider LiteLLM supports. Qasd trims what gets sent, keeps the provider's prompt cache hitting, and logs the cost of every request next to what it would have cost without the gateway.

## What it does

| Feature | What it does | Why it's safe |
| --- | --- | --- |
| **Prefix stabilizer** | Sorts tools by name and JSON schema keys, so the start of every prompt is byte-for-byte identical between calls. Adds Anthropic `cache_control` breakpoints (tools, system prompt, latest user message) automatically. | Changes the byte order only, never the meaning. Skipped when the client sets its own breakpoints. |
| **Cache-aware compaction** | When a conversation passes 70% of the model's input window, older turns are replaced by one summary. The cut point moves in fixed steps, so the compacted prompt stays identical across turns and the provider cache keeps hitting. | Keeps the system prompt and the first user message (the task). Never separates a tool call from its result. Each summary is made once and cached. |
| **Exact response cache** | Returns a stored answer for a repeated request. Streaming requests get the stored answer as a stream. | Only when `temperature` is 0, no tools are offered and `n` is 1. The key covers the tenant, model and every output-changing parameter. |
| **Opt-in router** | `model: "auto"` picks a cheap, mid or frontier model from rules in `config/routing.yaml`. `auto:mid` pins a tier. | Routes only when asked. A model the client names is never changed. Every decision is returned in `x-qasd-route` and logged. |
| **Savings ledger and dashboard** | Records tokens before and after, provider-cached tokens, cache hits, cost and baseline cost for each request. `/dashboard` shows the totals and a per-day chart. | Each API key sees only its own numbers. The admin key sees everything. |

Streaming, tool calls, images and provider-specific parameters pass through unchanged.

## Works with

| Client | How to connect |
| --- | --- |
| **Claude Code** | `export ANTHROPIC_BASE_URL=http://localhost:8787` and `export ANTHROPIC_AUTH_TOKEN=<gateway key>` |
| **Claude Agent SDK** and apps built on the Anthropic SDK | Same two variables, or `Anthropic(base_url=..., api_key=<gateway key>)` |
| **Claude Desktop** | Developer menu, Configure Third-Party Inference, gateway URL and gateway key |
| **Cursor, Continue, Cline, Aider, Open WebUI, LibreChat, n8n, LangChain** | Any "OpenAI-compatible" base URL setting: `http://localhost:8787/v1` |
| **Your own code** | Any OpenAI SDK with `base_url="http://localhost:8787/v1"` |

The upstream provider keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and so on) live on the gateway, not on the clients.

**Note on Claude subscriptions:** when a Claude app sends traffic through a gateway credential, it bills at API rates on the gateway's provider account instead of a Pro or Max plan.

### Claude Code specifics

The `/v1/messages` endpoint follows Claude Code's gateway compatibility rules:

- It forwards every `anthropic-*` header and every body field unchanged, so new betas keep working.
- It streams events as they arrive, including keep-alive pings.
- It relays upstream error bodies and the `retry-after` and `x-should-retry` headers as-is, so Claude Code's automatic recovery works.
- It serves `/v1/messages/count_tokens` for accurate `/context` numbers.
- It serves `/v1/models` for the model picker.
- It leaves tool order alone when the client places its own cache breakpoints, which Claude Code always does.
- Compaction is off on this endpoint by default, because Claude Code compacts on its own and rewriting earlier turns can invalidate preserved thinking. Turn it on for other Anthropic-SDK apps with `QASD_MESSAGES_COMPACTION=true` or the `x-qasd-compact: on` header. It is always skipped on Claude Code's own compaction requests.

For Claude Code, the main benefits are per-session cost visibility, the savings ledger and `auto` routing. Claude Code already manages its own prompt caching.

## Quick start

### Docker Compose (gateway + Postgres + Redis)

```bash
cp .env.example .env        # set QASD_API_KEYS, QASD_ADMIN_KEY and your provider keys
docker compose up -d --build
```

### Windows (runs in the background, starts at boot)

From an elevated PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-windows.ps1
```

The script installs Python if needed, installs Qasd into `C:\ProgramData\Qasd` and creates keys. It also registers a "Qasd Gateway" scheduled task that starts at boot and restarts if it stops. By default Qasd listens on localhost only. Add `-Listen` to accept connections from other machines, and `-Update` to update the code while keeping your keys.

### Local, without Docker

```bash
pip install -e ".[dev]"
export OPENAI_API_KEY=sk-...  QASD_API_KEYS=my-app-key
python -m qasd                # listens on :8787, SQLite ledger, in-memory cache
```

### Use it

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8787/v1", api_key="my-app-key")
client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello"}])
client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "Hello"}])  # routed
```

Open `http://localhost:8787/dashboard` and enter your key to see savings.

## Response headers

| Header | Meaning |
| --- | --- |
| `x-qasd-model` | Model the request was sent to |
| `x-qasd-route` | Tier and rule that chose it (only for `auto`) |
| `x-qasd-cache` | `hit`, `miss`, or `skip (reason)` |
| `x-qasd-compacted` | Number of older messages replaced by a summary |

## Request headers

| Header | Effect |
| --- | --- |
| `x-qasd-cache: off` | Skip the exact cache for this request |
| `x-qasd-compact: off` | Skip compaction for this request |
| `x-qasd-compact: on` | Compact this `/v1/messages` request even when `QASD_MESSAGES_COMPACTION` is off |

## Configuration

All settings are environment variables. See `.env.example`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `QASD_API_KEYS` | empty | Comma-separated gateway keys. Empty means anyone can call it, for local use only. |
| `QASD_ADMIN_KEY` | none | Key that sees every tenant's stats |
| `QASD_DATABASE_URL` | `sqlite+aiosqlite:///./qasd.db` | Ledger database. Use `postgresql+asyncpg://...` in production. |
| `QASD_REDIS_URL` | none | Shared cache. Without it, an in-memory cache is used per process. |
| `QASD_ROUTING_FILE` | `config/routing.yaml` | Tiers and rules for `auto` |
| `QASD_CACHE_ENABLED` / `QASD_CACHE_TTL_SECONDS` | `true` / `3600` | Exact cache |
| `QASD_COMPACTION_ENABLED` | `true` | Compaction on or off |
| `QASD_COMPACT_TRIGGER_RATIO` | `0.7` | Compact above this share of the model's input window |
| `QASD_COMPACT_TRIGGER_TOKENS` | `0` | Fixed trigger in tokens (overrides the ratio) |
| `QASD_COMPACT_TARGET_RATIO` | `0.5` | Size to compact down to, as a share of the trigger |
| `QASD_COMPACT_CHUNK` | `16` | Cut point moves in steps of this many messages. Larger means fewer cache resets. |
| `QASD_SUMMARY_MODEL` | none | Model that writes summaries. Without it, a short note replaces old turns. |
| `QASD_MESSAGES_COMPACTION` | `false` | Compaction on `/v1/messages` (off because Claude Code compacts itself) |
| `QASD_ANTHROPIC_BREAKPOINTS` | `true` | Add Anthropic cache breakpoints automatically |
| `QASD_MOCK_RESPONSE` | none | Answer every request with this text instead of calling a provider |
| `QASD_HOST` / `QASD_PORT` | `0.0.0.0` / `8787` | Listen address |

## How savings are measured

For each request, the ledger stores:

- **Cost**: what the provider charged, from LiteLLM's price table, including prompt-cache discounts.
- **Baseline cost**: the same completion with the original, uncompacted prompt, at full price, on the model the client asked for. For `auto` requests, the baseline is the tier set as `baseline` in `routing.yaml` (frontier by default), so routing savings show up.

Gateway cache hits cost nothing and count their stored baseline as saved. Token counts for the original prompt are estimates when compaction ran. Provider usage is used everywhere else.

## Measured results

A/B benchmark on the real Anthropic API, Claude Haiku 4.5, 28 September 2026. Every request was sent twice with identical content: once straight to Anthropic, once through Qasd. Costs come from the usage Anthropic returned on each response.

| Workload | Direct | Through Qasd | Saved |
| --- | --- | --- | --- |
| Agent session: 40 turns, ~4,600-token system prompt, tool output every turn | $1.0946 | $0.1763 | **83.9%** |
| Same deterministic question asked 10 times | $0.0008 | $0.0001 | 89.4% |
| **Total** | **$1.0955** | **$0.1764** | **83.9%** |

Almost all of the saving comes from prompt caching. Qasd adds cache breakpoints that the direct client didn't set, so 1.03 million input tokens were billed at the cached-read rate instead of full price. The run through Qasd also finished slightly faster (70.6 s against 76.0 s).

**What this measures:** a client that doesn't manage prompt caching itself, which is common in scripts, agents and many SDK integrations. Apps that already set their own `cache_control` breakpoints will see much smaller gains; Qasd leaves their breakpoints alone. Savings depend on how much of each prompt repeats between calls.

Reproduce it with your own key (about $1.30 for both arms):

```bash
python scripts/real_bench.py            # reads ANTHROPIC_API_KEY and the Qasd key from the Qasd .env
```

## Benchmark (mock mode)

```bash
QASD_MOCK_RESPONSE=ok QASD_COMPACT_TRIGGER_TOKENS=20000 python -m qasd &
python scripts/bench.py --synthetic --turns 60
python scripts/bench.py --replay my-requests.jsonl   # one request body per line
```

Mock mode needs no provider keys and shows compaction and exact-cache savings. Provider prompt caching only shows up with real traffic, so publish benchmark numbers from a real run.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Roadmap

- Semantic cache for FAQ-style, read-only endpoints, scoped per tenant
- Tool-schema pruning for large tool sets
- Arabic-aware token accounting
- Per-key budgets, alerts and monthly cost reports
- Hosted edition in a UAE region

## License

MIT

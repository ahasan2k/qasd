import json

import litellm
import pytest
from fastapi.testclient import TestClient

from qasd.app import create_app
from qasd.config import Settings
from qasd.messages_api import StreamTracker, add_breakpoints, synth_stream


@pytest.fixture
def client(tmp_path):
    def make(**overrides):
        overrides.setdefault("mock_response", "Hi from mock")
        s = Settings(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db", routing_file="config/routing.yaml", **overrides)
        return TestClient(create_app(s))
    return make


BODY = {"model": "claude-haiku-4-5", "max_tokens": 100, "system": "Be terse.",
        "messages": [{"role": "user", "content": "hello"}]}


def events(text):
    return [json.loads(line[5:]) for line in text.splitlines() if line.startswith("data:")]


def test_non_stream_and_ledger(client):
    with client() as c:
        r = c.post("/v1/messages", json=BODY, headers={"x-api-key": "anything"})
        assert r.status_code == 200
        assert r.json()["content"][0]["text"] == "Hi from mock"
        assert r.headers["x-qasd-model"] == "claude-haiku-4-5"
        assert c.get("/qasd/stats").json()["totals"]["requests"] == 1


def test_stream_mock_is_valid_anthropic_sse(client):
    with client() as c:
        r = c.post("/v1/messages", json=BODY | {"stream": True})
        evs = events(r.text)
        assert [e["type"] for e in evs][0] == "message_start" and evs[-1]["type"] == "message_stop"
        assert "".join(e["delta"].get("text", "") for e in evs if e["type"] == "content_block_delta") == "Hi from mock"


def test_real_stream_passthrough_counts_cache_tokens(client, monkeypatch):
    """Upstream SSE bytes pass through untouched; usage incl. cache reads lands in the ledger."""
    msg = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6",
           "content": [{"type": "text", "text": "Hello there"}], "stop_reason": "end_turn",
           "usage": {"input_tokens": 50, "output_tokens": 7, "cache_read_input_tokens": 9000,
                     "cache_creation_input_tokens": 0}}
    raw = "".join(synth_stream(msg))
    # put the full usage on message_start like Anthropic does
    raw = raw.replace('"usage": {"input_tokens": 50, "output_tokens": 0}',
                      '"usage": {"input_tokens": 50, "output_tokens": 1, "cache_read_input_tokens": 9000, "cache_creation_input_tokens": 0}')
    seen = {}

    async def fake(**kwargs):
        seen.update(kwargs)

        async def gen():
            for i in range(0, len(raw), 37):  # arbitrary chunk boundaries
                yield raw[i:i + 37].encode()
        return gen()

    monkeypatch.setattr(litellm, "anthropic_messages", fake)
    with client(mock_response=None) as c:
        r = c.post("/v1/messages", json=BODY | {"model": "claude-sonnet-4-6", "stream": True},
                   headers={"anthropic-beta": "some-beta"})
        assert r.text == raw
        assert seen["extra_headers"] == {"anthropic-beta": "some-beta"}
        t = c.get("/qasd/stats").json()["totals"]
        assert t["prompt_tokens_sent"] == 9050 and t["provider_cached_tokens"] == 9000
        assert t["completion_tokens"] == 7
        assert 0 < t["cost_usd"] < t["baseline_cost_usd"]  # cache reads are billed at a discount


def test_breakpoints_added_and_respected():
    tools = [{"name": "b", "input_schema": {}}, {"name": "a", "input_schema": {}}]
    msgs = [{"role": "user", "content": "q"}]
    system, m2, t2, n = add_breakpoints("S", msgs, tools)
    assert n == 3 and system[0]["cache_control"] and t2[-1]["cache_control"] and m2[-1]["content"][0]["cache_control"]
    _, _, _, n2 = add_breakpoints(system, m2, t2)
    assert n2 == 0


def test_client_tool_order_kept_when_it_sets_breakpoints(client, monkeypatch):
    seen = {}
    real = litellm.anthropic_messages

    async def spy(**kwargs):
        seen.update(kwargs)
        return await real(**kwargs)

    monkeypatch.setattr(litellm, "anthropic_messages", spy)
    tools = [{"name": "zeta", "input_schema": {}, "cache_control": {"type": "ephemeral"}},
             {"name": "alpha", "input_schema": {}}]
    with client() as c:
        c.post("/v1/messages", json=BODY | {"tools": tools})
    assert [t["name"] for t in seen["tools"]] == ["zeta", "alpha"]
    with client() as c:
        c.post("/v1/messages", json=BODY | {"tools": [{"name": "zeta", "input_schema": {}}, {"name": "alpha", "input_schema": {}}]})
    assert [t["name"] for t in seen["tools"]] == ["alpha", "zeta"]


def test_exact_cache_and_stream_replay(client):
    with client() as c:
        b = BODY | {"temperature": 0}
        assert c.post("/v1/messages", json=b).headers["x-qasd-cache"] == "miss"
        assert c.post("/v1/messages", json=b).headers["x-qasd-cache"] == "hit"
        r = c.post("/v1/messages", json=b | {"stream": True})
        assert r.headers["x-qasd-cache"] == "hit"
        assert events(r.text)[-1]["type"] == "message_stop"


def test_auto_routing_and_count_tokens(client):
    with client() as c:
        r = c.post("/v1/messages", json=BODY | {"model": "auto"})
        assert r.headers["x-qasd-route"].startswith("cheap")
        assert c.post("/v1/messages/count_tokens", json=BODY).json()["input_tokens"] > 0


def test_compaction_opt_in(client):
    msgs = [{"role": "user", "content": "TASK"}]
    for i in range(30):
        msgs += [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "read", "input": {"i": i}}]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "data " * 40}]}]
    body = BODY | {"messages": msgs}
    with client(compact_trigger_tokens=400, compact_chunk=4) as c:
        assert "x-qasd-compacted" not in c.post("/v1/messages", json=body).headers  # off by default
        r = c.post("/v1/messages", json=body, headers={"x-qasd-compact": "on"})
        assert int(r.headers["x-qasd-compacted"]) > 0


def test_errors_are_anthropic_shaped(client, monkeypatch):
    async def boom(**kwargs):
        raise litellm.RateLimitError("slow down", llm_provider="anthropic", model="claude-haiku-4-5")

    monkeypatch.setattr(litellm, "anthropic_messages", boom)
    with client() as c:
        r = c.post("/v1/messages", json=BODY)
        assert r.status_code == 429 and r.json()["error"]["type"] == "rate_limit_error"
        assert c.post("/v1/messages", json={"model": "x"}).json()["type"] == "error"


def test_tracker_handles_split_events():
    t = StreamTracker()
    raw = "".join(synth_stream({"id": "m", "type": "message", "role": "assistant", "model": "x",
                                "content": [{"type": "text", "text": "ab"}], "usage": {"input_tokens": 3, "output_tokens": 2}}))
    for ch in raw:
        t.feed(ch)
    assert t.rebuilt()["content"] == [{"type": "text", "text": "ab"}]


def test_upstream_error_body_relayed_unchanged(client, monkeypatch):
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": "thinking: Input tag 'x' is invalid"}}

    async def boom(**kwargs):
        e = litellm.BadRequestError(f"AnthropicException - {json.dumps(body)}", model="claude-haiku-4-5",
                                    llm_provider="anthropic")
        e.litellm_response_headers = {"x-should-retry": "false", "retry-after": "3"}
        raise e

    monkeypatch.setattr(litellm, "anthropic_messages", boom)
    with client() as c:
        r = c.post("/v1/messages?beta=true", json=BODY)
        assert r.status_code == 400 and r.json() == body
        assert r.headers["x-should-retry"] == "false" and r.headers["retry-after"] == "3"


def test_all_anthropic_headers_forwarded_and_models_discoverable(client, monkeypatch):
    seen = {}
    real = litellm.anthropic_messages

    async def spy(**kwargs):
        seen.update(kwargs)
        return await real(**kwargs)

    monkeypatch.setattr(litellm, "anthropic_messages", spy)
    with client() as c:
        c.post("/v1/messages", json=BODY, headers={"anthropic-beta": "a,b", "anthropic-version": "2023-06-01",
                                                   "x-claude-code-session-id": "s1"})
        assert seen["extra_headers"] == {"anthropic-beta": "a,b", "anthropic-version": "2023-06-01"}
        ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
        assert "auto" in ids and any("claude" in i for i in ids)

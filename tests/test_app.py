import json

import litellm
import pytest
from fastapi.testclient import TestClient

from qasd.app import create_app
from qasd.config import Settings


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def _make(**overrides):
        settings = Settings(
            database_url=f"sqlite+aiosqlite:///{tmp_path}/qasd-{len(clients)}.db",
            routing_file="config/routing.yaml",
            mock_response="Hello from the mock.",
            **overrides,
        )
        client = TestClient(create_app(settings))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def chat(client, key=None, **body):
    body.setdefault("model", "gpt-4o-mini")
    body.setdefault("messages", [{"role": "user", "content": "Say hello"}])
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return client.post("/v1/chat/completions", json=body, headers=headers)


def sse_events(text):
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


def test_basic_completion_and_ledger(make_client):
    c = make_client()
    r = chat(c)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Hello from the mock."
    assert r.headers["x-qasd-model"] == "gpt-4o-mini"
    assert r.headers["x-qasd-cache"].startswith("skip")
    stats = c.get("/qasd/stats").json()
    assert stats["totals"]["requests"] == 1
    assert stats["totals"]["cost_usd"] > 0


def test_auth_required_when_keys_configured(make_client):
    c = make_client(api_keys=["k1", "k2"], admin_key="adm")
    r = chat(c)
    assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"
    assert chat(c, key="wrong").status_code == 401
    assert chat(c, key="k1").status_code == 200
    assert chat(c, key="k2").status_code == 200
    own = c.get("/qasd/stats", headers={"Authorization": "Bearer k1"}).json()
    assert own["totals"]["requests"] == 1
    everyone = c.get("/qasd/stats", headers={"Authorization": "Bearer adm"}).json()
    assert everyone["totals"]["requests"] == 2


def test_exact_cache_only_for_deterministic_requests(make_client):
    c = make_client()
    assert chat(c, temperature=0).headers["x-qasd-cache"] == "miss"
    hit = chat(c, temperature=0)
    assert hit.headers["x-qasd-cache"] == "hit"
    assert hit.json()["choices"][0]["message"]["content"] == "Hello from the mock."
    assert chat(c, temperature=0.7).headers["x-qasd-cache"].startswith("skip")
    r = c.post("/v1/chat/completions", headers={"x-qasd-cache": "off"},
               json={"model": "gpt-4o-mini", "temperature": 0, "messages": [{"role": "user", "content": "Say hello"}]})
    assert r.headers["x-qasd-cache"].startswith("skip")
    stats = c.get("/qasd/stats").json()
    assert stats["gateway_cache"]["hit"] == 1
    assert stats["totals"]["saved_usd"] > 0


def test_cache_is_per_tenant(make_client):
    c = make_client(api_keys=["a", "b"])
    assert chat(c, key="a", temperature=0).headers["x-qasd-cache"] == "miss"
    assert chat(c, key="b", temperature=0).headers["x-qasd-cache"] == "miss"
    assert chat(c, key="a", temperature=0).headers["x-qasd-cache"] == "hit"


def test_streaming(make_client):
    c = make_client()
    r = chat(c, stream=True)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(ch["choices"][0]["delta"].get("content") or "" for ch in chunks if ch.get("choices"))
    assert text == "Hello from the mock."
    assert not any("usage" in ch for ch in chunks)  # client did not ask for usage
    r2 = chat(c, stream=True, stream_options={"include_usage": True})
    assert any("usage" in json.loads(e) for e in sse_events(r2.text)[:-1])
    assert c.get("/qasd/stats").json()["totals"]["requests"] == 2


def test_streaming_cache_replay(make_client):
    c = make_client()
    chat(c, temperature=0, stream=True)
    r = chat(c, temperature=0, stream=True)
    assert r.headers["x-qasd-cache"] == "hit"
    events = sse_events(r.text)
    assert events[-1] == "[DONE]"
    assert json.loads(events[0])["choices"][0]["delta"]["content"] == "Hello from the mock."


def test_auto_routing_and_explicit_model_untouched(make_client):
    c = make_client()
    r = chat(c, model="auto", messages=[{"role": "user", "content": "Capital of France?"}])
    assert r.headers["x-qasd-model"] == "claude-haiku-4-5"
    assert r.headers["x-qasd-route"].startswith("cheap")
    r = chat(c, model="auto", messages=[{"role": "user", "content": "Do a threat model for our API"}])
    assert r.headers["x-qasd-route"].startswith("frontier")
    r = chat(c, model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert r.headers["x-qasd-model"] == "gpt-4o" and "x-qasd-route" not in r.headers
    assert chat(c, model="auto:nope").status_code == 400


def test_compaction_through_gateway(make_client):
    c = make_client(compact_trigger_tokens=300, compact_chunk=4)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "TASK: write a report"}]
    for i in range(20):
        msgs += [{"role": "assistant", "content": f"working on part {i} " + "detail " * 20},
                 {"role": "user", "content": f"continue {i}"}]
    r = chat(c, messages=msgs)
    assert r.status_code == 200
    assert int(r.headers["x-qasd-compacted"]) > 0
    t = c.get("/qasd/stats").json()["totals"]
    assert t["prompt_tokens_sent"] < t["prompt_tokens_original"]
    assert c.get("/qasd/stats").json()["compacted_requests"] == 1


def test_anthropic_model_gets_breakpoints(make_client, monkeypatch):
    seen = {}
    real = litellm.acompletion

    async def spy(**kwargs):
        seen.update(kwargs)
        return await real(**kwargs)

    monkeypatch.setattr(litellm, "acompletion", spy)
    c = make_client()
    r = chat(c, model="claude-haiku-4-5",
             messages=[{"role": "system", "content": "You are terse."}, {"role": "user", "content": "hi"}])
    assert r.status_code == 200
    assert seen["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_upstream_error_is_openai_shaped(make_client, monkeypatch):
    async def boom(**kwargs):
        raise litellm.RateLimitError("slow down", llm_provider="openai", model="gpt-4o-mini")

    monkeypatch.setattr(litellm, "acompletion", boom)
    c = make_client()
    r = chat(c)
    assert r.status_code == 429
    assert "slow down" in r.json()["error"]["message"]
    assert c.get("/qasd/stats").json()["totals"]["requests"] == 1


def test_bad_requests(make_client):
    c = make_client()
    assert c.post("/v1/chat/completions", json={"messages": []}).status_code == 400
    assert c.post("/v1/chat/completions", content=b"nope").status_code == 400
    assert c.get("/v1/models").json()["data"][0]["id"] == "auto"
    assert "Qasd savings" in c.get("/dashboard").text

import pytest

from qasd import cache as rcache
from qasd.compaction import SUMMARY_HEADER, compact, plan
from qasd.prefix import add_anthropic_breakpoints, prefix_hash, stabilize_tools
from qasd.router import DEFAULT_CONFIG, Router
from qasd.store import MemoryStore


def tool(name, props):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": props}}}


# ---------- prefix ----------

def test_tools_sorted_and_schema_keys_sorted():
    a = [tool("zeta", {"b": {"type": "string"}, "a": {"type": "string"}}), tool("alpha", {})]
    b = [tool("alpha", {}), tool("zeta", {"a": {"type": "string"}, "b": {"type": "string"}})]
    sa, sb = stabilize_tools(a), stabilize_tools(b)
    assert [t["function"]["name"] for t in sa] == ["alpha", "zeta"]
    assert sa == sb
    assert list(sa[1]["function"]["parameters"]["properties"]) == ["a", "b"]
    msgs = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}]
    assert prefix_hash(msgs, sa) == prefix_hash(msgs, sb)


def test_prefix_hash_ignores_conversation_but_not_system():
    t = stabilize_tools([tool("x", {})])
    base = [{"role": "system", "content": "S"}, {"role": "user", "content": "one"}]
    later = base + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "two"}]
    assert prefix_hash(base, t) == prefix_hash(later, t)
    assert prefix_hash([{"role": "system", "content": "S2"}], t) != prefix_hash(base, t)


def test_anthropic_breakpoints_added_once():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]
    tools = [tool("x", {})]
    m2, t2, n = add_anthropic_breakpoints(msgs, tools)
    assert n == 3
    assert t2[-1]["cache_control"] == {"type": "ephemeral"}
    assert m2[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert m2[-1]["content"][0]["text"] == "q"
    assert msgs[0]["content"] == "S"  # original untouched
    # client already manages caching -> leave it alone
    _, _, n2 = add_anthropic_breakpoints(m2, t2)
    assert n2 == 0


# ---------- compaction ----------

def convo(turns):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "TASK"}]
    for i in range(turns):
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "read", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"})
        msgs.append({"role": "assistant", "content": f"step {i}"})
        msgs.append({"role": "user", "content": f"next {i}"})
    return msgs


def test_no_compaction_under_trigger():
    msgs = convo(3)
    assert plan(msgs, [10] * len(msgs), trigger=10_000, target=5_000, chunk=4) is None


def test_cut_never_splits_tool_pair_and_keeps_task():
    msgs = convo(20)
    counts = [100] * len(msgs)
    p = plan(msgs, counts, trigger=2_000, target=1_500, chunk=5)
    assert p is not None
    assert p.body_start == 2  # system + first user kept
    first_kept = msgs[p.body_start + p.cut]
    assert first_kept["role"] != "tool"


def test_cut_is_stable_across_turns():
    """Adding a turn must not move the cut, so the provider cache keeps hitting."""
    msgs = convo(10)
    cuts = []
    for i in range(40):  # 40 more user/assistant turns
        msgs = msgs + [{"role": "assistant", "content": f"a{i}"}, {"role": "user", "content": f"u{i}"}]
        p = plan(msgs, [100] * len(msgs), trigger=2_000, target=1_800 + 1_000, chunk=16)
        cuts.append(p.cut)
    assert cuts == sorted(cuts)            # only ever moves forward
    changes = sum(1 for a, b in zip(cuts, cuts[1:]) if a != b)
    assert changes <= 40 // 8              # a jump at most every 8 turns (16 messages)


async def test_compact_merges_summary_and_caches_it():
    calls = []

    async def summarizer(removed):
        calls.append(len(removed))
        return "- did things"

    store = MemoryStore()
    msgs = convo(30)
    kwargs = dict(store=store, trigger=500, target=400, chunk=8, summarizer=summarizer, summary_model="m")
    out, info = await compact(msgs, "gpt-4o-mini", **kwargs)
    assert info["removed_messages"] > 0 and info["summary_source"] == "llm"
    assert out[0]["role"] == "system" and out[1]["role"] == "user"
    assert out[1]["content"].startswith("TASK") and SUMMARY_HEADER in out[1]["content"]
    assert len(out) < len(msgs)
    assert out[2]["role"] != "tool"
    out2, info2 = await compact(msgs, "gpt-4o-mini", **kwargs)
    assert out2 == out and info2["summary_source"] == "cache" and len(calls) == 1


async def test_compact_falls_back_when_summarizer_fails():
    async def broken(_):
        raise RuntimeError("down")

    out, info = await compact(convo(30), "gpt-4o-mini", store=MemoryStore(), trigger=500, target=400,
                              chunk=8, summarizer=broken)
    assert info["summary_source"] == "deterministic"
    assert "earlier messages" in out[1]["content"]


# ---------- router ----------

@pytest.fixture
def router():
    return Router.load(None)


def test_router_defaults_to_cheap(router):
    d = router.route("auto", [{"role": "user", "content": "What is the capital of France?"}], None, 20)
    assert d.tier == "cheap" and d.model == DEFAULT_CONFIG["tiers"]["cheap"]


def test_router_keywords_use_word_boundaries(router):
    assert router.route("auto", [{"role": "user", "content": "Please prove this lemma"}], None, 10).tier == "frontier"
    assert router.route("auto", [{"role": "user", "content": "How can I improve my CV?"}], None, 10).tier == "cheap"


def test_router_tools_code_and_pinning(router):
    assert router.route("auto", [{"role": "user", "content": "hi"}], [tool("x", {})], 10).tier == "mid"
    assert router.route("auto", [{"role": "user", "content": "```py\nprint(1)\n```"}], None, 10).tier == "mid"
    assert router.route("auto:frontier", [{"role": "user", "content": "hi"}], None, 10).tier == "frontier"
    with pytest.raises(ValueError):
        router.route("auto:huge", [{"role": "user", "content": "hi"}], None, 10)


def test_router_only_for_auto():
    assert Router.is_auto("auto") and Router.is_auto("auto:mid")
    assert not Router.is_auto("gpt-4o") and not Router.is_auto("claude-sonnet-4-6")


# ---------- exact cache ----------

def test_cache_eligibility():
    assert rcache.eligible({"temperature": 0}, False)[0]
    assert not rcache.eligible({}, False)[0]
    assert not rcache.eligible({"temperature": 0.7}, False)[0]
    assert not rcache.eligible({"temperature": 0, "tools": [1]}, False)[0]
    assert not rcache.eligible({"temperature": 0, "n": 2}, False)[0]
    assert not rcache.eligible({"temperature": 0}, True)[0]


def test_cache_key_scoped_by_tenant_and_params():
    m = [{"role": "user", "content": "hi"}]
    k = rcache.make_key("t1", "gpt-4o", m, {"temperature": 0})
    assert k == rcache.make_key("t1", "gpt-4o", m, {"temperature": 0})
    assert k != rcache.make_key("t2", "gpt-4o", m, {"temperature": 0})
    assert k != rcache.make_key("t1", "gpt-4o", m, {"temperature": 0, "max_tokens": 5})
    assert k != rcache.make_key("t1", "gpt-4o", m, {"temperature": 0, "response_format": {"type": "json_object"}})

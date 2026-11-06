"""Failover behaviour of the multi-provider client, with HTTP faked (no network, no quota)."""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from promo import llm as L  # noqa: E402
from promo import pipeline  # noqa: E402

OK = lambda text: ({"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                   None, None, None)


@pytest.fixture(autouse=True)
def fresh_state(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "STATE", L._State())
    monkeypatch.setattr(L, "load_dotenv", lambda *a, **k: None)  # never read the developer's real .env
    for k in ("GEMINI_API_KEY", "LLM_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("MISTRAL_API_KEY", "m")
    monkeypatch.setenv("GROQ_API_KEY", "q")
    monkeypatch.chdir(tmp_path)  # isolate .env / cache
    yield


def make(chain, script, tmp_path, **kw):
    """script: dict entry -> list of responses (callables or tuples) consumed in order."""
    calls = []
    client = L.LLM("propose", chain=chain, cache_dir=str(tmp_path / "c"), discover=False, **kw)

    def fake(entry, system, user, temperature):
        calls.append(entry)
        seq = script[entry]
        r = seq.pop(0) if len(seq) > 1 else seq[0]
        return r(user) if callable(r) else r
    client._attempt = fake
    return client, calls


def test_daily_quota_fails_over_instantly_and_is_remembered(tmp_path):
    chain = ["gemini:a", "mistral:b"]
    c, calls = make(chain, {"gemini:a": [(None, "daily", "HTTP 429 PerDay", 50000)], "mistral:b": [OK('{"x": 1}')]}, tmp_path)
    t = time.time()
    out, model = c.chat_json_m("s", "u1")
    assert out == {"x": 1} and model == "mistral:b" and calls == ["gemini:a", "mistral:b"]
    assert time.time() - t < 2  # no sleeping on a daily quota
    saved = json.load(open(tmp_path / "c" / "model_state.json"))
    assert "gemini:a" in saved  # persisted: next run skips it immediately
    c2, calls2 = make(chain, {"gemini:a": [OK('{"y": 1}')], "mistral:b": [OK('{"y": 2}')]}, tmp_path)
    L.STATE.path = None
    L.STATE.load(str(tmp_path / "c" / "model_state.json"))
    assert c2.chat_json_m("s", "u2")[1] == "mistral:b" and calls2 == ["mistral:b"]


def test_rate_limit_moves_on_without_waiting(tmp_path):
    c, calls = make(["gemini:a", "groq:b"], {"gemini:a": [(None, "rate", "HTTP 429", 30)], "groq:b": [OK('{"ok": 1}')]}, tmp_path)
    t = time.time()
    assert c.chat_json_m("s", "u")[1] == "groq:b"
    assert time.time() - t < 2


def test_overload_then_recovery_order(tmp_path):
    c, calls = make(["gemini:a", "mistral:b"], {"gemini:a": [(None, "overload", "503", None)], "mistral:b": [OK("{}")]}, tmp_path)
    c.chat_json_m("s", "u")
    assert calls == ["gemini:a", "mistral:b"]
    assert L.STATE.cool["gemini:a"][0] > time.time()  # short cool-down, not persisted
    assert c.dead and "gemini:a" in c.dead


def test_auth_failure_disables_provider_only(tmp_path):
    c, calls = make(["groq:a", "groq:b", "mistral:c"],
                    {"groq:a": [(None, "auth", "401", None)], "groq:b": [OK("{}")], "mistral:c": [OK('{"m": 1}')]}, tmp_path)
    assert c.chat_json_m("s", "u")[1] == "mistral:c"
    assert calls == ["groq:a", "mistral:c"]  # groq:b skipped: same rejected key


def test_too_large_learned_then_no_model_fits(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_INPUT", "100")  # tiny tier
    monkeypatch.setenv("GROQ_MAX_INPUT", "100")
    c, _ = make(["gemini:a", "groq:b"], {"gemini:a": [OK("{}")], "groq:b": [OK("{}")]}, tmp_path)
    with pytest.raises(L.NoModelFits):
        c.chat_json_m("s", "x" * 2000)
    # a provider that reports "too large" is remembered for that size
    c2, calls = make(["mistral:a", "groq:b"], {"mistral:a": [(None, "too_large", "413", None)], "groq:b": [OK('{"g": 1}')]}, tmp_path)
    monkeypatch.setenv("GROQ_MAX_INPUT", "100000")
    assert c2.chat_json_m("s", "y" * 400)[1] == "groq:b"
    assert L.STATE.max_fit["mistral:a"] < 200


def test_invalid_json_retries_then_falls_over(tmp_path):
    c, calls = make(["gemini:a", "mistral:b"], {"gemini:a": [OK("not json")], "mistral:b": [OK('{"fine": true}')]}, tmp_path)
    assert c.chat_json_m("s", "u") == ({"fine": True}, "mistral:b")
    assert calls == ["gemini:a", "mistral:b"]  # a ready model beats waiting for gemini's rate slot


def test_invalid_json_single_model_retries_with_reminder(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_RPM", "6000")
    seen = []
    c, calls = make(["gemini:a"], {"gemini:a": [OK("nope"), OK('{"ok": 1}')]}, tmp_path)
    orig = c._attempt
    c._attempt = lambda e, s, u, t: (seen.append(u), orig(e, s, u, t))[1]
    assert c.chat_json_m("s", "u")[0] == {"ok": 1}
    assert "not valid JSON" in seen[1]


def test_cache_hit_avoids_calls(tmp_path):
    c, calls = make(["gemini:a"], {"gemini:a": [OK('{"v": 1}')]}, tmp_path)
    c.chat_json_m("s", "same")
    c.chat_json_m("s", "same")
    assert calls == ["gemini:a"] and c.stats["cache_hits"] == 1


def test_run_batches_splits_when_nothing_fits(tmp_path, monkeypatch):
    """Pipeline-level: a batch too big for every model is halved until it fits."""
    monkeypatch.setenv("GEMINI_MAX_INPUT", "400")
    c, calls = make(["gemini:a"], {"gemini:a": [lambda user: OK(json.dumps({"n": user.count("ITEM")}))[:4] if True else None]}, tmp_path)
    c._attempt = lambda e, s, u, t: OK(json.dumps({"n": u.count("ITEM")}))
    build = lambda batch: ("sys", " ".join(["ITEM " + "z" * 300] * len(batch)), 0.0)
    results, errors = pipeline.run_batches([list(range(8))], build, c, "test", lambda m: None, lambda b: "b")
    parts = results[0]
    assert not errors and len(parts) > 1 and sum(r["n"] for _, r, _ in parts) == 8


def test_provider_without_key_is_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    c = L.LLM("judge", chain=["groq:x", "gemini:y"], cache_dir=str(tmp_path / "c"), discover=False)
    assert c.entries == ["gemini:y"] and c.skipped["groq:x"] == "no API key"

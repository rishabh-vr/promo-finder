"""Bring-your-own-key behaviour used by the web app: the host's keys are never used, keys passed by a
visitor are used only by that visitor's LLM objects, and per-session state is isolated."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from promo.llm import LLM, _State  # noqa: E402

CHAIN = ["gemini:gemini-x", "groq:openai/gpt-oss-120b", "openrouter:some/model:free"]


def test_host_env_keys_are_ignored_in_byok_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "HOST-SECRET-gemini")
    monkeypatch.setenv("GROQ_API_KEY", "HOST-SECRET-groq")
    llm = LLM("propose", chain=CHAIN, keys={}, cache_dir=str(tmp_path), discover=False,
              use_env_keys=False, state=_State())
    assert llm.entries == []                                   # nothing usable without the visitor's key
    assert llm._key("gemini") is None and llm._key("groq") is None


def test_visitor_key_is_the_only_key_used(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "HOST-SECRET-gemini")
    llm = LLM("propose", chain=CHAIN, keys={"groq": "visitor-key"}, cache_dir=str(tmp_path), discover=False,
              use_env_keys=False, state=_State())
    assert llm.entries == ["groq:openai/gpt-oss-120b"]
    assert llm._key("groq") == "visitor-key" and llm._key("gemini") is None


def test_custom_endpoint_env_ignored_in_byok_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BASE_URL", "https://host.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "HOST-SECRET-custom")
    llm = LLM("propose", chain=CHAIN, keys={}, cache_dir=str(tmp_path), discover=False,
              use_env_keys=False, state=_State())
    assert llm._custom is None and "custom" not in llm.keys


def test_cli_mode_still_reads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "my-own-key")
    llm = LLM("propose", chain=CHAIN, cache_dir=str(tmp_path), discover=False, state=_State())
    assert "groq:openai/gpt-oss-120b" in llm.entries and llm._key("groq") == "my-own-key"


def test_sessions_do_not_share_bad_key_or_cooldown_state(tmp_path):
    a = LLM("propose", chain=CHAIN, keys={"groq": "bad"}, cache_dir=str(tmp_path / "a"), discover=False,
            use_env_keys=False, state=_State())
    b = LLM("propose", chain=CHAIN, keys={"groq": "good"}, cache_dir=str(tmp_path / "b"), discover=False,
            use_env_keys=False, state=_State())
    a.state.auth_bad.add("groq")
    a.state.cool_down("groq:openai/gpt-oss-120b", 3600, "daily quota")
    assert "groq" not in b.state.auth_bad
    assert "groq:openai/gpt-oss-120b" not in b.state.cool
    assert b.model == "groq:openai/gpt-oss-120b"


def test_keys_never_written_to_cache_or_state_files(tmp_path):
    llm = LLM("propose", chain=CHAIN, keys={"groq": "visitor-key-123456"}, cache_dir=str(tmp_path),
              discover=False, use_env_keys=False, state=_State())
    llm.state.path = str(tmp_path / "model_state.json")
    llm.state.cool_down("groq:openai/gpt-oss-120b", 7200, "daily quota")
    for root, _, files in os.walk(tmp_path):
        for f in files:
            assert "visitor-key-123456" not in open(os.path.join(root, f), encoding="utf-8", errors="ignore").read()

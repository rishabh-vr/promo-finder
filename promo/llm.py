"""Provider-agnostic LLM client.

Talks to any OpenAI-compatible /chat/completions endpoint, so swapping models is env vars
only (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY). Built for free tiers:
  * a fallback chain of models: when one model's daily quota is spent (or it is overloaded)
    the client moves to the next model instead of sleeping for hours;
  * client-side rate limit (RPM) and backoff on transient errors;
  * an on-disk cache so re-runs and interrupted runs never pay twice for the same prompt.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time

import requests

GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"
PRESETS = {
    # preset: (base_url, default model chain for generation, default model chain for judging)
    # The judge chain is ordered differently on purpose: a different model grades the clips than
    # the one that proposed them, which reduces a model's bias toward its own picks.
    "gemini": (GEMINI,
               ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3-flash-preview", "gemini-3.5-flash"],
               ["gemini-3-flash-preview", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.1-flash-lite"]),
    "groq": ("https://api.groq.com/openai/v1", ["llama-3.3-70b-versatile"], ["llama-3.3-70b-versatile"]),
    "openrouter": ("https://openrouter.ai/api/v1", ["meta-llama/llama-3.3-70b-instruct:free"], ["meta-llama/llama-3.3-70b-instruct:free"]),
    "anthropic": ("https://api.anthropic.com/v1", ["claude-sonnet-5-5"], ["claude-haiku-5-5"]),
    "openai": ("https://api.openai.com/v1", ["gpt-4o-mini"], ["gpt-4o-mini"]),
}


def load_dotenv(path: str | None = None) -> None:
    for p in [path] if path else [".env", os.path.join(os.path.dirname(__file__), "..", ".env")]:
        if p and os.path.exists(p):
            for ln in open(p, encoding="utf-8"):
                ln = ln.strip()
                if ln and not ln.startswith("#") and "=" in ln:
                    k, v = ln.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


class LLMError(RuntimeError):
    pass


def _split(s: str | None) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


class LLM:
    """role = 'propose' or 'judge'. Model chain resolution (first hit wins):
    explicit `models` arg > LLM_<ROLE>_MODELS > LLM_MODEL (+ LLM_FALLBACK_MODELS) > preset chain."""

    def __init__(self, role: str = "propose", models: list[str] | None = None, base_url: str | None = None,
                 api_key: str | None = None, rpm: float | None = None, cache_dir: str = ".cache",
                 max_tokens: int = 16000, timeout: int = 150):
        load_dotenv()
        preset = os.environ.get("LLM_PRESET", "gemini")
        pb, gen_chain, judge_chain = PRESETS.get(preset, PRESETS["gemini"])
        self.role = role
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or pb).rstrip("/")
        env_role = _split(os.environ.get(f"LLM_{role.upper()}_MODELS"))
        env_one = _split(os.environ.get("LLM_MODEL")) + _split(os.environ.get("LLM_FALLBACK_MODELS"))
        if role == "audit" and not env_role:  # auditor = strongest generator-class model by default
            env_role = _split(os.environ.get("LLM_PROPOSE_MODELS"))
        self.models = models or env_role or env_one or (judge_chain if role == "judge" else gen_chain)
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.rpm = float(rpm or os.environ.get("LLM_RPM", 8))
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.extra = json.loads(os.environ.get("LLM_EXTRA", "{}"))
        self.dead: dict[str, str] = {}  # model -> reason it was dropped this run
        self.stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "retries": 0, "by_model": {}, "failovers": []}
        self._lock = threading.Lock()
        self._last = 0.0

    @property
    def model(self) -> str:
        live = [m for m in self.models if m not in self.dead]
        return live[0] if live else self.models[-1]

    # -- plumbing ---------------------------------------------------------
    def _key(self, system, user, temperature):
        # keyed on the role's chain head, not the model that answered, so a run that failed over
        # half-way can resume from cache tomorrow without re-paying.
        h = hashlib.sha256(json.dumps([self.models[0], system, user, temperature], ensure_ascii=False).encode())
        return os.path.join(self.cache_dir, h.hexdigest()[:32] + ".json")

    def _throttle(self):
        with self._lock:
            wait = self._last + 60.0 / self.rpm - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

    def _drop(self, model: str, reason: str):
        self.dead[model] = reason
        self.stats["failovers"].append({"model": model, "reason": reason[:160]})

    def _post_one(self, model, system, user, temperature):
        """Returns (json, None) on success or (None, 'fatal'|'next'|'retry', err)."""
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            **self.extra,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        self._throttle()
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
        except requests.Timeout as e:
            return None, "timeout", f"timeout after {self.timeout}s: {e}"
        except requests.RequestException as e:
            return None, "retry", f"network error: {e}"
        if r.status_code == 200:
            return r.json(), None, None
        txt = r.text[:600]
        if r.status_code == 429:
            # daily quota (or very long retry delay) -> this model is done for today
            daily = "PerDay" in txt or re.search(r'"retryDelay":\s*"(\d{3,})s"', txt)
            return None, ("next" if daily else "retry"), f"HTTP 429: {txt[:200]}"
        if r.status_code in (404, 400) and ("no longer available" in txt or "not found" in txt.lower()):
            return None, "next", f"HTTP {r.status_code}: model unavailable"
        if r.status_code in (500, 502, 503, 504, 408):
            return None, "retry", f"HTTP {r.status_code}: {txt[:200]}"
        return None, "fatal", f"HTTP {r.status_code}: {txt}"

    def _post(self, system, user, temperature):
        if not self.api_key:
            raise LLMError("No LLM_API_KEY set. See README (a free Gemini key takes a minute).")
        last = "no models configured"
        while True:
            live = [m for m in self.models if m not in self.dead]
            if not live:
                raise LLMError(f"all models exhausted for role '{self.role}': {self.dead}; last error: {last}")
            model = live[0]
            delay = 4.0
            for attempt in range(4):
                data, kind, err = self._post_one(model, system, user, temperature)
                if data is not None:
                    return data, model
                last = err
                if kind == "fatal":
                    raise LLMError(f"{model}: {err}")
                if kind == "next" or (kind == "timeout" and attempt >= 1):
                    break  # quota spent, model gone, or hanging under load -> next model
                self.stats["retries"] += 1
                time.sleep(delay)
                delay = min(delay * 2, 30)
            # persistent overload or quota -> fail over to the next model in the chain
            self._drop(model, last)

    @staticmethod
    def parse_json(text: str):
        text = (text or "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"[\[{].*[\]}]", text, re.S)
            if m:
                return json.loads(m.group(0))
            raise

    # -- public -----------------------------------------------------------
    def chat_json(self, system: str, user: str, temperature: float = 0.2):
        path = self._key(system, user, temperature)
        if os.path.exists(path):
            self.stats["cache_hits"] += 1
            blob = json.load(open(path, encoding="utf-8"))
            return blob["out"] if isinstance(blob, dict) and "out" in blob and "_model" in blob else blob
        last_err = None
        for attempt in range(2):
            u = user if attempt == 0 else user + "\n\nYour previous reply was not valid JSON. Return ONLY valid JSON."
            data, model = self._post(system, u, temperature)
            self.stats["calls"] += 1
            self.stats["by_model"][model] = self.stats["by_model"].get(model, 0) + 1
            usage = data.get("usage") or {}
            self.stats["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
            self.stats["completion_tokens"] += usage.get("completion_tokens", 0) or 0
            try:
                out = self.parse_json(data["choices"][0]["message"]["content"])
            except Exception as e:
                last_err = e
                continue
            json.dump({"_model": model, "out": out}, open(path, "w", encoding="utf-8"), ensure_ascii=False)
            return out
        raise LLMError(f"model did not return valid JSON twice: {last_err}")


def items_of(res, key: str) -> list:
    """Models sometimes return the bare list instead of {key: [...]}, or a single object. Accept all."""
    if isinstance(res, list):
        return res
    if isinstance(res, dict):
        v = res.get(key)
        if isinstance(v, list):
            return v
        lists = [x for x in res.values() if isinstance(x, list)]
        if len(lists) == 1:
            return lists[0]
    return []

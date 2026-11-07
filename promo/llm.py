"""Multi-provider LLM client with automatic failover.

Every role (propose / judge / audit) has an ordered CHAIN of "provider:model" entries spread over
several free providers. A request goes to the best entry that is usable right now; the moment an
entry fails (daily quota, per-minute limit, overload, timeout, model gone, prompt too big) the
client moves on to the next one instead of sleeping, and remembers why:

  * daily-quota exhaustion is persisted to disk until the quota resets, so the next run (or the other
    roles in this run) skips that model instantly;
  * per-minute limits and overloads put the entry on a short cool-down and traffic flows to the next
    entry meanwhile;
  * "prompt too large for this provider" is learned per entry; if NO entry can fit a prompt the client
    raises NoModelFits so the caller can split the work into smaller requests;
  * models missing from a provider's catalogue are dropped up front (catalogue cached 12h).

Rate limits are tracked per model and thread-safe, so the pipeline can run requests in parallel.
Every response is cached on disk: a re-run never pays twice for the same prompt.

All providers speak the OpenAI /chat/completions dialect, so adding one is one entry in PROVIDERS.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass

import requests


# ------------------------------------------------------------------ providers
@dataclass
class Provider:
    name: str
    base_url: str
    key_envs: tuple[str, ...]  # env vars checked in order for the key
    rpm: float                  # client-side requests/minute per model (free-tier safe)
    max_input: int              # approx prompt tokens per request this tier accepts
    timeout: int                # seconds
    needs_key: bool = True
    models_url: str | None = None
    signup: str = ""


GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"
PROVIDERS: dict[str, Provider] = {p.name: p for p in [
    Provider("gemini", GEMINI, ("GEMINI_API_KEY", "LLM_API_KEY"), 8, 900_000, 150,
             signup="https://aistudio.google.com/apikey"),
    Provider("mistral", "https://api.mistral.ai/v1", ("MISTRAL_API_KEY",), 30, 120_000, 150,
             signup="https://console.mistral.ai/api-keys"),
    Provider("openrouter", "https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",), 15, 60_000, 150,
             signup="https://openrouter.ai/settings/keys"),
    Provider("groq", "https://api.groq.com/openai/v1", ("GROQ_API_KEY",), 20, 7_000, 60,
             signup="https://console.groq.com/keys"),
    Provider("cerebras", "https://api.cerebras.ai/v1", ("CEREBRAS_API_KEY",), 25, 8_000, 60,
             signup="https://cloud.cerebras.ai"),
    Provider("github", "https://models.github.ai/inference", ("GITHUB_MODELS_TOKEN", "GITHUB_TOKEN"), 10, 7_500, 90,
             models_url="https://models.github.ai/catalog/models",
             signup="https://github.com/settings/personal-access-tokens"),
    Provider("huggingface", "https://router.huggingface.co/v1", ("HF_TOKEN",), 10, 30_000, 120,
             signup="https://huggingface.co/settings/tokens"),
    Provider("ollama", "http://localhost:11434/v1", (), 600, 30_000, 600, needs_key=False,
             signup="https://ollama.com/download"),
    Provider("openai", "https://api.openai.com/v1", ("OPENAI_API_KEY",), 60, 120_000, 120),
    Provider("anthropic", "https://api.anthropic.com/v1", ("ANTHROPIC_API_KEY",), 50, 180_000, 150),
]}

# Default chains: strongest/most reliable first; providers without a key are skipped automatically.
# Judge and audit lead with different providers than the generator, for independent opinions.
# Verified live on 2026-11-06 (python doctor.py --ping). "openrouter:auto-free" expands at start-up to
# whatever large-context free models OpenRouter lists that day, so the chain survives model churn.
DEFAULT_CHAINS = {
    "propose": [
        "gemini:gemini-3-flash-preview", "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free",
        "gemini:gemini-3.6-flash", "gemini:gemini-3.5-flash", "openrouter:nvidia/nemotron-3.5-super-vl:free",
        "openrouter:dots-studio/dots-3-note-preview:free", "gemini:gemini-3.8-flash", "gemini:gemini-3.7-flash",
        "openrouter:google/gemma-4-31b-it:free", "groq:openai/gpt-oss-120b", "groq:qwen/qwen3.8-27b",
        "openrouter:auto-free", "mistral:mistral-large-latest", "cerebras:gpt-oss-120b",
        "huggingface:meta-llama/Llama-3.3-70B-Instruct", "ollama:qwen2.5:14b",
    ],
    "judge": [
        "gemini:gemini-3.1-flash-lite", "groq:openai/gpt-oss-120b", "groq:qwen/qwen3.8-27b",
        "gemini:gemini-flash-lite-latest", "openrouter:nvidia/nemotron-3.5-super-vl:free",
        "gemini:gemini-3.5-flash-lite", "groq:openai/gpt-oss-20b", "openrouter:nvidia/nemotron-3.5-lightning:free",
        "openrouter:auto-free", "mistral:mistral-small-latest", "cerebras:gpt-oss-120b",
        "huggingface:meta-llama/Llama-3.3-70B-Instruct", "ollama:qwen2.5:14b",
    ],
    # auditor leads with a different model family than the generator, for an independent second opinion
    "audit": [
        "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free", "gemini:gemini-3-flash-preview", "gemini:gemini-3.6-flash",
        "openrouter:nvidia/nemotron-3.5-super-vl:free", "gemini:gemini-3.5-flash",
        "openrouter:dots-studio/dots-3-note-preview:free", "groq:openai/gpt-oss-120b", "openrouter:auto-free",
        "mistral:mistral-large-latest", "cerebras:gpt-oss-120b", "ollama:qwen2.5:14b",
    ],
}
DEFAULT_CHAINS["rank"] = DEFAULT_CHAINS["judge"]


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


class NoModelFits(LLMError):
    """No available model accepts a prompt this large: the caller should split the work."""


def _split(s: str | None) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def est_tokens(*texts: str) -> int:
    return int(sum(len(t) for t in texts) / 3.6) + 50


def provider_cfg(name: str) -> Provider:
    p = PROVIDERS[name]
    up = name.upper()
    return Provider(p.name, os.environ.get(f"{up}_BASE_URL", p.base_url), p.key_envs,
                    float(os.environ.get(f"{up}_RPM", p.rpm)), int(os.environ.get(f"{up}_MAX_INPUT", p.max_input)),
                    int(os.environ.get(f"{up}_TIMEOUT", p.timeout)), p.needs_key, p.models_url, p.signup)


def provider_key(name: str, keys: dict | None = None) -> str | None:
    if keys and keys.get(name):
        return keys[name]
    p = PROVIDERS[name]
    if not p.needs_key:
        return "ollama" if os.environ.get("OLLAMA_ENABLED") else None
    for env in p.key_envs:
        if os.environ.get(env):
            return os.environ[env]
    return None


def parse_entry(s: str, default_provider: str = "gemini") -> tuple[str, str]:
    head, _, rest = s.partition(":")
    if rest and head in PROVIDERS:
        return head, rest
    return default_provider, s


# ------------------------------------------------------------------ shared state (all roles, all threads)
class _State:
    """Cool-downs, learned size limits and per-model throttles, shared across LLM instances."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cool: dict[str, tuple[float, str]] = {}   # entry -> (until_epoch, reason)
        self.max_fit: dict[str, int] = {}               # entry -> largest prompt (tokens) known to fit... upper bound
        self.next_slot: dict[str, float] = {}           # entry -> earliest next request time
        self.auth_bad: set[str] = set()                 # providers whose key was rejected
        self.no_json: set[str] = set()                  # providers that reject response_format
        self.path: str | None = None

    def load(self, path: str):
        with self.lock:
            if self.path == path:
                return
            self.path = path
            try:
                data = json.load(open(path, encoding="utf-8"))
                now = time.time()
                for k, (until, why) in data.items():
                    if until > now:
                        self.cool[k] = (until, why)
            except (OSError, ValueError):
                pass

    def save(self):
        if not self.path:
            return
        now = time.time()
        keep = {k: v for k, v in self.cool.items() if v[0] - now > 600}  # only long cool-downs
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(keep, f, indent=1)
        except OSError:
            pass

    def cool_down(self, entry: str, seconds: float, reason: str):
        with self.lock:
            until = time.time() + seconds
            if until > self.cool.get(entry, (0, ""))[0]:
                self.cool[entry] = (until, reason[:200])
            if seconds > 600:
                self.save()


STATE = _State()
_CATALOG: dict[str, set[str] | None] = {}
_CATALOG_LOCK = threading.Lock()


def catalog(provider: str, key: str | None, cache_dir: str) -> set[str] | None:
    """Model ids the provider currently serves (None = unknown; don't filter)."""
    with _CATALOG_LOCK:
        if provider in _CATALOG:
            return _CATALOG[provider]
        path = os.path.join(cache_dir, f"catalog_{provider}.json")
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < 12 * 3600:
            try:
                ids = set(json.load(open(path, encoding="utf-8")))
                _CATALOG[provider] = ids
                return ids
            except (OSError, ValueError):
                pass
        p = provider_cfg(provider)
        url = p.models_url or f"{p.base_url}/models"
        ids = None
        try:
            r = requests.get(url, headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=8)
            if r.ok:
                js = r.json()
                items = js.get("data", js) if isinstance(js, dict) else js
                found = set()
                for it in items if isinstance(items, list) else []:
                    mid = it.get("id") or it.get("name") if isinstance(it, dict) else None
                    if mid:
                        found.add(mid.removeprefix("models/"))
                ids = found or None
        except (requests.RequestException, ValueError):
            ids = None
        if ids:
            try:
                json.dump(sorted(ids), open(path, "w", encoding="utf-8"))
            except OSError:
                pass
        _CATALOG[provider] = ids
        return ids


def openrouter_free(cache_dir: str, min_ctx: int = 64_000, limit: int = 6) -> list[str]:
    """Today's large-context free OpenRouter models (the free line-up changes often)."""
    path = os.path.join(cache_dir, "openrouter_free.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 12 * 3600:
        try:
            return json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            pass
    try:
        data = requests.get("https://openrouter.ai/api/v1/models", timeout=10).json().get("data", [])
    except (requests.RequestException, ValueError):
        return []
    small = re.compile(r"(nano|mini|lite|[^\d][1-9](\.\d)?b\b)", re.I)  # skip tiny models
    free = [m for m in data if str(m.get("id", "")).endswith(":free") and (m.get("context_length") or 0) >= min_ctx
            and not small.search(m["id"])]
    free.sort(key=lambda m: -(m.get("context_length") or 0))
    ids = [m["id"] for m in free[:limit]]
    try:
        json.dump(ids, open(path, "w", encoding="utf-8"))
    except OSError:
        pass
    return ids


# ------------------------------------------------------------------ client
class LLM:
    """LLM(role) builds its chain from env / defaults and the keys you have.

    Chain resolution (first hit wins): `chain` arg > LLM_<ROLE>_CHAIN > legacy LLM_<ROLE>_MODELS /
    `models` arg (bare model ids = gemini, or LLM_BASE_URL if set) > DEFAULT_CHAINS[role].
    """

    def __init__(self, role: str = "propose", chain: list[str] | None = None, keys: dict | None = None,
                 models: list[str] | None = None, base_url: str | None = None, api_key: str | None = None,
                 rpm: float | None = None, cache_dir: str | None = None, max_tokens: int = 16000,
                 discover: bool = True, timeout: int | None = None, use_env_keys: bool = True,
                 state: "_State | None" = None, **_):
        """use_env_keys=False: only keys passed in `keys` are used (never .env / host secrets / env vars).
        state: cool-downs, bad-key flags and throttles; pass a fresh _State() per user so one person's
        bad key or spent quota never blocks another user. Defaults to the process-wide STATE (CLI)."""
        self.use_env_keys = use_env_keys
        self.state = state if state is not None else STATE
        if use_env_keys:
            load_dotenv()
        self.role = role
        self.keys = dict(keys or {})
        self.max_tokens = max_tokens
        cache_dir = cache_dir or os.environ.get("LLM_CACHE_DIR", ".cache")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.state.load(os.path.join(cache_dir, "model_state.json"))
        self.extra = json.loads(os.environ.get("LLM_EXTRA", "{}"))
        self.patience = float(os.environ.get("LLM_PATIENCE", 4))      # max throttle wait before trying the next model
        self.max_wait = float(os.environ.get("LLM_MAX_WAIT", 120))    # max wait when EVERY model is cooling down
        self._custom = None
        custom_url = base_url or (os.environ.get("LLM_BASE_URL") if use_env_keys else None)
        if custom_url:  # a single explicit OpenAI-compatible endpoint
            self._custom = Provider("custom", custom_url.rstrip("/"), (), float(rpm or os.environ.get("LLM_RPM", 30)),
                                    int(os.environ.get("LLM_MAX_INPUT", 120_000)), timeout or 150)
            self.keys["custom"] = api_key or (os.environ.get("LLM_API_KEY", "") if use_env_keys else "")
        elif api_key:
            self.keys.setdefault("gemini", api_key)
        self.rpm_override = rpm
        self.timeout_override = timeout

        raw = (chain or _split(os.environ.get(f"LLM_{role.upper()}_CHAIN"))
               or models or _split(os.environ.get(f"LLM_{role.upper()}_MODELS"))
               or (_split(os.environ.get("LLM_PROPOSE_CHAIN")) if role == "audit" else [])
               or DEFAULT_CHAINS.get(role, DEFAULT_CHAINS["propose"]))
        default_prov = "custom" if self._custom else "gemini"
        entries, self.skipped = [], {}
        for s in raw:
            prov, model = parse_entry(s, default_prov)
            label = f"{prov}:{model}"
            if label in entries:
                continue
            if label == "openrouter:auto-free":
                if self._key("openrouter"):
                    for mid in openrouter_free(cache_dir):
                        if f"openrouter:{mid}" not in entries:
                            entries.append(f"openrouter:{mid}")
                continue
            if prov != "custom" and not self._key(prov):
                self.skipped[label] = "no API key"
                continue
            entries.append(label)
        if discover:
            for label in list(entries):
                prov, model = label.split(":", 1)
                if prov in ("custom", "ollama"):
                    continue
                ids = catalog(prov, self._key(prov), cache_dir)
                if ids is not None and model not in ids:
                    entries.remove(label)
                    self.skipped[label] = "not in provider catalogue"
        self.entries = entries
        self.cache_tag = os.environ.get("LLM_CACHE_TAG") or (entries[0].split(":", 1)[1] if entries else role)
        self.stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0,
                      "by_model": {}, "failovers": [], "splits": 0}
        self._slock = threading.Lock()

    # -- introspection -----------------------------------------------------
    @property
    def models(self) -> list[str]:
        return list(self.entries)

    @property
    def dead(self) -> dict:
        now = time.time()
        return {e: self.state.cool[e][1] for e in self.entries if e in self.state.cool and self.state.cool[e][0] > now}

    @property
    def model(self) -> str:
        now = time.time()
        for e in self.entries:
            if self.state.cool.get(e, (0, ""))[0] <= now and e.split(":")[0] not in self.state.auth_bad:
                return e
        return self.entries[0] if self.entries else "none"

    def _key(self, prov: str) -> str | None:
        if prov == "custom":
            return self.keys.get("custom", "")
        if not self.use_env_keys:
            p = PROVIDERS.get(prov)
            if p is not None and not p.needs_key:
                return provider_key(prov)        # local Ollama: no secret involved
            return self.keys.get(prov) or None
        return provider_key(prov, self.keys)

    def _prov(self, prov: str) -> Provider:
        if prov == "custom":
            return self._custom
        p = provider_cfg(prov)
        if self.rpm_override:
            p.rpm = float(self.rpm_override)
        if self.timeout_override:
            p.timeout = int(self.timeout_override)
        return p

    def _bump(self, k, n=1):
        with self._slock:
            self.stats[k] += n

    # -- selection -----------------------------------------------------------
    def _fits(self, entry: str, need: int) -> bool:
        prov = entry.split(":", 1)[0]
        return need <= min(self._prov(prov).max_input, self.state.max_fit.get(entry, 10 ** 9))

    def _pick(self, need: int) -> tuple[str, float]:
        """Choose the entry to call now. Returns (entry, seconds to wait for its rate slot)."""
        while True:
            now = time.time()
            fitting = [e for e in self.entries if e.split(":")[0] not in self.state.auth_bad and self._fits(e, need)]
            if not fitting:
                if any(e.split(":")[0] not in self.state.auth_bad for e in self.entries):
                    raise NoModelFits(f"no available model accepts a ~{need}-token prompt for role '{self.role}'")
                raise LLMError(f"no usable model for role '{self.role}' (keys rejected or none configured)")
            ready = [e for e in fitting if self.state.cool.get(e, (0, ""))[0] <= now]
            if not ready:
                soonest = min(self.state.cool[e][0] for e in fitting)
                if soonest - now > self.max_wait:
                    reasons = {e: self.state.cool[e][1][:80] for e in fitting}
                    raise LLMError(f"all models for role '{self.role}' are exhausted or cooling down: {reasons}")
                time.sleep(max(0.5, soonest - now))
                continue
            with self.state.lock:
                waits = [(e, max(0.0, self.state.next_slot.get(e, 0) - now)) for e in ready]
                choice = next(((e, w) for e, w in waits if w <= self.patience), None) or min(waits, key=lambda x: x[1])
                e, w = choice
                gap = 60.0 / max(self._prov(e.split(":")[0]).rpm, 0.1)
                self.state.next_slot[e] = max(now, self.state.next_slot.get(e, 0)) + gap
            return e, w

    # -- one HTTP attempt ------------------------------------------------------
    def _attempt(self, entry: str, system: str, user: str, temperature: float):
        prov, model = entry.split(":", 1)
        p = self._prov(prov)
        body = {"model": model, "temperature": temperature, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **self.extra}
        if prov not in self.state.no_json:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        key = self._key(prov)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if prov == "openrouter":
            headers.update({"HTTP-Referer": "https://github.com/promo-finder", "X-Title": "Promo Finder"})
        try:
            r = requests.post(f"{p.base_url}/chat/completions", json=body, headers=headers, timeout=p.timeout)
        except requests.Timeout:
            return None, "timeout", f"timeout after {p.timeout}s", None
        except requests.RequestException as e:
            return None, "network", f"network error: {e}", None
        txt = r.text[:800]
        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError:  # e.g. a retired endpoint answering plain "OK"
                return None, "gone", f"endpoint returned non-JSON: {txt[:60]!r}", None
            try:
                content = data["choices"][0]["message"]["content"]
                if not content:
                    raise ValueError("empty content")
                return data, None, None, None
            except (ValueError, KeyError, IndexError, TypeError) as e:
                return None, "bad_output", f"unusable response: {e}", None
        low = txt.lower()
        if r.status_code == 400 and ("failed to validate json" in low or "json_validate_failed" in low):
            return None, "no_json", "provider's JSON mode rejected the output; retrying without it", None
        ra = r.headers.get("retry-after")
        m = re.search(r'"retryDelay":\s*"([\d.]+)s"', txt) or re.search(r"try again in ([\d.]+)s", low)
        retry_s = float(ra) if ra and ra.replace(".", "").isdigit() else (float(m.group(1)) if m else None)
        if r.status_code in (401, 403):
            return None, "auth", f"HTTP {r.status_code}: key rejected", None
        if r.status_code == 404 or "no longer available" in low or "does not exist" in low or "model_not_found" in low:
            return None, "gone", f"HTTP {r.status_code}: model unavailable", None
        if r.status_code == 413 or any(s in low for s in ("context length", "context_length", "maximum context",
                                                             "too large", "reduce the length", "too many tokens")):
            return None, "too_large", f"HTTP {r.status_code}: prompt too large", None
        if r.status_code == 400 and "response_format" in low:
            return None, "no_json", "provider rejects response_format", None
        if r.status_code == 429:
            tpm = re.search(r"limit (\d+), requested (\d+)", low)
            if tpm and int(tpm.group(2)) > int(tpm.group(1)):
                return None, "too_large", f"request exceeds per-minute token limit ({tpm.group(2)} > {tpm.group(1)})", None
            daily = any(s in txt for s in ("PerDay", "per day", "per-day", "daily", "TPD", "RPD", "free-models-per-day"))
            if daily or (retry_s and retry_s > 600):
                return None, "daily", f"HTTP 429 daily quota: {txt[:160]}", retry_s
            return None, "rate", f"HTTP 429 rate limit: {txt[:160]}", retry_s
        if r.status_code == 402:
            return None, "daily", "HTTP 402: out of free credits", None
        if r.status_code in (408, 500, 502, 503, 504, 529):
            return None, "overload", f"HTTP {r.status_code}: {txt[:160]}", retry_s
        return None, "bad_request", f"HTTP {r.status_code}: {txt[:300]}", None

    # -- public ----------------------------------------------------------------
    def chat_json_m(self, system: str, user: str, temperature: float = 0.2):
        """Returns (parsed_json, 'provider:model' that answered)."""
        key = hashlib.sha256(json.dumps([self.cache_tag, system, user, temperature], ensure_ascii=False).encode())
        path = os.path.join(self.cache_dir, key.hexdigest()[:32] + ".json")
        if os.path.exists(path):
            self._bump("cache_hits")
            blob = json.load(open(path, encoding="utf-8"))
            if isinstance(blob, dict) and "_model" in blob and "out" in blob:
                return blob["out"], blob["_model"]
            return blob, "cache"
        if not self.entries:
            raise LLMError(f"no models configured for role '{self.role}'. Skipped: {self.skipped}. "
                           f"Add at least one API key to .env (see .env.example).")
        need = est_tokens(system, user) + 0
        attempts, json_fail, last = 0, {}, ""
        while attempts < 3 * len(self.entries) + 6:
            attempts += 1
            entry, wait = self._pick(need)
            if wait:
                time.sleep(wait)
            prov = entry.split(":", 1)[0]
            u = user if not json_fail.get(entry) else user + "\n\nYour previous reply was not valid JSON. Return ONLY valid JSON."
            data, kind, err, retry_s = self._attempt(entry, system, u, temperature)
            if data is not None:
                usage = data.get("usage") or {}
                with self._slock:
                    self.stats["calls"] += 1
                    self.stats["by_model"][entry] = self.stats["by_model"].get(entry, 0) + 1
                    self.stats["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
                    self.stats["completion_tokens"] += usage.get("completion_tokens", 0) or 0
                try:
                    out = self.parse_json(data["choices"][0]["message"]["content"])
                except (ValueError, TypeError) as e:
                    json_fail[entry] = json_fail.get(entry, 0) + 1
                    last = f"{entry}: invalid JSON ({e})"
                    if json_fail[entry] >= 2:
                        self.state.cool_down(entry, 600, "returned invalid JSON twice")
                    continue
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"_model": entry, "out": out}, f, ensure_ascii=False)
                return out, entry
            last = f"{entry}: {err}"
            self._bump("retries")
            with self._slock:
                self.stats["failovers"].append({"model": entry, "kind": kind, "reason": (err or "")[:160]})
            if kind == "auth":
                self.state.auth_bad.add(prov)
            elif kind == "gone":
                self.state.cool_down(entry, 24 * 3600, err)
            elif kind == "too_large":
                with self.state.lock:
                    self.state.max_fit[entry] = min(self.state.max_fit.get(entry, 10 ** 9), int(need * 0.9))
            elif kind == "no_json":
                self.state.no_json.add(prov)
            elif kind == "daily":
                # trust the provider's own retry hint when it gives one (Gemini does); else assume 6h
                self.state.cool_down(entry, max(retry_s, 60) if retry_s else 6 * 3600, err)
            elif kind == "rate":
                self.state.cool_down(entry, min(max(retry_s or 20, 5), 120), err)
            elif kind == "overload":
                self.state.cool_down(entry, 90, err)
            elif kind == "timeout":
                self.state.cool_down(entry, 300, err)
            elif kind in ("network", "bad_output"):
                self.state.cool_down(entry, 30, err)
            else:  # bad_request: this entry can't serve this request shape; try others
                self.state.cool_down(entry, 3600, err)
        raise LLMError(f"gave up after {attempts} attempts for role '{self.role}'; last error: {last}")

    def chat_json(self, system: str, user: str, temperature: float = 0.2):
        return self.chat_json_m(system, user, temperature)[0]

    @staticmethod
    def parse_json(text: str):
        text = (text or "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
        text = re.sub(r"^<think>.*?</think>\s*", "", text, flags=re.S)  # reasoning models
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"[\[{].*[\]}]", text, re.S)
            if m:
                return json.loads(m.group(0))
            raise


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


# Back-compat for older imports
PRESETS = {name: (p.base_url, [m for m in DEFAULT_CHAINS["propose"] if m.startswith(name + ":")],
                  [m for m in DEFAULT_CHAINS["judge"] if m.startswith(name + ":")]) for name, p in PROVIDERS.items()}

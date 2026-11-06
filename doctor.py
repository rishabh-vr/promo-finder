"""Check which providers and models are usable right now.

  python doctor.py            # keys, catalogues, resolved chains per role (free: no generation calls)
  python doctor.py --ping     # also send a tiny request to every model in the chains (uses quota)
  python doctor.py --reset    # forget remembered quota exhaustion / cool-downs
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from promo.llm import LLM, PROVIDERS, STATE, catalog, load_dotenv, provider_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--cache", default=".cache")
    a = ap.parse_args()
    load_dotenv()
    os.makedirs(a.cache, exist_ok=True)
    if a.reset:
        path = os.path.join(a.cache, "model_state.json")
        if os.path.exists(path):
            os.remove(path)
        for f in os.listdir(a.cache):
            if f.startswith("catalog_"):
                os.remove(os.path.join(a.cache, f))
        print("cleared remembered cool-downs and model catalogues\n")

    print("PROVIDERS")
    for name, p in PROVIDERS.items():
        key = provider_key(name)
        if not key:
            print(f"  ✗ {name:<12} no key   {p.signup}")
            continue
        ids = catalog(name, key, a.cache) if name != "ollama" else None
        info = f"{len(ids)} models listed" if ids else "catalogue unavailable (models tried directly)"
        print(f"  ✓ {name:<12} key set  {info}")

    print("\nCHAINS (in failover order)")
    llms = {}
    for role in ("propose", "judge", "audit"):
        llm = LLM(role, cache_dir=a.cache)
        llms[role] = llm
        print(f"  {role}:")
        now = time.time()
        for e in llm.entries:
            cool = STATE.cool.get(e)
            status = f"cooling {int((cool[0] - now) / 60)} min: " + " ".join(cool[1].split())[:60] if cool and cool[0] > now else "ready"
            print(f"    • {e:<55} {status}")
        if not llm.entries:
            print("    (none usable: add a key to .env)")
        for e, why in llm.skipped.items():
            if why != "no API key":
                print(f"    – {e:<55} skipped: {why}")

    if a.ping:
        print("\nPING (one tiny request per model)")
        entries = sorted({e for l in llms.values() for e in l.entries})

        def ping(e):
            l = LLM("ping", chain=[e], cache_dir=a.cache, discover=False, max_tokens=50)
            t = time.time()
            data, kind, err, _ = l._attempt(e, "Reply with JSON only.", 'Return {"ok": true}', 0.0)
            return e, data is not None, time.time() - t, err or ""

        with ThreadPoolExecutor(max_workers=6) as ex:
            for e, ok, dt, err in ex.map(ping, entries):
                print(f"  {chr(10003) if ok else chr(10007)} {e:<55} {dt:5.1f}s " + " ".join(err.split())[:80])


if __name__ == "__main__":
    main()

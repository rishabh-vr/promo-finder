"""Run the pipeline on many shows and score each against its gold labels.

  python tools/benchmark.py --shows ../test_data --out bench_out --mode baseline     # free, no key
  python tools/benchmark.py --shows ../test_data --out bench_out --mode llm --episodes 1-10
  python tools/benchmark.py --score-only --shows ../test_data --out bench_out         # re-score saved runs

A "show" is any folder under --shows holding episode .txt files (gold.json optional). For each show it
writes <out>/<show>/ (the pipeline's normal outputs + gold_eval.json) and a combined
<out>/benchmark.json + benchmark.md with recall/precision@K, hook/cut accuracy, decoy rate and lift
over random clips. Shows without gold still get run-health numbers (clips, qualify rate, rejections).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from promo import goldeval, pipeline  # noqa: E402
from promo.llm import LLM  # noqa: E402
from promo.parse import load_show  # noqa: E402


def find_shows(root: str) -> list[str]:
    out = []
    for d, _, files in os.walk(root):
        if "_raw" in d.replace("\\", "/").split("/"):
            continue
        if any(f.lower().endswith(".txt") and f.lower().startswith("episode") for f in files):
            out.append(d)
    return sorted(out)


def parse_eps(s):
    if not s:
        return None
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["llm", "baseline"], default="baseline")
    ap.add_argument("--episodes", help="subset per show, e.g. 1-10 (saves quota on long shows)")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--no-audit", action="store_true")
    a = ap.parse_args()
    llm = judge = auditor = None
    rows_out = []
    for show in find_shows(a.shows):
        name = os.path.relpath(show, a.shows).replace(os.sep, "__")
        out = os.path.join(a.out, name)
        eps = load_show(show)
        subset = parse_eps(a.episodes)
        t0 = time.time()
        if not a.score_only:
            print(f"\n=== {name}: {len(eps)} episodes ===")
            if a.mode == "llm":
                # fresh clients per show, so run_stats.json counts only this show's calls; cool-downs
                # (shared process state) and the on-disk response cache still carry over between shows
                llm, judge = LLM(role="propose"), LLM(role="judge")
                auditor = None if a.no_audit else LLM(role="audit")
            try:
                pipeline.run(show, out, a.mode, llm, judge, subset, eps=eps, show_name=name, audit_llm=auditor)
            except Exception as e:  # keep the benchmark going; record the failure
                rows_out.append({"show": name, "error": str(e)[:300]})
                continue
        pj = os.path.join(out, "promos.json")
        if not os.path.exists(pj):
            continue
        rows = json.load(open(pj, encoding="utf-8"))
        st = json.load(open(os.path.join(out, "run_stats.json"), encoding="utf-8"))
        rec = {"show": name, "episodes": len(subset or eps), "clips": len(rows),
               "rejected": st.get("n_rejected"), "qualify_rate": st["summary"].get("qualify_rate"),
               "hook_late": st["summary"].get("hook_late"), "seconds": round(time.time() - t0, 1)}
        gp = os.path.join(show, "gold.json")
        if os.path.exists(gp):
            gold = goldeval.load(gp)
            if subset:
                gold = {**gold, "moments": [m for m in gold["moments"] if m["episode"] in subset],
                        "decoys": [d for d in gold.get("decoys", []) if d["episode"] in subset]}
            durations = {e.ep: e.duration for e in eps if not subset or e.ep in subset}
            ev = goldeval.evaluate(rows, gold, durations)
            json.dump(ev, open(os.path.join(out, "gold_eval.json"), "w", encoding="utf-8"), indent=1)
            rec["gold"] = ev["at"]
            rec["gold_source"] = gold.get("source")
        rows_out.append(rec)
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows_out, open(os.path.join(a.out, "benchmark.json"), "w", encoding="utf-8"), indent=1)
    md = ["# Benchmark", "", f"mode: {a.mode}", "",
          "| show | eps | clips | qualify | gold | recall@10 | prec@10 (random) | p | hook ok | cut ok | decoy@10 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows_out:
        if "error" in r:
            md.append(f"| {r['show']} | error: {r['error'][:60]} |")
            continue
        g = (r.get("gold") or {}).get("top10") or (r.get("gold") or {}).get("all") or {}
        f = lambda v: "–" if v is None else v
        md.append(f"| {r['show']} | {r['episodes']} | {r['clips']} | {f(r['qualify_rate'])} | "
                  f"{(r.get('gold_source') or '–')[:18]} | {f(g.get('recall'))} | {f(g.get('precision'))} "
                  f"({f(g.get('random_precision'))}) | {f(g.get('p_precision_vs_random'))} | {f(g.get('hook_ok'))} | "
                  f"{f(g.get('cut_ok'))} | {f(g.get('decoy_rate'))} |")
    open(os.path.join(a.out, "benchmark.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

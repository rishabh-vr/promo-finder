"""CLI.

  python run.py --data <folder of episode .txt files> --out out/              # full pipeline
  python run.py --data ... --mode baseline                                    # rules only, no key
  python run.py --data ... --episodes 1-10                                    # subset
  python run.py --data ... --compare-baseline 50                              # quality experiment
  python run.py --data ... --models gemini-3.8-flash --judge-models gemini-3-flash-preview
"""
import argparse
import json
import os
import sys

from promo import pipeline
from promo.llm import LLM, PRESETS


def parse_eps(s):
    if not s:
        return None
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description="Find promo clips in microdrama screenplays.")
    ap.add_argument("--data", required=True, help="folder containing the episode .txt screenplays")
    ap.add_argument("--out", default="out", help="output folder")
    ap.add_argument("--mode", choices=["llm", "baseline"], default="llm",
                    help="llm = full pipeline; baseline = rule-based candidates, no API key needed")
    ap.add_argument("--episodes", help="subset, e.g. 1-10 or 1,5,9 (default: all)")
    ap.add_argument("--preset", choices=list(PRESETS), help="provider preset (default from .env, else gemini)")
    ap.add_argument("--models", help="comma-separated generator model chain (first = preferred)")
    ap.add_argument("--judge-models", help="comma-separated judge model chain (ideally different models)")
    ap.add_argument("--chunk", type=int, help="episodes per proposal call (default 5)")
    ap.add_argument("--rpm", type=float, help="client-side requests per minute")
    ap.add_argument("--audit-models", help="comma-separated show-aware auditor chain (default: generator chain)")
    ap.add_argument("--no-audit", action="store_true", help="skip the show-aware audit stage")
    ap.add_argument("--compare-baseline", type=int, metavar="N", default=0,
                    help="also judge N rule-based baseline clips with the same judge (quality experiment)")
    a = ap.parse_args()
    if a.preset:
        os.environ["LLM_PRESET"] = a.preset
    if a.chunk:
        pipeline.CHUNK = a.chunk
    split = lambda s: [x.strip() for x in s.split(",")] if s else None

    llm = judge = auditor = None
    if a.mode == "llm" or a.compare_baseline:
        judge = LLM(role="judge", models=split(a.judge_models), rpm=a.rpm)
    if a.mode == "llm":
        llm = LLM(role="propose", models=split(a.models), rpm=a.rpm)
        auditor = None if a.no_audit else LLM(role="audit", models=split(a.audit_models), rpm=a.rpm)
        print(f"generator chain: {llm.models}\njudge chain:     {judge.models}\n"
              f"audit chain:     {auditor.models if auditor else 'off'}")
        rows = pipeline.run(a.data, a.out, "llm", llm, judge, parse_eps(a.episodes), audit_llm=auditor)
        print(f"\nDone: {len(rows)} ranked clips -> {a.out}/report.md, promos.json, promos.csv")
    elif a.mode == "baseline" and not a.compare_baseline:
        rows = pipeline.run(a.data, a.out, "baseline", None, None, parse_eps(a.episodes))
        print(f"\nDone: {len(rows)} rule-based clips -> {a.out}")

    if a.compare_baseline:
        bdir = os.path.join(a.out, "baseline_judged")
        rows = pipeline.compare_baseline(a.data, bdir, judge, n=a.compare_baseline)
        print(f"\nBaseline experiment: {len(rows)} clips judged -> {bdir}")
        print(json.dumps(pipeline.summarize(rows), indent=1))
    for name, l in (("generator", llm), ("judge", judge), ("audit", auditor)):
        if l:
            print(f"{name} usage:", json.dumps(l.stats))


if __name__ == "__main__":
    sys.exit(main())

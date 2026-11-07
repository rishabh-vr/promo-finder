"""Recompute every number the solution doc / README quotes, straight from the raw outputs.

  python tools/verify_claims.py --raw ../submission/raw_outputs [--data "<show folder>"]

Prints PASS/FAIL per claim, plus the statistics a careful reader would ask for (significance of the
pairwise result, whether the pairwise judge is also the generator). With --data it also checks every
timestamp against the source screenplays: cuts on block boundaries, no line cut in half, 20-60s.
Exit code 1 if any claim fails, so it can run in CI before a doc is shipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from promo import stats  # noqa: E402

# claim text -> expected value (as quoted in the doc). Edit alongside the doc.
CLAIMS = {
    "valid LLM clips": 44,
    "candidate proposals": 46,
    "rejected proposals": 2,
    "validation findings": 3,
    "episodes with an LLM clip": 43,
    "episodes with no clip": [8, 14, 17, 25, 27, 29, 49],
    "stitched clips": 3,
    "cross-episode clips": 0,
    "blind-judge qualifying (LLM)": 31,
    "blind-judge qualifying (baseline, of 50)": 30,
    "audited qualifying": 29,
    "tiers acquisition/re-engagement/weak": [19, 10, 15],
    "mean blind hook/cliff/standalone LLM": [3.39, 4.05, 3.55],
    "mean blind hook/cliff/standalone baseline": [3.02, 3.80, 3.50],
    "pairwise llm/baseline/tie": [23, 17, 3],
    "pairwise win rate (decided)": 0.57,
    "baseline clips ending at episode end": 26,
    "judge-vs-audit cliffhanger within1 / spearman": [0.91, 0.72],
    "judge-vs-audit hook within1 / spearman": [0.89, 0.51],
    "judge-vs-audit spoiler within1 / spearman": [0.59, 0.21],
    "auditor spoiler by story third": [1.4, 2.7, 4.1],
    "rules-only clips": 129,
    "#1 clip": "E3 00:58.2-01:25.5 + E3 02:12.7-02:20.5",
    "#2 clip": "E11 01:01.8-01:52.8",
    "#3 clip": "E23 01:18.5-02:11.5",
}


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def compute(raw: str, generator: str | None = None) -> tuple[dict, dict]:
    P = load(os.path.join(raw, "promos.json"))
    B = load(os.path.join(raw, "baseline_judged", "promos.json"))
    BR = load(os.path.join(raw, "baseline_rules_only", "promos.json"))
    REJ = load(os.path.join(raw, "rejected.json"))
    PW = load(os.path.join(raw, "eval_pairwise.json"))
    RS = load(os.path.join(raw, "run_stats.json"))
    q = lambda j: bool(j) and float(j["hook"]) >= 3 and float(j["cliffhanger"]) >= 3
    mean = lambda rows, k: round(sum(float(r["judge"][k]) for r in rows) / len(rows), 2)
    covered = {s["episode"] for r in P for s in r["segments"]}
    both = [r for r in P if r.get("audit")]

    def agree(k):
        a = [float(r["judge"][k]) for r in both]
        b = [float(r["audit"][k]) for r in both]
        return [round(sum(abs(x - y) <= 1 for x, y in zip(a, b)) / len(a), 2), stats.spearman(a, b)]

    thirds = []
    for lo, hi in ((1, 17), (18, 34), (35, 50)):
        xs = [float(r["audit"]["spoiler_risk"]) for r in both if lo <= r["episode"] <= hi]
        thirds.append(round(sum(xs) / len(xs), 1))
    where = lambda r: " + ".join(f"E{s['episode']} {s['start']}-{s['end']}" for s in r["segments"])
    tiers = Counter(r["tier"] for r in P)
    t = PW["tally"]
    got = {
        "valid LLM clips": len(P),
        "candidate proposals": RS["n_raw_candidates"],
        "rejected proposals": len(REJ),
        "validation findings": sum(len(r["problems"]) for r in REJ),
        "episodes with an LLM clip": len({r["episode"] for r in P}),
        "episodes with no clip": sorted(set(range(1, 51)) - covered),
        "stitched clips": sum(len(r["segments"]) > 1 for r in P),
        "cross-episode clips": sum(len({s["episode"] for s in r["segments"]}) > 1 for r in P),
        "blind-judge qualifying (LLM)": sum(q(r["judge"]) for r in P),
        "blind-judge qualifying (baseline, of 50)": sum(q(r["judge"]) for r in B),
        "audited qualifying": sum(bool(r["qualifies"]) for r in P),
        "tiers acquisition/re-engagement/weak": [tiers["acquisition"], tiers["re-engagement (spoils the plot)"], tiers["weak"]],
        "mean blind hook/cliff/standalone LLM": [mean(P, k) for k in ("hook", "cliffhanger", "standalone")],
        "mean blind hook/cliff/standalone baseline": [mean(B, k) for k in ("hook", "cliffhanger", "standalone")],
        "pairwise llm/baseline/tie": [t["llm"], t["baseline"], t["tie"]],
        "pairwise win rate (decided)": round(t["llm"] / (t["llm"] + t["baseline"]), 2),
        "baseline clips ending at episode end": sum(bool(r["checks"].get("ends_at_episode_end")) for r in B),
        "judge-vs-audit cliffhanger within1 / spearman": agree("cliffhanger"),
        "judge-vs-audit hook within1 / spearman": agree("hook"),
        "judge-vs-audit spoiler within1 / spearman": agree("spoiler_risk"),
        "auditor spoiler by story third": thirds,
        "rules-only clips": len(BR),
        "#1 clip": where(P[0]), "#2 clip": where(P[1]), "#3 clip": where(P[2]),
    }
    gen = sorted({r.get("gen_model") for r in P if r.get("gen_model")})
    gen_note = "recorded in promos.json"
    if not gen and generator:
        gen, gen_note = [generator], "NOT recorded in promos.json; taken from --generator (QUALITY_REVIEW.md)"
    judges = sorted({r.get("judge_model") for r in P if r.get("judge_model")})
    auditors = sorted({r.get("audit_model") for r in P if r.get("audit_model")})
    w, l = t["llm"], t["baseline"]
    extra = {
        "pairwise sign-test p (two-sided)": stats.sign_test(w, l),
        "pairwise win-rate 95% Wilson CI": stats.wilson(w, w + l),
        "decided pairs needed for p<0.05 at this win rate": stats.min_pairs_for_significance(w / (w + l)),
        "pairwise judge model": PW.get("model"),
        "generator model(s)": f"{gen} ({gen_note})", "blind judge model(s)": judges, "auditor model(s)": auditors,
        "pairwise judge == generator": PW.get("model") in gen,
        "auditor == generator": bool(set(auditors) & set(gen)),
        "pairwise position counts A/B": [PW.get("a_picked"), PW.get("b_picked")],
        "LLM clip placed as A/B": list(Counter(p["llm_is"] for p in PW["pairs"]).values()),
        "run served from cache (propose/judge/audit calls)": {k: v["stats"]["calls"] for k, v in RS["llm"].items()},
        "clips whose first beat is a quiet action (salient beat >3s, heuristic)": None,
    }
    return got, extra


def check_timestamps(raw: str, data: str) -> dict:
    from promo.clips import is_salient
    from promo.parse import load_show
    E = {e.ep: e for e in load_show(data)}
    P = load(os.path.join(raw, "promos.json"))
    bad, late = [], []
    for r in P:
        for s in r["segments"]:
            ep = E[s["episode"]]
            for b in ep.blocks:
                for cut in (s["start_s"], s["end_s"]):
                    if b.start < cut - 0.05 and b.end > cut + 0.05:
                        bad.append((r["rank"], s["episode"], cut, b.id))
        if not 19.95 <= r["duration_s"] <= 60.05:
            bad.append((r["rank"], "duration", r["duration_s"]))
        s0 = r["segments"][0]
        ep = E[s0["episode"]]
        sal = [b.start - s0["start_s"] for b in ep.blocks
               if b.start >= s0["start_s"] - 0.05 and b.end <= s0["end_s"] + 0.05 and is_salient(b)]
        if not sal or min(sal) > 3.0:
            late.append((r["rank"], r["episode"], round(min(sal), 1) if sal else None))
    return {"cut_or_duration_violations": bad, "salient_beat_after_3s": late}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--data")
    ap.add_argument("--generator", help="generator model, if the run predates per-clip gen_model provenance")
    a = ap.parse_args()
    got, extra = compute(a.raw, a.generator)
    fails = 0
    print(f"{'claim':52} {'doc':>28} {'recomputed':>28}  result")
    for k, want in CLAIMS.items():
        have = got[k]
        ok = have == want or (isinstance(want, list) and isinstance(have, list) and len(want) == len(have)
                              and all(abs(float(x) - float(y)) < 0.051 for x, y in zip(want, have)))
        fails += not ok
        print(f"{k:52} {str(want):>28} {str(have):>28}  {'PASS' if ok else 'FAIL'}")
    if a.data:
        ts = check_timestamps(a.raw, a.data)
        extra["timestamp/cut violations vs source"] = len(ts["cut_or_duration_violations"])
        extra["clips whose first beat is a quiet action (salient beat >3s, heuristic)"] = ts["salient_beat_after_3s"]
        fails += bool(ts["cut_or_duration_violations"])
    print("\nContext a reviewer would ask for:")
    for k, v in extra.items():
        if v is not None:
            print(f"  {k}: {v}")
    print(f"\n{len(CLAIMS) - fails if fails <= len(CLAIMS) else 0}/{len(CLAIMS)} claims reproduce")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

"""Score a run against gold moments (labelled data), with a random-clip null so numbers mean something.

gold.json
  {"show": "...", "source": "synthetic | tripod | editor | ...",
   "moments": [{"id": "g1", "episode": 3, "start_s": 41.0, "end_s": 75.5, "kind": "promo",
                "hook_s": 41.2, "cliff_s": 74.9, "label": "..."}],
   "decoys":  [{"id": "d1", "episode": 3, "start_s": 80.0, "end_s": 110.0, "kind": "resolved_ending"}]}

  moments : where a good promo should come from (planted promos, editor picks, screenplay turning points)
  decoys  : stretches that look dramatic but are NOT good promos (e.g. an ending that resolves). Optional.

Metrics (for top-K of the ranked list, K = 5, 10, 20, all):
  recall@K     fraction of gold moments hit by at least one clip
  precision@K  fraction of clips that hit some gold moment
  decoy@K      fraction of clips that mostly sit on a decoy (lower is better)
  hook_ok      among hits with a gold hook time: clip starts within 3s before the hook (hook lands in 3s)
  cut_ok       among hits with a gold cliff time: clip ends within [-0.5s, +4s] of the cliffhanger
  random_*     the same metrics for random clips with the same durations, averaged over many draws,
               and p = Monte Carlo share of random draws doing at least as well
A hit = same-episode temporal overlap >= 50% of the shorter of (clip, gold moment).
"""
from __future__ import annotations

import json
import random

HIT_OVERLAP = 0.5
KS = (5, 10, 20, None)


def _segs(row: dict) -> list[tuple[int, float, float]]:
    return [(int(s["episode"]), float(s["start_s"]), float(s["end_s"])) for s in row["segments"]]


def _overlap(segs, m) -> float:
    inter = sum(max(0.0, min(e, m["end_s"]) - max(s, m["start_s"])) for ep, s, e in segs if ep == m["episode"])
    clip_dur = sum(e - s for _, s, e in segs)
    shorter = max(1e-9, min(clip_dur, m["end_s"] - m["start_s"]))
    return inter / shorter


def _metrics(clips: list[list[tuple[int, float, float]]], gold: dict) -> dict:
    moments, decoys = gold.get("moments", []), gold.get("decoys", [])
    hit_m = set()
    clip_hits = 0
    decoy_hits = 0
    hook_n = hook_ok = cut_n = cut_ok = 0
    for segs in clips:
        hits = [m for m in moments if _overlap(segs, m) >= HIT_OVERLAP]
        if hits:
            clip_hits += 1
            best = max(hits, key=lambda m: _overlap(segs, m))
            hit_m.update(m["id"] for m in hits)
            ep0, s0, _ = segs[0]
            if best.get("hook_s") is not None and ep0 == best["episode"]:
                hook_n += 1
                hook_ok += (best["hook_s"] - 3.0 <= s0 <= best["hook_s"] + 0.5)
            epl, _, el = segs[-1]
            if best.get("cliff_s") is not None and epl == best["episode"]:
                cut_n += 1
                cut_ok += (best["cliff_s"] - 0.5 <= el <= best["cliff_s"] + 4.0)
        elif any(_overlap(segs, d) >= HIT_OVERLAP for d in decoys):
            decoy_hits += 1
    n = len(clips)
    return {
        "n_clips": n,
        "recall": round(len(hit_m) / len(moments), 3) if moments else None,
        "precision": round(clip_hits / n, 3) if n else None,
        "decoy_rate": round(decoy_hits / n, 3) if n and decoys else None,
        "hook_ok": round(hook_ok / hook_n, 3) if hook_n else None, "hook_n": hook_n,
        "cut_ok": round(cut_ok / cut_n, 3) if cut_n else None, "cut_n": cut_n,
    }


def _random_like(clips, durations: dict[int, float], rng: random.Random):
    """Random single-stretch clips with exactly matched durations.

    Return None when any clip cannot fit in any episode. Silently generating an overlong
    random clip would make the null artificially weak and its precision incomparable.
    """
    out = []
    for segs in clips:
        dur = sum(e - s for _, s, e in segs)
        eligible = [(ep, span) for ep, span in durations.items() if span >= dur and span > 0]
        if not eligible:
            return None
        # uniform over all valid (episode, start) placements: weight = number of valid starts
        eps, weights = zip(*[(ep, max(span - dur, 1e-6)) for ep, span in eligible])
        ep = rng.choices(eps, weights)[0]
        start = rng.uniform(0, durations[ep] - dur)
        out.append([(ep, start, start + dur)])
    return out


def evaluate(rows: list[dict], gold: dict, durations: dict[int, float], n_random: int = 300, seed: int = 0) -> dict:
    ranked = sorted([r for r in rows if r.get("rank") is not None], key=lambda r: r["rank"])
    eps_gold = {m["episode"] for m in gold.get("moments", [])}
    out = {"show": gold.get("show"), "source": gold.get("source"), "n_gold": len(gold.get("moments", [])),
           "n_decoys": len(gold.get("decoys", [])), "gold_episodes": len(eps_gold), "at": {}}
    rng = random.Random(seed)
    for k in KS:
        top = [_segs(r) for r in (ranked[:k] if k else ranked)]
        if not top:
            continue
        obs = _metrics(top, gold)
        random_sets = [_random_like(top, durations, rng) for _ in range(n_random)]
        # If even one observed clip cannot fit in any episode, don't compare a partial slate
        # or manufacture an invalid null clip. Mark the matched-duration null unavailable.
        comparable = all(rows is not None for rows in random_sets)
        rand = [_metrics(rows, gold) for rows in random_sets if rows is not None] if comparable else []
        avg = lambda key: (round(sum(r[key] for r in rand if r[key] is not None) / max(1, sum(r[key] is not None for r in rand)), 3)
                           if any(r[key] is not None for r in rand) else None)
        def p(key):
            valid = [r[key] for r in rand if r[key] is not None]
            if obs[key] is None or not valid:
                return None
            return round((1 + sum(value >= obs[key] for value in valid)) / (1 + len(valid)), 4)
        out["at"][f"top{k}" if k else "all"] = {
            **obs,
            "random_comparable": comparable,
            "random_recall": avg("recall"), "random_precision": avg("precision"),
            "random_decoy_rate": avg("decoy_rate"),
            "precision_lift": (round(obs["precision"] / avg("precision"), 2)
                               if obs["precision"] is not None and avg("precision") else None),
            "p_precision_vs_random": p("precision"), "p_recall_vs_random": p("recall"),
        }
    return out


def sanitize(gold: dict) -> dict:
    """Labels from a writer model can be self-inconsistent (a 'cliff' tagged outside its own promo).
    Drop hook/cliff times that fall outside their moment, drop decoys that overlap a moment, and record
    every fix in gold['label_warnings'] so the cleanup is visible, never silent."""
    warns = list(gold.get("label_warnings", []))
    moments = []
    for m in gold.get("moments", []):
        m = dict(m)
        if m["end_s"] <= m["start_s"]:
            warns.append(f"{m.get('id')}: empty span dropped")
            continue
        for key in ("hook_s", "cliff_s"):
            t = m.get(key)
            if t is not None and not (m["start_s"] - 0.5 <= t <= m["end_s"] + 0.5):
                warns.append(f"{m.get('id')}: {key}={t} outside [{m['start_s']}, {m['end_s']}]; ignored")
                m[key] = None
        moments.append(m)
    decoys = []
    for d in gold.get("decoys", []):
        clash = [m for m in moments if m["episode"] == d["episode"]
                 and min(m["end_s"], d["end_s"]) - max(m["start_s"], d["start_s"]) > 1.0]
        if clash:
            warns.append(f"{d.get('id')}: overlaps gold moment {clash[0].get('id')}; decoy dropped")
        else:
            decoys.append(d)
    return {**gold, "moments": moments, "decoys": decoys, "label_warnings": warns}


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return sanitize(json.load(f))

"""Offline evaluation from saved outputs (no re-run of the pipeline).

pairwise(): for every episode where both the LLM pipeline and the rule-based baseline produced a
clip, a strong model sees the two clips blind (random A/B order) and picks the one a stranger is
more likely to stop for and follow into the show. Absolute 1-5 scores bunch up at the top
(ceiling effect); a forced choice is a much sharper test of "is the AI better than simple rules?".
"""
from __future__ import annotations

import json
import os
import random

from . import prompts
from .clips import build_clip, clip_text
from .llm import LLM, LLMError, items_of
from .parse import Episode

PAIR_BATCH = int(os.environ.get("PROMO_PAIR_BATCH", 22))


def rebuild(row: dict, episodes: dict[int, Episode]):
    specs = []
    for s in row["segments"]:
        a, b = s["blocks"].split("-")
        specs.append({"episode": s["episode"], "start_block": a, "end_block": b})
    return build_clip(episodes, specs)


def pairwise(llm_rows: list[dict], base_rows: list[dict], episodes: dict[int, Episode], llm: LLM,
             seed: int = 11, cb=print) -> dict:
    best_llm, best_base = {}, {}
    for r in sorted(llm_rows, key=lambda r: r["rank"] or 1e9):
        best_llm.setdefault(r["episode"], r)
    for r in sorted(base_rows, key=lambda r: r["rank"] or 1e9):
        best_base.setdefault(r["episode"], r)
    eps = sorted(set(best_llm) & set(best_base))
    rng = random.Random(seed)
    pairs, meta = [], {}
    for e in eps:
        L, B = best_llm[e], best_base[e]
        lt, bt = clip_text(episodes, rebuild(L, episodes), True, True), clip_text(episodes, rebuild(B, episodes), True, True)
        flip = rng.random() < 0.5
        pid = f"P{e:02d}"
        pairs.append((pid, bt if flip else lt, lt if flip else bt))
        meta[pid] = {"episode": e, "llm_is": "B" if flip else "A", "llm_id": L["id"], "base_id": B["id"],
                     "llm_where": " + ".join(f"{s['start']}-{s['end']}" for s in L["segments"]),
                     "base_where": " + ".join(f"{s['start']}-{s['end']}" for s in B["segments"])}
    results = {}
    for i in range(0, len(pairs), PAIR_BATCH):
        batch = pairs[i:i + PAIR_BATCH]
        cb(f"[pairwise] pairs {i + 1}-{i + len(batch)} of {len(pairs)}")
        try:
            res = llm.chat_json(prompts.PAIR_SYSTEM, prompts.pair_user(batch), temperature=0.0)
        except LLMError as e:
            cb(f"[pairwise]   failed: {str(e)[:160]}")
            continue
        for r in items_of(res, "results"):
            if isinstance(r, dict) and r.get("pair") in meta:
                results[r["pair"]] = r
    rows, tally = [], {"llm": 0, "baseline": 0, "tie": 0}
    for pid, m in meta.items():
        r = results.get(pid)
        if not r:
            continue
        w = str(r.get("winner", "")).strip().upper()
        who = "tie" if w not in ("A", "B") else ("llm" if w == m["llm_is"] else "baseline")
        tally[who] += 1
        rows.append({**m, "winner": who, "reason": r.get("reason", "")})
    n = sum(tally.values())
    decided = tally["llm"] + tally["baseline"]
    return {"model": llm.model, "pairs_compared": n, "tally": tally,
            "llm_win_rate_decided": round(tally["llm"] / decided, 2) if decided else None,
            "llm_win_or_tie_rate": round((tally["llm"] + tally["tie"]) / n, 2) if n else None,
            "a_picked": sum(1 for p, r in results.items() if str(r.get("winner", "")).upper() == "A"),
            "b_picked": sum(1 for p, r in results.items() if str(r.get("winner", "")).upper() == "B"),
            "pairs": rows}


def write(out_dir: str, result: dict):
    with open(os.path.join(out_dir, "eval_pairwise.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

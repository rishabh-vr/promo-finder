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

from . import prompts, stats
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
             seed: int = 11, cb=print, both_orders: bool = True) -> dict:
    """Same-episode blind A/B. With both_orders=True every pair is judged twice with A and B swapped;
    a pair counts as a win only when both orders pick the same system ("consistent"), otherwise it
    is a position-dependent tie. This removes the LLM-judge position bias instead of hoping it averages out."""
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
        orders = [("", flip)] + ([("x", not flip)] if both_orders else [])
        for suffix, f in orders:
            pairs.append((pid + suffix, bt if f else lt, lt if f else bt))
            meta[pid + suffix] = {"pair": pid, "llm_is": "B" if f else "A"}
        meta[pid].update({"episode": e, "llm_id": L["id"], "base_id": B["id"],
                          "llm_where": " + ".join(f"{s['start']}-{s['end']}" for s in L["segments"]),
                          "base_where": " + ".join(f"{s['start']}-{s['end']}" for s in B["segments"])})
    rng.shuffle(pairs)  # the two orders of one pair should not sit side by side in a prompt
    results = {}
    for i in range(0, len(pairs), PAIR_BATCH):
        batch = pairs[i:i + PAIR_BATCH]
        cb(f"[pairwise] judgments {i + 1}-{i + len(batch)} of {len(pairs)}")
        try:
            res = llm.chat_json(prompts.PAIR_SYSTEM, prompts.pair_user(batch), temperature=0.0)
        except LLMError as e:
            cb(f"[pairwise]   failed: {str(e)[:160]}")
            continue
        for r in items_of(res, "results"):
            if isinstance(r, dict) and r.get("pair") in meta:
                results[r["pair"]] = r

    def who(pid):
        r = results.get(pid)
        if not r:
            return None
        w = str(r.get("winner", "")).strip().upper()
        return "tie" if w not in ("A", "B") else ("llm" if w == meta[pid]["llm_is"] else "baseline")

    rows, tally = [], {"llm": 0, "baseline": 0, "tie": 0}
    inconsistent = 0
    for pid in sorted(k for k in meta if not k.endswith("x")):
        first = who(pid)
        second = who(pid + "x") if both_orders else first
        if first is None or second is None:
            continue
        if first == second:
            verdict = first
        else:
            verdict = "tie"
            inconsistent += 1
        tally[verdict] += 1
        rows.append({**meta[pid], "winner": verdict, "order1": first, "order2": second if both_orders else None,
                     "reason": results[pid].get("reason", "")})
    n = sum(tally.values())
    decided = tally["llm"] + tally["baseline"]
    raw_letters = [str(r.get("winner", "")).upper() for r in results.values()]
    a_picked, b_picked = raw_letters.count("A"), raw_letters.count("B")
    gen_models = sorted({str(r.get("gen_model")) for r in llm_rows if r.get("gen_model")})
    return {"model": llm.model, "generator_models": gen_models,
            "judge_is_generator": llm.model.split(":", 1)[-1] in {g.split(":", 1)[-1] for g in gen_models},
            "both_orders": both_orders, "pairs_compared": n, "tally": tally,
            "order_inconsistent": inconsistent,
            "llm_win_rate_decided": round(tally["llm"] / decided, 2) if decided else None,
            "llm_win_rate_wilson95": stats.wilson(tally["llm"], decided),
            "sign_test_p": stats.sign_test(tally["llm"], tally["baseline"]),
            "pairs_needed_for_p05_at_this_rate": (stats.min_pairs_for_significance(tally["llm"] / decided)
                                                  if decided and tally["llm"] != tally["baseline"] else None),
            "llm_win_or_tie_rate": round((tally["llm"] + tally["tie"]) / n, 2) if n else None,
            "a_picked": a_picked, "b_picked": b_picked,
            "position_bias_p": stats.sign_test(a_picked, b_picked),
            "pairs": rows}


def write(out_dir: str, result: dict):
    with open(os.path.join(out_dir, "eval_pairwise.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

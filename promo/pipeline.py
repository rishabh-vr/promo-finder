"""End-to-end pipeline: parse -> propose -> validate -> judge -> score -> rank -> export.

What is AI and what is code
  AI   : proposing candidate moments (block ids only), judging each clip as a stranger would
         (a different model, shown no show context), writing the one-line posting rationale.
  Code : parsing, timestamps, the 20-60s rule, snapping cuts so no line is cut in half,
         repairs, de-duplication, scoring maths, the ranking itself, exports.
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from typing import Callable

from . import baseline, prompts
from .clips import (build_clip, clip_text, code_checks, describe, normalize_spec,
                    overlap_ratio)
from .llm import LLM, LLMError, items_of
from .parse import Episode, fmt, load_show, render_transcript

W = {"hook": 0.30, "cliffhanger": 0.30, "standalone": 0.25, "spoiler": 0.15}
CHUNK = int(os.environ.get("PROMO_CHUNK", 5))       # episodes per proposal call
JUDGE_BATCH = int(os.environ.get("PROMO_JUDGE_BATCH", 12))
RANK_TOP = 20
DEDUPE_OVERLAP = 0.4
MAX_PER_EPISODE_TOP10 = 2
MIN_HOOK, MIN_CLIFF = 3, 3


def story_bible(eps: list[Episode]) -> str:
    return "\n".join(f"E{e.ep}: {e.logline}" for e in eps)


def _log(cb, msg):
    if cb is None:
        print(msg, flush=True)
    else:
        cb(msg)


# ---------------------------------------------------------------- stage 1: propose
def propose_llm(eps: list[Episode], all_eps: list[Episode], llm: LLM, cb=None, chunk: int = CHUNK):
    bible = story_bible(all_eps)
    cands, errors = [], []
    chunks = [eps[i:i + chunk] for i in range(0, len(eps), chunk)]
    for k, ch in enumerate(chunks, 1):
        nums = [e.ep for e in ch]
        _log(cb, f"[propose] chunk {k}/{len(chunks)}: episodes {nums[0]}-{nums[-1]}")
        transcripts = "\n\n".join(render_transcript(e, qualify=True) for e in ch)
        try:
            res = llm.chat_json(prompts.PROPOSE_SYSTEM, prompts.propose_user(bible, nums, transcripts))
        except LLMError as e:
            errors.append({"episodes": nums, "stage": "propose", "error": str(e)[:500]})
            _log(cb, f"[propose]   failed: {str(e)[:160]}")
            continue
        for p in items_of(res, "promos"):
            # keep the model's ids as given (e.g. "E5-B012"); build_clip normalises and validates them
            segs = [{"episode": s.get("episode", nums[0]), "start_block": s.get("start_block"),
                     "end_block": s.get("end_block")} for s in (p.get("segments") or []) if isinstance(s, dict)]
            try:
                first_ep = int(normalize_spec(segs[0])["episode"]) if segs else nums[0]
            except (TypeError, ValueError):
                first_ep = nums[0]
            cands.append({
                "source": "llm", "episode": first_ep,
                "segments": segs,
                "raw_segments": p.get("segments"),
                "hook": p.get("hook", ""), "cliffhanger": p.get("cliffhanger", ""), "why": p.get("why", ""),
                "gen_hook_strength": p.get("hook_strength"), "gen_cliff_strength": p.get("cliffhanger_strength"),
                "context_needed": p.get("context_needed", ""),
            })
    if chunks and len(errors) == len(chunks):
        raise LLMError(f"every proposal call failed; first error: {errors[0]['error'][:300]}")
    return cands, errors


# ---------------------------------------------------------------- stage 2: validate
def _repair_long(c: dict, episodes: dict[int, Episode], max_trim: float = 6.0) -> bool:
    """Slightly over 60s: drop leading blocks of the FIRST segment (never the ending, which carries
    the cliffhanger) until it fits, trimming at most max_trim seconds. Marked so QA can see it."""
    specs = [normalize_spec(s) for s in c["segments"]]
    if not specs or specs[0].get("_cross"):
        return False
    first = specs[0]
    try:
        ep = episodes[int(first["episode"])]
    except (KeyError, ValueError, TypeError):
        return False
    a, b = ep.block(first["start_block"]), ep.block(first["end_block"])
    if a is None or b is None:
        return False
    orig_start = c["clip"].segments[0].start
    for idx in range(a.idx + 1, min(b.idx, a.idx + 10) + 1):
        trial = [{**first, "start_block": ep.blocks[idx].id}] + specs[1:]
        clip = build_clip(episodes, trial)
        if not clip.segments or clip.segments[0].start - orig_start > max_trim:
            return False
        if clip.valid:
            c["segments"], c["clip"] = trial, clip
            c["repaired"] = f"trimmed {clip.segments[0].start - orig_start:.1f}s off the start to fit 60s"
            return True
    return False


def validate(cands: list[dict], episodes: dict[int, Episode]):
    ok, rejected = [], []
    for c in cands:
        c["clip"] = build_clip(episodes, c["segments"])
        if not c["clip"].valid and all(p.startswith("too long") for p in c["clip"].problems):
            _repair_long(c, episodes)
        (ok if c["clip"].valid else rejected).append(c)
    # de-dupe near-identical windows; keep the one the generator rated higher
    ok.sort(key=lambda c: -((c.get("gen_hook_strength") or 0) + (c.get("gen_cliff_strength") or 0)))
    kept = []
    for c in ok:
        if all(overlap_ratio(c["clip"], k["clip"]) < DEDUPE_OVERLAP for k in kept):
            kept.append(c)
        else:
            c["clip"].problems.append("duplicate: overlaps a better clip by >40%")
            rejected.append(c)
    kept.sort(key=lambda c: (c["episode"], c["clip"].segments[0].start))
    return kept, rejected


# ---------------------------------------------------------------- stage 3: judge
def judge(cands: list[dict], episodes, llm: LLM, cb=None, label="judge") -> list[dict]:
    errors = []
    todo = list(cands)
    for attempt in range(2):  # second pass picks up any clip the judge skipped
        missing = []
        for i in range(0, len(todo), JUDGE_BATCH):
            batch = todo[i:i + JUDGE_BATCH]
            _log(cb, f"[{label}] clips {i + 1}-{i + len(batch)} of {len(todo)}" + (" (retry pass)" if attempt else ""))
            items = [(c["id"], clip_text(episodes, c["clip"], with_times=True, scrub=True)) for c in batch]
            try:
                res = llm.chat_json(prompts.JUDGE_SYSTEM, prompts.judge_user(items), temperature=0.0)
            except LLMError as e:
                errors.append({"stage": label, "error": str(e)[:500]})
                _log(cb, f"[{label}]   failed: {str(e)[:160]}")
                continue
            by_id = {str(r.get("id")): r for r in items_of(res, "results") if isinstance(r, dict)}
            for c in batch:
                j = by_id.get(c["id"])
                if j and all(k in j for k in ("hook", "cliffhanger", "standalone", "spoiler_risk")):
                    c["judge"] = j
                    c["judge_model"] = llm.model
                else:
                    missing.append(c)
        if not missing:
            break
        todo = missing
    return errors


AUDIT_BATCH = int(os.environ.get("PROMO_AUDIT_BATCH", 15))
DIMS = ("hook", "cliffhanger", "standalone", "spoiler_risk")


def final_scores(c: dict) -> dict | None:
    """Blind judge alone, or (when audited) the mean of blind judge and show-aware auditor for
    hook / cliffhanger / standalone, with spoiler risk taken from the auditor only: a judge with no
    show context cannot know whether a moment is a major twist."""
    j, a = c.get("judge"), c.get("audit")
    try:
        js = {k: float(j[k]) for k in DIMS}
    except (TypeError, KeyError, ValueError):
        return None
    try:
        au = {k: float(a[k]) for k in DIMS}
    except (TypeError, KeyError, ValueError):
        return js
    out = {k: (js[k] + au[k]) / 2 for k in ("hook", "cliffhanger", "standalone")}
    out["spoiler_risk"] = au["spoiler_risk"]
    return out


def composite(c: dict) -> float | None:
    f = final_scores(c)
    if f is None:
        return None
    h, k, s, sp = (f[x] for x in DIMS)
    n = lambda v: (min(5, max(1, v)) - 1) / 4
    return round(100 * (W["hook"] * n(h) + W["cliffhanger"] * n(k) + W["standalone"] * n(s)
                        + W["spoiler"] * (1 - n(sp))), 1)


def qualifies(c: dict) -> bool:
    """The brief's definition of a promo needs BOTH a hook and a cliffhanger."""
    f = final_scores(c)
    return bool(f) and f["hook"] >= MIN_HOOK and f["cliffhanger"] >= MIN_CLIFF


# ---------------------------------------------------------------- stage 3b: audit
def audit(cands: list[dict], episodes, all_eps: list[Episode], llm: LLM, cb=None) -> list[dict]:
    """Second opinion from a stronger, show-aware model (it gets the story bible). Catches what a
    blind judge structurally cannot (spoilers) and damps the cheap judge's leniency."""
    errors, bible = [], story_bible(all_eps)
    todo = list(cands)
    for attempt in range(2):
        missing = []
        for i in range(0, len(todo), AUDIT_BATCH):
            batch = todo[i:i + AUDIT_BATCH]
            _log(cb, f"[audit] clips {i + 1}-{i + len(batch)} of {len(todo)}" + (" (retry pass)" if attempt else ""))
            items = [(c["id"], clip_text(episodes, c["clip"], with_times=True, scrub=True), c["episode"]) for c in batch]
            try:
                res = llm.chat_json(prompts.AUDIT_SYSTEM, prompts.audit_user(bible, items), temperature=0.0)
            except LLMError as e:
                errors.append({"stage": "audit", "error": str(e)[:500]})
                _log(cb, f"[audit]   failed: {str(e)[:160]}")
                continue
            by_id = {str(r.get("id")): r for r in items_of(res, "results") if isinstance(r, dict)}
            for c in batch:
                r = by_id.get(c["id"])
                if r and all(k in r for k in DIMS):
                    c["audit"], c["audit_model"] = r, llm.model
                else:
                    missing.append(c)
        if not missing:
            break
        todo = missing
    return errors


# ---------------------------------------------------------------- stage 4: rank
MAX_SPOILER_ACQUISITION = 3
TIERS = {0: "acquisition", 1: "re-engagement (spoils the plot)", 2: "weak"}


def tier(c: dict) -> int:
    """0 = post to strangers; 1 = a strong promo that gives away a major turn, so it suits
    re-targeting people already watching, not first contact; 2 = misses the hook or cliffhanger bar."""
    if not qualifies(c):
        return 2
    return 1 if final_scores(c)["spoiler_risk"] > MAX_SPOILER_ACQUISITION else 0


def order_clips(cands: list[dict]) -> list[dict]:
    """Deterministic and auditable: tier first (acquisition > re-engagement > weak), then composite
    score, with a per-episode cap in the top 10 so the first posts are not all from one episode.
    Ties -> earlier episode (fewer spoilers for a new viewer)."""
    scored = [c for c in cands if c.get("score") is not None]
    pool = sorted(scored, key=lambda c: (tier(c), -c["score"], c["episode"]))
    out, deferred, per_ep = [], [], {}
    for c in pool:
        if len(out) < 10 and per_ep.get(c["episode"], 0) >= MAX_PER_EPISODE_TOP10:
            deferred.append(c)
            continue
        out.append(c)
        per_ep[c["episode"]] = per_ep.get(c["episode"], 0) + 1
    return out + deferred


def rank(cands: list[dict], llm: LLM | None, cb=None) -> list[dict]:
    errors = []
    ordered = order_clips(cands)
    for i, c in enumerate(ordered):
        c["rank"], c["rank_reason"] = i + 1, ""
    for c in cands:
        if c.get("score") is None:
            c["rank"], c["rank_reason"] = None, ""
    if llm is None or not ordered:
        return errors
    top = ordered[:RANK_TOP]
    _log(cb, f"[rank] writing posting rationale for top {len(top)}")
    items = [{
        "id": c["id"], "position": c["rank"], "where": describe(c["clip"]),
        "hook_score": round(final_scores(c)["hook"], 1), "cliff_score": round(final_scores(c)["cliffhanger"], 1),
        "standalone": round(final_scores(c)["standalone"], 1),
        "spoiler_risk": round(final_scores(c)["spoiler_risk"], 1),
        "content_flag": c["judge"].get("content_flag", "?"), "hook": c["hook"], "cliffhanger": c["cliffhanger"],
        "verdict": (c.get("audit") or {}).get("verdict") or c["judge"].get("verdict", ""), "qualifies": qualifies(c),
        "tier": TIERS[tier(c)],
    } for c in top]
    try:
        res = llm.chat_json(prompts.RANK_SYSTEM, prompts.rank_user(items), temperature=0.0)
        by_id = {r.get("id"): r.get("reason", "") for r in items_of(res, "ranking") if isinstance(r, dict)}
        for c in top:
            c["rank_reason"] = by_id.get(c["id"], "")
    except LLMError as e:
        errors.append({"stage": "rank", "error": str(e)[:500]})
        _log(cb, f"[rank] could not write rationale (order unaffected): {str(e)[:160]}")
    return errors


# ---------------------------------------------------------------- export
def _row(c: dict, episodes) -> dict:
    return {
        "rank": c.get("rank"), "id": c["id"], "episode": c["episode"], "source": c["source"],
        "qualifies": qualifies(c), "tier": TIERS[tier(c)] if final_scores(c) else None,
        "segments": [{"episode": s.ep, "start": fmt(s.start), "end": fmt(s.end), "start_s": round(s.start, 2),
                      "end_s": round(s.end, 2), "duration_s": round(s.dur, 1), "blocks": f"{s.first_block}-{s.last_block}"}
                     for s in c["clip"].segments],
        "duration_s": round(c["clip"].duration, 1),
        "hook": c["hook"], "cliffhanger": c["cliffhanger"], "why": c["why"],
        "context_needed": c.get("context_needed", ""), "repaired": c.get("repaired", ""),
        "score": c.get("score"),
        "gen_hook_strength": c.get("gen_hook_strength"), "gen_cliff_strength": c.get("gen_cliff_strength"),
        "judge": c.get("judge") or {}, "judge_model": c.get("judge_model"),
        "audit": c.get("audit") or {}, "audit_model": c.get("audit_model"),
        "final": {k: round(v, 2) for k, v in (final_scores(c) or {}).items()},
        "checks": c.get("checks", {}), "rank_reason": c.get("rank_reason", ""),
        "clip_text": clip_text(episodes, c["clip"], with_times=True),
    }


def summarize(rows: list[dict]) -> dict:
    judged = [r for r in rows if r["judge"]]
    q = [r for r in judged if r["qualifies"]]
    avg = lambda xs: round(sum(xs) / len(xs), 2) if xs else None
    jv = lambda k: [float(r["judge"][k]) for r in judged if k in r["judge"]]
    gen_c = [(float(r["gen_cliff_strength"]) / 2, float(r["judge"]["cliffhanger"])) for r in judged
             if r.get("gen_cliff_strength") is not None]
    return {
        "clips": len(rows), "judged": len(judged), "qualifying": len(q),
        "by_tier": {t: sum(1 for r in rows if r.get("tier") == t) for t in TIERS.values()},
        "qualify_rate": round(len(q) / len(judged), 2) if judged else None,
        "avg_judge": {k: avg(jv(k)) for k in ("hook", "cliffhanger", "standalone", "spoiler_risk")},
        "avg_score": avg([r["score"] for r in judged if r["score"] is not None]),
        "episodes_covered": len({s["episode"] for r in rows for s in r["segments"]}),
        "episodes_with_qualifying": len({r["episode"] for r in q}),
        "stitched": sum(1 for r in rows if len(r["segments"]) > 1),
        "cross_episode": sum(1 for r in rows if len({s["episode"] for s in r["segments"]}) > 1),
        "repaired": sum(1 for r in rows if r["repaired"]),
        "content_flagged": sum(1 for r in judged if r["judge"].get("content_flag", "none") not in ("none", None)),
        "first_beat_late": sum(1 for r in rows if (r["checks"].get("first_beat_s") or 0) > 3),
        "ends_at_episode_end": sum(1 for r in rows if r["checks"].get("ends_at_episode_end")),
        "generator_overconfidence_cliff": avg([g - j for g, j in gen_c]),  # gen score /2 vs judge (both ~1-5)
        "judge_vs_audit": agreement(rows),
    }


def agreement(rows: list[dict]) -> dict | None:
    """How far can the cheap blind judge be trusted? Compare it with the show-aware auditor."""
    both = [r for r in rows if r.get("judge") and r.get("audit")]
    if not both:
        return None
    out = {"n": len(both)}
    for k in DIMS:
        pairs = [(float(r["judge"][k]), float(r["audit"][k])) for r in both]
        out[k] = {"exact": round(sum(a == b for a, b in pairs) / len(pairs), 2),
                  "within_1": round(sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs), 2),
                  "judge_minus_audit": round(sum(a - b for a, b in pairs) / len(pairs), 2),
                  "spearman": spearman([a for a, _ in pairs], [b for _, b in pairs])}
    q = lambda d: float(d["hook"]) >= MIN_HOOK and float(d["cliffhanger"]) >= MIN_CLIFF
    jq = {r["id"] for r in both if q(r["judge"])}
    aq = {r["id"] for r in both if q(r["audit"])}
    out["qualify_agreement"] = round(1 - len(jq ^ aq) / len(both), 2)
    out["judge_only_qualifies"] = len(jq - aq)
    out["audit_only_qualifies"] = len(aq - jq)
    return out


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2
            i = j + 1
        return r
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 2) if den else None


def export(out_dir, ranked, rejected, errors, episodes, llms, meta):
    os.makedirs(out_dir, exist_ok=True)
    rows = [_row(c, episodes) for c in sorted(ranked, key=lambda c: c.get("rank") or 1e9)]
    with open(os.path.join(out_dir, "promos.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    rej = [{"episode": c["episode"], "source": c["source"], "problems": c["clip"].problems,
            "segments": c.get("raw_segments") or c["segments"], "hook": c["hook"], "cliffhanger": c["cliffhanger"]}
           for c in rejected]
    with open(os.path.join(out_dir, "rejected.json"), "w", encoding="utf-8") as f:
        json.dump(rej, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "promos.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "tier", "qualifies", "episode", "segments (episode start-end)", "duration_s", "score", "hook",
                    "cliffhanger", "why", "rank_reason", "judge_hook", "judge_cliff", "judge_standalone",
                    "judge_spoiler", "content_flag", "judge_verdict", "first_beat_s", "repaired"])
        for r in rows:
            j = r["judge"]
            w.writerow([r["rank"], r["tier"], r["qualifies"], r["episode"],
                        " + ".join(f"E{s['episode']} {s['start']}-{s['end']}" for s in r["segments"]),
                        r["duration_s"], r["score"], r["hook"], r["cliffhanger"], r["why"], r["rank_reason"],
                        j.get("hook"), j.get("cliffhanger"), j.get("standalone"), j.get("spoiler_risk"),
                        j.get("content_flag"), j.get("verdict"), r["checks"].get("first_beat_s"), r["repaired"]])
    summary = summarize(rows)
    stats = {**meta, "summary": summary, "n_ranked": len(rows), "n_rejected": len(rej),
             "rejection_reasons": _reasons(rej), "errors": errors,
             "llm": {name: {"models": l.models, "stats": l.stats, "dropped": l.dead} for name, l in llms.items() if l}}
    with open(os.path.join(out_dir, "run_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    _write_report(out_dir, rows, stats)
    return rows


def _reasons(rej):
    out = {}
    for r in rej:
        for p in r["problems"]:
            key = p.split("(")[0].strip()
            out[key] = out.get(key, 0) + 1
    return out


def _write_report(out_dir, rows, stats):
    s = stats["summary"]
    models = {n: v["stats"].get("by_model") for n, v in (stats.get("llm") or {}).items()}
    L = [f"# Promo shortlist: {stats.get('show', '')}", "",
         f"Mode **{stats.get('mode')}** · models used {models or 'none (rules only)'} · "
         f"run {stats.get('elapsed_s')}s", "",
         f"- {stats['n_raw_candidates']} proposals → {stats['n_ranked']} valid clips, {stats['n_rejected']} rejected "
         f"by code checks {stats['rejection_reasons']}",
         f"- {s['qualifying']}/{s['judged']} clips **qualify** (stranger-judge hook ≥{MIN_HOOK} and cliffhanger ≥{MIN_CLIFF}); "
         f"covering {s['episodes_with_qualifying']} episodes. Tiers: {s['by_tier']}",
         f"- {s['stitched']} stitched clips ({s['cross_episode']} cross-episode), {s['repaired']} auto-trimmed to fit 60s, "
         f"{s['content_flagged']} carry a content flag",
         f"- Avg blind-judge scores (1-5): {s['avg_judge']}",
         f"- Blind judge vs show-aware auditor: {s.get('judge_vs_audit')}", "",
         "Timestamps are relative to the start of each episode (mm:ss.s). Ranked = order to post.", ""]
    for r in rows[:RANK_TOP]:
        seg = "  +  ".join(f"**E{x['episode']} {x['start']} → {x['end']}**" for x in r["segments"])
        j = r["judge"]
        tag = {"acquisition": "", "weak": "  ⚠️ weak"}.get(r.get("tier"), "  🔁 re-engagement (spoils the plot)")
        L += [f"### #{r['rank']}  {seg}  ({r['duration_s']}s · score {r['score']}){tag}",
              f"- **Hook:** {r['hook']}", f"- **Cliffhanger:** {r['cliffhanger']}", f"- **Why it works:** {r['why']}",
              f"- **Stranger judge (1-5):** hook {j.get('hook')} · cliffhanger {j.get('cliffhanger')} · standalone "
              f"{j.get('standalone')} · spoiler {j.get('spoiler_risk')} · flag {j.get('content_flag')} — {j.get('verdict', '')}"]
        a = r.get("audit") or {}
        if a:
            L.append(f"- **Show-aware auditor (1-5):** hook {a.get('hook')} · cliffhanger {a.get('cliffhanger')} · "
                     f"standalone {a.get('standalone')} · spoiler {a.get('spoiler_risk')} — {a.get('verdict', '')}"
                     + (f" _Fix: {a.get('fix')}_" if a.get("fix") not in (None, "", "none") else ""))
        if r["rank_reason"]:
            L.append(f"- **Why post it here:** {r['rank_reason']}")
        if r["repaired"]:
            L.append(f"- _Auto-repair: {r['repaired']}_")
        L.append("")
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ---------------------------------------------------------------- orchestrator
def run(data_dir: str, out_dir: str, mode: str = "llm", llm: LLM | None = None, judge_llm: LLM | None = None,
        episodes_subset: set[int] | None = None, cb: Callable[[str], None] | None = None,
        eps: list[Episode] | None = None, show_name: str | None = None, audit_llm: LLM | None = None) -> list[dict]:
    t0 = time.time()
    eps = eps or load_show(data_dir)
    eps_run = [e for e in eps if e.ep in episodes_subset] if episodes_subset else eps
    episodes = {e.ep: e for e in eps}
    judge_llm = judge_llm or llm
    errors: list[dict] = []
    _log(cb, f"Parsed {len(eps)} episodes, {sum(len(e.blocks) for e in eps)} blocks")

    if mode == "baseline":
        raw = baseline.propose(eps_run)
    else:
        raw, errors = propose_llm(eps_run, eps, llm, cb)
    for i, c in enumerate(raw):
        c["id"] = f"C{i + 1:03d}"
    kept, rejected = validate(raw, episodes)
    _log(cb, f"[validate] {len(raw)} proposals -> kept {len(kept)}, rejected {len(rejected)}")
    for c in kept:
        c["checks"] = code_checks(episodes, c["clip"])

    if judge_llm is not None:
        errors += judge(kept, episodes, judge_llm, cb)
    if audit_llm is not None:
        errors += audit(kept, episodes, eps, audit_llm, cb)
    for c in kept:
        c["score"] = composite(c)
    errors += rank(kept, judge_llm, cb) if judge_llm is not None else []  # rationale text is low-stakes: cheap model
    if judge_llm is None:
        for i, c in enumerate(kept):
            c["rank"] = i + 1
    meta = {"show": show_name or os.path.basename(os.path.normpath(data_dir)), "mode": mode,
            "n_episodes": len(eps_run), "n_raw_candidates": len(raw), "elapsed_s": round(time.time() - t0)}
    return export(out_dir, kept, rejected, errors, episodes, {"propose": llm, "judge": judge_llm, "audit": audit_llm}, meta)


def compare_baseline(data_dir: str, out_dir: str, judge_llm: LLM, per_episode: int = 1,
                     sample_seed: int = 7, n: int | None = None, cb=None) -> dict:
    """Quality experiment: send the rule-based baseline's best clip per episode through the SAME
    stranger judge, so LLM picks and rule picks are scored on one scale."""
    eps = load_show(data_dir)
    episodes = {e.ep: e for e in eps}
    raw = baseline.propose(eps, per_episode=per_episode)
    if n and len(raw) > n:
        random.Random(sample_seed).shuffle(raw)
        raw = sorted(raw[:n], key=lambda c: c["episode"])
    for i, c in enumerate(raw):
        c["id"] = f"R{i + 1:03d}"
    kept, rejected = validate(raw, episodes)
    for c in kept:
        c["checks"] = code_checks(episodes, c["clip"])
    errors = judge(kept, episodes, judge_llm, cb, label="judge-baseline")
    for c in kept:
        c["score"] = composite(c)
    order = order_clips(kept)
    for i, c in enumerate(order):
        c["rank"] = i + 1
    return export(out_dir, kept, rejected, errors, episodes, {"judge": judge_llm},
                  {"show": os.path.basename(os.path.normpath(data_dir)), "mode": "baseline+judge",
                   "n_episodes": len(eps), "n_raw_candidates": len(raw), "elapsed_s": None})

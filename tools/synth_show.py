"""Generate a NEW microdrama in Kuku's exact screenplay format, with planted gold labels.

Why: the real test (and the interview) is "run it on a show it has never seen". Public microdrama
scripts don't exist, so we write fresh ones with an LLM. A different model family from the
pipeline's generator writes them by default, so the tool is not grading its own style. While writing,
the model TAGS blocks, which become ground truth:
  promo_start / hook / cliff / promo_end   one planted 25-50s promo per episode (hook within 3s of
                                           promo_start, cliffhanger cut BEFORE the answer)
  decoy_start / decoy_end                  a dramatic-looking stretch that ends RESOLVED (a decision,
                                           a reconciliation): the failure mode the quality review found
Half the episodes are told to place the planted promo mid-episode (not at the episode's ending), so
"just take the episode's final minute" cannot score well.

Code, not the model, assigns every timestamp (from per-block durations), so the files are always
well-formed; the result is parsed back with the pipeline's own parser before it is written.

  python tools/synth_show.py --out ../test_data/synthetic/ceo_revenge --episodes 8 --premise "..."
  python tools/synth_show.py --out ../test_data/synthetic/random --episodes 8            # random premise
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from promo.kukufmt import RawBlock, RawEpisode, RawScene, write_show  # noqa: E402
from promo.llm import LLM, LLMError  # noqa: E402
from promo.parse import load_show  # noqa: E402

PREMISES = [
    "A janitor humiliated at a billionaire's gala is secretly the heir who owns the company.",
    "A werewolf alpha rejects his fated mate in front of the whole pack; she returns as the rival pack's luna.",
    "A divorced housewife discovers her ex-husband's new fiancée is the woman who framed her for theft.",
    "A palace maid falsely accused of poisoning the empress must find the real poisoner before her execution.",
    "A surgeon finds her twin sister, presumed dead, alive and married to the surgeon's husband.",
    "A mafia boss's contract bride learns her father sold her to pay a debt to the man who killed her mother.",
    "A small-town bride in India is abandoned at the mandap; the groom's elder brother offers to marry her.",
]

SHOW_SYSTEM = """You are a showrunner for vertical microdramas (ReelShort/DramaBox style: 1-2 minute episodes, \
melodrama, a slap or shocking reveal every episode, a cliffhanger at every episode end). Respond with JSON only."""

SHOW_USER = """Create an ORIGINAL show (invent all names). Premise: {premise}
Return JSON: {{"title": "...", "characters": [{{"name": "UPPERCASE NAME", "role": "..."}}], \
"episodes": [{{"n": 1, "logline": "one sentence", "continuity": "one sentence"}}, ... exactly {n} episodes]}}"""

EP_SYSTEM = """You write one episode of a vertical microdrama as a list of screenplay blocks. Respond with JSON only.

Block kinds: "camera" (shot description), "action", "sound", "dialogue" (needs "speaker" in UPPERCASE and \
"emotion" like "Cold, Mocking"). Give every block "seconds" (camera 1.5-4, action 1.5-4, sound 1-2, \
dialogue = about 2.6 words per second, min 1). The episode must total 90-150 seconds of non-camera blocks.

You must TAG blocks (field "tag") to mark ground truth:
- "promo_start": first block of the best 25-50 second promo in this episode
- "hook": the block with the scroll-stopping moment (slap, reveal, shocking line); at most 3 seconds \
after promo_start (it may be the same block, then use tag "promo_start+hook")
- "cliff": the block carrying the unresolved moment (threat, question, interrupted action) BEFORE any answer
- "promo_end": last block of the promo (can be the cliff block, then use "cliff+promo_end")
- "decoy_start" / "decoy_end": a 15-40 second dramatic stretch that ENDS RESOLVED (a decision is made, \
people reconcile, a question is answered). It must not overlap the promo.
{placement}

Return: {{"scenes": [{{"heading": "INT. PLACE - DAY", "blocks": [{{"kind": "...", "speaker": "", "emotion": "", \
"text": "...", "seconds": 2.5, "tag": ""}}]}}]}}"""

EP_USER = """SHOW: {title}
CHARACTERS: {chars}
ALL EPISODE LOGLINES:
{bible}

Write episode {n}: {logline}"""

MID = "The promo must END at least 25 seconds BEFORE the episode ends (the episode continues afterwards), \
and the episode should still end on its own cliffhanger."
END = "The promo should end at the episode's final cliffhanger."


def render(n: int, logline: str, continuity: str, data: dict) -> tuple[RawEpisode, dict]:
    ep = RawEpisode(n, logline, continuity)
    t, tags = 0.0, {}
    for sc in data.get("scenes") or []:
        rs = RawScene(str(sc.get("heading") or "INT. UNKNOWN - DAY"))
        for b in sc.get("blocks") or []:
            kind = str(b.get("kind", "action")).lower()
            kind = {"sfx": "sound", "sound fx": "sound", "cam": "camera", "act": "action"}.get(kind, kind)
            if kind not in ("camera", "action", "sound", "dialogue"):
                kind = "action"
            text = str(b.get("text") or "").strip()
            if not text:
                continue
            try:
                sec = float(b.get("seconds") or 2)
            except (TypeError, ValueError):
                sec = 2.0
            sec = max(0.8, min(15.0, sec))
            if kind == "camera":           # a shot runs alongside the next beat, as in Kuku's scripts
                blk = RawBlock("camera", round(t, 2), round(t + sec, 2), text)
            else:
                blk = RawBlock(kind, round(t, 2), round(t + sec, 2), text,
                               str(b.get("speaker") or "") if kind == "dialogue" else "",
                               str(b.get("emotion") or "") if kind == "dialogue" else "")
                t += sec
            rs.blocks.append(blk)
            for tag in str(b.get("tag") or "").replace(" ", "").split("+"):
                if tag:
                    tags.setdefault(tag, blk)
        if rs.blocks:
            ep.scenes.append(rs)
    return ep, tags


def gold_for(n: int, tags: dict) -> tuple[list, list, list]:
    moments, decoys, problems = [], [], []
    need = ("promo_start", "hook", "cliff", "promo_end")
    if all(k in tags for k in need):
        a, h, c, e = (tags[k] for k in need)
        dur = e.end - a.start
        m = {"id": f"E{n}-promo", "episode": n, "start_s": a.start, "end_s": e.end, "kind": "promo",
             "hook_s": h.start, "cliff_s": c.end, "label": c.text[:120]}
        if not 18 <= dur <= 65:
            problems.append(f"E{n}: planted promo is {dur:.0f}s")
        if h.start - a.start > 3.5:
            problems.append(f"E{n}: planted hook lands {h.start - a.start:.1f}s in")
        moments.append(m)
    else:
        problems.append(f"E{n}: missing tags {[k for k in need if k not in tags]}")
    if "decoy_start" in tags and "decoy_end" in tags:
        decoys.append({"id": f"E{n}-decoy", "episode": n, "start_s": tags["decoy_start"].start,
                       "end_s": tags["decoy_end"].end, "kind": "resolved_ending", "label": tags["decoy_end"].text[:120]})
    return moments, decoys, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--premise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="groq:openai/gpt-oss-120b,openrouter:nvidia/nemotron-3-ultra-550b-a55b:free,"
                    "groq:qwen/qwen3.8-27b,gemini:gemini-3.5-flash,gemini:gemini-3-flash-preview",
                    help="writer chain; default leads with a different family than the pipeline's generator")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    premise = a.premise or rng.choice(PREMISES)
    llm = LLM(role="synth", chain=[m.strip() for m in a.models.split(",")], max_tokens=12000)
    print("writer chain:", llm.models)
    show = llm.chat_json(SHOW_SYSTEM, SHOW_USER.format(premise=premise, n=a.episodes), temperature=0.9)
    eps_meta = (show.get("episodes") or [])[:a.episodes]
    chars = ", ".join(f"{c.get('name')} ({c.get('role')})" for c in show.get("characters") or [])
    bible = "\n".join(f"E{e.get('n')}: {e.get('logline')}" for e in eps_meta)
    placements = {i + 1: (MID if rng.random() < 0.5 else END) for i in range(len(eps_meta))}

    def one(meta):
        n = int(meta.get("n"))
        sysmsg = EP_SYSTEM.format(placement=placements[n])
        user = EP_USER.format(title=show.get("title"), chars=chars, bible=bible, n=n, logline=meta.get("logline"))
        for attempt in range(2):
            try:
                data = llm.chat_json(sysmsg, user + ("" if not attempt else "\n(Retry: include ALL required tags.)"),
                                     temperature=0.8)
                ep, tags = render(n, meta.get("logline", ""), meta.get("continuity", ""), data)
                mom, dec, probs = gold_for(n, tags)
                if mom or attempt:
                    return ep, mom, dec, probs
            except LLMError as e:
                if attempt:
                    return None, [], [], [f"E{n}: {str(e)[:200]}"]
        return None, [], [], [f"E{n}: failed"]

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(one, eps_meta))
    episodes = [r[0] for r in results if r[0] is not None]
    write_show(episodes, a.out)
    parsed = load_show(a.out)  # round-trip through the pipeline's own parser
    gold = {"show": show.get("title"), "source": "synthetic (LLM-written, planted labels)", "premise": premise,
            "writer_models": sorted(llm.stats["by_model"]),
            "placement": {str(k): ("mid-episode" if v == MID else "episode end") for k, v in placements.items()},
            "moments": [m for r in results for m in r[1]], "decoys": [d for r in results for d in r[2]],
            "problems": [p for r in results for p in r[3]],
            "caveat": "Labels are the writer model's intent, not audience data. Good for recall/regression tests."}
    from promo.goldeval import sanitize
    gold = sanitize(gold)
    with open(os.path.join(a.out, "gold.json"), "w", encoding="utf-8") as f:
        json.dump(gold, f, indent=1, ensure_ascii=False)
    print(json.dumps({"title": gold["show"], "episodes_written": len(parsed),
                      "blocks": sum(len(e.blocks) for e in parsed),
                      "minutes": round(sum(e.duration for e in parsed) / 60, 1),
                      "gold_moments": len(gold["moments"]), "decoys": len(gold["decoys"]),
                      "problems": gold["problems"], "usage": llm.stats["by_model"]}, indent=1))


if __name__ == "__main__":
    main()

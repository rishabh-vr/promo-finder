"""Rule-based baseline: no LLM. Enumerates every valid 20-60s window on block boundaries
and scores it with cheap text features. Two jobs:
  1. a free control for the quality experiment (does the LLM beat simple rules?);
  2. a sanity check that the clip machinery works without any API key.
"""
from __future__ import annotations

import re

from .clips import MAX_LEN, MIN_LEN, build_clip, hook_window, overlap_ratio, segment_blocks
from .parse import Episode

_HOOK_WORDS = re.compile(
    r"\b(slap|kill|dead|die|murder|divorce|pregnan|betray|lie[sd]?|secret|truth|"
    r"marry|married|wife|husband|duke|king|poison|blood|never|how dare|get out|guards?)\b", re.I
)


def _dialogue(blocks):
    return [b for b in blocks if b.kind == "dialogue"]


def _score(ep: Episode, clip, episodes) -> tuple[float, str, str]:
    seg = clip.segments[0]
    blocks = segment_blocks(ep, seg)
    dl = _dialogue(blocks)
    hook_blocks = _dialogue(hook_window(episodes, clip))
    hook_txt = hook_blocks[0].text if hook_blocks else ""
    last = dl[-1].text if dl else ""
    s = 0.0
    # hook: speech starts inside 3s, with punch
    if hook_blocks:
        s += 2
        if re.search(r"[!?]", hook_txt):
            s += 1
        if _HOOK_WORDS.search(hook_txt):
            s += 1
    # cliffhanger: ends on a question/exclamation/trailing off, or at a scene/episode end
    if re.search(r"[?!]|\.\.\.|—|-$", last):
        s += 2
    if abs(seg.end - ep.duration) < 0.5:
        s += 2
    elif any(abs(sc.end - seg.end) < 0.3 for sc in ep.scenes if sc.end is not None):
        s += 1
    # density of conflict words and speech
    s += min(2, len(_HOOK_WORDS.findall(" ".join(b.text for b in dl))) * 0.4)
    s += min(1.5, len(dl) / 12)
    # being the very start of the episode is context-free-ish; being mid-scene is not
    if seg.start < 1.0:
        s += 0.5
    return s, hook_txt, last


def propose(episodes_list: list[Episode], per_episode: int = 3, step_blocks: int = 1):
    episodes = {e.ep: e for e in episodes_list}
    out = []
    for ep in episodes_list:
        cands = []
        for i, a in enumerate(ep.blocks):
            for j in range(i + 1, len(ep.blocks)):
                b = ep.blocks[j]
                if b.end - a.start > MAX_LEN:
                    break
                if b.end - a.start < MIN_LEN:
                    continue
                clip = build_clip(episodes, [{"episode": ep.ep, "start_block": a.id, "end_block": b.id}])
                if not clip.valid:
                    continue
                sc, hook, last = _score(ep, clip, episodes)
                cands.append((sc, clip, hook, last))
        cands.sort(key=lambda x: -x[0])
        picked = []
        for sc, clip, hook, last in cands:
            if all(overlap_ratio(clip, p[1]) < 0.2 for p in picked):
                picked.append((sc, clip, hook, last))
            if len(picked) >= per_episode:
                break
        for sc, clip, hook, last in picked:
            seg = clip.segments[0]
            out.append(
                {
                    "source": "baseline",
                    "episode": ep.ep,
                    "segments": [{"episode": ep.ep, "start_block": seg.first_block, "end_block": seg.last_block}],
                    "hook": hook,
                    "cliffhanger": last,
                    "why": f"rule score {sc:.1f}",
                    "gen_hook_strength": None,
                    "gen_cliff_strength": None,
                }
            )
    return out

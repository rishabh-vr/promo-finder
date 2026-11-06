"""Deterministic clip construction + validation.

The LLM only ever names block ids. Everything numeric (timestamps, durations, snapping
cuts so no line of dialogue is cut in half, the 20-60s rule) is computed here, so the
model cannot invent a timestamp or quietly break the length rule.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parse import Block, Episode, fmt

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_WINDOW = 3.0
EPS = 0.05


@dataclass
class Segment:
    ep: int
    start: float
    end: float
    first_block: str
    last_block: str

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class Clip:
    segments: list[Segment]
    problems: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(s.dur for s in self.segments)

    @property
    def valid(self) -> bool:
        return not self.problems


def _snap(ep: Episode, start: float, end: float) -> tuple[float, float]:
    """Move cut points outward so neither falls inside any block (no half-cut lines)."""
    changed = True
    while changed:
        changed = False
        for b in ep.blocks:
            if b.start < start - EPS and b.end > start + EPS:
                start, changed = b.start, True
            if b.start < end - EPS and b.end > end + EPS:
                end, changed = b.end, True
    return start, end


def build_segment(ep: Episode, first_id: str, last_id: str) -> tuple[Segment | None, str | None]:
    a, b = ep.block(first_id), ep.block(last_id)
    if a is None or b is None:
        return None, f"unknown block id ({first_id}..{last_id}) in episode {ep.ep}"
    if b.idx < a.idx:
        a, b = b, a
    span = ep.blocks[a.idx : b.idx + 1]
    start = min(x.start for x in span)
    end = max(x.end for x in span)
    start, end = _snap(ep, start, end)
    return Segment(ep.ep, start, end, a.id, b.id), None


_QID = re.compile(r"^\s*E(\d+)\s*[-:.]\s*(B\d+)\s*$", re.I)
MAX_SEGMENTS = 3
MIN_SEGMENT = 2.0


def normalize_spec(spec: dict, default_ep: int | None = None) -> dict:
    """Accept 'E5-B012' or ('B012' + episode) and return {'episode', 'start_block', 'end_block'}."""
    out = {"episode": spec.get("episode", default_ep)}
    for k in ("start_block", "end_block"):
        v = str(spec.get(k, "")).strip()
        m = _QID.match(v)
        if m:
            out["episode"] = int(m.group(1)) if k == "start_block" or out["episode"] is None else out["episode"]
            if k == "end_block" and int(m.group(1)) != int(out["episode"]):
                out["_cross"] = True  # a single segment cannot span two episodes
            v = m.group(2).upper()
        out[k] = v
    return out


def build_clip(episodes: dict[int, Episode], seg_specs: list[dict]) -> Clip:
    """seg_specs: [{'episode': int, 'start_block': 'B012', 'end_block': 'B040'}, ...] (or E5-B012 ids)"""
    segs, problems = [], []
    if len(seg_specs) > MAX_SEGMENTS:
        problems.append(f"too many segments ({len(seg_specs)} > {MAX_SEGMENTS})")
    for raw in seg_specs:
        spec = normalize_spec(raw)
        if spec.get("_cross"):
            problems.append(f"one segment spans two episodes ({raw})")
            continue
        try:
            ep_no = int(spec["episode"])
            ep = episodes[ep_no]
        except (KeyError, ValueError, TypeError):
            problems.append(f"bad episode in segment spec {raw}")
            continue
        seg, err = build_segment(ep, str(spec.get("start_block")), str(spec.get("end_block")))
        if err:
            problems.append(err)
        else:
            segs.append(seg)
    for i, s in enumerate(segs):
        if len(segs) > 1 and s.dur < MIN_SEGMENT:
            problems.append(f"segment {i + 1} too short to read ({s.dur:.1f}s)")
        for t in segs[i + 1:]:
            if s.ep == t.ep and min(s.end, t.end) - max(s.start, t.start) > EPS:
                problems.append("segments overlap")
    clip = Clip(segs, problems)
    if segs:
        d = clip.duration
        if d < MIN_LEN - EPS:
            clip.problems.append(f"too short ({d:.1f}s < {MIN_LEN:.0f}s)")
        if d > MAX_LEN + EPS:
            clip.problems.append(f"too long ({d:.1f}s > {MAX_LEN:.0f}s)")
    else:
        clip.problems.append("no valid segments")
    return clip


def segment_blocks(ep: Episode, seg: Segment) -> list[Block]:
    return [b for b in ep.blocks if b.start >= seg.start - EPS and b.end <= seg.end + EPS]


def clip_blocks(episodes: dict[int, Episode], clip: Clip) -> list[tuple[Segment, list[Block]]]:
    return [(s, segment_blocks(episodes[s.ep], s)) for s in clip.segments]


def hook_window(episodes: dict[int, Episode], clip: Clip) -> list[Block]:
    """Blocks active in the first HOOK_WINDOW seconds of the clip."""
    seg = clip.segments[0]
    ep = episodes[seg.ep]
    t0, t1 = seg.start, seg.start + HOOK_WINDOW
    return [b for b in segment_blocks(ep, seg) if b.start < t1 and b.end > t0]


_META = re.compile(r"\b(ominous )?cliffhanger( peak| moment| ending)?\b", re.I)


def clip_text(episodes: dict[int, Episode], clip: Clip, with_times: bool = False, scrub: bool = False) -> str:
    """Clip as plain script text, for the isolated 'stranger' judge (no show context)."""
    parts = []
    for i, (seg, blocks) in enumerate(clip_blocks(episodes, clip)):
        if i:
            parts.append("[CUT TO A LATER MOMENT]")
        for b in blocks:
            t = f"[{fmt(b.start - seg.start)}] " if with_times else ""
            txt = _META.sub("dramatic", b.text) if scrub else b.text
            if b.kind == "dialogue":
                emo = f" ({b.emotion})" if b.emotion else ""
                parts.append(f'{t}{b.speaker}{emo}: "{txt}"')
            else:
                label = {"camera": "CAMERA", "action": "ACTION", "sound": "SOUND"}[b.kind]
                parts.append(f"{t}{label}: {txt}")
    return "\n".join(parts)


def code_checks(episodes: dict[int, Episode], clip: Clip) -> dict:
    """Cheap deterministic signals, independent of any model:
    first_beat_s : seconds until the first line/action/sound (a hook needs one inside 3s)
    hook_has_line: a line of dialogue starts inside the first 3s
    ends_at_episode_end: the cut is the episode's own ending (usually a written cliffhanger)
    last_line    : the final line of dialogue in the clip"""
    first = clip.segments[0]
    blocks = segment_blocks(episodes[first.ep], first)
    beats = [b.start - first.start for b in blocks if b.kind != "camera"]
    lines = [b for b in blocks if b.kind == "dialogue"]
    last = clip.segments[-1]
    last_blocks = segment_blocks(episodes[last.ep], last)
    last_lines = [b for b in last_blocks if b.kind == "dialogue"]
    return {
        "first_beat_s": round(min(beats), 1) if beats else None,
        "hook_has_line": any(b.start - first.start < HOOK_WINDOW for b in lines),
        "ends_at_episode_end": abs(last.end - episodes[last.ep].duration) < 0.6,
        "last_line": (f'{last_lines[-1].speaker}: "{last_lines[-1].text}"' if last_lines else ""),
        "n_segments": len(clip.segments),
        "cross_episode": len({s.ep for s in clip.segments}) > 1,
    }


def describe(clip: Clip) -> str:
    return " + ".join(
        f"E{s.ep} {fmt(s.start)}-{fmt(s.end)} ({s.dur:.1f}s)" for s in clip.segments
    )


def overlap_ratio(a: Clip, b: Clip) -> float:
    """Fraction of the shorter clip covered by the other (same-episode time overlap)."""
    inter = 0.0
    for s in a.segments:
        for t in b.segments:
            if s.ep == t.ep:
                inter += max(0.0, min(s.end, t.end) - max(s.start, t.start))
    return inter / max(1e-9, min(a.duration, b.duration))

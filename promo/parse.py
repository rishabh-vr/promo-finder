"""Parse Kuku screenplay .txt files into Episode / Block objects.

The format is regular (see README): a header, then SCENE blocks, each holding tagged
blocks (CAMERA / ACTION / SOUND FX / <CHARACTER (emotion)>) with Start/End timestamps.
Parsing is deliberately tolerant: case-insensitive labels (episode 6 uses
"SCENE HEADING:" in capitals), missing scene times, multi-line text.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

_BLOCK_HDR = re.compile(r"^\*\*(.+?):\*\*\s*$")
_TIMES = re.compile(
    r"^Start:\s*\((\d+):(\d+(?:\.\d+)?)\)\s*\|\s*End:\s*\((\d+):(\d+(?:\.\d+)?)\)", re.I
)
_SCENE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
_SCENE_FIELD = re.compile(r"^SCENE\s+(HEADING|START|END)\s*:\s*(.*)$", re.I)
_EP_NUM = re.compile(r"^EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", re.I)

NON_DIALOGUE = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sound"}


def _mmss(m: str, s: str) -> float:
    return int(m) * 60 + float(s)


def fmt(t: float) -> str:
    """Seconds -> mm:ss.s (matches the source files' style)."""
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


@dataclass
class Block:
    id: str          # e.g. "B012" (unique within the episode)
    ep: int
    idx: int         # position in the episode's block list
    scene: int
    kind: str        # camera | action | sound | dialogue
    speaker: str     # only for dialogue
    emotion: str     # only for dialogue
    start: float
    end: float
    text: str


@dataclass
class Scene:
    n: int
    heading: str = ""
    start: float | None = None
    end: float | None = None


@dataclass
class Episode:
    ep: int
    logline: str = ""
    continuity: str = ""
    scenes: list[Scene] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((b.end for b in self.blocks), default=0.0)

    def block(self, bid: str) -> Block | None:
        for b in self.blocks:
            if b.id == bid:
                return b
        return None


def parse_episode(path: str) -> Episode:
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    ep = Episode(ep=-1)
    m = re.search(r"episode-(\d+)", os.path.basename(path), re.I)
    if m:
        ep.ep = int(m.group(1))

    scene_no = 0
    cur: dict | None = None  # block being assembled

    def flush():
        nonlocal cur
        if cur is None:
            return
        text = " ".join(t.strip() for t in cur["text"] if t.strip()).strip()
        if text.startswith('"') and text.endswith('"') and len(text) > 1:
            text = text[1:-1]
        idx = len(ep.blocks)
        ep.blocks.append(
            Block(
                id=f"B{idx + 1:03d}", ep=ep.ep, idx=idx, scene=cur["scene"],
                kind=cur["kind"], speaker=cur["speaker"], emotion=cur["emotion"],
                start=cur["start"], end=cur["end"], text=text,
            )
        )
        cur = None

    in_script = False
    for ln in lines:
        s = ln.strip()
        if not in_script:
            if (mm := _EP_NUM.match(s)):
                ep.ep = int(mm.group(1))
            elif s.upper().startswith("EPISODE LOGLINE:"):
                ep.logline = s.split(":", 1)[1].strip()
            elif s.upper().startswith("CONTINUITY NOTES:"):
                ep.continuity = s.split(":", 1)[1].strip()
            elif s.upper().startswith("--- SCRIPT START"):
                in_script = True
            continue
        if s.upper().startswith("--- SCRIPT END"):
            flush()
            break
        if (mm := _SCENE.match(s)):
            flush()
            scene_no = int(mm.group(1))
            ep.scenes.append(Scene(n=scene_no))
            continue
        if (mm := _SCENE_FIELD.match(s)) and ep.scenes:
            field_, val = mm.group(1).upper(), mm.group(2).strip()
            sc = ep.scenes[-1]
            if field_ == "HEADING":
                sc.heading = val
            else:
                tm = re.match(r"(\d+):(\d+(?:\.\d+)?)", val)
                if tm:
                    setattr(sc, field_.lower(), _mmss(*tm.groups()))
            continue
        if s.lower().startswith("actions and dialogue"):
            continue
        if (mm := _BLOCK_HDR.match(s)):
            flush()
            label = mm.group(1).strip()
            base = re.sub(r"\s*\(.*\)\s*$", "", label).strip()
            emo = re.search(r"\((.*)\)\s*$", label)
            if base.upper() in NON_DIALOGUE:
                kind, speaker, emotion = NON_DIALOGUE[base.upper()], "", ""
            else:
                kind, speaker, emotion = "dialogue", base, (emo.group(1) if emo else "")
            cur = dict(scene=scene_no, kind=kind, speaker=speaker, emotion=emotion,
                       start=None, end=None, text=[])
            continue
        if cur is not None:
            if (mm := _TIMES.match(s)) and cur["start"] is None:
                a, b, c, d = mm.groups()
                cur["start"], cur["end"] = _mmss(a, b), _mmss(c, d)
            elif s:
                cur["text"].append(s)
    flush()

    # Drop any block that never got timestamps (malformed) rather than guess.
    ep.blocks = [b for b in ep.blocks if b.start is not None]
    for i, b in enumerate(ep.blocks):
        b.idx = i
    # Fill missing scene times from blocks.
    for sc in ep.scenes:
        bs = [b for b in ep.blocks if b.scene == sc.n]
        if bs:
            if sc.start is None:
                sc.start = min(b.start for b in bs)
            if sc.end is None:
                sc.end = max(b.end for b in bs)
    return ep


def load_show(folder: str) -> list[Episode]:
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt screenplay files in {folder!r}")
    eps = [parse_episode(p) for p in files]
    eps.sort(key=lambda e: e.ep)
    return eps


def render_transcript(ep: Episode, blocks: list[Block] | None = None, qualify: bool = False) -> str:
    """Compact, LLM-friendly view with block ids and timestamps.
    qualify=True prefixes ids with the episode (E5-B012) for multi-episode prompts."""
    out, last_scene = [], None
    sc_by_n = {s.n: s for s in ep.scenes}
    if qualify:
        out.append(f"===== EPISODE {ep.ep} (length {fmt(ep.duration)}) =====")
    for b in blocks if blocks is not None else ep.blocks:
        if b.scene != last_scene:
            sc = sc_by_n.get(b.scene)
            out.append(f"--- SCENE {b.scene}: {sc.heading if sc else ''} ---")
            last_scene = b.scene
        bid = f"E{ep.ep}-{b.id}" if qualify else b.id
        tag = f"[{bid} {fmt(b.start)}-{fmt(b.end)}]"
        if b.kind == "dialogue":
            emo = f" ({b.emotion})" if b.emotion else ""
            out.append(f'{tag} {b.speaker}{emo}: "{b.text}"')
        else:
            label = {"camera": "CAM", "action": "ACT", "sound": "SFX"}[b.kind]
            out.append(f"{tag} {label}: {b.text}")
    return "\n".join(out)

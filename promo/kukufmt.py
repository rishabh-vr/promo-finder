"""Write episodes in Kuku's screenplay format, so any source can be fed to the same pipeline.

Converters (subtitles, film screenplays, LLM-generated shows) build a list of `RawEpisode`s with
timed blocks and call `write_show`. The output is read back by `promo.parse.load_show`, which is
tested round-trip, so a converter can never silently produce a file the pipeline mis-reads.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .parse import fmt


@dataclass
class RawBlock:
    kind: str              # camera | action | sound | dialogue
    start: float
    end: float
    text: str
    speaker: str = ""
    emotion: str = ""


@dataclass
class RawScene:
    heading: str
    blocks: list[RawBlock] = field(default_factory=list)


@dataclass
class RawEpisode:
    n: int
    logline: str = ""
    continuity: str = ""
    scenes: list[RawScene] = field(default_factory=list)


_LABEL = {"camera": "CAMERA", "action": "ACTION", "sound": "SOUND FX"}


def _ts(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:05.2f}"


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("*", "")).strip()


def render_episode(ep: RawEpisode) -> str:
    out = [f"EPISODE NUMBER: Episode {ep.n}",
           f"EPISODE LOGLINE: {_clean(ep.logline)}",
           f"CONTINUITY NOTES: {_clean(ep.continuity) or 'none'}", "", "--- SCRIPT START ---", ""]
    for i, sc in enumerate(ep.scenes, 1):
        if not sc.blocks:
            continue
        out += [f"SCENE {i:02d}", f"Scene heading: {_clean(sc.heading) or 'UNKNOWN'}",
                f"Scene start: {_ts(min(b.start for b in sc.blocks))}",
                f"Scene end: {_ts(max(b.end for b in sc.blocks))}", "", "Actions and dialogue:", ""]
        for b in sc.blocks:
            if b.kind == "dialogue":
                spk = _clean(b.speaker).upper() or "SPEAKER"
                spk = re.sub(r"[():]", "", spk)
                label = f"{spk} ({_clean(b.emotion)})" if _clean(b.emotion) else spk
                body = f'"{_clean(b.text)}"'
            else:
                label, body = _LABEL[b.kind], _clean(b.text)
            out += [f"**{label}:**", f"Start: ({_ts(b.start)}) | End: ({_ts(b.end)})", body, ""]
    out.append("--- SCRIPT END ---")
    return "\n".join(out) + "\n"


def write_show(episodes: list[RawEpisode], folder: str) -> list[str]:
    os.makedirs(folder, exist_ok=True)
    paths = []
    for ep in episodes:
        p = os.path.join(folder, f"episode-{ep.n:03d}.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(render_episode(ep))
        paths.append(p)
    return paths


def episode_span(ep: RawEpisode) -> float:
    return max((b.end for sc in ep.scenes for b in sc.blocks), default=0.0)


__all__ = ["RawBlock", "RawScene", "RawEpisode", "render_episode", "write_show", "episode_span", "fmt"]

"""Convert subtitle files (.srt / .vtt) into Kuku-format episodes.

Subtitles are the most widely available *timestamped* script-like data (open movies, your own
YouTube uploads, ASR output from Whisper). They lack camera/action lines and usually speaker names,
so they are a deliberately harsher test: the tool must find hooks from dialogue alone.

  python tools/convert_srt.py --in TOS-en.srt --out test_data/tears_of_steel      # one file -> ~2-min episodes
  python tools/convert_srt.py --in folder_of_srts/ --out test_data/myshow          # one file per episode

What it does
  * "NAME: line" or "[NAME] line" -> speaker NAME; "- a / - b" two-speaker cues are split in two
  * "[door slams]", "(gasps)", "♪ music ♪" -> SOUND FX blocks
  * a silence longer than --scene-gap seconds starts a new scene
  * a single long file is cut into episodes of ~--episode-seconds at the longest nearby silence
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from promo.kukufmt import RawBlock, RawEpisode, RawScene, write_show  # noqa: E402

_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})\s*-->\s*(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})")
_TAG = re.compile(r"<[^>]+>|\{\\[^}]*\}")
_SPK = re.compile(r"^\s*(?:\[([A-Z][A-Za-z0-9 .'\-]{0,30})\]|([A-Z][A-Z0-9 .'\-]{1,30}):)\s*(.+)$")
_SOUND = re.compile(r"^\s*[\[(♪#].*[\])♪#]\s*$")


def _sec(h, m, s, ms):
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_subs(text: str) -> list[tuple[float, float, list[str]]]:
    cues, cur = [], None
    for ln in text.replace("﻿", "").splitlines():
        m = _TS.search(ln)
        if m:
            g = m.groups()
            cur = (_sec(*g[:4]), _sec(*g[4:]), [])
            cues.append(cur)
        elif cur is not None and ln.strip() and not ln.strip().isdigit():
            cur[2].append(_TAG.sub("", ln).strip())
        elif not ln.strip():
            cur = None
    return [c for c in cues if c[2]]


def cue_blocks(start: float, end: float, lines: list[str]) -> list[RawBlock]:
    text = " ".join(lines).strip()
    parts = [p.strip() for p in re.split(r"(?:^|\s)-\s+", text) if p.strip()] if text.startswith("-") else [text]
    out, span = [], (end - start) / max(1, len(parts))
    for i, p in enumerate(parts):
        a, b = start + i * span, start + (i + 1) * span
        if _SOUND.match(p):
            out.append(RawBlock("sound", a, b, p.strip("[]()♪# ")))
            continue
        m = _SPK.match(p)
        if m:
            out.append(RawBlock("dialogue", a, b, m.group(3), speaker=(m.group(1) or m.group(2)).strip()))
        else:
            out.append(RawBlock("dialogue", a, b, p, speaker="SPEAKER"))
    return out


def _cut_points(blocks: list[RawBlock], target: float) -> list[int]:
    """Indices where a new episode starts: near every `target` seconds, at the largest silence."""
    cuts, last_t = [0], blocks[0].start
    i = 1
    while i < len(blocks):
        if blocks[i].start - last_t >= target:
            window = [j for j in range(max(cuts[-1] + 1, i - 12), min(len(blocks), i + 12))
                      if blocks[j].start - last_t >= target * 0.6]
            j = max(window, key=lambda j: blocks[j].start - blocks[j - 1].end) if window else i
            cuts.append(j)
            last_t = blocks[j].start
            i = j + 1
        else:
            i += 1
    return cuts


def to_episode(n: int, blocks: list[RawBlock], scene_gap: float, title: str) -> RawEpisode:
    t0 = blocks[0].start
    ep = RawEpisode(n, logline="")
    sc = None
    prev_end = None
    for b in blocks:
        if sc is None or (prev_end is not None and b.start - prev_end > scene_gap):
            sc = RawScene(f"SCENE {len(ep.scenes) + 1}")
            ep.scenes.append(sc)
        sc.blocks.append(RawBlock(b.kind, round(b.start - t0, 2), round(b.end - t0, 2), b.text, b.speaker))
        prev_end = b.end
    lines = [b.text for s in ep.scenes for b in s.blocks if b.kind == "dialogue"]
    ep.logline = f"{title}, part {n}: \"{(lines[0] if lines else '')[:80]}\" ... \"{(lines[-1] if lines else '')[:80]}\""
    return ep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="inp", required=True, help=".srt/.vtt file, or a folder of them (one per episode)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episode-seconds", type=float, default=120.0)
    ap.add_argument("--scene-gap", type=float, default=5.0)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.inp, "*.srt")) + glob.glob(os.path.join(a.inp, "*.vtt"))) \
        if os.path.isdir(a.inp) else [a.inp]
    title = a.title or os.path.splitext(os.path.basename(os.path.normpath(a.inp)))[0]
    episodes = []
    for f in files:
        with open(f, encoding="utf-8-sig", errors="replace") as fh:
            blocks = [b for s, e, ls in parse_subs(fh.read()) for b in cue_blocks(s, e, ls)]
        if not blocks:
            continue
        if len(files) == 1:
            cuts = _cut_points(blocks, a.episode_seconds) + [len(blocks)]
            for i in range(len(cuts) - 1):
                episodes.append(to_episode(len(episodes) + 1, blocks[cuts[i]:cuts[i + 1]], a.scene_gap, title))
        else:
            episodes.append(to_episode(len(episodes) + 1, blocks, a.scene_gap, title))
    write_show(episodes, a.out)
    print(json.dumps({"episodes": len(episodes), "blocks": sum(len(s.blocks) for e in episodes for s in e.scenes)}))


if __name__ == "__main__":
    main()

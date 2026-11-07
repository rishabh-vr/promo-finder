"""Convert a standard (US-format) film screenplay into Kuku-format "episodes", optionally with gold
turning-point labels from the TRIPOD dataset (Papalampidi et al., EMNLP 2019).

Why: Kuku's input is a timestamped microdrama screenplay. Public screenplays have no timestamps and
are 2 hours long, so this tool
  1. segments scenes exactly like TRIPOD's own script (a line containing INT./EXT. starts a scene),
     so TRIPOD's gold scene indices line up;
  2. parses each scene into dialogue (CHARACTER cue + lines, parentheticals -> emotion) and action;
  3. estimates timing from word counts (dialogue ~2.6 words/s, action ~3.5 words/s, clamped);
  4. chunks consecutive scenes into ~2-minute episodes (long scenes are split at block boundaries);
  5. writes gold.json: each turning point's scenes -> (episode, start_s, end_s) moments.

  python tools/convert_screenplay.py --script "Juno (film)_script.txt" --out test_data/tripod/juno \
      --tripod-gold TRIPOD_screenplays_test.csv --movie "Juno (film)"

Caveat (also written into gold.json): timing is estimated, and a turning point is a story beat, not
an editor's promo pick. Use it to test "does the tool find the dramatic moments?" and "does it break on
a different format/genre?", not as a promo ground truth.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from promo.kukufmt import RawBlock, RawEpisode, RawScene, write_show  # noqa: E402

HEADING_TOKENS = {"INT.", "EXT.", "INT/EXT.", "EXT./INT.", "INT./EXT."}
TRANSITION = re.compile(r"^(FADE (IN|OUT|TO)|CUT TO|DISSOLVE TO|SMASH CUT|MATCH CUT|THE END|CONTINUED|"
                        r"\(CONTINUED\)|CONT'D|MORE|\(MORE\)|BACK TO|INTERCUT|END OF|TITLE:|SUPER:)", re.I)
CUE = re.compile(r"^([A-Z][A-Z0-9 .'&\-/]{0,34}?)\s*(\((?:V\.?O\.?|O\.?S\.?|O\.?C\.?|CONT'?D|CONT|"
                 r"INTO PHONE|ON PHONE|PRE-?LAP|FILTERED)[^)]*\)\s*)*$")
DIALOGUE_WPS, ACTION_WPS = 2.6, 3.5


def split_scenes(lines: list[str]) -> list[list[str]]:
    """TRIPOD-compatible: a line whose whitespace tokens include INT./EXT. starts a new scene.
    (TRIPOD's script drops the final scene; we keep it, which leaves indices 0..n-2 identical.)"""
    scenes, cur, started = [], [], False
    for ln in lines:
        toks = re.sub(" +", " ", ln).split()
        if any(t in HEADING_TOKENS for t in toks):
            if started:
                scenes.append(cur)
            cur, started = [ln], True
        elif started:
            cur.append(ln)
    if started:
        scenes.append(cur)
    return scenes


def _paragraphs(lines: list[str]) -> list[list[str]]:
    out, cur = [], []
    for ln in lines:
        if ln.strip():
            cur.append(ln.strip())
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _is_cue(s: str) -> bool:
    return bool(CUE.match(s)) and s.upper() == s and any(c.isalpha() for c in s) and not TRANSITION.match(s) \
        and len(s.split()) <= 5


def parse_scene(lines: list[str]) -> tuple[str, list[RawBlock]]:
    heading = re.sub(r"\s+\d+[A-Z]?\s*$", "", lines[0].strip())  # drop trailing scene numbers
    blocks: list[RawBlock] = []
    for para in _paragraphs(lines[1:]):
        if TRANSITION.match(para[0]) and len(para) == 1:
            continue
        if _is_cue(para[0]) and len(para) > 1:
            speaker = re.sub(r"\s*\(.*$", "", para[0]).strip()
            emotion, text = [], []
            for ln in para[1:]:
                if ln.startswith("(") and ln.endswith(")") and not text:
                    emotion.append(ln.strip("()"))
                elif ln.startswith("(") and ln.endswith(")"):
                    text.append("")  # mid-speech parenthetical: keep the line, drop the direction
                else:
                    text.append(ln)
            t = " ".join(x for x in text if x).strip()
            if t:
                blocks.append(RawBlock("dialogue", 0, 0, t, speaker=speaker, emotion=", ".join(emotion)))
        else:
            t = " ".join(para).strip()
            if t and not TRANSITION.match(t):
                blocks.append(RawBlock("action", 0, 0, t))
    return heading, blocks


def _dur(b: RawBlock) -> float:
    w = len(b.text.split())
    if b.kind == "dialogue":
        return round(max(1.0, min(14.0, w / DIALOGUE_WPS)), 2)
    return round(max(1.5, min(8.0, w / ACTION_WPS)), 2)


def chunk(scenes: list[tuple[str, list[RawBlock]]], target: float = 120.0, max_len: float = 170.0):
    """Greedy: pack whole scenes into episodes of ~target seconds; split scenes longer than max_len.
    Returns episodes and, per scene index, the list of (episode_no, start_s, end_s) it occupies."""
    episodes: list[RawEpisode] = []
    where: dict[int, list[tuple[int, float, float]]] = {}
    cur = RawEpisode(1)
    t = 0.0

    def close():
        nonlocal cur, t
        if any(sc.blocks for sc in cur.scenes):
            episodes.append(cur)
            cur = RawEpisode(len(episodes) + 1)
            t = 0.0

    for si, (heading, blocks) in enumerate(scenes):
        if not blocks:
            continue
        scene_len = sum(_dur(b) for b in blocks)
        if t > 0 and t + scene_len > max_len and t >= target * 0.5:
            close()
        sc = RawScene(heading)
        cur.scenes.append(sc)
        start_t = t
        for b in blocks:
            d = _dur(b)
            if t + d > max_len and t >= target * 0.5:
                where.setdefault(si, []).append((cur.n, start_t, t))
                close()
                sc = RawScene(heading + " (CONTINUED)")
                cur.scenes.append(sc)
                start_t = t
            sc.blocks.append(RawBlock(b.kind, round(t, 2), round(t + d, 2), b.text, b.speaker, b.emotion))
            t += d
        where.setdefault(si, []).append((cur.n, start_t, t))
        if t >= target:
            close()
    close()
    for ep in episodes:
        ep.scenes = [s for s in ep.scenes if s.blocks]
        heads = [re.sub(r"^(INT\.|EXT\.|INT/EXT\.|INT\./EXT\.|EXT\./INT\.)\s*", "", s.heading) for s in ep.scenes]
        names = []
        for s in ep.scenes:
            for b in s.blocks:
                if b.kind == "dialogue" and b.speaker.title() not in names:
                    names.append(b.speaker.title())
        ep.logline = (f"{', '.join(names[:4]) or 'Characters'} at {heads[0].title() if heads else 'unknown'}"
                      + (f", then {heads[-1].title()}" if len(heads) > 1 else "") + ".")
    return episodes, where


def tripod_gold(csv_path: str, movie: str, where: dict, n_scenes: int) -> dict:
    with open(csv_path, encoding="utf-8") as f:
        row = next((r for r in csv.DictReader(f) if r["movie_name"] == movie), None)
    if row is None:
        raise SystemExit(f"{movie!r} not in {csv_path}")
    moments = []
    for tp in ("tp1", "tp2", "tp3", "tp4", "tp5"):
        idxs = ast.literal_eval(row[tp])
        spans: dict[int, list[float]] = {}
        for si in idxs:
            for ep, a, b in where.get(si, []):
                s = spans.setdefault(ep, [a, b])
                s[0], s[1] = min(s[0], a), max(s[1], b)
        for ep, (a, b) in sorted(spans.items()):
            moments.append({"id": f"{tp}-E{ep}", "episode": ep, "start_s": round(a, 2), "end_s": round(b, 2),
                            "kind": "turning_point", "label": tp.upper(), "scenes": idxs})
    return {"show": movie, "source": "tripod-test (Papalampidi et al. 2019), scene-level TP gold",
            "n_scenes": n_scenes,
            "caveat": "Timing estimated from word counts. Turning points are story beats, not editor promo picks.",
            "moments": moments, "decoys": []}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--script", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episode-seconds", type=float, default=120.0)
    ap.add_argument("--tripod-gold", help="TRIPOD_screenplays_test.csv")
    ap.add_argument("--movie", help="movie_name in the TRIPOD csv")
    a = ap.parse_args()
    with open(a.script, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    raw_scenes = split_scenes(lines)
    scenes = [parse_scene(s) for s in raw_scenes]
    episodes, where = chunk(scenes, target=a.episode_seconds)
    write_show(episodes, a.out)
    info = {"episodes": len(episodes), "scenes": len(scenes),
            "minutes": round(sum(max(b.end for s in e.scenes for b in s.blocks) for e in episodes) / 60, 1),
            "dialogue_blocks": sum(b.kind == "dialogue" for e in episodes for s in e.scenes for b in s.blocks),
            "action_blocks": sum(b.kind == "action" for e in episodes for s in e.scenes for b in s.blocks)}
    if a.tripod_gold:
        gold = tripod_gold(a.tripod_gold, a.movie, where, len(scenes))
        with open(os.path.join(a.out, "gold.json"), "w", encoding="utf-8") as f:
            json.dump(gold, f, indent=1)
        info["gold_moments"] = len(gold["moments"])
    print(json.dumps(info))


if __name__ == "__main__":
    main()

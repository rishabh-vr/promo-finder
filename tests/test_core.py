"""Tests for the deterministic core (parser, clip builder, validator, ranking).
Uses a tiny synthetic screenplay in the same format, so no confidential show data is needed."""
import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from promo import pipeline  # noqa: E402
from promo.clips import build_clip, code_checks, hook_window, normalize_spec  # noqa: E402
from promo.parse import load_show, parse_episode, render_transcript  # noqa: E402


def _block(label, a, b, text):
    return f"**{label}:**\nStart: ({a}) | End: ({b})\n{text}\n"


def _episode(n, upper=False):
    sc = "SCENE HEADING: INT. HALL – DAY\nSCENE START: 00:00.00\nSCENE END: 01:00.00" if upper else \
        "Scene heading: INT. HALL – DAY\nScene start: 00:00.00\nScene end: 01:00.00"
    blocks = [
        _block("CAMERA", "00:00.00", "00:05.00", "Wide shot of the hall."),
        _block("ACTION", "00:01.00", "00:05.00", "Anna walks in."),
        _block("BEN (Cold)", "00:05.50", "00:08.00", '"You are not welcome here."'),
        _block("CAMERA", "00:05.00", "00:20.00", "Close on Anna."),
        _block("ANNA (Shocked)", "00:09.00", "00:19.50", '"Then who is the woman\nwearing my ring?"'),
        _block("SOUND FX", "00:20.00", "00:22.00", "A slap echoes."),
        _block("CAMERA", "00:20.00", "00:45.00", "Ben turns away."),
        _block("BEN (Quiet)", "00:30.00", "00:44.00", '"Ask your sister."'),
        _block("CAMERA", "00:45.00", "01:00.00", "Fade to black."),
    ]
    return textwrap.dedent(f"""\
        EPISODE NUMBER: Episode {n}
        EPISODE LOGLINE: Anna learns the truth in episode {n}.
        CONTINUITY NOTES: none

        --- SCRIPT START ---

        SCENE 01
        {sc}

        Actions and dialogue:

        """) + "\n".join(blocks) + "\n--- SCRIPT END ---\n"


@pytest.fixture()
def show(tmp_path):
    for n in (1, 2):
        (tmp_path / f"episode-00{n}-Episode {n}.txt").write_text(_episode(n, upper=(n == 2)), encoding="utf-8")
    eps = load_show(str(tmp_path))
    return eps, {e.ep: e for e in eps}


def test_parse_basic(show):
    eps, _ = show
    assert [e.ep for e in eps] == [1, 2]
    e = eps[0]
    assert len(e.blocks) == 9 and e.duration == 60.0
    anna = e.blocks[4]
    assert anna.kind == "dialogue" and anna.speaker == "ANNA" and anna.emotion == "Shocked"
    assert anna.text == "Then who is the woman wearing my ring?"  # multi-line joined, quotes stripped
    assert e.blocks[5].kind == "sound"


def test_parse_uppercase_scene_fields(show):
    eps, _ = show
    sc = eps[1].scenes[0]
    assert sc.heading.startswith("INT. HALL") and sc.start == 0 and sc.end == 60


def test_snap_never_cuts_a_line(show):
    _, E = show
    # start inside the shot that runs 05-20, end on the line that runs 30-44
    clip = build_clip(E, [{"episode": 1, "start_block": "B005", "end_block": "B008"}])
    s = clip.segments[0]
    assert s.start == 5.0  # widened to the enclosing shot start
    assert s.end == 45.0  # widened to the end of the shot containing the last line
    assert clip.valid


def test_length_rules(show):
    _, E = show
    short = build_clip(E, [{"episode": 1, "start_block": "B003", "end_block": "B003"}])
    assert not short.valid and "too short" in short.problems[0]
    assert build_clip(E, [{"episode": 1, "start_block": "B001", "end_block": "B009"}]).valid  # exactly 60s


def test_qualified_ids_and_cross_episode(show):
    _, E = show
    assert normalize_spec({"start_block": "E2-B003", "end_block": "E2-B005"})["episode"] == 2
    bad = build_clip(E, [{"start_block": "E1-B003", "end_block": "E2-B005"}])
    assert not bad.valid and any("spans two episodes" in p for p in bad.problems)
    stitched = build_clip(E, [{"start_block": "E1-B003", "end_block": "E1-B003"},  # snaps to shot 05-20
                              {"start_block": "E2-B008", "end_block": "E2-B008"}])
    assert stitched.valid and len(stitched.segments) == 2
    assert code_checks(E, stitched)["cross_episode"]


def test_overlap_and_unknown_ids(show):
    _, E = show
    ov = build_clip(E, [{"episode": 1, "start_block": "B001", "end_block": "B005"},
                        {"episode": 1, "start_block": "B004", "end_block": "B008"}])
    assert any("overlap" in p for p in ov.problems)
    assert not build_clip(E, [{"episode": 1, "start_block": "B999", "end_block": "B001"}]).valid


def test_hook_window_and_transcript(show):
    eps, E = show
    clip = build_clip(E, [{"episode": 1, "start_block": "B003", "end_block": "B008"}])
    assert any(b.kind == "dialogue" for b in hook_window(E, clip))
    t = render_transcript(eps[0], qualify=True)
    assert "[E1-B003 00:05.5-00:08.0] BEN (Cold)" in t


def test_ranking_qualifiers_first_and_episode_cap():
    def c(i, ep, h, k, s=4, sp=2):
        x = {"id": i, "episode": ep, "judge": {"hook": h, "cliffhanger": k, "standalone": s, "spoiler_risk": sp}}
        x["score"] = pipeline.composite(x)
        return x
    cands = [c("a", 1, 5, 2), c("b", 2, 3, 3), c("c", 1, 4, 4), c("d", 1, 4, 4, 5), c("e", 1, 3, 4)]
    order = [x["id"] for x in pipeline.order_clips(cands)]
    assert order[-1] == "a"  # high hook but no cliffhanger -> never above a qualifying clip
    assert order[:2] == ["d", "c"]
    assert order.index("e") > order.index("b")  # third clip from ep1 deferred by the per-episode cap


def test_composite_bounds():
    hi = pipeline.composite({"judge": {"hook": 5, "cliffhanger": 5, "standalone": 5, "spoiler_risk": 1}})
    lo = pipeline.composite({"judge": {"hook": 1, "cliffhanger": 1, "standalone": 1, "spoiler_risk": 5}})
    assert hi == 100 and lo == 0
    assert pipeline.composite({"judge": None}) is None


def test_tiers_and_audit_blend():
    def c(i, ep, judge, audit=None):
        x = {"id": i, "episode": ep, "judge": dict(zip(("hook", "cliffhanger", "standalone", "spoiler_risk"), judge))}
        if audit:
            x["audit"] = dict(zip(("hook", "cliffhanger", "standalone", "spoiler_risk"), audit))
        x["score"] = pipeline.composite(x)
        return x
    spoiler = c("s", 40, (5, 5, 5, 1), (5, 5, 5, 5))      # blind judge misses the spoiler; auditor catches it
    safe = c("a", 2, (4, 4, 4, 1), (4, 4, 4, 2))
    weak = c("w", 3, (5, 2, 5, 1), (5, 2, 5, 1))
    f = pipeline.final_scores(spoiler)
    assert f["spoiler_risk"] == 5 and f["hook"] == 5        # spoiler from auditor only
    assert pipeline.final_scores(c("m", 1, (4, 2, 4, 1), (2, 4, 4, 1)))["hook"] == 3  # mean of the two
    order = [x["id"] for x in pipeline.order_clips([spoiler, weak, safe])]
    assert order == ["a", "s", "w"]                           # acquisition > re-engagement > weak
    assert pipeline.TIERS[pipeline.tier(spoiler)].startswith("re-engagement")

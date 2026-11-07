"""Tests for the evaluation and data tools: statistics, anchors, gold scoring, converters, round-trips.
All synthetic, so no confidential data is needed."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from promo import goldeval, pipeline, stats  # noqa: E402
from promo.clips import build_clip, code_checks, is_salient  # noqa: E402
from promo.kukufmt import RawBlock, RawEpisode, RawScene, render_episode, write_show  # noqa: E402
from promo.parse import Block, load_show  # noqa: E402


# ------------------------------------------------------------------ stats
def test_sign_test_matches_known_values():
    assert stats.sign_test(23, 17) == 0.4296          # the saved run: not significant
    assert stats.sign_test(15, 2) < 0.01
    assert stats.sign_test(0, 0) is None
    assert stats.sign_test(5, 5) == 1.0


def test_wilson_interval_contains_point_and_is_bounded():
    lo, hi = stats.wilson(23, 40)
    assert lo < 23 / 40 < hi and 0 <= lo and hi <= 1
    assert stats.wilson(0, 10)[0] == 0.0
    assert stats.wilson(0, 0) is None


def test_min_pairs_for_significance():
    n = stats.min_pairs_for_significance(0.575)
    assert n is not None and n > 100                  # a 57% win rate needs a lot of pairs
    assert stats.min_pairs_for_significance(0.8) < 25


def test_kappa_quadratic():
    assert stats.kappa_quadratic([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert stats.kappa_quadratic([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) < 0
    assert stats.kappa_quadratic([], []) is None


def test_spearman_ties():
    assert stats.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert stats.spearman([1, 1, 2, 2], [2, 2, 1, 1]) == -1.0


# ------------------------------------------------------------------ kuku writer round-trip
def _raw_episode(n=1):
    sc = RawScene("INT. HALL - NIGHT", [
        RawBlock("camera", 0.0, 3.0, "Wide on the hall."),
        RawBlock("action", 0.0, 2.0, "Mia sits quietly."),
        RawBlock("action", 2.0, 4.0, "Rex slaps the glass from her hand."),
        RawBlock("dialogue", 4.0, 7.0, "You were never invited.", "REX", "Cold"),
        RawBlock("sound", 7.0, 8.0, "Glass shatters."),
        RawBlock("dialogue", 8.0, 30.0, "Then who sent me the letter with your seal on it?", "MIA", "Shaking"),
        RawBlock("dialogue", 30.0, 33.0, "Open the door.", "REX", ""),
        RawBlock("action", 33.0, 40.0, "The door handle turns."),
    ])
    return RawEpisode(n, "Mia is humiliated.", "none", [sc])


def test_kukufmt_round_trip(tmp_path):
    write_show([_raw_episode(1), _raw_episode(2)], str(tmp_path))
    eps = load_show(str(tmp_path))
    assert [e.ep for e in eps] == [1, 2]
    e = eps[0]
    assert len(e.blocks) == 8
    assert e.blocks[3].speaker == "REX" and e.blocks[3].emotion == "Cold"
    assert e.blocks[3].text == "You were never invited."
    assert abs(e.duration - 40.0) < 1e-6
    assert "**SOUND FX:**" in render_episode(_raw_episode())


# ------------------------------------------------------------------ anchors / hook timing
def _eps(tmp_path):
    write_show([_raw_episode(1)], str(tmp_path))
    return {e.ep: e for e in load_show(str(tmp_path))}


def test_salient_beats_ignore_quiet_actions():
    quiet = Block("B1", 1, 0, 1, "action", "", "", 0, 1, "Mia sits quietly on the bed.")
    loud = Block("B2", 1, 1, 1, "action", "", "", 0, 1, "He slams the door.")
    assert not is_salient(quiet) and is_salient(loud)


def test_hook_anchor_is_measured_not_claimed(tmp_path):
    E = _eps(tmp_path)
    clip = build_clip(E, [{"episode": 1, "start_block": "B002", "end_block": "B007"}])
    assert clip.valid
    ok = code_checks(E, clip, hook_block="E1-B003", cliff_block="E1-B007")
    assert ok["hook_offset_s"] == 2.0 and ok["hook_event_within_3s"] and ok["hook_anchor"] == "measured"
    assert ok["cliff_tail_s"] == 0.0
    late = code_checks(E, clip, hook_block="E1-B006")      # the named hook lands 8s in
    assert late["hook_offset_s"] == 8.0 and late["hook_event_within_3s"] is False
    # no anchor -> heuristic: the slap at 2s is the first salient beat (the quiet sit at 0s is not)
    heur = code_checks(E, clip)
    assert heur["salient_beat_s"] == 2.0 and heur["first_beat_s"] == 0.0 and heur["hook_anchor"].startswith("none")


def test_anchor_outside_clip_fails_hook(tmp_path):
    E = _eps(tmp_path)
    clip = build_clip(E, [{"episode": 1, "start_block": "B004", "end_block": "B007"}])
    chk = code_checks(E, clip, hook_block="B002")
    assert chk["hook_anchor"] == "outside clip" and chk["hook_event_within_3s"] is False


def test_qualifies_uses_hook_timing_and_cliff_tail():
    base = {"judge": {"hook": 4, "cliffhanger": 4, "standalone": 4, "spoiler_risk": 1}}
    assert pipeline.qualifies({**base, "checks": {"hook_event_within_3s": True, "cliff_tail_s": 1.0,
                                                   "cliff_anchor": "measured"}})
    assert not pipeline.qualifies({**base, "checks": {"hook_event_within_3s": False}})
    assert not pipeline.qualifies({**base, "checks": {"hook_event_within_3s": True, "cliff_tail_s": 12.0,
                                                       "cliff_anchor": "measured"}})
    # a missing cliff anchor is unverified, not failed; one named outside the clip fails
    assert pipeline.qualifies({**base, "checks": {"hook_event_within_3s": True, "cliff_anchor": "missing"}})
    assert not pipeline.qualifies({**base, "checks": {"hook_event_within_3s": True, "cliff_anchor": "outside clip"}})
    assert pipeline.qualifies(base)                          # legacy rows without checks still score


# ------------------------------------------------------------------ gold evaluation
def _row(rank, ep, a, b):
    return {"rank": rank, "segments": [{"episode": ep, "start_s": a, "end_s": b}]}


def test_goldeval_hits_hooks_cuts_and_decoys():
    gold = {"moments": [{"id": "g1", "episode": 1, "start_s": 10, "end_s": 40, "hook_s": 11, "cliff_s": 39.5},
                        {"id": "g2", "episode": 2, "start_s": 60, "end_s": 90}],
            "decoys": [{"id": "d1", "episode": 3, "start_s": 0, "end_s": 30}]}
    rows = [_row(1, 1, 9.0, 41.0),    # hits g1, hook within 3s, cut within window
            _row(2, 3, 2.0, 28.0),    # sits on the decoy
            _row(3, 4, 0.0, 30.0)]    # misses everything
    ev = goldeval.evaluate(rows, gold, {1: 120, 2: 120, 3: 120, 4: 120}, n_random=50)
    a = ev["at"]["all"]
    assert a["recall"] == 0.5 and a["precision"] == round(1 / 3, 3)
    assert a["hook_ok"] == 1.0 and a["cut_ok"] == 1.0 and a["decoy_rate"] == round(1 / 3, 3)
    assert a["random_precision"] is not None and 0 < a["p_precision_vs_random"] <= 1


def test_random_null_matches_duration_and_respects_episode_bounds():
    import random
    sampled = goldeval._random_like([[(1, 10.0, 35.0)]], {1: 30.0, 2: 60.0}, random.Random(7))
    assert sampled is not None
    ep, start, end = sampled[0][0]
    assert abs((end - start) - 25.0) < 1e-9
    assert 0.0 <= start <= end <= {1: 30.0, 2: 60.0}[ep]
    assert goldeval._random_like([[(1, 0.0, 40.0)]], {1: 30.0, 2: 35.0}, random.Random(7)) is None


def test_goldeval_marks_unmatched_random_null_unavailable():
    gold = {"moments": [{"id": "g1", "episode": 1, "start_s": 0, "end_s": 30}], "decoys": []}
    ev = goldeval.evaluate([_row(1, 1, 0, 30)], gold, {1: 20}, n_random=10)
    top = ev["at"]["top5"]
    assert top["random_comparable"] is False
    assert top["random_precision"] is None and top["p_precision_vs_random"] is None


def test_baseline_mode_ranks_by_rule_score(tmp_path):
    write_show([_raw_episode(n) for n in (1, 2, 3)], str(tmp_path / "show"))
    rows = pipeline.run(str(tmp_path / "show"), str(tmp_path / "out"), "baseline", None, None, cb=lambda m: None)
    ranked = sorted(rows, key=lambda r: r["rank"])
    scores = [float(r["why"].split()[-1]) for r in ranked]
    assert scores[0] == max(scores)


# ------------------------------------------------------------------ converters
def test_srt_converter(tmp_path):
    import convert_srt
    srt = ("1\n00:00:01,000 --> 00:00:03,000\nANNA: You lied to me!\n\n"
           "2\n00:00:03,500 --> 00:00:05,000\n[door slams]\n\n"
           "3\n00:00:20,000 --> 00:00:24,000\n- Who are you?\n- Your husband's wife.\n")
    cues = convert_srt.parse_subs(srt)
    blocks = [b for s, e, ls in cues for b in convert_srt.cue_blocks(s, e, ls)]
    kinds = [(b.kind, b.speaker) for b in blocks]
    assert kinds == [("dialogue", "ANNA"), ("sound", ""), ("dialogue", "SPEAKER"), ("dialogue", "SPEAKER")]
    ep = convert_srt.to_episode(1, blocks, scene_gap=5.0, title="t")
    assert len(ep.scenes) == 2 and ep.scenes[0].blocks[0].start == 0.0
    write_show([ep], str(tmp_path))
    assert len(load_show(str(tmp_path))[0].blocks) == 4


def test_screenplay_converter_and_tripod_alignment(tmp_path):
    import convert_screenplay as cs
    script = """FADE IN:

INT. KITCHEN - NIGHT 1

Anna drops the plate. It shatters.

                    ANNA
          (whispering)
     Who told you?

                    BEN (V.O.)
     Your sister did.

EXT. STREET - DAY 2

Ben runs.

INT. CAR - NIGHT 3

                    ANNA
     Drive.
"""
    scenes = cs.split_scenes(script.splitlines())
    assert len(scenes) == 3                      # TRIPOD-compatible: one per INT./EXT. line
    heading, blocks = cs.parse_scene(scenes[0])
    assert heading == "INT. KITCHEN - NIGHT"
    assert [(b.kind, b.speaker, b.emotion) for b in blocks] == [
        ("action", "", ""), ("dialogue", "ANNA", "whispering"), ("dialogue", "BEN", "")]
    eps, where = cs.chunk([cs.parse_scene(s) for s in scenes], target=120)
    assert set(where) == {0, 1, 2} and len(eps) == 1
    csv_path = tmp_path / "gold.csv"
    csv_path.write_text('movie_name,tp1,tp2,tp3,tp4,tp5\nX,[0],[1],[2],[2],[2]\n', encoding="utf-8")
    gold = cs.tripod_gold(str(csv_path), "X", where, 3)
    assert len(gold["moments"]) == 5 and gold["moments"][0]["start_s"] == 0.0
    json.dumps(gold)


def test_synth_render_assigns_times_and_tags():
    import synth_show
    data = {"scenes": [{"heading": "INT. GALA - NIGHT", "blocks": [
        {"kind": "camera", "text": "Wide on the gala.", "seconds": 3},
        {"kind": "action", "text": "She slaps him.", "seconds": 2, "tag": "promo_start+hook"},
        {"kind": "dialogue", "speaker": "LIA", "emotion": "Cold", "text": "You sold my mother's ring.", "seconds": 14},
        {"kind": "dialogue", "speaker": "MAX", "text": "Then who is wearing it?", "seconds": 5, "tag": "cliff+promo_end"},
        {"kind": "action", "text": "They embrace and agree to talk.", "seconds": 18, "tag": "decoy_start+decoy_end"},
    ]}]}
    ep, tags = synth_show.render(1, "log", "cont", data)
    assert tags["hook"].start == 0.0 and tags["cliff"].end == 21.0
    moments, decoys, problems = synth_show.gold_for(1, tags)
    assert moments[0]["start_s"] == 0.0 and moments[0]["end_s"] == 21.0 and not problems
    assert decoys[0]["start_s"] == 21.0


def test_gold_sanitize_flags_inconsistent_labels():
    g = goldeval.sanitize({"moments": [{"id": "m", "episode": 1, "start_s": 0, "end_s": 41, "hook_s": 0, "cliff_s": 92}],
                           "decoys": [{"id": "d", "episode": 1, "start_s": 30, "end_s": 60}]})
    assert g["moments"][0]["cliff_s"] is None and g["moments"][0]["hook_s"] == 0
    assert g["decoys"] == [] and len(g["label_warnings"]) == 2


class _FakeJudge:
    """Always answers 'A': pure position bias, no preference at all."""
    model = "fake:always-a"

    def chat_json(self, system, user, temperature=0.0):
        import re
        return {"results": [{"pair": p, "winner": "A", "reason": "first"} for p in re.findall(r"##### PAIR (\S+)", user)]}


def test_pairwise_both_orders_cancels_position_bias(tmp_path):
    from promo.evaluate import pairwise
    write_show([_raw_episode(n) for n in (1, 2, 3, 4)], str(tmp_path))
    E = {e.ep: e for e in load_show(str(tmp_path))}
    seg = lambda ep, a, b: [{"episode": ep, "start": "", "end": "", "blocks": f"{a}-{b}"}]
    llm_rows = [{"rank": i, "episode": i, "id": f"L{i}", "segments": seg(i, "B002", "B007")} for i in (1, 2, 3, 4)]
    base_rows = [{"rank": i, "episode": i, "id": f"R{i}", "segments": seg(i, "B004", "B008")} for i in (1, 2, 3, 4)]
    single = pairwise(llm_rows, base_rows, E, _FakeJudge(), cb=lambda m: None, both_orders=False)
    assert single["tally"]["tie"] == 0 and single["tally"]["llm"] + single["tally"]["baseline"] == 4
    both = pairwise(llm_rows, base_rows, E, _FakeJudge(), cb=lambda m: None, both_orders=True)
    assert both["tally"] == {"llm": 0, "baseline": 0, "tie": 4} and both["order_inconsistent"] == 4
    assert both["position_bias_p"] < 0.01


def test_saved_legacy_rows_keep_their_qualification():
    """Regression: rows saved before anchors existed have `checks` but no anchor fields; they must
    not all flip to non-qualifying (this happened once: 29 -> 0 on the saved Kuku run)."""
    row = {"judge": {"hook": 4, "cliffhanger": 5, "standalone": 4, "spoiler_risk": 1},
           "checks": {"first_beat_s": 0.0, "hook_has_line": True, "ends_at_episode_end": False}}
    assert pipeline.qualifies(row)

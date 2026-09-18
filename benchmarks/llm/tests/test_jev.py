from __future__ import annotations

import pytest

from benchmark import jev
from benchmark.truth_parser import Ad


def seg(sid: int, start: float, end: float, text: str = "words") -> dict:
    return {"sid": sid, "start": start, "end": end, "text": text}


def contiguous(n: int, *, length: float = 10.0, start: float = 0.0) -> list[dict]:
    return [seg(i, start + i * length, start + (i + 1) * length) for i in range(n)]


class TestSpansFromProbabilities:
    def test_empty_when_nothing_clears_enter(self):
        segs = contiguous(5)
        probs = {i: 0.5 for i in range(5)}
        assert jev.spans_from_probabilities(segs, probs) == []

    def test_single_run_spans_segment_edges(self):
        segs = contiguous(5)
        probs = {0: 0.0, 1: 0.9, 2: 0.9, 3: 0.0, 4: 0.0}
        (ad,) = jev.spans_from_probabilities(segs, probs)
        assert (ad["start"], ad["end"]) == (10.0, 30.0)
        assert (ad["start_id"], ad["end_id"]) == (1, 2)

    def test_boundaries_are_never_interpolated(self):
        segs = [seg(0, 0.0, 7.3), seg(1, 7.3, 21.9), seg(2, 21.9, 30.0)]
        probs = {0: 0.0, 1: 0.95, 2: 0.0}
        (ad,) = jev.spans_from_probabilities(segs, probs)
        assert ad["start"] == 7.3
        assert ad["end"] == 21.9

    def test_hysteresis_bridges_a_weak_middle_segment(self):
        segs = contiguous(5)
        probs = {0: 0.0, 1: 0.9, 2: 0.45, 3: 0.9, 4: 0.0}
        spans = jev.spans_from_probabilities(segs, probs)
        assert len(spans) == 1
        assert (spans[0]["start"], spans[0]["end"]) == (10.0, 40.0)

    def test_run_of_only_weak_segments_is_not_an_ad(self):
        segs = contiguous(4)
        probs = {i: 0.45 for i in range(4)}
        assert jev.spans_from_probabilities(segs, probs) == []

    def test_confidence_is_the_run_maximum(self):
        segs = contiguous(3)
        probs = {0: 0.62, 1: 0.97, 2: 0.55}
        (ad,) = jev.spans_from_probabilities(segs, probs)
        assert ad["confidence"] == pytest.approx(0.97)

    def test_run_breaks_across_a_wide_silence_gap(self):
        # Adjacent in the list, 40s apart in time: two breaks, not one.
        segs = [seg(0, 0.0, 30.0), seg(1, 70.0, 100.0)]
        probs = {0: 0.9, 1: 0.9}
        spans = jev.spans_from_probabilities(segs, probs)
        assert [(s["start"], s["end"]) for s in spans] == [(0.0, 30.0), (70.0, 100.0)]

    def test_run_survives_a_narrow_silence_gap(self):
        segs = [seg(0, 0.0, 30.0), seg(1, 35.0, 60.0)]
        probs = {0: 0.9, 1: 0.9}
        (ad,) = jev.spans_from_probabilities(segs, probs)
        assert (ad["start"], ad["end"]) == (0.0, 60.0)

    def test_missing_probability_reads_as_not_an_ad(self):
        segs = contiguous(3)
        (ad,) = jev.spans_from_probabilities(segs, {1: 0.9})
        assert (ad["start_id"], ad["end_id"]) == (1, 1)

    def test_unsorted_input_is_ordered_by_time(self):
        segs = list(reversed(contiguous(4)))
        probs = {1: 0.9, 2: 0.9}
        (ad,) = jev.spans_from_probabilities(segs, probs)
        assert (ad["start"], ad["end"]) == (10.0, 30.0)

    def test_stay_above_enter_is_rejected(self):
        with pytest.raises(ValueError):
            jev.spans_from_probabilities(contiguous(2), {}, enter=0.4, stay=0.8)


class TestOracleProbabilities:
    def test_overlap_marks_any_intersection(self):
        segs = [seg(0, 0.0, 20.0), seg(1, 20.0, 40.0)]
        probs = jev.oracle_probabilities(segs, [Ad(18.0, 25.0, "x")], policy="overlap")
        assert probs == {0: 1.0, 1: 1.0}

    def test_majority_requires_more_than_half_the_segment(self):
        segs = [seg(0, 0.0, 20.0), seg(1, 20.0, 40.0)]
        probs = jev.oracle_probabilities(segs, [Ad(18.0, 35.0, "x")], policy="majority")
        assert probs == {0: 0.0, 1: 1.0}

    def test_no_truth_ads_marks_nothing(self):
        probs = jev.oracle_probabilities(contiguous(3), [], policy="overlap")
        assert set(probs.values()) == {0.0}

    def test_unknown_policy_is_rejected(self):
        with pytest.raises(ValueError):
            jev.oracle_probabilities(contiguous(1), [], policy="nonsense")


class TestPayload:
    def test_criteria_is_not_repeated_per_question(self):
        payload = jev.build_payload(contiguous(3))
        assert all("criteria" not in q for q in payload["questions"].values())
        assert "guidance" in payload["state"]

    def test_instructions_name_the_line_id(self):
        payload = jev.build_payload([seg(42, 0.0, 1.0)])
        assert "L0042" in payload["questions"]["s42"]["instructions"]

    def test_state_lines_carry_ids(self):
        state = jev.build_state([seg(7, 0.0, 1.0, "hello there")])
        assert state == "L0007| hello there"

    def test_uid_makes_a_repeat_a_distinct_draw(self):
        a = jev.build_payload(contiguous(2), uid="pass-1")
        b = jev.build_payload(contiguous(2), uid="pass-2")
        assert a["state"]["uid"] != b["state"]["uid"]
        assert a["questions"] == b["questions"]

    def test_no_uid_key_when_not_requested(self):
        assert "uid" not in jev.build_payload(contiguous(2))["state"]


class TestParseResponse:
    def test_reads_noul_probabilities_and_usage(self):
        body = {
            "answers": {
                "s0": {"type": "noul", "noul": 0.91},
                "s1": {"type": "noul", "noul": 0.02},
            },
            "usage": {"input_tokens": 1234, "output_tokens": 0},
        }
        result = jev.parse_response(body)
        assert result.probabilities == {0: 0.91, 1: 0.02}
        assert result.input_tokens == 1234

    def test_ignores_answers_outside_the_segment_namespace(self):
        body = {"answers": {"s3": {"noul": 0.5}, "summary": {"choice": "x"}}}
        assert jev.parse_response(body).probabilities == {3: 0.5}

    def test_empty_body_is_not_an_error(self):
        assert jev.parse_response({}).probabilities == {}

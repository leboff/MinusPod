from __future__ import annotations

import pytest

from benchmark import jev
from benchmark.truth_parser import Ad


def seg(sid: int, start: float, end: float, text: str = "words") -> dict:
    return {"sid": sid, "start": start, "end": end, "text": text}


def contiguous(n: int, *, length: float = 10.0, start: float = 0.0) -> list[dict]:
    return [seg(i, start + i * length, start + (i + 1) * length) for i in range(n)]


# Run recovery is tested against fixed thresholds, not the tuned defaults:
# ENTER/STAY are fitted to corpus data and are expected to move.
ENTER, STAY = 0.60, 0.40


def spans(segments, probabilities, **kw):
    kw.setdefault('enter', ENTER)
    kw.setdefault('stay', STAY)
    return jev.spans_from_probabilities(segments, probabilities, **kw)


class TestSpansFromProbabilities:
    def test_empty_when_nothing_clears_enter(self):
        segs = contiguous(5)
        probs = {i: 0.5 for i in range(5)}
        assert spans(segs, probs) == []

    def test_single_run_spans_segment_edges(self):
        segs = contiguous(5)
        probs = {0: 0.0, 1: 0.9, 2: 0.9, 3: 0.0, 4: 0.0}
        (ad,) = spans(segs, probs)
        assert (ad["start"], ad["end"]) == (10.0, 30.0)
        assert (ad["start_id"], ad["end_id"]) == (1, 2)

    def test_boundaries_are_never_interpolated(self):
        segs = [seg(0, 0.0, 7.3), seg(1, 7.3, 21.9), seg(2, 21.9, 30.0)]
        probs = {0: 0.0, 1: 0.95, 2: 0.0}
        (ad,) = spans(segs, probs)
        assert ad["start"] == 7.3
        assert ad["end"] == 21.9

    def test_hysteresis_bridges_a_weak_middle_segment(self):
        segs = contiguous(5)
        probs = {0: 0.0, 1: 0.9, 2: 0.45, 3: 0.9, 4: 0.0}
        recovered = spans(segs, probs)
        assert len(recovered) == 1
        assert (recovered[0]["start"], recovered[0]["end"]) == (10.0, 40.0)

    def test_run_of_only_weak_segments_is_not_an_ad(self):
        segs = contiguous(4)
        probs = {i: 0.45 for i in range(4)}
        assert spans(segs, probs) == []

    def test_confidence_is_the_run_maximum(self):
        segs = contiguous(3)
        probs = {0: 0.62, 1: 0.97, 2: 0.55}
        (ad,) = spans(segs, probs)
        assert ad["confidence"] == pytest.approx(0.97)

    def test_run_breaks_across_a_wide_silence_gap(self):
        # Adjacent in the list, 40s apart in time: two breaks, not one.
        segs = [seg(0, 0.0, 30.0), seg(1, 70.0, 100.0)]
        probs = {0: 0.9, 1: 0.9}
        recovered = spans(segs, probs)
        assert [(s["start"], s["end"]) for s in recovered] == [(0.0, 30.0), (70.0, 100.0)]

    def test_run_survives_a_narrow_silence_gap(self):
        segs = [seg(0, 0.0, 30.0), seg(1, 35.0, 60.0)]
        probs = {0: 0.9, 1: 0.9}
        (ad,) = spans(segs, probs)
        assert (ad["start"], ad["end"]) == (0.0, 60.0)

    def test_missing_probability_reads_as_not_an_ad(self):
        segs = contiguous(3)
        (ad,) = spans(segs, {1: 0.9})
        assert (ad["start_id"], ad["end_id"]) == (1, 1)

    def test_unsorted_input_is_ordered_by_time(self):
        segs = list(reversed(contiguous(4)))
        probs = {1: 0.9, 2: 0.9}
        (ad,) = spans(segs, probs)
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


class TestConfirmPolicy:
    def test_accepts_a_clear_ad(self):
        assert jev.ConfirmPolicy().accepts(
            {"is_ad": 0.98, "promotional_language": 0.98})

    def test_rejects_below_is_ad(self):
        assert not jev.ConfirmPolicy().accepts(
            {"is_ad": 0.1, "promotional_language": 0.98})

    def test_rejects_without_promotional_language(self):
        assert not jev.ConfirmPolicy().accepts(
            {"is_ad": 0.98, "promotional_language": 0.05})

    def test_either_exclusion_signal_rejects(self):
        base = {"is_ad": 0.98, "promotional_language": 0.98}
        assert not jev.ConfirmPolicy().accepts({**base, "guest_own_work": 0.9})
        assert not jev.ConfirmPolicy().accepts({**base, "host_organic": 0.9})

    def test_missing_signals_read_as_zero(self):
        assert not jev.ConfirmPolicy().accepts({})

    def test_bounds_are_independent(self):
        answers = {"is_ad": 0.6, "promotional_language": 0.98}
        assert jev.ConfirmPolicy(min_is_ad=0.5).accepts(answers)
        assert not jev.ConfirmPolicy(min_is_ad=0.7).accepts(answers)


class TestConfirmSpans:
    def test_span_members_are_selected_by_time_not_id(self):
        segs = contiguous(6)
        # A merged span carries a stale end_id; time must still bound it.
        ad = {"start": 10.0, "end": 40.0, "start_id": 1, "end_id": 1}
        seen = {}

        def source(payload):
            seen["span"] = payload["state"]["span"]
            return {"is_ad": 0.99, "promotional_language": 0.99}

        kept = jev.confirm_spans([ad], segs, source, policy=jev.ConfirmPolicy())
        assert len(kept) == 1
        assert seen["span"].split().count("words") == 3

    def test_rejected_span_is_dropped(self):
        segs = contiguous(3)
        ad = {"start": 0.0, "end": 10.0, "start_id": 0, "end_id": 0}
        kept = jev.confirm_spans(
            [ad], segs, lambda p: {"is_ad": 0.01}, policy=jev.ConfirmPolicy())
        assert kept == []

    def test_answers_are_attached_to_survivors(self):
        segs = contiguous(3)
        ad = {"start": 0.0, "end": 10.0, "start_id": 0, "end_id": 0}
        answers = {"is_ad": 0.99, "promotional_language": 0.99}
        (kept,) = jev.confirm_spans(
            [ad], segs, lambda p: answers, policy=jev.ConfirmPolicy())
        assert kept["confirm"] == answers

    def test_payload_carries_surrounding_context(self):
        segs = contiguous(6)
        payload = jev.build_confirm_payload([segs[2]], segs)
        assert payload["state"]["content_before"]
        assert payload["state"]["content_after"]
        assert set(payload["questions"]) == set(jev.CONFIRM_QUESTIONS)

    def test_span_at_episode_start_has_empty_before(self):
        segs = contiguous(4)
        payload = jev.build_confirm_payload([segs[0]], segs)
        assert payload["state"]["content_before"] == ""


class TestAggregatePasses:
    def test_mean_across_passes(self):
        a = jev.WindowResult({0: 1.0, 1: 0.0})
        b = jev.WindowResult({0: 0.0, 1: 0.0})
        assert jev.aggregate_passes([a, b]).probabilities == {0: 0.5, 1: 0.0}

    def test_a_contested_segment_falls_below_enter(self):
        # One pass says yes, one says no: the mean must not open a run.
        agreed = jev.aggregate_passes(
            [jev.WindowResult({0: 0.98}), jev.WindowResult({0: 0.02})])
        assert agreed.probabilities[0] < jev.ENTER_THRESHOLD

    def test_agreement_survives_averaging(self):
        agreed = jev.aggregate_passes(
            [jev.WindowResult({0: 0.98}), jev.WindowResult({0: 0.98})])
        assert agreed.probabilities[0] >= jev.ENTER_THRESHOLD

    def test_missing_segment_counts_as_zero(self):
        out = jev.aggregate_passes(
            [jev.WindowResult({0: 1.0}), jev.WindowResult({})])
        assert out.probabilities == {0: 0.5}

    def test_tokens_sum_across_passes(self):
        out = jev.aggregate_passes([
            jev.WindowResult({}, input_tokens=100),
            jev.WindowResult({}, input_tokens=150),
        ])
        assert out.input_tokens == 250

    def test_empty_is_not_an_error(self):
        assert jev.aggregate_passes([]).probabilities == {}


class TestPassSpread:
    def test_spread_is_max_minus_min(self):
        spread = jev.pass_spread(
            [jev.WindowResult({0: 0.9, 1: 0.5}), jev.WindowResult({0: 0.3, 1: 0.5})])
        assert spread == {0: pytest.approx(0.6), 1: pytest.approx(0.0)}

    def test_single_pass_has_no_spread(self):
        assert jev.pass_spread([jev.WindowResult({0: 0.9})]) == {}


class TestProbabilityCache:
    def test_round_trips_through_disk(self, tmp_path):
        segs = contiguous(2)
        cache = jev.ProbabilityCache(tmp_path / "c.json")
        key = jev.payload_key(segs)
        cache._data[key] = {"probabilities": {"0": 0.9}, "input_tokens": 5}
        cache.save()

        reloaded = jev.ProbabilityCache(tmp_path / "c.json")
        result = reloaded.get_or_call(segs, api_key=None)
        assert result.probabilities == {0: 0.9}
        assert reloaded.hits == 1

    def test_miss_without_a_key_raises_rather_than_scoring_zeros(self, tmp_path):
        cache = jev.ProbabilityCache(tmp_path / "c.json")
        with pytest.raises(KeyError):
            cache.get_or_call(contiguous(2), api_key=None)

    def test_changing_a_question_invalidates_the_key(self, tmp_path, monkeypatch):
        segs = contiguous(2)
        before = jev.payload_key(segs)
        monkeypatch.setattr(jev, "GUIDANCE", "different definition")
        assert jev.payload_key(segs) != before

    def test_distinct_uids_are_distinct_draws(self):
        segs = contiguous(2)
        assert jev.payload_key(segs, uid="pass-0") != jev.payload_key(segs, uid="pass-1")

    def test_missing_file_starts_empty(self, tmp_path):
        assert jev.ProbabilityCache(tmp_path / "absent.json")._data == {}


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

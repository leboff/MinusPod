"""Pass-A spike: per-segment ad-ness judgments via TypeSafe Jev.

Asks one Noul per transcript segment ("is this line advertising?") over a
window of transcript state, then recovers ad spans from the resulting
probability curve by taking contiguous runs. Every boundary is a real
segment edge, so no timestamp is ever generated.

Scoring reuses ``metrics`` so numbers are comparable to the chat-model rows
in ``results/report.md``.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import requests

from . import metrics
from .corpus import Episode, stamp_id_windows
from .truth_parser import Ad

API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

# $ per million input tokens (docs.typesafe.ai/models). Output is not billed.
INPUT_COST_PER_MTOK = 0.042

# A run must enter above ENTER and may continue while above STAY. Ad breaks
# here span 3-5 segments and a mid-break segment often reads weaker than its
# neighbours (station ident, a beat of silence), which a single threshold
# would split into two spans.
# Swept against the cached corpus. The signal is strongly bimodal -- 39% of
# segments come back at 0.02 and the top bucket is 0.98 -- so a run should
# only open on near-certainty, then extend generously across the weaker
# shoulders of the same break. Loosening `enter` to 0.6 costs 27 points of
# precision (0.929 -> 0.632) for 5 points of recall. Jev reports two decimal
# places and tops out at 0.99, so anything above 0.99 matches nothing at all.
ENTER_THRESHOLD = 0.98
STAY_THRESHOLD = 0.50

# A run breaks across a silence gap wider than this. Swept against the oracle
# over the corpus: precision climbs to 1.000 at 30s and is flat from there to
# unbounded, because speech between two breaks produces its own low-scoring
# segments and ends the run without help. Only a pure-silence gap can bridge
# two breaks, so this is the smallest value that costs nothing.
MAX_RUN_GAP_SECONDS = 30.0


def line_id(sid: int) -> str:
    return f"L{sid:04d}"


# State is sent once per request; per-question text is sent once per segment.
# Defining "advertising" here rather than in each question's criteria is what
# keeps a 300-segment window affordable.
GUIDANCE = (
    "Each line of `transcript` is one segment of a podcast episode, prefixed "
    "with its line id. A line is ADVERTISING when it is a sponsor read, a "
    "produced ad spot, a dynamically inserted ad, a hosting-platform pre-roll "
    "or post-roll, a cross-promotion for another show, or a produced segment "
    "asking listeners to subscribe, rate, or follow. Signs of advertising: a "
    "sponsor or brand name, a URL, a promo code, a product pitch, a call to "
    "action, or concentrated marketing copy that is tonally separate from the "
    "conversation. A line is EDITORIAL CONTENT when it is the host or a guest "
    "discussing the episode's subject. A guest talking about their own book or "
    "project, and the host mentioning their own show or Patreon in passing "
    "during conversation, are editorial content, not advertising."
)

NOUL_INSTRUCTIONS = "Line {lid} of `transcript` is advertising, not editorial content."


def build_state(segments: Sequence[dict]) -> str:
    """Window transcript as ID-prefixed lines, the semantic_find shape."""
    return "\n".join(
        f"{line_id(seg['sid'])}| {seg.get('text', '').strip()}" for seg in segments
    )


def build_questions(segments: Sequence[dict]) -> dict[str, dict]:
    """One Noul per segment. Question keys are internal, so the line id is
    repeated inside ``instructions`` where the model can actually see it.

    No per-question ``criteria``: the definition lives in the shared state,
    so adding a segment costs one short sentence rather than a rubric.
    """
    return {
        f"s{seg['sid']}": {
            "type": "noul",
            "instructions": NOUL_INSTRUCTIONS.format(lid=line_id(seg["sid"])),
        }
        for seg in segments
    }


def build_payload(segments: Sequence[dict], *, model: str = DEFAULT_MODEL,
                  uid: str | None = None) -> dict:
    """One request covering a whole window.

    ``uid`` makes an otherwise identical repeat a distinct draw, which is what
    multi-pass self-consistency needs.
    """
    state: dict[str, object] = {
        "guidance": GUIDANCE,
        "transcript": build_state(segments),
    }
    if uid is not None:
        state["uid"] = uid
    return {
        "state": state,
        "model": model,
        "questions": build_questions(segments),
    }


@dataclass
class WindowResult:
    probabilities: dict[int, float]
    input_tokens: int = 0
    output_tokens: int = 0


def parse_response(body: dict) -> WindowResult:
    answers = body.get("answers") or {}
    probs: dict[int, float] = {}
    for key, ans in answers.items():
        if not key.startswith("s"):
            continue
        value = ans.get("noul") if isinstance(ans, dict) else None
        if isinstance(value, (int, float)):
            probs[int(key[1:])] = float(value)
    usage = body.get("usage") or {}
    return WindowResult(
        probabilities=probs,
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
    )


def call_window(segments: Sequence[dict], *, api_key: str,
                model: str = DEFAULT_MODEL, uid: str | None = None,
                timeout: float = 60.0) -> WindowResult:
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json=build_payload(segments, model=model, uid=uid),
        timeout=timeout,
    )
    resp.raise_for_status()
    return parse_response(resp.json())


def payload_key(segments: Sequence[dict], *, model: str = DEFAULT_MODEL,
                uid: str | None = None) -> str:
    """Cache key covering everything that would change the answer."""
    blob = json.dumps(build_payload(segments, model=model, uid=uid),
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class ProbabilityCache:
    """Disk cache of per-window probabilities.

    Thresholds are tuned by re-reading these, not by re-asking: a sweep over
    a cached corpus costs nothing, so the only spend is the first pass.
    Keyed by payload hash, so editing a question invalidates its entries
    rather than silently scoring stale answers.
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, dict] = {}
        if path.is_file():
            self._data = json.loads(path.read_text())
        self.hits = 0
        self.misses = 0

    def get_or_call(self, segments: Sequence[dict], *, api_key: str | None,
                    model: str = DEFAULT_MODEL,
                    uid: str | None = None) -> WindowResult:
        key = payload_key(segments, model=model, uid=uid)
        entry = self._data.get(key)
        if entry is not None:
            self.hits += 1
            return WindowResult(
                probabilities={int(k): v for k, v in entry["probabilities"].items()},
                input_tokens=entry.get("input_tokens", 0),
                output_tokens=entry.get("output_tokens", 0),
            )
        if api_key is None:
            raise KeyError(
                f"no cached probabilities for window {key} and no API key to fetch them")
        result = call_window(segments, api_key=api_key, model=model, uid=uid)
        self.misses += 1
        self._data[key] = {
            "probabilities": {str(k): v for k, v in result.probabilities.items()},
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }
        return result

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=0, sort_keys=True))


# --- probability curve -> spans ------------------------------------------

def spans_from_probabilities(
    segments: Sequence[dict],
    probabilities: dict[int, float],
    *,
    enter: float = ENTER_THRESHOLD,
    stay: float = STAY_THRESHOLD,
    max_gap: float = MAX_RUN_GAP_SECONDS,
) -> list[dict]:
    """Contiguous runs of ad-ish segments, as ad dicts.

    A run opens on a segment at or above ``enter`` and extends across
    neighbours at or above ``stay``. Boundaries are segment edges, never
    interpolated.

    Adjacency is in time, not in list position: VAD drops silence, so two
    consecutive segment dicts can sit half a minute apart and belong to
    different ad breaks. A run breaks across a gap wider than ``max_gap``.
    """
    if stay > enter:
        raise ValueError("stay threshold must not exceed enter threshold")

    ordered = sorted(segments, key=lambda s: s["start"])
    above_stay = [probabilities.get(s["sid"], 0.0) >= stay for s in ordered]
    opens = [probabilities.get(s["sid"], 0.0) >= enter for s in ordered]

    ads: list[dict] = []
    i = 0
    while i < len(ordered):
        if not above_stay[i]:
            i += 1
            continue
        j = i
        while (j + 1 < len(ordered) and above_stay[j + 1]
               and ordered[j + 1]["start"] - ordered[j]["end"] <= max_gap):
            j += 1
        # A run of merely-above-stay segments with no confident member is not
        # an ad; it is the tail of an ordinary conversation.
        if any(opens[k] for k in range(i, j + 1)):
            members = ordered[i:j + 1]
            ads.append({
                "start": members[0]["start"],
                "end": members[-1]["end"],
                "confidence": max(
                    probabilities.get(m["sid"], 0.0) for m in members),
                "start_id": members[0]["sid"],
                "end_id": members[-1]["sid"],
            })
        i = j + 1
    return ads


# --- oracle --------------------------------------------------------------

def oracle_probabilities(
    segments: Sequence[dict],
    truth_ads: Sequence[Ad],
    *,
    policy: str = "overlap",
) -> dict[int, float]:
    """Per-segment probabilities a perfect judge would return.

    ``overlap``  - any intersection with a truth ad marks the segment.
    ``majority`` - more than half the segment's duration must fall inside one.

    The gap between the two is the cost of segment granularity: with 26s
    segments, ``overlap`` over-extends spans and ``majority`` clips them.
    """
    if policy not in ("overlap", "majority"):
        raise ValueError(f"unknown oracle policy: {policy}")

    probs: dict[int, float] = {}
    for seg in segments:
        duration = max(0.0, seg["end"] - seg["start"])
        covered = 0.0
        for ad in truth_ads:
            covered += max(
                0.0, min(seg["end"], ad.end) - max(seg["start"], ad.start))
        if policy == "overlap":
            hit = covered > 0
        else:
            hit = duration > 0 and covered / duration > 0.5
        probs[seg["sid"]] = 1.0 if hit else 0.0
    return probs


# --- episode scoring -----------------------------------------------------

def aggregate_passes(results: Sequence[WindowResult]) -> WindowResult:
    """Mean probability per segment across independent passes.

    Averaging is the point of repeating: a segment both passes agree on keeps
    its value, while one they split on lands mid-scale and falls below the
    enter threshold instead of opening a run on a coin flip.
    """
    if not results:
        return WindowResult(probabilities={})
    sids = {sid for r in results for sid in r.probabilities}
    return WindowResult(
        probabilities={
            sid: sum(r.probabilities.get(sid, 0.0) for r in results) / len(results)
            for sid in sids
        },
        input_tokens=sum(r.input_tokens for r in results),
        output_tokens=sum(r.output_tokens for r in results),
    )


def pass_spread(results: Sequence[WindowResult]) -> dict[int, float]:
    """Max-minus-min probability per segment across passes.

    A segment with a wide spread is genuinely contested rather than merely
    mid-confidence, which is the distinction a single pass cannot make.
    """
    if len(results) < 2:
        return {}
    sids = {sid for r in results for sid in r.probabilities}
    spread = {}
    for sid in sids:
        vals = [r.probabilities.get(sid, 0.0) for r in results]
        spread[sid] = max(vals) - min(vals)
    return spread


@dataclass
class EpisodeScore:
    ep_id: str
    is_no_ad: bool
    f1: float = 0.0
    f05: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    start_mae: float | None = None
    end_mae: float | None = None
    no_ad_passed: bool | None = None
    no_ad_fps: int = 0
    predictions: list[dict] = field(default_factory=list)
    input_tokens: int = 0


def score_episode(
    episode: Episode,
    windows: Sequence[Sequence[dict]],
    probability_source: Callable[[Sequence[dict]], WindowResult],
    *,
    enter: float = ENTER_THRESHOLD,
    stay: float = STAY_THRESHOLD,
) -> EpisodeScore:
    """Run Pass A over every window, stitch, and score.

    Mirrors ``report.aggregate``: flatten per-window ads, canonicalize both
    sides at a 15s gap, then greedy IoU match at 0.5.
    """
    per_window_ads: list[list[dict]] = []
    input_tokens = 0
    for segs in windows:
        result = probability_source(segs)
        input_tokens += result.input_tokens
        per_window_ads.append(
            spans_from_probabilities(
                segs, result.probabilities, enter=enter, stay=stay))

    flat = [ad for window in per_window_ads for ad in window]
    flat = metrics.canonicalize_ads(flat)
    preds = [(ad["start"], ad["end"]) for ad in flat]

    score = EpisodeScore(
        ep_id=episode.ep_id,
        is_no_ad=episode.truth.is_no_ad_episode,
        predictions=flat,
        input_tokens=input_tokens,
    )

    if episode.truth.is_no_ad_episode:
        per_window_spans = [[(a["start"], a["end"]) for a in w]
                            for w in per_window_ads]
        res = metrics.no_ad_score(per_window_spans)
        score.no_ad_passed = res.passed
        score.no_ad_fps = res.false_positive_count
        return score

    truth_ranges = metrics.canonicalize_spans(
        [(ad.start, ad.end) for ad in episode.truth.ads])
    result = metrics.match_predictions(
        preds, truth_ranges, threshold=metrics_iou_threshold())
    score.f1 = result.f1
    score.f05 = result.fbeta(0.5)
    score.precision = result.precision
    score.recall = result.recall
    boundary = metrics.boundary_error(preds, truth_ranges, result.matches)
    if boundary is not None:
        score.start_mae = boundary.start_mae
        score.end_mae = boundary.end_mae
    return score


def metrics_iou_threshold() -> float:
    from .report.aggregate import DEFAULT_IOU_THRESHOLD
    return DEFAULT_IOU_THRESHOLD


def episode_windows(episode: Episode) -> list[list[dict]]:
    return stamp_id_windows(episode)


def estimate_input_tokens(windows: Iterable[Sequence[dict]]) -> int:
    """Rough token count for a whole-episode pass: state plus questions.

    Four characters per token, the same approximation the corpus sizing used.
    """
    total = 0
    for segs in windows:
        payload = build_payload(segs)
        total += len(json.dumps(payload)) // 4
    return total


def api_key_from_env() -> str | None:
    return os.environ.get("TYPESAFE_API_KEY")

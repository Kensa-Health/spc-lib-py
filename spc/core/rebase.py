"""suggest_rebase() -- decision support for *when* to rebase, never
automatic rebasing itself.

Deliberately reuses the rule engine's own output (already present on
ChartResult.points) rather than a separate detection algorithm. Returns
suggestions only -- never mutates the ChartResult it was given, and never
triggers a recalculation.
"""

from __future__ import annotations

from typing import Literal

from spc.core.models import ChartResult, PointResult, RebaseSuggestion

# Confidence grows with how long the signal has persisted, relative to
# min_points_since -- roughly double the minimum run length reads as "high".
_HIGH_CONFIDENCE_POINTS = 14
_MEDIUM_CONFIDENCE_POINTS = 10


def _direction_for(
        point_type: str
) -> Literal["improvement", "concern", "neutral"]:
    """Maps a point's classification to a rebase-suggestion direction."""
    if point_type == "special_cause_improvement":
        return "improvement"
    if point_type == "special_cause_concern":
        return "concern"
    return "neutral"


def _confidence_for(
        points_since: int
) -> Literal["low", "medium", "high"]:
    """Confidence grows with how long the signal has persisted."""
    if points_since >= _HIGH_CONFIDENCE_POINTS:
        return "high"
    if points_since >= _MEDIUM_CONFIDENCE_POINTS:
        return "medium"
    return "low"


def _side(
        point: PointResult
) -> float:
    """Signed distance from the mean -- sign indicates which side."""
    return point.y - point.mean


def suggest_rebase(
        result: ChartResult,
        *,
        min_points_since: int = 7
) -> list[RebaseSuggestion]:
    """Scans `result.points` for maximal runs of flagged (non-common-cause)
    points that stay on the same side of the mean, within one facet/segment
    (a signal cannot span a rebase boundary, since each segment's rules
    were computed independently), and proposes the earliest point of each
    qualifying run as a rebase candidate.

    Grouping by "same side of the mean" rather than "same special_cause_rule
    name" is deliberate: the rule engine's priority order means a single
    sustained shift or trend can have its per-point label interrupted
    mid-run -- e.g. the first few points of an 8-point shift are labelled
    "seven_point_shift", but once a later point also crosses a control
    limit, priority 1's "outside_limits" wins that point's label instead.
    A same-name grouping would see that as two short runs and miss the
    rebase candidate entirely, even though the underlying signal (points
    sustained on one side of the mean) never stopped. This still respects
    the "not a single one-off point" guardrail: an isolated outside_limits
    point that isn't part of a longer same-side run still won't meet
    `min_points_since`.

    A run must be at least `min_points_since` (>= 1) points long to be surfaced.
    A run that starts at the first point of its segment is not suggested:
    rebasing there would change nothing.

    Two properties to know about. The candidate is the first point where a rule
    fired, so it can trail a real change by up to a run length (a step is only
    flagged once enough points have followed it). And the suggestions inherit
    the rules' false-alarm rate: on stationary normal data, roughly one 40-point
    series in five gets a suggestion, nearly all of them at "low" confidence, so
    treat a low-confidence suggestion as a prompt to look, not as evidence.
    """
    if min_points_since < 1:
        raise ValueError(f"min_points_since must be >= 1, got {min_points_since}")
    by_segment: dict[tuple[str | None, int], list[PointResult]] = {}
    for point in result.points:
        by_segment.setdefault((point.facet, point.segment_id), []).append(point)

    suggestions: list[RebaseSuggestion] = []
    for (facet, _segment_id), points in by_segment.items():
        points = sorted(points, key=lambda p: p.x)
        i = 0
        n = len(points)
        while i < n:
            if points[i].special_cause_rule is None:
                i += 1
                continue
            side = _side(points[i]) >= 0
            j = i
            while j < n and points[j].special_cause_rule is not None and (_side(points[j]) >= 0) == side:
                j += 1
            run = points[i:j]
            points_since = len(run)
            if points_since >= min_points_since and i > 0:
                rule = run[0].special_cause_rule
                assert rule is not None  # guaranteed: a run only starts at a flagged point (see loop guard above)
                suggestions.append(
                    RebaseSuggestion(
                        facet=facet,
                        candidate_date=run[0].x,
                        rule=rule,
                        direction=_direction_for(run[0].point_type),
                        points_since=points_since,
                        confidence=_confidence_for(points_since),
                    )
                )
            i = j
    return suggestions

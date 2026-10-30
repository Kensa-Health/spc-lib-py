"""Special-cause rule engine.

Four standard Western Electric/Wheeler-family rules (the definitions and labels follow
NHSRplotthedots; see THIRD_PARTY_NOTICES.md), implemented here as a
pluggable registry rather than a hardcoded case_when -- a deployment or
user role can drop a rule (e.g. "2-in-3 near limit" judged too noisy for
a clinician-facing view) or add new rules without touching this module's
callers.

seven_point_trend/seven_point_shift's run length is configurable via
run_rule_length (BaseChartOptions) rather than hardcoded -- see
_resolve_run_length and _run_length_rules. DEFAULT_RULES/
DEFAULT_RULE_REGISTRY stay fixed at the historical run-length-7 versions
(the field's own default), used directly whenever the caller doesn't
override run_rule_length.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from spc.core.models import DEFAULT_RULE_NAMES, ImprovementDirection, PointType, RunRuleLength
from spc.core.segments import SegmentArrays


@dataclass
class Rule:
    """Plain dataclass, not pydantic: carries Python callables, which
    aren't meaningfully serialisable/validatable data. Pydantic is reserved
    for the boundary types in core.models.
    """

    name: str
    priority: int  # lower = evaluated first, first match wins
    detector: Callable[[SegmentArrays], np.ndarray]
    label_fn: Callable[[SegmentArrays, np.ndarray], list[str]]


# --- Detectors -----------------------------------------------------------
# Each detector returns one value per point in the segment. For the
# direction-bearing rules (trend, shift) the value is +1/-1/0 (direction),
# consumed by the matching label_fn; for the others it's a plain 0/1 flag.


def _outside_limits_detector(
        seg: SegmentArrays
) -> np.ndarray:
    """Flags every point outside its own control limits."""
    return seg.outside_limits.to_numpy().astype(int)


def _outside_limits_label(
        seg: SegmentArrays,
        flags: np.ndarray
) -> list[str]:
    """ "Above UCL"/"Below LCL" per flagged point."""
    rel = seg.relative_to_mean.to_numpy()
    return ["Above UCL" if rel[i] > 0 else "Below LCL" for i in range(len(flags))]


def _seven_point_trend_ends(
        y: np.ndarray,
        *,
        run_length: int = 7
) -> np.ndarray:
    """+1/-1/0 per point, for points that are the *end* of a run of
    `run_length` monotonically increasing/decreasing points (run_length-1
    consecutive same-sign diffs). Named for the historical default of 7;
    see BaseChartOptions.run_rule_length for the configurable case.
    """
    n = len(y)
    out = np.zeros(n)
    if n < run_length:
        return out
    diffs = np.sign(np.diff(y))
    w = run_length - 1
    for i in range(w, n):
        window = diffs[i - w : i]
        if np.all(window == 1):
            out[i] = 1
        elif np.all(window == -1):
            out[i] = -1
    return out


def _part_of_run_of_seven(
        end_flags: np.ndarray,
        *,
        run_length: int = 7
) -> np.ndarray:
    """A point counts if it, or any of the following run_length-1 points,
    is flagged as the end of a run -- i.e. the point is somewhere inside
    that run.
    """
    n = len(end_flags)
    w = run_length - 1
    padded = np.concatenate([end_flags, np.zeros(w)])
    return np.array([np.any(np.abs(padded[i : i + run_length]) == 1) for i in range(n)])


def _seven_point_trend_detector(
        seg: SegmentArrays,
        *,
        run_length: int = 7
) -> np.ndarray:
    """Direction per point, for points that are part of a run_length-point
    trend (default 7 -- see BaseChartOptions.run_rule_length).

    This is NOT simply "the sign of the run this point belongs to" -- a
    point sitting exactly at a reversal (e.g. the valley between a
    decreasing run and an increasing run that share that point) is
    genuinely ambiguous. Resolved by looking one point ahead: if the
    *next* point is also part of a trend run, use the forward diff
    (sign(y[i+1] - y[i])); otherwise (this is the last point of its run,
    or the series) use the backward diff (sign(y[i] - y[i-1])). At a
    decreasing-then-increasing valley, the shared point is followed by an
    increasing point, so the valley itself is classified "Increasing".
    """
    y = seg.y.to_numpy()
    n = len(y)
    ends = _seven_point_trend_ends(y, run_length=run_length)
    in_run = _part_of_run_of_seven(ends, run_length=run_length)
    if n < 2:
        return np.zeros(n)

    diffs = np.sign(np.diff(y))  # length n-1; diffs[k] = sign(y[k+1] - y[k])

    next_also_in_run = np.zeros(n, dtype=bool)
    next_also_in_run[:-1] = in_run[1:] & in_run[:-1]  # last point can never take the forward branch

    forward_diff = np.zeros(n)
    forward_diff[:-1] = diffs
    forward_diff[-1] = diffs[-1]  # unused unless in_run[-1]

    backward_diff = np.zeros(n)
    backward_diff[1:] = diffs  # backward_diff[0] stays 0 -- no prior point to diff against

    sdy = np.where(next_also_in_run, forward_diff, backward_diff)
    return np.where(in_run, sdy, 0)


def _seven_point_trend_label(
        seg: SegmentArrays,
        direction: np.ndarray
) -> list[str]:
    """ "7 Point Trend (Increasing/Decreasing)" per flagged point."""
    return ["7 Point Trend (Increasing)" if d > 0 else "7 Point Trend (Decreasing)" for d in direction]


def _two_in_three_detector(
        seg: SegmentArrays
) -> np.ndarray:
    """Flags a point when it's one of at least 2 of 3 consecutive points
    inside the near-limit zone, all on the same side of the mean.
    """
    close = seg.close_to_limits.to_numpy().astype(int)
    rel = seg.relative_to_mean.to_numpy()
    n = len(close)
    cf = np.concatenate([[0, 0], close, [0, 0]])
    rt = np.concatenate([[0, 0], rel, [0, 0]])
    raw = np.zeros(n)
    for i in range(n):
        for offset in (0, 1, 2):
            window_c = cf[i + offset : i + offset + 3]
            window_r = rt[i + offset : i + offset + 3]
            if window_c.sum() >= 2 and abs(window_r.sum()) == 3:
                raw[i] = np.sign(window_r.sum())
                break
    # A point only counts if it's itself one of the near-limit points AND
    # covered by a qualifying window.
    return np.where(close == 1, raw, 0)


def _two_in_three_label(
        seg: SegmentArrays,
        direction: np.ndarray
) -> list[str]:
    """ "2 in 3 Above/Below CL" per flagged point."""
    return ["2 in 3 Above CL" if d > 0 else "2 in 3 Below CL" for d in direction]


def _seven_point_shift_detector(
        seg: SegmentArrays,
        *,
        run_length: int = 7
) -> np.ndarray:
    """Flags a point as part of a run of run_length consecutive points on
    the same side of the mean (default 7 -- a sustained shift, as opposed
    to a monotonic trend; see BaseChartOptions.run_rule_length).
    """
    rel = seg.relative_to_mean.to_numpy()
    n = len(rel)
    w = run_length - 1
    padded = np.concatenate([np.zeros(w), rel])
    out = np.zeros(n)
    for i in range(n):
        idx = i + w
        window = padded[idx - w : idx + 1]
        if padded[idx] != 0 and np.all(window == padded[idx]):
            out[i] = padded[idx]
    # Same "end of run, or within run_length-1 points before it" spread as
    # the trend rule.
    ends = out.copy()
    in_run = _part_of_run_of_seven(ends, run_length=run_length)
    direction = np.zeros(n)
    for i in range(n):
        if not in_run[i]:
            continue
        for j in range(i, min(i + run_length, n)):
            if ends[j] != 0:
                direction[i] = ends[j]
                break
    return direction


def _seven_point_shift_label(
        seg: SegmentArrays,
        direction: np.ndarray
) -> list[str]:
    """ "7 Points Above/Below CL" per flagged point."""
    return ["7 Points Above CL" if d > 0 else "7 Points Below CL" for d in direction]


DEFAULT_RULES: list[Rule] = [
    Rule("outside_limits", priority=1, detector=_outside_limits_detector, label_fn=_outside_limits_label),
    Rule("seven_point_trend", priority=2, detector=_seven_point_trend_detector, label_fn=_seven_point_trend_label),
    Rule("two_in_three_near_limit", priority=3, detector=_two_in_three_detector, label_fn=_two_in_three_label),
    Rule("seven_point_shift", priority=4, detector=_seven_point_shift_detector, label_fn=_seven_point_shift_label),
]

DEFAULT_RULE_REGISTRY: dict[str, Rule] = {r.name: r for r in DEFAULT_RULES}

if {r.name for r in DEFAULT_RULES} != set(DEFAULT_RULE_NAMES):
    # Not `assert`: this guards a module-level invariant every chart kind's
    # default rule set depends on. assert is compiled out entirely under
    # `python -O`/PYTHONOPTIMIZE=1, which would turn a future mismatch here
    # into silently-wrong rule evaluation instead of a failure at import time.
    raise AssertionError("DEFAULT_RULES names must match core.models.DEFAULT_RULE_NAMES exactly")


def _resolve_run_length(
        run_rule_length: RunRuleLength,
        n: int
) -> int:
    """Fixed run length, or a sample-size-scaled bound when configured as
    "proportional": round(log2(n)) + 3, always >= 3 for n >= 1. Keeps the
    run-length rules' false-alarm rate roughly constant as segment length
    grows, rather than a fixed count under-triggering on short segments
    and over-triggering on long ones -- see BaseChartOptions.run_rule_length.
    """
    if run_rule_length != "proportional":
        return run_rule_length
    return round(math.log2(max(n, 1))) + 3


def _run_length_rules(
        run_length: int
) -> dict[str, Rule]:
    """seven_point_trend/seven_point_shift, rebuilt for a specific
    resolved run length. Names stay fixed regardless of run_length -- see
    BaseChartOptions.run_rule_length.
    """
    return {
        "seven_point_trend": Rule(
            "seven_point_trend",
            priority=2,
            detector=lambda seg: _seven_point_trend_detector(seg, run_length=run_length),
            label_fn=lambda seg, direction: [
                f"{run_length} Point Trend (Increasing)" if d > 0 else f"{run_length} Point Trend (Decreasing)"
                for d in direction
            ],
        ),
        "seven_point_shift": Rule(
            "seven_point_shift",
            priority=4,
            detector=lambda seg: _seven_point_shift_detector(seg, run_length=run_length),
            label_fn=lambda seg, direction: [
                f"{run_length} Points Above CL" if d > 0 else f"{run_length} Points Below CL" for d in direction
            ],
        ),
    }


def evaluate_rules(
        seg: SegmentArrays,
        rule_names: list[str],
        *,
        run_rule_length: RunRuleLength = 7
) -> tuple[list[str | None], list[str]]:
    """Runs each detector exactly once, in priority order, first match
    wins. Returns (winning rule name per point, human-readable label per
    point) together so callers needing both don't pay for every detector
    twice.
    """
    run_length = _resolve_run_length(run_rule_length, len(seg.y))
    registry = {**DEFAULT_RULE_REGISTRY, **_run_length_rules(run_length)}
    unknown = [name for name in rule_names if name not in registry]
    if unknown:
        raise ValueError(f"unknown rule name(s) {unknown}; available: {sorted(registry)}")
    active = sorted((registry[name] for name in rule_names), key=lambda r: r.priority)
    n = len(seg.y)
    winner_rule: list[str | None] = [None] * n
    labels = ["Common Cause"] * n

    for rule in active:
        result = rule.detector(seg)
        idx = [i for i in range(n) if winner_rule[i] is None and result[i]]
        if not idx:
            continue
        rule_labels_full = rule.label_fn(seg, result)
        for i in idx:
            winner_rule[i] = rule.name
            labels[i] = rule_labels_full[i]

    return winner_rule, labels


def apply_rules(
        seg: SegmentArrays,
        rule_names: list[str],
        *,
        run_rule_length: RunRuleLength = 7
) -> list[str | None]:
    """Runs the given rules (by name, resolved against DEFAULT_RULE_REGISTRY)
    against a segment in priority order, first match wins. Returns one
    special_cause_rule value per point -- the winning rule's *name*, or
    None for common-cause points.
    """
    winners, _labels = evaluate_rules(seg, rule_names, run_rule_length=run_rule_length)
    return winners


def rule_labels(
        seg: SegmentArrays,
        rule_names: list[str],
        *,
        run_rule_length: RunRuleLength = 7
) -> list[str]:
    """Human-readable label per point (e.g. "Above UCL", "7 Points Below CL"),
    using each winning rule's own label_fn.
    """
    _winners, labels = evaluate_rules(seg, rule_names, run_rule_length=run_rule_length)
    return labels


def point_type(
        special_cause_rule: str | None,
        label: str,
        improvement_direction: ImprovementDirection
) -> PointType:
    """Classifies a flagged point by direction and improvement_direction."""
    if special_cause_rule is None:
        return "common_cause"

    if "Above" in label or "Increasing" in label:
        v = 1
    elif "Below" in label or "Decreasing" in label:
        v = -1
    else:
        v = 0

    if improvement_direction == "neutral":
        if v > 0:
            return "special_cause_neutral_high"
        if v < 0:
            return "special_cause_neutral_low"
        return "common_cause"

    signed = v * (1 if improvement_direction == "increase" else -1)
    if signed > 0:
        return "special_cause_improvement"
    if signed < 0:
        return "special_cause_concern"
    return "common_cause"

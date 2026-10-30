from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from spc.core.models import DEFAULT_RULE_NAMES
from spc.core.rules import (
    DEFAULT_RULES,
    _outside_limits_detector,
    _part_of_run_of_seven,
    _resolve_run_length,
    _seven_point_shift_detector,
    _seven_point_trend_detector,
    _seven_point_trend_ends,
    _two_in_three_detector,
    apply_rules,
    evaluate_rules,
    point_type,
)
from spc.core.segments import SegmentArrays


def _series(
        values
):
    n = len(values)
    idx = pd.RangeIndex(n)
    return pd.Series(values, index=idx, dtype=float)


def make_segment(
        y=None,
        relative_to_mean=None,
        close_to_limits=None,
        outside_limits=None
) -> SegmentArrays:
    lengths = [len(v) for v in (y, relative_to_mean, close_to_limits, outside_limits) if v is not None]
    n = lengths[0]
    y = _series(y) if y is not None else _series([0] * n)
    relative_to_mean = _series(relative_to_mean) if relative_to_mean is not None else _series([0] * n)
    close_to_limits = pd.Series(close_to_limits, dtype=bool) if close_to_limits is not None else pd.Series([False] * n)
    outside_limits = pd.Series(outside_limits, dtype=bool) if outside_limits is not None else pd.Series([False] * n)
    x = pd.Series([date(2024, 1, 1) + timedelta(weeks=i) for i in range(n)])
    zeros = _series([0] * n)
    return SegmentArrays(
        facet=None,
        segment_id=0,
        x=x,
        y=y,
        centre_line=zeros,
        ucl=zeros,
        lcl=zeros,
        amr=zeros,
        outside_limits=outside_limits,
        relative_to_mean=relative_to_mean,
        close_to_limits=close_to_limits,
        low_count_warning=pd.Series([False] * n),
        autocorrelation=pd.Series([float("nan")] * n),
    )


def test_default_rule_names_match_models():
    assert {r.name for r in DEFAULT_RULES} == set(DEFAULT_RULE_NAMES)


def test_outside_limits_detector():
    seg = make_segment(y=[10, 30, 10, -30], relative_to_mean=[0, 1, 0, -1], outside_limits=[False, True, False, True])
    flags = _outside_limits_detector(seg)
    assert list(flags) == [0, 1, 0, 1]


def test_seven_point_trend_ends_pure_monotonic_run():
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    ends = _seven_point_trend_ends(y)
    # 8 points, 7 diffs; two overlapping windows of 6 diffs each qualify: [0:6] and [1:7]
    assert list(ends) == [0, 0, 0, 0, 0, 0, 1, 1]


def test_seven_point_trend_ends_too_short():
    y = np.array([1, 2, 3], dtype=float)
    assert list(_seven_point_trend_ends(y)) == [0, 0, 0]


def test_part_of_run_of_seven_spreads_to_all_points_in_an_exact_run():
    # A single end-flag at the last of exactly 7 points should mark all 7.
    ends = np.array([0, 0, 0, 0, 0, 0, 1], dtype=float)
    assert list(_part_of_run_of_seven(ends)) == [True] * 7


def test_seven_point_trend_detector_flags_whole_run_with_direction():
    y = list(range(1, 9))  # 8 strictly increasing points
    seg = make_segment(y=y)
    direction = _seven_point_trend_detector(seg)
    assert list(direction) == [1.0] * 8

    y_decreasing = list(range(8, 0, -1))
    seg_dec = make_segment(y=y_decreasing)
    direction_dec = _seven_point_trend_detector(seg_dec)
    assert list(direction_dec) == [-1.0] * 8


def test_seven_point_trend_detector_no_trend_in_noise():
    seg = make_segment(y=[5, 3, 6, 2, 7, 1, 8, 4])
    direction = _seven_point_trend_detector(seg)
    assert list(direction) == [0.0] * 8


def test_seven_point_trend_direction_at_a_reversal_valley():
    # A 7-point decreasing run immediately followed by a 7-point increasing
    # run, sharing the valley point. Direction is resolved by looking one
    # point ahead: since the next point continues the increasing run, the
    # valley itself is classified "Increasing", not "Decreasing" (which a
    # naive "which run's end-flag is nearest" reading would give instead).
    y = [10, 9, 8, 7, 6, 5, 4, 5, 6, 7, 8, 9, 10]
    seg = make_segment(y=y)
    direction = _seven_point_trend_detector(seg)
    assert direction[6] == 1.0  # the valley (y=4): Increasing, per R
    assert list(direction) == [-1.0] * 6 + [1.0] * 7


def test_two_in_three_detector_hand_computed():
    close = [1, 1, 0]
    rel = [1, 1, 1]
    seg = make_segment(y=[0, 0, 0], relative_to_mean=rel, close_to_limits=close)
    result = _two_in_three_detector(seg)
    assert list(result) == [1, 1, 0]


def test_two_in_three_detector_no_signal_when_sides_disagree():
    close = [1, 1, 1]
    rel = [1, -1, 1]  # not all on the same side -> should never qualify
    seg = make_segment(y=[0, 0, 0], relative_to_mean=rel, close_to_limits=close)
    result = _two_in_three_detector(seg)
    assert list(result) == [0, 0, 0]


def test_seven_point_shift_detector_flags_whole_run_with_direction():
    rel = [1] * 7
    seg = make_segment(y=[0] * 7, relative_to_mean=rel)
    direction = _seven_point_shift_detector(seg)
    assert list(direction) == [1.0] * 7

    rel_below = [-1] * 7
    seg_below = make_segment(y=[0] * 7, relative_to_mean=rel_below)
    direction_below = _seven_point_shift_detector(seg_below)
    assert list(direction_below) == [-1.0] * 7


def test_seven_point_shift_detector_no_signal_under_seven_points():
    seg = make_segment(y=[0] * 6, relative_to_mean=[1] * 6)
    assert list(_seven_point_shift_detector(seg)) == [0.0] * 6


def test_apply_rules_priority_outside_limits_beats_shift():
    # A point that is both outside limits AND part of a 7-point shift run
    # must be labelled "outside_limits" (priority 1), not "seven_point_shift"
    # (priority 4) -- strict priority order, first match wins.
    # y is deliberately non-monotonic (only the last diff is large) so the
    # seven_point_trend detector (priority 2) doesn't also fire and confound
    # the comparison -- relative_to_mean is set independently of y here.
    rel = [1] * 7
    outside = [False] * 6 + [True]
    seg = make_segment(y=[10, 12, 9, 13, 8, 11, 100], relative_to_mean=rel, outside_limits=outside)
    winners = apply_rules(seg, list(DEFAULT_RULE_NAMES))
    assert winners[-1] == "outside_limits"
    assert winners[0] == "seven_point_shift"


@pytest.mark.parametrize(
    "rule,label,direction,expected",
    [
        (None, "Common Cause", "increase", "common_cause"),
        ("outside_limits", "Above UCL", "increase", "special_cause_improvement"),
        ("outside_limits", "Below LCL", "increase", "special_cause_concern"),
        ("outside_limits", "Above UCL", "decrease", "special_cause_concern"),
        ("outside_limits", "Below LCL", "decrease", "special_cause_improvement"),
        ("outside_limits", "Above UCL", "neutral", "special_cause_neutral_high"),
        ("outside_limits", "Below LCL", "neutral", "special_cause_neutral_low"),
    ],
)
def test_point_type_classification(
        rule,
        label,
        direction,
        expected
):
    assert point_type(rule, label, direction) == expected


# --- Configurable run length ---------------------------------------------


@pytest.mark.parametrize(
    "n,expected",
    [(4, 5), (8, 6), (10, 6), (12, 7), (16, 7), (20, 7), (24, 8), (100, 10)],
)
def test_resolve_run_length_proportional_matches_schilling_formula(
        n,
        expected
):
    assert _resolve_run_length("proportional", n) == expected


def test_resolve_run_length_fixed_value_passes_through_unchanged():
    assert _resolve_run_length(5, n=1000) == 5


def test_seven_point_trend_detector_respects_custom_run_length():
    # Only 5 strictly increasing points -- doesn't qualify at the default
    # run length of 7, but does at 5.
    y = [1, 2, 3, 4, 5]
    seg = make_segment(y=y)
    assert list(_seven_point_trend_detector(seg)) == [0.0] * 5
    assert list(_seven_point_trend_detector(seg, run_length=5)) == [1.0] * 5


def test_seven_point_shift_detector_respects_custom_run_length():
    rel = [1] * 5
    seg = make_segment(y=[0] * 5, relative_to_mean=rel)
    assert list(_seven_point_shift_detector(seg)) == [0.0] * 5
    assert list(_seven_point_shift_detector(seg, run_length=5)) == [1.0] * 5


def test_evaluate_rules_default_run_length_is_seven():
    rel = [1] * 7
    seg = make_segment(y=[0] * 7, relative_to_mean=rel)
    winners, labels = evaluate_rules(seg, list(DEFAULT_RULE_NAMES))
    assert winners == ["seven_point_shift"] * 7
    assert labels == ["7 Points Above CL"] * 7


def test_evaluate_rules_custom_run_length_changes_both_detection_and_label():
    rel = [1] * 5
    seg = make_segment(y=[0] * 5, relative_to_mean=rel)
    # Not flagged at the default of 7...
    winners_default, _ = evaluate_rules(seg, list(DEFAULT_RULE_NAMES))
    assert winners_default == [None] * 5
    # ...but is at a configured run length of 5, with a label reflecting it.
    winners, labels = evaluate_rules(seg, list(DEFAULT_RULE_NAMES), run_rule_length=5)
    assert winners == ["seven_point_shift"] * 5
    assert labels == ["5 Points Above CL"] * 5


def test_evaluate_rules_proportional_run_length_label_reflects_resolved_length():
    # 24 points -> proportional run length resolves to 8 (see the
    # parametrized _resolve_run_length test above).
    rel = [1] * 24
    seg = make_segment(y=[0] * 24, relative_to_mean=rel)
    winners, labels = evaluate_rules(seg, list(DEFAULT_RULE_NAMES), run_rule_length="proportional")
    assert winners == ["seven_point_shift"] * 24
    assert labels == ["8 Points Above CL"] * 24


def test_default_rules_constant_is_unaffected_by_run_rule_length_feature():
    # DEFAULT_RULES/DEFAULT_RULE_REGISTRY stay the fixed, run-length-7
    # versions -- evaluate_rules only special-cases seven_point_trend/
    # seven_point_shift per call, it doesn't mutate the module-level constant.
    assert {r.name for r in DEFAULT_RULES} == set(DEFAULT_RULE_NAMES)

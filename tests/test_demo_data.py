"""Integration tests against data/*.csv -- realistic-shaped series run
through the whole pipeline, as opposed to tests/test_*.py's minimal,
precise per-assertion fixtures. Each assertion here mirrors the specific
claim made in data/README.md / data/generate.py's docstrings for that file
-- if one of these ever fails, that documentation is now wrong too.
"""

from pathlib import Path

import pandas as pd
import pytest

import spc
from spc.core.models import BaseChartOptions, LimitMode

DATA_DIR = Path(__file__).parent.parent / "data"


def _load(
        name: str
) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, parse_dates=["period"])


def test_stable_is_clean():
    df = _load("xmr_stable.csv")
    result = spc.chart(df, kind="xmr", options=BaseChartOptions(value_field="value", date_field="period"))
    assert all(p.special_cause_rule is None for p in result.points)


def test_outlier_screening_modes_diverge():
    df = _load("xmr_outlier.csv")
    single = spc.chart(
        df,
        kind="xmr",
        options=BaseChartOptions(value_field="value", date_field="period", screen_outliers="single_pass"),
    )
    iterative = spc.chart(
        df, kind="xmr", options=BaseChartOptions(value_field="value", date_field="period", screen_outliers="iterative")
    )
    assert single.segments[0].ucl != pytest.approx(iterative.segments[0].ucl)


def test_shift_detected_only_via_seven_point_shift_with_clean_baseline():
    df = _load("xmr_shift.csv")
    options = BaseChartOptions(
        value_field="value", date_field="period", limit_mode=LimitMode.FIXED_N, fix_after_n_points=10
    )
    result = spc.chart(df, kind="xmr", options=options)
    assert not any(p.outside_limits for p in result.points)

    suggestions = spc.suggest_rebase(result)
    assert len(suggestions) == 1
    assert suggestions[0].points_since == 10
    assert str(suggestions[0].candidate_date) == "2024-03-11"


def test_trend_isolated_with_clean_baseline():
    df = _load("xmr_trend.csv")
    options = BaseChartOptions(
        value_field="value", date_field="period", limit_mode=LimitMode.FIXED_N, fix_after_n_points=8
    )
    result = spc.chart(df, kind="xmr", options=options)
    rules = {p.special_cause_rule for p in result.points if p.special_cause_rule}
    assert rules == {"seven_point_trend"}


def test_two_facets_computed_independently():
    df = _load("xmr_two_facets.csv")
    options = BaseChartOptions(
        value_field="value",
        date_field="period",
        facet_field="org",
        limit_mode=LimitMode.FIXED_N,
        fix_after_n_points=10,
    )
    result = spc.chart(df, kind="xmr", options=options)
    north_flags = [p for p in result.points if p.facet == "north" and p.special_cause_rule]
    south_flags = [p for p in result.points if p.facet == "south" and p.special_cause_rule]
    assert north_flags == []
    assert south_flags and {p.special_cause_rule for p in south_flags} == {"seven_point_shift"}


def test_rebase_option_cleans_up_a_pooled_regime_change():
    df = _load("xmr_rebase.csv")
    without = spc.chart(df, kind="xmr", options=BaseChartOptions(value_field="value", date_field="period"))
    with_rebase = spc.chart(
        df,
        kind="xmr",
        options=BaseChartOptions(value_field="value", date_field="period", rebase=[pd.Timestamp("2024-03-25")]),
    )
    flagged_without = sum(1 for p in without.points if p.special_cause_rule)
    flagged_with = sum(1 for p in with_rebase.points if p.special_cause_rule)
    assert flagged_without > flagged_with
    assert flagged_with <= 2

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import spc
from spc.core.models import BaseChartOptions, LimitMode


def _dates(
        n,
        start=date(2024, 1, 1)
):
    return [start + timedelta(weeks=i) for i in range(n)]


def test_suggest_rebase_never_mutates_result():
    dates = _dates(20)
    values = [10, 11, 9, 10, 11, 9, 10, 11, 9, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    df = pd.DataFrame({"period": dates, "y": values})
    options = BaseChartOptions(
        value_field="y", date_field="period", limit_mode=LimitMode.FIXED_N, fix_after_n_points=10
    )
    result = spc.chart(df, kind="xmr", options=options)
    before = result.model_dump_json()
    spc.suggest_rebase(result)
    after = result.model_dump_json()
    assert before == after


def test_suggest_rebase_finds_sustained_shift_that_stays_within_limits():
    # A stable baseline, then a shift up that's real but small enough to
    # stay within the (locked, wide) fixed_n control limits -- this should
    # only be catchable via the seven_point_shift rule, not outside_limits.
    dates = _dates(20)
    # Alternates around its own mean, and deliberately ends on a
    # below-mean point so the shift run below starts cleanly at index 10
    # rather than absorbing a same-signed tail point from the baseline.
    baseline = [100, 105, 98, 103, 97, 102, 99, 104, 101, 96]
    shifted = [106, 107, 106, 108, 107, 106, 108, 107, 106, 108]  # consistently above baseline mean, within limits
    values = baseline + shifted
    df = pd.DataFrame({"period": dates, "y": values})
    options = BaseChartOptions(
        value_field="y", date_field="period", limit_mode=LimitMode.FIXED_N, fix_after_n_points=10
    )
    result = spc.chart(df, kind="xmr", options=options)

    # sanity: none of the shifted points should be outside_limits (otherwise
    # this test isn't isolating the shift-detection path)
    shifted_points = [p for p in result.points if p.x >= dates[10]]
    assert not any(p.outside_limits for p in shifted_points)

    suggestions = spc.suggest_rebase(result, min_points_since=7)
    assert len(suggestions) == 1
    assert suggestions[0].candidate_date == dates[10]
    assert suggestions[0].points_since == 10
    assert suggestions[0].rule == "seven_point_shift"


def test_suggest_rebase_ignores_isolated_outside_limits_point():
    dates = _dates(15)
    values = [10, 11, 9, 10, 11, 9, 10, 11, 9, 10, 100, 10, 11, 9, 10]  # one spike at index 10
    df = pd.DataFrame({"period": dates, "y": values})
    options = BaseChartOptions(
        value_field="y", date_field="period", limit_mode=LimitMode.FIXED_N, fix_after_n_points=10
    )
    result = spc.chart(df, kind="xmr", options=options)
    suggestions = spc.suggest_rebase(result, min_points_since=7)
    assert suggestions == []


def test_suggest_rebase_respects_min_points_since():
    dates = _dates(15)
    baseline = [100, 105, 98, 103, 97, 102, 99, 104, 96, 101]
    short_shift = [110, 111, 110]  # only 3 points -- shorter than default min_points_since
    values = baseline + short_shift + [100, 101]
    df = pd.DataFrame({"period": dates, "y": values})
    options = BaseChartOptions(
        value_field="y", date_field="period", limit_mode=LimitMode.FIXED_N, fix_after_n_points=10
    )
    result = spc.chart(df, kind="xmr", options=options)
    assert spc.suggest_rebase(result, min_points_since=7) == []


def test_min_points_since_must_be_at_least_one():
    df = pd.DataFrame({"p": pd.date_range("2024-01-01", periods=10, freq="W"), "y": range(10)})
    result = spc.chart(df, kind="xmr", options=BaseChartOptions(value_field="y", date_field="p"))
    for bad in (0, -3):
        with pytest.raises(ValueError, match="min_points_since"):
            spc.suggest_rebase(result, min_points_since=bad)


def test_a_run_starting_at_the_first_point_is_not_suggested():
    rng = np.random.default_rng(3)
    y = np.r_[rng.normal(10, 1, 30), rng.normal(15, 1, 30)]
    df = pd.DataFrame({"p": pd.date_range("2024-01-01", periods=60, freq="W"), "y": y})
    result = spc.chart(df, kind="xmr", options=BaseChartOptions(value_field="y", date_field="p"))
    first = min(p.x for p in result.points)
    suggestions = spc.suggest_rebase(result)
    assert suggestions  # the step is still found
    assert all(s.candidate_date != first for s in suggestions)

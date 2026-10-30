"""BaseChartOptions.autocorrelation_adjustment end-to-end tests -- xmr
only, deliberately ignored by every other chart kind. Unit coverage of
the underlying formulas lives in tests/test_limits.py; this file covers
wiring through spc.chart() and the non-fatal diagnostics on ChartResult/
SegmentSummary.
"""

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import spc
from spc.core.models import BaseChartOptions, LimitMode, ProportionChartOptions


def _dates(
        n,
        start=date(2024, 1, 1)
):
    return [start + timedelta(weeks=i) for i in range(n)]


def _random_walk(
        n,
        seed
):
    rng = np.random.default_rng(seed)
    return list(np.cumsum(rng.normal(size=n)))


# --- Option 4: autocorrelation_uncorrected_rules ----------------------------


def test_uncorrected_rules_empty_when_adjustment_off():
    df = pd.DataFrame({"period": _dates(20), "y": _random_walk(20, 0)})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)
    assert result.autocorrelation_uncorrected_rules == []


def test_uncorrected_rules_empty_when_rules_already_only_outside_limits():
    df = pd.DataFrame({"period": _dates(20), "y": _random_walk(20, 0)})
    options = BaseChartOptions(
        value_field="y", date_field="period", autocorrelation_adjustment=True, rules=["outside_limits"]
    )
    result = spc.chart(df, kind="xmr", options=options)
    assert result.autocorrelation_uncorrected_rules == []


def test_uncorrected_rules_lists_exactly_the_non_outside_limits_rules_present():
    df = pd.DataFrame({"period": _dates(20), "y": _random_walk(20, 0)})
    options = BaseChartOptions(
        value_field="y",
        date_field="period",
        autocorrelation_adjustment=True,
        rules=["outside_limits", "seven_point_shift"],
    )
    result = spc.chart(df, kind="xmr", options=options)
    assert result.autocorrelation_uncorrected_rules == ["seven_point_shift"]


def test_uncorrected_rules_default_rules_lists_all_three_non_outside_limits_names():
    df = pd.DataFrame({"period": _dates(20), "y": _random_walk(20, 0)})
    options = BaseChartOptions(value_field="y", date_field="period", autocorrelation_adjustment=True)
    result = spc.chart(df, kind="xmr", options=options)
    assert result.autocorrelation_uncorrected_rules == [
        "seven_point_trend",
        "two_in_three_near_limit",
        "seven_point_shift",
    ]


def test_uncorrected_rules_never_mutates_options_rules():
    df = pd.DataFrame({"period": _dates(20), "y": _random_walk(20, 0)})
    options = BaseChartOptions(value_field="y", date_field="period", autocorrelation_adjustment=True)
    original_rules = list(options.rules)
    spc.chart(df, kind="xmr", options=options)
    assert options.rules == original_rules


# --- SegmentSummary.autocorrelation_estimate / high_autocorrelation_warning -


def test_autocorrelation_estimate_is_none_for_a_too_short_segment():
    df = pd.DataFrame({"period": _dates(1), "y": [5.0]})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)
    assert result.segments[0].autocorrelation_estimate is None


def test_autocorrelation_estimate_populated_regardless_of_adjustment():
    df = pd.DataFrame({"period": _dates(20), "y": _random_walk(20, 0)})
    options = BaseChartOptions(value_field="y", date_field="period", autocorrelation_adjustment=False)
    result = spc.chart(df, kind="xmr", options=options)
    assert result.segments[0].autocorrelation_estimate is not None


def test_high_autocorrelation_warning_true_only_when_adjustment_on_and_rho_high():
    df = pd.DataFrame({"period": _dates(60), "y": _random_walk(60, 5)})  # random walk -> rho typically > 0.8
    result_off = spc.chart(df, kind="xmr", options=BaseChartOptions(value_field="y", date_field="period"))
    result_on = spc.chart(
        df, kind="xmr", options=BaseChartOptions(value_field="y", date_field="period", autocorrelation_adjustment=True)
    )
    assert result_off.segments[0].high_autocorrelation_warning is False  # never True with the adjustment off
    assert result_on.segments[0].autocorrelation_estimate >= 0.8
    assert result_on.segments[0].high_autocorrelation_warning is True


def test_high_autocorrelation_warning_false_for_low_rho():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"period": _dates(30), "y": rng.normal(size=30)})  # white noise -> rho near 0
    options = BaseChartOptions(value_field="y", date_field="period", autocorrelation_adjustment=True)
    result = spc.chart(df, kind="xmr", options=options)
    assert result.segments[0].high_autocorrelation_warning is False


# --- Attribute chart kinds: no effect, no error -----------------------------


@pytest.mark.parametrize("kind", ["p", "np", "c", "u", "p_prime", "u_prime"])
def test_attribute_chart_kinds_ignore_autocorrelation_adjustment(
        kind
):
    df = pd.DataFrame({"period": _dates(15), "d": [2, 3, 2, 4, 1, 3, 2, 5, 2, 3, 1, 2, 4, 2, 3], "n": [10] * 15})
    kwargs = {"denominator_field": "n"} if kind != "c" else {}
    options_cls = ProportionChartOptions if kind != "c" else BaseChartOptions
    options_on = options_cls(value_field="d", date_field="period", autocorrelation_adjustment=True, **kwargs)
    options_off = options_cls(value_field="d", date_field="period", autocorrelation_adjustment=False, **kwargs)

    result_on = spc.chart(df, kind=kind, options=options_on)
    result_off = spc.chart(df, kind=kind, options=options_off)

    assert [p.ucl for p in result_on.points] == [p.ucl for p in result_off.points]
    assert [p.lcl for p in result_on.points] == [p.lcl for p in result_off.points]
    assert result_on.segments[0].autocorrelation_estimate is None
    assert result_on.segments[0].high_autocorrelation_warning is False
    assert result_on.autocorrelation_uncorrected_rules == []  # Option 4 diagnostic is xmr-only too


# --- EXPANDING warm-up: bounded, not unbounded ------------------------------


def test_expanding_warm_up_width_stays_within_the_clamp_bound():
    # Strong injected autocorrelation, so early warm-up points (tiny
    # windows) are exactly the case a noisy small-n rho estimate risks --
    # width still can't exceed the 0.95-cap-implied ~4.47x factor.
    df = pd.DataFrame({"period": _dates(20), "y": _random_walk(20, 7)})
    off = spc.chart(
        df,
        kind="xmr",
        options=BaseChartOptions(
            value_field="y", date_field="period", limit_mode=LimitMode.EXPANDING, fix_after_n_points=8
        ),
    )
    on = spc.chart(
        df,
        kind="xmr",
        options=BaseChartOptions(
            value_field="y",
            date_field="period",
            limit_mode=LimitMode.EXPANDING,
            fix_after_n_points=8,
            autocorrelation_adjustment=True,
        ),
    )
    max_factor = 1 / math.sqrt(1 - 0.95)
    for p_off, p_on in zip(off.points, on.points, strict=True):
        if p_off.ucl is None or p_off.lcl is None or p_on.ucl is None or p_on.lcl is None:
            continue
        off_width = p_off.ucl - p_off.lcl
        on_width = p_on.ucl - p_on.lcl
        if off_width > 0:
            assert on_width <= off_width * max_factor + 1e-9


# --- Default-unchanged regression -------------------------------------------


def test_default_unchanged_regression_across_limit_modes():
    df = pd.DataFrame({"period": _dates(20), "y": _random_walk(20, 8)})
    for limit_mode, extra in [
        (LimitMode.SEGMENT, {}),
        (LimitMode.FIXED_N, {"fix_after_n_points": 10}),
        (LimitMode.EXPANDING, {"fix_after_n_points": 10}),
    ]:
        with_default = spc.chart(
            df,
            kind="xmr",
            options=BaseChartOptions(value_field="y", date_field="period", limit_mode=limit_mode, **extra),
        )
        explicit_off = spc.chart(
            df,
            kind="xmr",
            options=BaseChartOptions(
                value_field="y", date_field="period", limit_mode=limit_mode, autocorrelation_adjustment=False, **extra
            ),
        )
        for p1, p2 in zip(with_default.points, explicit_off.points, strict=True):
            assert p1.ucl == p2.ucl
            assert p1.lcl == p2.lcl

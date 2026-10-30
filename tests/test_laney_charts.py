"""P'/U' (Laney overdispersion-corrected) chart tests -- verified against
Laney's own method (standardise, treat as an XmR series, floor the
dispersion factor at 1) via hand computation and synthetic data with/
without genuine overdispersion.
"""

import math
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import spc
from spc.core.limits import laney_sigma_z, p_prime_chart_limits, u_prime_chart_limits
from spc.core.models import BaseChartOptions, ExternalBaseline, ProportionChartOptions


def _dates(
        n,
        start=date(2024, 1, 1)
):
    return [start + timedelta(weeks=i) for i in range(n)]


# --- laney_sigma_z (core/limits.py) -----------------------------------------


def test_laney_sigma_z_is_nan_for_fewer_than_two_points():
    assert math.isnan(laney_sigma_z(np.array([0.1]), np.array([0.1]), np.array([0.03])))
    assert math.isnan(laney_sigma_z(np.array([]), np.array([]), np.array([])))


def test_laney_sigma_z_hand_computed():
    # Z = (y-centre)/se = [1, -1, 1, -1] -- moving range of Z is [2,2,2],
    # AmR_z = 2, sigma_z = 2/1.128 = 1.7730...
    y = np.array([1.03, 0.97, 1.03, 0.97])
    centre = np.array([1.0, 1.0, 1.0, 1.0])
    se = np.array([0.03, 0.03, 0.03, 0.03])
    assert laney_sigma_z(y, centre, se) == pytest.approx(2 / 1.128, abs=1e-4)


def test_laney_sigma_z_floored_at_one_not_below():
    # Z essentially constant (tiny moving range) -- would compute sigma_z
    # well below 1 unfloored; must clip to 1, not narrow the limits.
    y = np.array([1.0001, 1.0000, 1.0001, 1.0000, 1.0001])
    centre = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    se = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert laney_sigma_z(y, centre, se) == 1.0


def test_p_prime_chart_limits_reduces_to_p_chart_limits_when_sigma_z_is_one():
    from spc.core.limits import p_chart_limits

    p_bar = np.array([0.1, 0.1, 0.1])
    n = np.array([50.0, 50.0, 50.0])
    prime_stats = p_prime_chart_limits(p_bar, n, sigma_z=1.0)
    plain_stats = p_chart_limits(p_bar, n)
    np.testing.assert_allclose(prime_stats.ucl, plain_stats.ucl)
    np.testing.assert_allclose(prime_stats.lcl, plain_stats.lcl)


def test_p_prime_chart_limits_widens_with_sigma_z():
    # p_bar=0.3 keeps the sigma_z=1 LCL comfortably above the zero floor,
    # so widening at sigma_z=3 has room to actually move it down (a small
    # p_bar would floor both cases at 0 and hide the widening on this side).
    p_bar = np.array([0.3])
    n = np.array([50.0])
    narrow = p_prime_chart_limits(p_bar, n, sigma_z=1.0)
    wide = p_prime_chart_limits(p_bar, n, sigma_z=3.0)
    assert wide.ucl[0] > narrow.ucl[0]
    assert wide.lcl[0] < narrow.lcl[0]


def test_u_prime_chart_limits_widens_with_sigma_z():
    u_bar = np.array([2.0])
    opportunity = np.array([100.0])
    narrow = u_prime_chart_limits(u_bar, opportunity, sigma_z=1.0)
    wide = u_prime_chart_limits(u_bar, opportunity, sigma_z=3.0)
    assert wide.ucl[0] > narrow.ucl[0]
    assert wide.lcl[0] < narrow.lcl[0]


# --- End-to-end P' chart -----------------------------------------------


def test_p_prime_matches_p_when_data_is_pure_binomial_noise():
    # No genuine overdispersion -- p and p_prime should produce
    # (near-)identical limits, sigma_z close to 1.
    rng = random.Random(42)
    dates = _dates(15)
    defects = [rng.randint(6, 14) for _ in range(15)]  # around p=0.1, n=100
    df = pd.DataFrame({"period": dates, "defects": defects, "n": [100] * 15})
    options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")

    result_p = spc.chart(df, kind="p", options=options)
    result_pp = spc.chart(df, kind="p_prime", options=options)

    for pt_p, pt_pp in zip(result_p.points, result_pp.points, strict=True):
        assert pt_pp.ucl == pytest.approx(pt_p.ucl, rel=0.05)
        assert pt_pp.lcl == pytest.approx(pt_p.lcl, abs=1e-3)
        assert pt_pp.mean == pytest.approx(pt_p.mean)


def test_p_prime_widens_limits_and_reduces_false_alarms_under_overdispersion():
    # Genuine overdispersion: the *true* underlying proportion swings
    # between ~0.05 and ~0.20 period to period, on top of binomial
    # sampling noise -- more process variation than the pure-binomial
    # model expects. Classical p should over-flag; p_prime should widen
    # its limits enough to stop treating routine swings as special-cause.
    rng = random.Random(7)
    dates = _dates(20)
    true_ps = [0.05, 0.20, 0.06, 0.18, 0.07, 0.19, 0.05, 0.21, 0.06, 0.20] * 2
    n = 200
    defects = [sum(1 for _ in range(n) if rng.random() < p) for p in true_ps]
    df = pd.DataFrame({"period": dates, "defects": defects, "n": [n] * 20})
    options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")

    result_p = spc.chart(df, kind="p", options=options)
    result_pp = spc.chart(df, kind="p_prime", options=options)

    p_flagged = sum(pt.outside_limits for pt in result_p.points)
    pp_flagged = sum(pt.outside_limits for pt in result_pp.points)
    assert p_flagged > pp_flagged  # classical over-flags a systematically volatile process
    assert result_pp.points[0].ucl - result_pp.points[0].mean > result_p.points[0].ucl - result_p.points[0].mean


def test_p_prime_wrong_options_type_raises_clear_error():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "y": [1, 2, 3]})
    with pytest.raises(TypeError, match="ProportionChartOptions"):
        spc.chart(df, kind="p_prime", options=BaseChartOptions(value_field="y", date_field="period"))


def test_p_prime_low_count_warning_behaves_like_p():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "defects": [1, 1, 1], "n": [5, 5, 5]})
    options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    result = spc.chart(df, kind="p_prime", options=options)
    assert all(p.low_count_warning for p in result.points)


def test_p_prime_rejects_numerator_exceeding_denominator():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "defects": [1, 15, 1], "n": [10, 10, 10]})
    options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    with pytest.raises(ValueError, match="exceeds"):
        spc.chart(df, kind="p_prime", options=options)


def test_p_prime_rejects_negative_count():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "defects": [1, -2, 1], "n": [10, 10, 10]})
    options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    with pytest.raises(ValueError, match="non-negative integer"):
        spc.chart(df, kind="p_prime", options=options)


def test_p_prime_never_auto_suggested():
    # suggest_chart_type() has no concept of p_prime -- kind="auto" on
    # p-shaped data must resolve to plain 'p', never 'p_prime' (opt-in only).
    dates = _dates(10)
    df = pd.DataFrame({"period": dates, "defects": [2, 3, 1, 4, 2, 3, 1, 4, 2, 3], "n": list(range(10, 20))})
    options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    result = spc.chart(df, kind="auto", options=options)
    assert result.kind == "p"


# --- End-to-end U' chart -----------------------------------------------


def test_u_prime_matches_u_when_data_is_pure_poisson_noise():
    rng = random.Random(3)
    dates = _dates(15)
    events = [rng.randint(3, 9) for _ in range(15)]  # around u=0.2 with opportunity=25
    df = pd.DataFrame({"period": dates, "events": events, "opportunity": [25] * 15})
    options = ProportionChartOptions(value_field="events", date_field="period", denominator_field="opportunity")

    result_u = spc.chart(df, kind="u", options=options)
    result_up = spc.chart(df, kind="u_prime", options=options)

    for pt_u, pt_up in zip(result_u.points, result_up.points, strict=True):
        assert pt_up.ucl == pytest.approx(pt_u.ucl, rel=0.1)
        assert pt_up.mean == pytest.approx(pt_u.mean)


def test_u_prime_wrong_options_type_raises_clear_error():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "y": [1, 2, 3]})
    with pytest.raises(TypeError, match="ProportionChartOptions"):
        spc.chart(df, kind="u_prime", options=BaseChartOptions(value_field="y", date_field="period"))


def test_u_prime_low_count_warning_always_false():
    # u/u_prime are Poisson-based -- the binomial low-count diagnostic
    # never applies, even for a very small count.
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "events": [1, 1, 1], "opportunity": [5, 5, 5]})
    options = ProportionChartOptions(value_field="events", date_field="period", denominator_field="opportunity")
    result = spc.chart(df, kind="u_prime", options=options)
    assert not any(p.low_count_warning for p in result.points)


# --- External baseline composes with sigma_z ------------------------------


def test_p_prime_baseline_widens_limits_and_reduces_false_alarms_under_overdispersion():
    # Baseline predicts the true generating rate closely (small residual
    # noise only) for the first half, then a period where the true rate
    # departs further from what the baseline predicts -- genuine
    # overdispersion relative to the baseline, not just sampling noise.
    # Classical p (baseline-centred) should over-flag; p_prime should
    # widen enough to stop treating the departure as special-cause.
    rng = random.Random(11)
    dates = _dates(10)
    baseline_rate = 0.10
    true_ps = [0.10, 0.11, 0.09, 0.25, 0.24, 0.26, 0.10, 0.09, 0.25, 0.24]
    n = 200
    defects = [sum(1 for _ in range(n) if rng.random() < p) for p in true_ps]
    df = pd.DataFrame({"period": dates, "defects": defects, "n": [n] * 10})
    baseline = [ExternalBaseline(x=d, expected_value=baseline_rate, source="test") for d in dates]
    options = ProportionChartOptions(
        value_field="defects", date_field="period", denominator_field="n", baseline=baseline
    )

    result_p = spc.chart(df, kind="p", options=options)
    result_pp = spc.chart(df, kind="p_prime", options=options)

    assert all(p.mean == pytest.approx(baseline_rate) for p in result_p.points)
    assert all(p.mean == pytest.approx(baseline_rate) for p in result_pp.points)
    p_flagged = sum(pt.outside_limits for pt in result_p.points)
    pp_flagged = sum(pt.outside_limits for pt in result_pp.points)
    assert p_flagged > pp_flagged
    assert result_pp.points[0].ucl - result_pp.points[0].mean > result_p.points[0].ucl - result_p.points[0].mean


def test_p_prime_baseline_reduces_to_flat_centre_when_baseline_matches_pooled_rate():
    dates = _dates(6)
    df = pd.DataFrame({"period": dates, "defects": [10, 12, 9, 11, 10, 8], "n": [100] * 6})
    plain_options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    result_plain = spc.chart(df, kind="p_prime", options=plain_options)
    pooled = result_plain.points[0].mean

    baseline = [ExternalBaseline(x=d, expected_value=pooled, source="test") for d in dates]
    baseline_options = ProportionChartOptions(
        value_field="defects", date_field="period", denominator_field="n", baseline=baseline
    )
    result_baseline = spc.chart(df, kind="p_prime", options=baseline_options)

    assert [p.ucl for p in result_baseline.points] == pytest.approx([p.ucl for p in result_plain.points])
    assert [p.lcl for p in result_baseline.points] == pytest.approx([p.lcl for p in result_plain.points])

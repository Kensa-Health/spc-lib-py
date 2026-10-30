import math

import numpy as np
import pandas as pd
import pytest

from spc.core.limits import (
    LIMIT,
    LIMIT_CLOSE,
    assurance_type,
    compute_limit_stats,
    compute_point_flags,
    estimate_autocorrelation,
    widen_amr_for_autocorrelation,
)


def test_compute_limit_stats_matches_textbook_formula_no_screening():
    y = pd.Series([10, 12, 11, 13, 9, 14, 10], dtype=float)
    stats = compute_limit_stats(y, screen_outliers="off")

    expected_mean = y.mean()
    expected_amr = y.diff().abs().mean()

    assert stats.mean == pytest.approx(expected_mean)
    assert stats.amr == pytest.approx(expected_amr)
    assert stats.ucl == pytest.approx(expected_mean + LIMIT * expected_amr)
    assert stats.lcl == pytest.approx(expected_mean - LIMIT * expected_amr)
    assert stats.near_ucl == pytest.approx(expected_mean + LIMIT_CLOSE * expected_amr)
    assert stats.near_lcl == pytest.approx(expected_mean - LIMIT_CLOSE * expected_amr)


def test_single_pass_screening_excludes_extreme_moving_range():
    # A single huge jump inflates amr; single-pass screening should exclude
    # whichever moving range(s) exceed OUTLIER_SCREEN_MULTIPLIER * (unscreened amr).
    y = pd.Series([10, 11, 9, 10, 11, 100, 10, 11, 9], dtype=float)
    off = compute_limit_stats(y, screen_outliers="off")
    screened = compute_limit_stats(y, screen_outliers="single_pass")
    assert screened.amr < off.amr


def test_iterative_screening_is_never_worse_than_single_pass():
    y = pd.Series([10, 11, 9, 10, 11, 100, 10, 11, 9], dtype=float)
    single = compute_limit_stats(y, screen_outliers="single_pass")
    iterative = compute_limit_stats(y, screen_outliers="iterative")
    # Iterative re-applies a (possibly lower) threshold repeatedly, so it
    # should exclude at least as much as a single pass.
    assert iterative.amr <= single.amr + 1e-9


def test_iterative_screening_converges_to_a_fixed_point():
    # A second call using the converged amr as the "unscreened" input should
    # reproduce the same excluded set (i.e. re-running screening is a no-op).
    y = pd.Series([10, 11, 9, 10, 11, 100, 10, 11, 9, 105, 11, 9], dtype=float)
    once = compute_limit_stats(y, screen_outliers="iterative")
    twice = compute_limit_stats(y, screen_outliers="iterative")
    assert once.amr == pytest.approx(twice.amr)


def test_off_leaves_full_unscreened_moving_range_average():
    y = pd.Series([10, 11, 9, 10, 11, 100, 10, 11, 9], dtype=float)
    stats = compute_limit_stats(y, screen_outliers="off")
    assert stats.amr == pytest.approx(y.diff().abs().mean())


def test_compute_point_flags():
    y = pd.Series([5, 30, 2, -30], dtype=float)
    mean = pd.Series(10.0, index=y.index)
    ucl = pd.Series(20.0, index=y.index)
    lcl = pd.Series(0.0, index=y.index)
    near_ucl = pd.Series(17.0, index=y.index)
    near_lcl = pd.Series(3.0, index=y.index)

    outside, rel, close = compute_point_flags(y, mean, ucl, lcl, near_ucl, near_lcl)

    assert list(outside) == [False, True, False, True]
    assert list(rel) == [-1, 1, -1, -1]
    assert list(close) == [False, False, True, False]


# --- estimate_autocorrelation ----------------------------------------------


def test_estimate_autocorrelation_matches_shared_mean_formula():
    x = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])
    xbar = x.mean()
    expected = ((x[1:].to_numpy() - xbar) * (x[:-1].to_numpy() - xbar)).sum() / ((x - xbar) ** 2).sum()
    assert estimate_autocorrelation(x) == pytest.approx(expected)


def test_estimate_autocorrelation_positive_for_a_random_walk():
    rng = np.random.default_rng(0)
    x = pd.Series(np.cumsum(rng.normal(size=200)))
    assert estimate_autocorrelation(x) > 0.8


def test_estimate_autocorrelation_negative_for_an_alternating_series():
    x = pd.Series([0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0])
    assert estimate_autocorrelation(x) < -0.5


def test_estimate_autocorrelation_nan_below_two_points():
    assert math.isnan(estimate_autocorrelation(pd.Series([5.0])))
    assert math.isnan(estimate_autocorrelation(pd.Series([], dtype=float)))


def test_estimate_autocorrelation_nan_for_zero_variance():
    assert math.isnan(estimate_autocorrelation(pd.Series([5.0] * 10)))


def test_estimate_autocorrelation_pairwise_deletion_differs_from_naive_dropna():
    # A NaN in the middle: pairwise deletion drops both pairs touching it,
    # naive dropna-then-correlate would instead treat the points either
    # side of the gap as adjacent -- a different lag-1 structure.
    with_gap = pd.Series([1.0, 2.0, np.nan, 10.0, 1.0, 2.0, 10.0, 1.0])
    naive = estimate_autocorrelation(pd.Series(with_gap.dropna().reset_index(drop=True)))
    pairwise = estimate_autocorrelation(with_gap)
    assert pairwise != pytest.approx(naive)


def test_estimate_autocorrelation_nan_when_too_few_valid_pairs_remain():
    # Each valid point is isolated between two NaNs -- no valid adjacent pair exists.
    x = pd.Series([1.0, np.nan, 5.0, np.nan, 9.0])
    assert math.isnan(estimate_autocorrelation(x))


# --- widen_amr_for_autocorrelation ------------------------------------------


def test_widen_amr_for_autocorrelation_is_a_no_op_when_off():
    assert widen_amr_for_autocorrelation(2.0, 0.9, autocorrelation_adjustment=False) == 2.0


def test_widen_amr_for_autocorrelation_is_a_no_op_for_nan_rho():
    assert widen_amr_for_autocorrelation(2.0, float("nan"), autocorrelation_adjustment=True) == 2.0


def test_widen_amr_for_autocorrelation_widens_for_positive_rho():
    widened = widen_amr_for_autocorrelation(2.0, 0.5, autocorrelation_adjustment=True)
    assert widened == pytest.approx(2.0 / math.sqrt(0.5))
    assert widened > 2.0


def test_widen_amr_for_autocorrelation_narrows_for_negative_rho():
    # No floor at 0 -- a genuinely negative rho is allowed to narrow the
    # result, not just clamp to a no-op.
    narrowed = widen_amr_for_autocorrelation(2.0, -0.5, autocorrelation_adjustment=True)
    assert narrowed == pytest.approx(2.0 / math.sqrt(1.5))
    assert narrowed < 2.0


def test_widen_amr_for_autocorrelation_caps_at_0_95():
    capped = widen_amr_for_autocorrelation(2.0, 0.999, autocorrelation_adjustment=True)
    uncapped_would_be = 2.0 / math.sqrt(1 - 0.999)
    assert capped == pytest.approx(2.0 / math.sqrt(1 - 0.95))
    assert capped < uncapped_would_be


# --- compute_limit_stats(autocorrelation_adjustment=...) -------------------


def test_compute_limit_stats_autocorrelation_field_always_populated():
    # Visible for review even when not applied -- same split as low_count_warning.
    y = pd.Series(np.cumsum(np.random.default_rng(1).normal(size=20)))
    off = compute_limit_stats(y, screen_outliers="off", autocorrelation_adjustment=False)
    assert not math.isnan(off.autocorrelation)


def test_compute_limit_stats_default_reproduces_unadjusted_numbers_exactly():
    y = pd.Series(np.cumsum(np.random.default_rng(2).normal(size=30)))
    with_default = compute_limit_stats(y, screen_outliers="single_pass")
    explicit_off = compute_limit_stats(y, screen_outliers="single_pass", autocorrelation_adjustment=False)
    assert with_default.ucl == explicit_off.ucl
    assert with_default.lcl == explicit_off.lcl
    assert with_default.amr == explicit_off.amr


def test_compute_limit_stats_widens_for_positive_rho_by_the_exact_factor():
    rng = np.random.default_rng(3)
    y = pd.Series(np.cumsum(rng.normal(size=40)))  # random walk -> strong positive rho
    off = compute_limit_stats(y, screen_outliers="off")
    on = compute_limit_stats(y, screen_outliers="off", autocorrelation_adjustment=True)
    assert on.autocorrelation > 0.5
    assert (on.ucl - on.lcl) > (off.ucl - off.lcl)
    expected_amr = off.amr / math.sqrt(1 - min(on.autocorrelation, 0.95))
    assert on.ucl - on.lcl == pytest.approx(2 * LIMIT * expected_amr)
    assert on.near_ucl - on.near_lcl == pytest.approx(2 * LIMIT_CLOSE * expected_amr)


def test_compute_limit_stats_narrows_for_negative_rho():
    y = pd.Series([0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0])
    off = compute_limit_stats(y, screen_outliers="off")
    on = compute_limit_stats(y, screen_outliers="off", autocorrelation_adjustment=True)
    assert on.autocorrelation < 0
    assert (on.ucl - on.lcl) < (off.ucl - off.lcl)


def test_compute_limit_stats_estimates_rho_from_outlier_excluded_residuals():
    # A single huge spike: rho estimated on the outlier-excluded residuals
    # this function actually uses should differ from rho on the raw,
    # unscreened residuals.
    y = pd.Series([10, 11, 9, 10, 11, 9, 10, 11, 9, 100, 10, 11, 9, 10, 11], dtype=float)
    screened = compute_limit_stats(y, screen_outliers="single_pass")
    raw_rho = estimate_autocorrelation(y - y.mean())
    assert screened.autocorrelation != pytest.approx(raw_rho)


@pytest.mark.parametrize(
    "target,ucl,lcl,direction,expected",
    [
        (25, 20, 10, "increase", "consistent_fail"),
        (5, 20, 10, "increase", "consistent_pass"),
        (15, 20, 10, "increase", "inconsistent"),
        (25, 20, 10, "decrease", "consistent_pass"),
        (5, 20, 10, "decrease", "consistent_fail"),
        (15, 20, 10, "decrease", "inconsistent"),
        (15, 20, 10, "neutral", None),
    ],
)
def test_assurance_type(
        target,
        ucl,
        lcl,
        direction,
        expected
):
    assert assurance_type(target, ucl, lcl, direction) == expected

"""The library against the naive reference implementations in reference.py, over
many seeded random cases. Where the two disagree, one of them has a bug.
"""

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import reference as ref

import spc
from spc.core.limits import assurance_type
from spc.core.models import BaseChartOptions, LimitMode, ProportionChartOptions
from spc.core.rules import point_type

BASE = date(2024, 1, 1)


def dates(
        n
):
    return [BASE + timedelta(weeks=i) for i in range(n)]


def points(
        result
):
    return sorted(result.points, key=lambda p: p.x)


def as_array(
        values
):
    return np.array([math.nan if v is None else v for v in values], dtype=float)


def same(
        a,
        b,
        tol=1e-8
):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.allclose(
        np.nan_to_num(a, nan=1e300), np.nan_to_num(b, nan=1e300), rtol=tol, atol=tol
    )


def draw_mode(
        rng
):
    mode = ["segment", "fixed_n", "expanding"][int(rng.integers(0, 3))]
    return mode, (int(rng.integers(2, 12)) if mode != "segment" else None)


def run_cases(
        count,
        seed_offset,
        body
):
    """Calls body(rng) for each seed; returns how many produced a comparison."""
    ran = 0
    for case in range(count):
        if body(np.random.default_rng(seed_offset + case)):
            ran += 1
    return ran


# --- XmR limits and the four rules ---------------------------------------------------------------


def _xmr_series(
        rng
):
    n = int(rng.choice([2, 3, 5, 8, 13, 21, 40, 80]))
    y = (
        rng.normal(20, 4, n)
        + (rng.random(n) < 0.08) * rng.normal(0, 25, n)
        + (np.arange(n) > n // 2) * rng.choice([0, 6])
    )
    if rng.random() < 0.3 and n >= 12:
        y[3:10] = np.sort(y[3:10])  # a monotone stretch
    if rng.random() < 0.3 and n >= 10:
        y[2:10] = np.arange(8) * 1.5 + 10  # a clean trend
    if rng.random() < 0.2 and n >= 6:
        y[1:6] = 15.0  # ties
    if rng.random() < 0.08:
        y[:] = 12.5  # constant: every point is exactly on the centre line
    return np.round(y, 3)


def test_xmr_limits_and_rules_match_the_reference():
    def body(
            rng
    ):
        y = _xmr_series(rng)
        n = len(y)
        mode, fix_n = draw_mode(rng)
        screen = ["off", "single_pass", "iterative"][int(rng.integers(0, 3))]
        run_option = [7, 4, "proportional"][int(rng.integers(0, 3))]
        df = pd.DataFrame({"p": dates(n), "y": y})
        options = BaseChartOptions(
            value_field="y",
            date_field="p",
            limit_mode=LimitMode(mode),
            fix_after_n_points=fix_n,
            screen_outliers=screen,
            run_rule_length=run_option,
        )
        pts = points(spc.chart(df, kind="xmr", options=options))
        mean, ucl, lcl = ref.xmr_limits(y, mode, fix_n, screen)
        context = (mode, screen, run_option, n, fix_n)
        assert same([p.mean for p in pts], mean), context
        assert same(as_array(p.ucl for p in pts), ucl), context
        assert same(as_array(p.lcl for p in pts), lcl), context
        expected = ref.rule_winners(y, mean, ucl, lcl, ref.run_length(run_option, n))
        assert [p.special_cause_rule for p in pts] == expected, context
        return True

    assert run_cases(400, 0, body) == 400


# --- attribute, Laney and subgroup limits ---------------------------------------------------------------


@pytest.mark.parametrize("kind", ["p", "np", "u", "c", "p_prime", "u_prime"])
def test_attribute_limits_match_the_reference(
        kind
):
    def body(
            rng
    ):
        n = int(rng.choice([3, 5, 8, 13, 21, 40]))
        mode, fix_n = draw_mode(rng)
        denominator = np.full(n, 150.0) if kind == "np" else rng.integers(40, 400, n).astype(float)
        if kind in ("p", "np", "p_prime"):
            y = rng.binomial(denominator.astype(int), rng.uniform(0.05, 0.45)).astype(float)
        elif kind in ("u", "u_prime"):
            y = rng.poisson(denominator * rng.uniform(0.01, 0.2)).astype(float)
        else:
            y = rng.poisson(rng.uniform(2, 30), n).astype(float)
        if rng.random() < 0.25:
            y[: min(4, n)] = 0  # a zero-event window: the Laney zero-centre case
        df = pd.DataFrame({"p": dates(n), "y": y, "n": denominator})
        common = {"value_field": "y", "date_field": "p", "limit_mode": LimitMode(mode), "fix_after_n_points": fix_n}
        options = BaseChartOptions(**common) if kind == "c" else ProportionChartOptions(denominator_field="n", **common)
        pts = points(spc.chart(df, kind=kind, options=options))
        plotted, centre, ucl, lcl = ref.attribute_limits(kind, y, denominator, mode, fix_n)
        context = (kind, mode, n, fix_n)
        assert same([p.y for p in pts], plotted), context
        assert same([p.mean for p in pts], centre), context
        assert same(as_array(p.ucl for p in pts), ucl), context
        assert same(as_array(p.lcl for p in pts), lcl), context
        return True

    assert run_cases(150, 10_000, body) == 150


@pytest.mark.parametrize("kind", ["xbar_r", "xbar_s"])
def test_subgroup_limits_match_the_reference(
        kind
):
    def body(
            rng
    ):
        k = int(rng.choice([3, 6, 10, 20]))
        size = int(rng.integers(2, 11))
        mode, fix_n = draw_mode(rng)
        rows = []
        for i in range(k):
            center = 50 + rng.normal(0, 1)
            rows += [{"p": dates(k)[i], "y": float(v)} for v in rng.normal(center, 3, size)]
        df = pd.DataFrame(rows)
        options = BaseChartOptions(
            value_field="y", date_field="p", limit_mode=LimitMode(mode), fix_after_n_points=fix_n
        )
        pts = points(spc.chart(df, kind=kind, options=options))
        grouped = df.groupby("p")["y"]
        means = grouped.mean().to_numpy()
        spreads = (grouped.max() - grouped.min()).to_numpy() if kind == "xbar_r" else grouped.std(ddof=1).to_numpy()
        centre, half_width = ref.subgroup_limits(kind, size, means, spreads, mode, fix_n)
        assert same([p.y for p in pts], means)
        assert same([p.mean for p in pts], centre)
        # the published constants are only given to 3-4 decimal places
        assert same(as_array(p.ucl for p in pts), centre + half_width, tol=2e-3), (kind, mode, size)
        assert same(as_array(p.lcl for p in pts), centre - half_width, tol=2e-3), (kind, mode, size)
        return True

    assert run_cases(150, 20_000, body) == 150


# --- autocorrelation adjustment -----------------------------------------------------------------------------


def test_autocorrelation_adjustment_matches_the_reference():
    def body(
            rng
    ):
        n = int(rng.choice([6, 10, 20, 40, 80]))
        phi = rng.uniform(-0.6, 0.95)
        y = np.zeros(n)
        y[0] = rng.normal()
        for i in range(1, n):
            y[i] = phi * y[i - 1] + rng.normal()
        y = y * 3 + 20
        df = pd.DataFrame({"p": dates(n), "y": y})
        common = {"value_field": "y", "date_field": "p", "screen_outliers": "off"}
        adjusted = spc.chart(df, kind="xmr", options=BaseChartOptions(autocorrelation_adjustment=True, **common))
        plain = spc.chart(df, kind="xmr", options=BaseChartOptions(**common))
        rho = ref.lag1_autocorrelation(y)
        mean_range = np.abs(np.diff(y)).mean()
        widened = mean_range / math.sqrt(1 - min(rho, 0.95))
        assert points(adjusted)[0].ucl == pytest.approx(y.mean() + 2.66 * widened, rel=1e-9)
        assert adjusted.segments[0].autocorrelation_estimate == pytest.approx(rho, rel=1e-9)
        assert points(plain)[0].ucl == pytest.approx(y.mean() + 2.66 * mean_range, rel=1e-9)
        return True

    assert run_cases(200, 30_000, body) == 200


# --- classification tables -----------------------------------------------------------------------------------

LABELS = [
    ("outside_limits", "Above UCL", True),
    ("outside_limits", "Below LCL", False),
    ("seven_point_shift", "7 Points Above CL", True),
    ("seven_point_shift", "7 Points Below CL", False),
    ("seven_point_trend", "7 Point Trend (Increasing)", True),
    ("seven_point_trend", "7 Point Trend (Decreasing)", False),
    ("two_in_three_near_limit", "2 in 3 Above CL", True),
    ("two_in_three_near_limit", "2 in 3 Below CL", False),
]


@pytest.mark.parametrize("improvement", ["increase", "decrease", "neutral"])
def test_point_type_follows_direction_and_improvement_direction(
        improvement
):
    for rule, label, upward in LABELS:
        if improvement == "neutral":
            expected = "special_cause_neutral_high" if upward else "special_cause_neutral_low"
        elif (improvement == "increase") == upward:
            expected = "special_cause_improvement"
        else:
            expected = "special_cause_concern"
        assert point_type(rule, label, improvement) == expected, (rule, label)
    assert point_type(None, "Common Cause", improvement) == "common_cause"


@pytest.mark.parametrize(("target", "ucl", "lcl"), [(10, 8, 4), (3, 8, 4), (6, 8, 4), (8, 8, 4), (4, 8, 4)])
def test_assurance_follows_where_the_target_sits(
        target,
        ucl,
        lcl
):
    above = "consistent_fail" if target > ucl else "consistent_pass" if target < lcl else "inconsistent"
    flipped = {
        "consistent_fail": "consistent_pass",
        "consistent_pass": "consistent_fail",
        "inconsistent": "inconsistent",
    }
    assert assurance_type(target, ucl, lcl, "increase") == above
    assert assurance_type(target, ucl, lcl, "decrease") == flipped[above]
    assert assurance_type(target, ucl, lcl, "neutral") is None


# --- a rebase is two independent charts -----------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["xmr", "p", "np", "u", "c", "p_prime", "u_prime", "g"])
def test_a_rebased_chart_equals_the_two_charts_on_either_side(
        kind
):
    def body(
            rng
    ):
        n = int(rng.choice([8, 13, 21, 40]))
        mode, fix_n = draw_mode(rng)
        denominator = np.full(n, 150.0) if kind == "np" else rng.integers(40, 400, n).astype(float)
        if kind == "xmr":
            y = rng.normal(20, 4, n)
        elif kind in ("p", "np", "p_prime"):
            y = rng.binomial(denominator.astype(int), 0.2).astype(float)
        elif kind in ("u", "u_prime"):
            y = rng.poisson(denominator * 0.05).astype(float)
        elif kind == "c":
            y = rng.poisson(15, n).astype(float)
        else:
            y = rng.integers(1, 200, n).astype(float)
        df = pd.DataFrame({"p": dates(n), "y": y, "n": denominator})
        common = {
            "value_field": "y",
            "date_field": "p",
            "limit_mode": LimitMode(mode),
            "fix_after_n_points": fix_n,
            "screen_outliers": ["off", "single_pass", "iterative"][int(rng.integers(0, 3))],
        }
        proportion = kind in ("p", "np", "u", "p_prime", "u_prime")

        def chart(
                frame,
                rebase=None
        ):
            if proportion:
                options = ProportionChartOptions(denominator_field="n", rebase=rebase, **common)
            else:
                options = BaseChartOptions(rebase=rebase, **common)
            return spc.chart(frame, kind=kind, options=options)

        cut = dates(n)[n // 2]
        try:
            whole = chart(df, rebase=[cut])
            left = chart(df[df["p"] < cut])
            right = chart(df[df["p"] >= cut])
        except ValueError:
            return False  # e.g. a half too short for the chosen limit mode

        def signature(
                pts
        ):
            return [
                (
                    p.x,
                    round(p.y, 9),
                    round(p.mean, 9),
                    p.ucl and round(p.ucl, 9),
                    p.lcl and round(p.lcl, 9),
                    p.special_cause_rule,
                )
                for p in pts
            ]

        assert signature([p for p in points(whole) if p.segment_id == 0]) == signature(points(left))
        assert signature([p for p in points(whole) if p.segment_id == 1]) == signature(points(right))
        return True

    assert run_cases(60, 40_000, body) >= 40


def test_a_constant_series_fires_no_rule():
    """Points exactly on the centre line are on neither side, so there is no run of 7 on one side."""
    df = pd.DataFrame({"p": dates(20), "y": [12.5] * 20})
    for run_option in (7, 4, "proportional"):
        result = spc.chart(
            df, kind="xmr", options=BaseChartOptions(value_field="y", date_field="p", run_rule_length=run_option)
        )
        assert {p.special_cause_rule for p in result.points} == {None}, run_option

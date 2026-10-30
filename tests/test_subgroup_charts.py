"""Xbar-R / Xbar-S chart tests, verified against hand computation."""

import statistics
from datetime import date, timedelta

import pandas as pd
import pytest

import spc
from spc.core.limits import XBAR_R_CONSTANTS, c4, xbar_r_constants, xbar_s_constants
from spc.core.models import BaseChartOptions, ExternalBaseline


def _dates(
        n,
        start=date(2024, 1, 1)
):
    return [start + timedelta(weeks=i) for i in range(n)]


def _subgroup_df(
        dates,
        subgroup_data,
        value_field="measurement",
        date_field="period"
):
    rows = [{date_field: d, value_field: v} for d, vals in zip(dates, subgroup_data, strict=True) for v in vals]
    return pd.DataFrame(rows)


# --- Constants -----------------------------------------------------------


def test_c4_matches_nist_reference_value():
    assert c4(10) == pytest.approx(0.9727, abs=1e-4)  # NIST e-Handbook states this exact value


@pytest.mark.parametrize("n,expected_a2", [(2, 1.880), (5, 0.577), (10, 0.308)])
def test_xbar_r_constants_match_reference_table(
        n,
        expected_a2
):
    a2, _d3, _d4 = xbar_r_constants(n)
    assert a2 == pytest.approx(expected_a2, abs=1e-3)


def test_xbar_r_constants_below_minimum_raises():
    with pytest.raises(ValueError, match="subgroup size"):
        xbar_r_constants(1)


def test_xbar_r_constants_extends_past_table_via_computed_moments():
    # n=15 is past XBAR_R_CONSTANTS' n=10 cap -- must compute rather than
    # raise -- computed via numerical integration over the order statistics
    # of the standard normal, same rigor tier as xbar_s's c4(n)-based
    # derivation.
    pytest.importorskip("scipy")
    a2, d3, d4 = xbar_r_constants(15)
    assert a2 > 0
    assert 0 <= d3 < d4
    # A2/D4 both decrease monotonically with subgroup size -- n=15 should
    # sit strictly below its n=10 table neighbour on both.
    a2_10, _d3_10, d4_10 = xbar_r_constants(10)
    assert a2 < a2_10
    assert d4 < d4_10


def test_xbar_r_constants_computed_matches_table_for_n_2_to_10():
    # The critical correctness check: the same numerical-integration
    # formula used to extend past n=10 must reproduce every existing,
    # NIST-cross-checked table value when run over the table's own range.
    pytest.importorskip("scipy")
    from spc.core.limits import _xbar_r_constants_computed

    for n, (table_a2, table_d3, table_d4) in XBAR_R_CONSTANTS.items():
        a2, d3, d4 = _xbar_r_constants_computed(n)
        assert a2 == pytest.approx(table_a2, abs=2e-3)
        assert d3 == pytest.approx(table_d3, abs=2e-3)
        assert d4 == pytest.approx(table_d4, abs=2e-3)


def test_xbar_s_constants_no_upper_cap():
    # xbar_s exists precisely because xbar_r's table stops at 10 -- must work past it.
    a3, b3, b4 = xbar_s_constants(20)
    assert a3 > 0
    assert 0 <= b3 < b4


# --- Xbar-R end to end -----------------------------------------------------


def test_xbar_r_hand_computed():
    dates = _dates(4)
    subgroup_data = [[10, 12, 11], [9, 10, 8], [13, 14, 12], [11, 10, 12]]
    df = _subgroup_df(dates, subgroup_data)
    options = BaseChartOptions(value_field="measurement", date_field="period")
    result = spc.chart(df, kind="xbar_r", options=options)

    means = [statistics.mean(v) for v in subgroup_data]
    ranges = [max(v) - min(v) for v in subgroup_data]
    grand_mean = statistics.mean(means)
    r_bar = statistics.mean(ranges)
    a2, _, _ = xbar_r_constants(3)

    assert [p.y for p in result.points] == pytest.approx(means)
    assert result.points[0].mean == pytest.approx(grand_mean)
    assert result.points[0].ucl == pytest.approx(grand_mean + a2 * r_bar)
    assert result.points[0].lcl == pytest.approx(grand_mean - a2 * r_bar)


def test_xbar_r_varying_subgroup_size_raises():
    dates = _dates(3)
    subgroup_data = [[10, 12, 11], [9, 10], [13, 14, 12]]  # middle subgroup has only 2
    df = _subgroup_df(dates, subgroup_data)
    options = BaseChartOptions(value_field="measurement", date_field="period")
    with pytest.raises(ValueError, match="constant subgroup size"):
        spc.chart(df, kind="xbar_r", options=options)


def test_xbar_r_single_measurement_subgroup_raises():
    dates = _dates(2)
    subgroup_data = [[10, 12, 11], [9]]  # second subgroup has only 1 row -- no range
    df = _subgroup_df(dates, subgroup_data)
    options = BaseChartOptions(value_field="measurement", date_field="period")
    with pytest.raises(ValueError, match="at least 2 raw measurements"):
        spc.chart(df, kind="xbar_r", options=options)


# --- Xbar-S end to end -----------------------------------------------------


def test_xbar_s_hand_computed():
    dates = _dates(3)
    subgroup_data = [[10, 12, 11, 9, 13], [9, 10, 8, 11, 12], [13, 14, 12, 15, 11]]
    df = _subgroup_df(dates, subgroup_data)
    options = BaseChartOptions(value_field="measurement", date_field="period")
    result = spc.chart(df, kind="xbar_s", options=options)

    means = [statistics.mean(v) for v in subgroup_data]
    stds = [statistics.stdev(v) for v in subgroup_data]  # ddof=1, matches pandas' default
    grand_mean = statistics.mean(means)
    s_bar = statistics.mean(stds)
    a3, _, _ = xbar_s_constants(5)

    assert [p.y for p in result.points] == pytest.approx(means)
    assert result.points[0].mean == pytest.approx(grand_mean)
    assert result.points[0].ucl == pytest.approx(grand_mean + a3 * s_bar)
    assert result.points[0].lcl == pytest.approx(grand_mean - a3 * s_bar)


def test_xbar_facets_computed_independently():
    dates = _dates(3)
    rows = []
    for d, vals in zip(dates, [[10, 12, 11], [9, 10, 8], [13, 14, 12]], strict=True):
        for v in vals:
            rows.append({"period": d, "measurement": v, "line": "A"})
    for d, vals in zip(dates, [[50, 52, 51], [49, 50, 48], [53, 54, 52]], strict=True):
        for v in vals:
            rows.append({"period": d, "measurement": v, "line": "B"})
    df = pd.DataFrame(rows)
    options = BaseChartOptions(value_field="measurement", date_field="period", facet_field="line")
    result = spc.chart(df, kind="xbar_r", options=options)

    a_means = [p.y for p in result.points if p.facet == "A"]
    b_means = [p.y for p in result.points if p.facet == "B"]
    assert min(a_means) > 8 and max(a_means) < 15
    assert min(b_means) > 47 and max(b_means) < 55


# --- External baseline -------------------------------------------------


def test_xbar_r_baseline_seam_reduces_to_flat_mean_when_baseline_is_zero():
    dates = _dates(4)
    subgroup_data = [[10, 12, 11], [9, 10, 8], [13, 14, 12], [11, 10, 12]]
    df = _subgroup_df(dates, subgroup_data)
    baseline = [ExternalBaseline(x=d, expected_value=0.0, source="test") for d in dates]

    plain_options = BaseChartOptions(value_field="measurement", date_field="period")
    baseline_options = BaseChartOptions(value_field="measurement", date_field="period", baseline=baseline)

    result_plain = spc.chart(df, kind="xbar_r", options=plain_options)
    result_baseline = spc.chart(df, kind="xbar_r", options=baseline_options)

    assert [p.mean for p in result_plain.points] == pytest.approx([p.mean for p in result_baseline.points])
    assert [p.ucl for p in result_plain.points] == pytest.approx([p.ucl for p in result_baseline.points])


def test_xbar_r_baseline_matching_a_genuine_alternating_pattern_reads_as_common_cause():
    # Without a baseline, an alternating 11/19/11/19 subgroup mean pools to
    # a flat 15 and every point reads as special-cause. With a baseline
    # that correctly predicts the alternation, each point matches its own
    # period's expectation exactly and none should be flagged.
    dates = _dates(4)
    subgroup_data = [[10, 12, 11], [19, 20, 18], [10, 12, 11], [19, 20, 18]]
    df = _subgroup_df(dates, subgroup_data)

    plain_options = BaseChartOptions(value_field="measurement", date_field="period")
    result_plain = spc.chart(df, kind="xbar_r", options=plain_options)
    assert any(p.point_type != "common_cause" for p in result_plain.points)

    baseline = [
        ExternalBaseline(x=d, expected_value=(11.0 if i % 2 == 0 else 19.0), source="test") for i, d in enumerate(dates)
    ]
    baseline_options = BaseChartOptions(value_field="measurement", date_field="period", baseline=baseline)
    result_baseline = spc.chart(df, kind="xbar_r", options=baseline_options)
    assert all(p.point_type == "common_cause" for p in result_baseline.points)
    assert [p.mean for p in result_baseline.points] == pytest.approx([11.0, 19.0, 11.0, 19.0])


def test_xbar_s_baseline_matching_a_genuine_alternating_pattern_reads_as_common_cause():
    dates = _dates(4)
    subgroup_data = [[10, 12, 11], [19, 20, 18], [10, 12, 11], [19, 20, 18]]
    df = _subgroup_df(dates, subgroup_data)
    baseline = [
        ExternalBaseline(x=d, expected_value=(11.0 if i % 2 == 0 else 19.0), source="test") for i, d in enumerate(dates)
    ]
    options = BaseChartOptions(value_field="measurement", date_field="period", baseline=baseline)
    result = spc.chart(df, kind="xbar_s", options=options)
    assert all(p.point_type == "common_cause" for p in result.points)
    assert [p.mean for p in result.points] == pytest.approx([11.0, 19.0, 11.0, 19.0])


def test_xbar_r_baseline_does_not_change_the_spread_estimate():
    # Subgroup spread (range) is translation-invariant -- a baseline
    # shifting the centre line shouldn't move the control-limit width.
    dates = _dates(4)
    subgroup_data = [[10, 12, 11], [19, 20, 18], [10, 12, 11], [19, 20, 18]]
    df = _subgroup_df(dates, subgroup_data)

    plain_options = BaseChartOptions(value_field="measurement", date_field="period")
    result_plain = spc.chart(df, kind="xbar_r", options=plain_options)
    width_plain = result_plain.points[0].ucl - result_plain.points[0].mean

    baseline = [
        ExternalBaseline(x=d, expected_value=(11.0 if i % 2 == 0 else 19.0), source="test") for i, d in enumerate(dates)
    ]
    baseline_options = BaseChartOptions(value_field="measurement", date_field="period", baseline=baseline)
    result_baseline = spc.chart(df, kind="xbar_r", options=baseline_options)
    width_baseline = result_baseline.points[0].ucl - result_baseline.points[0].mean

    assert width_plain == pytest.approx(width_baseline)


def test_xbar_r_baseline_missing_coverage_raises():
    dates = _dates(3)
    subgroup_data = [[10, 12, 11], [9, 10, 8], [13, 14, 12]]
    df = _subgroup_df(dates, subgroup_data)
    baseline = [ExternalBaseline(x=dates[0], expected_value=0.0, source="test")]  # only 1 of 3 covered
    options = BaseChartOptions(value_field="measurement", date_field="period", baseline=baseline)
    with pytest.raises(ValueError, match="no entry for facet"):
        spc.chart(df, kind="xbar_r", options=options)

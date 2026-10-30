"""P/NP/C/U chart tests -- binomial/Poisson formulas, verified against
hand computation.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

import spc
from spc.core.limits import c_chart_limits, np_chart_limits, p_chart_limits, u_chart_limits
from spc.core.models import BaseChartOptions, ExternalBaseline, LimitMode, ProportionChartOptions


def _dates(
        n,
        start=date(2024, 1, 1)
):
    return [start + timedelta(weeks=i) for i in range(n)]


# --- Pure formula tests (core/limits.py) -----------------------------------


def test_p_chart_limits_hand_computed():
    stats = p_chart_limits(0.5, 10)
    assert stats.ucl == pytest.approx(0.5 + 3 * (0.5 * 0.5 / 10) ** 0.5)
    assert stats.lcl == pytest.approx(0.5 - 3 * (0.5 * 0.5 / 10) ** 0.5)
    assert stats.near_ucl == pytest.approx(0.5 + 2 * (0.5 * 0.5 / 10) ** 0.5)


def test_p_chart_limits_clip_lower_at_zero_not_upper_at_one():
    stats = p_chart_limits(0.02, 5)  # small p, small n -> naive lcl would go negative
    assert stats.lcl == 0.0
    assert stats.ucl > 0  # not clipped, even though it could exceed 1 for extreme inputs


def test_np_chart_limits_hand_computed():
    stats = np_chart_limits(6.0, 50.0)
    expected_se = (6.0 * (1 - 6.0 / 50.0)) ** 0.5
    assert stats.ucl == pytest.approx(6.0 + 3 * expected_se)
    assert stats.lcl == pytest.approx(max(6.0 - 3 * expected_se, 0.0))


def test_c_chart_limits_hand_computed():
    stats = c_chart_limits(4.0)
    assert stats.ucl == pytest.approx(4.0 + 3 * (4.0**0.5))
    assert stats.lcl == pytest.approx(0.0)  # 4 - 6 = -2, clipped


def test_u_chart_limits_hand_computed():
    stats = u_chart_limits(0.5, 20.0)
    expected_se = (0.5 / 20.0) ** 0.5
    assert stats.ucl == pytest.approx(0.5 + 3 * expected_se)
    assert stats.lcl == pytest.approx(0.5 - 3 * expected_se)


# --- End-to-end P chart ------------------------------------------------


def test_p_chart_pooled_mean_not_simple_average():
    # p_bar = Sum(defects)/Sum(n), NOT mean(defects/n) -- these differ once
    # n varies. 2 periods: (defects=1, n=10) -> 0.1; (defects=8, n=90) -> 0.0889.
    # Simple average of ratios = 0.0944; pooled = 9/100 = 0.09.
    dates = _dates(2)
    df = pd.DataFrame({"period": dates, "defects": [1, 8], "n": [10, 90]})
    options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    result = spc.chart(df, kind="p", options=options)
    assert result.points[0].mean == pytest.approx(9 / 100)


def test_p_chart_plots_proportion_not_raw_count():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "defects": [2, 4, 6], "n": [10, 20, 30]})
    options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    result = spc.chart(df, kind="p", options=options)
    assert [p.y for p in result.points] == pytest.approx([0.2, 0.2, 0.2])


def test_p_chart_flags_a_genuine_out_of_control_proportion():
    # SEGMENT mode pools the whole series (including the spike itself) to
    # estimate p_bar, which the spike inflates -- so the 8 preceding
    # low-but-normal points also end up flagged (seven_point_shift), the
    # same Phase I/II pooling effect documented in core/segments.py, now
    # confirmed to apply to attribute charts too. What matters here is that
    # the actual spike itself is correctly caught as outside_limits.
    dates = _dates(10)
    defects = [2, 3, 2, 3, 2, 3, 2, 3, 25, 2]  # one wildly high proportion
    df = pd.DataFrame({"period": dates, "defects": defects, "n": [50] * 10})
    options = ProportionChartOptions(
        value_field="defects", date_field="period", denominator_field="n", improvement_direction="decrease"
    )
    result = spc.chart(df, kind="p", options=options)
    spike = next(p for p in result.points if p.x == dates[8])
    assert spike.special_cause_rule == "outside_limits"
    assert (
        spike.point_type == "special_cause_concern"
    )  # a higher defect rate is bad, per improvement_direction="decrease"


def test_p_chart_wrong_options_type_raises_clear_error():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "y": [1, 2, 3]})
    with pytest.raises(TypeError, match="ProportionChartOptions"):
        spc.chart(df, kind="p", options=BaseChartOptions(value_field="y", date_field="period"))


def test_p_chart_fixed_n_locks_pooled_mean_from_first_n():
    dates = _dates(6)
    # 9/10 is a valid (if extreme) proportion -- defects can't exceed n.
    df = pd.DataFrame({"period": dates, "defects": [1, 1, 1, 9, 9, 9], "n": [10] * 6})
    options = ProportionChartOptions(
        value_field="defects",
        date_field="period",
        denominator_field="n",
        limit_mode=LimitMode.FIXED_N,
        fix_after_n_points=3,
    )
    result = spc.chart(df, kind="p", options=options)
    # p_bar locked from first 3 points only: 3/30 = 0.1, not pulled toward the later spike
    assert result.points[0].mean == pytest.approx(0.1)
    assert result.points[-1].mean == pytest.approx(0.1)


# --- End-to-end U chart ------------------------------------------------


def test_u_chart_plots_rate_per_opportunity():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "events": [2, 4, 6], "opportunity": [10, 20, 30]})
    options = ProportionChartOptions(value_field="events", date_field="period", denominator_field="opportunity")
    result = spc.chart(df, kind="u", options=options)
    assert [p.y for p in result.points] == pytest.approx([0.2, 0.2, 0.2])


# --- End-to-end NP chart -------------------------------------------------


def test_np_chart_requires_constant_denominator():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "count": [5, 6, 7], "n": [50, 50, 60]})  # n varies -- should be 'p' instead
    options = ProportionChartOptions(value_field="count", date_field="period", denominator_field="n")
    with pytest.raises(ValueError, match="constant denominator"):
        spc.chart(df, kind="np", options=options)


def test_np_chart_plots_raw_count():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "count": [5, 6, 7], "n": [50, 50, 50]})
    options = ProportionChartOptions(value_field="count", date_field="period", denominator_field="n")
    result = spc.chart(df, kind="np", options=options)
    assert [p.y for p in result.points] == [5.0, 6.0, 7.0]


# --- End-to-end C chart -------------------------------------------------


def test_c_chart_needs_no_denominator_field():
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "falls": [3, 5, 4, 6, 2]})
    options = BaseChartOptions(value_field="falls", date_field="period")
    result = spc.chart(df, kind="c", options=options)
    assert len(result.points) == 5
    assert result.points[0].mean == pytest.approx(4.0)


# --- low_count_warning (normal-approximation-to-binomial diagnostic) ----


def test_low_count_warning_true_for_p_chart_when_np_bar_below_five():
    dates = _dates(3)
    # p_bar = 3/15 = 0.2; n*p_bar = 5*0.2 = 1 < 5 at every point.
    df = pd.DataFrame({"period": dates, "defects": [1, 1, 1], "n": [5, 5, 5]})
    options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    result = spc.chart(df, kind="p", options=options)
    assert all(p.low_count_warning for p in result.points)


def test_low_count_warning_false_for_p_chart_when_approximation_is_valid():
    dates = _dates(3)
    # p_bar = 30/60 = 0.5; n*p_bar = n*(1-p_bar) = 10 >= 5 at every point.
    df = pd.DataFrame({"period": dates, "defects": [10, 10, 10], "n": [20, 20, 20]})
    options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    result = spc.chart(df, kind="p", options=options)
    assert not any(p.low_count_warning for p in result.points)


def test_low_count_warning_true_for_np_chart_when_np_bar_below_five():
    dates = _dates(3)
    # np_bar = 1, p_bar = 1/5 = 0.2; n*p_bar = 1 < 5.
    df = pd.DataFrame({"period": dates, "count": [1, 1, 1], "n": [5, 5, 5]})
    options = ProportionChartOptions(value_field="count", date_field="period", denominator_field="n")
    result = spc.chart(df, kind="np", options=options)
    assert all(p.low_count_warning for p in result.points)


def test_low_count_warning_false_for_np_chart_when_approximation_is_valid():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "count": [10, 10, 10], "n": [20, 20, 20]})
    options = ProportionChartOptions(value_field="count", date_field="period", denominator_field="n")
    result = spc.chart(df, kind="np", options=options)
    assert not any(p.low_count_warning for p in result.points)


def test_low_count_warning_always_false_for_c_chart():
    # c/u are Poisson-based, not the binomial approximation this diagnostic
    # checks -- always False here even with a very small c_bar, by design
    # (see core/attribute_segments.py).
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "falls": [1, 1, 1]})
    options = BaseChartOptions(value_field="falls", date_field="period")
    result = spc.chart(df, kind="c", options=options)
    assert not any(p.low_count_warning for p in result.points)


def test_low_count_warning_always_false_for_xmr():
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)
    assert not any(p.low_count_warning for p in result.points)


# --- suggest_chart_type consistency with the newly-implemented kinds ----


def test_suggest_chart_type_p_suggestion_is_now_actually_runnable():
    dates = _dates(10)
    df = pd.DataFrame({"period": dates, "defects": [2, 3, 1, 4, 2, 3, 1, 4, 2, 3], "n": list(range(10, 20))})
    suggestion = spc.suggest_chart_type(df, "defects", denominator_field="n", date_field="period")
    assert suggestion.kind == "p"
    options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    result = spc.chart(df, kind=suggestion.kind, options=options)  # doesn't raise NotImplementedError anymore
    assert len(result.points) == 10


# --- External baseline ---------------------------------------------------


def test_p_baseline_seam_reduces_to_pooled_ratio_when_baseline_matches_it():
    dates = _dates(4)
    df = pd.DataFrame({"period": dates, "defects": [1, 2, 1, 2], "n": [10, 10, 10, 10]})
    plain_options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    result_plain = spc.chart(df, kind="p", options=plain_options)
    pooled = result_plain.points[0].mean

    baseline = [ExternalBaseline(x=d, expected_value=pooled, source="test") for d in dates]
    baseline_options = ProportionChartOptions(
        value_field="defects", date_field="period", denominator_field="n", baseline=baseline
    )
    result_baseline = spc.chart(df, kind="p", options=baseline_options)

    assert [p.mean for p in result_baseline.points] == pytest.approx([p.mean for p in result_plain.points])
    assert [p.ucl for p in result_baseline.points] == pytest.approx([p.ucl for p in result_plain.points])


def test_p_baseline_matching_a_genuine_alternating_pattern_reads_as_common_cause():
    # Pooled p_bar for [0.1, 0.5, 0.1, 0.5] with equal n is 0.3, which puts
    # the true alternating pattern outside the pooled limits. A baseline
    # that correctly predicts the alternation should read every point as
    # common-cause instead.
    dates = _dates(4)
    df = pd.DataFrame({"period": dates, "defects": [10, 50, 10, 50], "n": [100] * 4})

    plain_options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    result_plain = spc.chart(df, kind="p", options=plain_options)
    assert any(p.outside_limits for p in result_plain.points)

    baseline = [
        ExternalBaseline(x=d, expected_value=(0.1 if i % 2 == 0 else 0.5), source="test") for i, d in enumerate(dates)
    ]
    baseline_options = ProportionChartOptions(
        value_field="defects", date_field="period", denominator_field="n", baseline=baseline
    )
    result_baseline = spc.chart(df, kind="p", options=baseline_options)
    assert not any(p.outside_limits for p in result_baseline.points)
    assert [p.mean for p in result_baseline.points] == pytest.approx([0.1, 0.5, 0.1, 0.5])


def test_np_baseline_centres_on_expected_count():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "count": [8, 12, 10], "n": [50, 50, 50]})
    baseline = [ExternalBaseline(x=d, expected_value=10.0, source="test") for d in dates]
    options = ProportionChartOptions(value_field="count", date_field="period", denominator_field="n", baseline=baseline)
    result = spc.chart(df, kind="np", options=options)
    assert [p.mean for p in result.points] == pytest.approx([10.0, 10.0, 10.0])


def test_c_baseline_centres_on_expected_count():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "falls": [2, 6, 4]})
    baseline = [ExternalBaseline(x=d, expected_value=4.0, source="test") for d in dates]
    options = BaseChartOptions(value_field="falls", date_field="period", baseline=baseline)
    result = spc.chart(df, kind="c", options=options)
    assert [p.mean for p in result.points] == pytest.approx([4.0, 4.0, 4.0])


def test_u_baseline_centres_on_expected_rate():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "events": [2, 8, 5], "opportunity": [10, 20, 15]})
    baseline = [ExternalBaseline(x=d, expected_value=0.3, source="test") for d in dates]
    options = ProportionChartOptions(
        value_field="events", date_field="period", denominator_field="opportunity", baseline=baseline
    )
    result = spc.chart(df, kind="u", options=options)
    assert [p.mean for p in result.points] == pytest.approx([0.3, 0.3, 0.3])


def test_p_baseline_low_count_warning_uses_baseline_rate_not_pooled():
    # Pooled p_bar here is moderate (n*p_bar comfortably >= 5), but the
    # baseline predicts a much smaller rate that would fail the check --
    # the diagnostic must reflect whichever centre is actually in play.
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "defects": [5, 5, 5], "n": [10, 10, 10]})
    baseline = [ExternalBaseline(x=d, expected_value=0.02, source="test") for d in dates]  # n*p_bar = 0.2
    options = ProportionChartOptions(
        value_field="defects", date_field="period", denominator_field="n", baseline=baseline
    )
    result = spc.chart(df, kind="p", options=options)
    assert all(p.low_count_warning for p in result.points)


def test_p_baseline_missing_coverage_raises():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "defects": [1, 2, 1], "n": [10, 10, 10]})
    baseline = [ExternalBaseline(x=dates[0], expected_value=0.1, source="test")]  # only 1 of 3 covered
    options = ProportionChartOptions(
        value_field="defects", date_field="period", denominator_field="n", baseline=baseline
    )
    with pytest.raises(ValueError, match="no entry for facet"):
        spc.chart(df, kind="p", options=options)


def test_attribute_and_subgroup_builders_reject_unknown_kinds():
    import pandas as pd

    from spc.core.attribute_segments import build_attribute_segment_arrays
    from spc.core.models import BaseChartOptions
    from spc.core.segments import assign_segments
    from spc.core.subgroup_segments import build_subgroup_segment_arrays

    df = pd.DataFrame({"period": pd.date_range("2024-01-01", periods=3), "y": [1, 2, 3]})
    opts = BaseChartOptions(value_field="y", date_field="period")
    segmented = assign_segments(df, opts)
    with pytest.raises(ValueError, match="attribute-chart kind"):
        build_attribute_segment_arrays(segmented, opts, "xmr")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="subgroup-chart kind"):
        build_subgroup_segment_arrays(segmented, opts, "xmr")

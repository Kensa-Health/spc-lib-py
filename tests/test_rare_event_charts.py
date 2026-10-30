"""T/G chart tests -- Nelson (1994) power transform. Verified against
hand computation (transform -> plain XmR formula -> back-transform).
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from approx_rows import assert_rows_approx

import spc
from spc.core.models import BaseChartOptions, ExternalBaseline
from spc.core.rare_event_segments import NELSON_POWER


def _dates(
        n,
        start=date(2024, 1, 1)
):
    return [start + timedelta(weeks=i) for i in range(n)]


def _hand_computed_xmr(
        values
):
    transformed = [v**NELSON_POWER for v in values]
    mean_t = sum(transformed) / len(transformed)
    mr = [abs(transformed[i] - transformed[i - 1]) for i in range(1, len(transformed))]
    amr = sum(mr) / len(mr)
    ucl_t = mean_t + 2.66 * amr
    lcl_t = max(mean_t - 2.66 * amr, 0.0)
    return mean_t ** (1 / NELSON_POWER), ucl_t ** (1 / NELSON_POWER), lcl_t ** (1 / NELSON_POWER)


# --- T chart ---------------------------------------------------------------


def test_t_chart_hand_computed():
    gaps = [5, 10, 3, 20, 8]
    event_dates = [date(2024, 1, 1)]
    for g in gaps:
        event_dates.append(event_dates[-1] + timedelta(days=g))
    df = pd.DataFrame({"incident": event_dates, "y": range(len(event_dates)), "period": event_dates})
    options = BaseChartOptions(value_field="incident", date_field="period")
    result = spc.chart(df, kind="t", options=options)

    assert [p.y for p in result.points] == pytest.approx([float(g) for g in gaps])
    expected_mean, expected_ucl, expected_lcl = _hand_computed_xmr(gaps)
    assert result.points[0].mean == pytest.approx(expected_mean)
    assert result.points[0].ucl == pytest.approx(expected_ucl)
    assert result.points[0].lcl == pytest.approx(expected_lcl)


def test_t_chart_x_position_is_the_later_event_of_each_gap():
    event_dates = [date(2024, 1, 1), date(2024, 1, 6), date(2024, 1, 20)]  # gaps: 5, 14 days
    df = pd.DataFrame({"incident": event_dates, "y": [0, 0, 0], "period": event_dates})
    options = BaseChartOptions(value_field="incident", date_field="period")
    result = spc.chart(df, kind="t", options=options)
    assert [p.x for p in result.points] == [date(2024, 1, 6), date(2024, 1, 20)]


def test_t_chart_requires_at_least_two_events():
    df = pd.DataFrame({"incident": [date(2024, 1, 1)], "y": [0], "period": [date(2024, 1, 1)]})
    options = BaseChartOptions(value_field="incident", date_field="period")
    with pytest.raises(ValueError, match="at least 2 events"):
        spc.chart(df, kind="t", options=options)


def test_t_chart_numeric_value_field_raises():
    # A numeric column would be misread by pd.to_datetime as epoch
    # nanoseconds -- plausible-looking but meaningless gaps.
    df = pd.DataFrame({"y": [1, 2, 3], "period": _dates(3)})
    with pytest.raises(ValueError, match="event timestamps"):
        spc.chart(df, kind="t", options=BaseChartOptions(value_field="y", date_field="period"))


def test_t_chart_ignores_date_field():
    event_dates = [date(2024, 1, 1), date(2024, 1, 6), date(2024, 1, 20)]
    df = pd.DataFrame({"incident": event_dates, "other": [date(2000, 1, 1)] * 3})
    result = spc.chart(df, kind="t", options=BaseChartOptions(value_field="incident", date_field="other"))
    assert [p.x for p in result.points] == [date(2024, 1, 6), date(2024, 1, 20)]


def test_g_chart_never_sets_suggested_kind():
    df = pd.DataFrame({"period": _dates(5), "cnt": [12, 25, 8, 40, 15]})
    result = spc.chart(df, kind="g", options=BaseChartOptions(value_field="cnt", date_field="period"))
    assert result.suggested_kind is None


def test_t_chart_limits_never_go_negative_despite_transformed_space_lcl():
    # A wide spread of gaps can push the transformed-space LCL negative;
    # back-transforming a negative number via **(1/NELSON_POWER) would be
    # complex-valued if not clipped first.
    gaps = [1, 1, 1, 1, 1, 1, 100]
    event_dates = [date(2024, 1, 1)]
    for g in gaps:
        event_dates.append(event_dates[-1] + timedelta(days=g))
    df = pd.DataFrame({"incident": event_dates, "y": range(len(event_dates)), "period": event_dates})
    options = BaseChartOptions(value_field="incident", date_field="period")
    result = spc.chart(df, kind="t", options=options)
    assert all(p.lcl is not None and p.lcl >= 0 for p in result.points)


# --- G chart ---------------------------------------------------------------


def test_g_chart_hand_computed():
    dates = _dates(5)
    counts = [12, 25, 8, 40, 15]
    df = pd.DataFrame({"period": dates, "cases_since_last": counts})
    options = BaseChartOptions(value_field="cases_since_last", date_field="period")
    result = spc.chart(df, kind="g", options=options)

    assert [p.y for p in result.points] == pytest.approx([float(c) for c in counts])
    expected_mean, expected_ucl, expected_lcl = _hand_computed_xmr(counts)
    assert result.points[0].mean == pytest.approx(expected_mean)
    assert result.points[0].ucl == pytest.approx(expected_ucl)
    assert result.points[0].lcl == pytest.approx(expected_lcl)


def test_g_chart_uses_date_field_directly_no_derivation():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "count": [5, 10, 15]})
    options = BaseChartOptions(value_field="count", date_field="period")
    result = spc.chart(df, kind="g", options=options)
    assert [p.x for p in result.points] == dates


# --- External baseline -------------------------------------------------

# 3 low gaps/counts then 7 high ones: a genuine step-shift, not an
# alternation -- large enough that every point reads as outside_limits
# against a flat pooled mean, which is what makes it a useful baseline
# demonstration (a baseline that predicts the step correctly should
# recentre every point onto its own local expectation instead).
_SHIFT_VALUES = [5, 5, 5, 30, 30, 30, 30, 30, 30, 30]


def _t_shift_df():
    event_dates = [date(2024, 1, 1)]
    for gap in _SHIFT_VALUES:
        event_dates.append(event_dates[-1] + timedelta(days=gap))
    df = pd.DataFrame({"incident": event_dates, "y": range(len(event_dates)), "period": event_dates})
    gap_dates = event_dates[1:]
    return df, gap_dates


def test_t_baseline_seam_reduces_to_plain_result_when_baseline_is_zero():
    df, gap_dates = _t_shift_df()
    baseline = [ExternalBaseline(x=d, expected_value=0.0, source="test") for d in gap_dates]
    plain_options = BaseChartOptions(value_field="incident", date_field="period")
    baseline_options = BaseChartOptions(value_field="incident", date_field="period", baseline=baseline)
    result_plain = spc.chart(df, kind="t", options=plain_options)
    result_baseline = spc.chart(df, kind="t", options=baseline_options)
    assert [p.mean for p in result_plain.points] == pytest.approx([p.mean for p in result_baseline.points])
    assert [p.ucl for p in result_plain.points] == pytest.approx([p.ucl for p in result_baseline.points])


def test_t_baseline_matching_a_genuine_shift_reads_as_common_cause():
    df, gap_dates = _t_shift_df()
    plain_options = BaseChartOptions(value_field="incident", date_field="period")
    result_plain = spc.chart(df, kind="t", options=plain_options)
    assert all(p.point_type != "common_cause" for p in result_plain.points)

    baseline = [
        ExternalBaseline(x=d, expected_value=float(v), source="test")
        for d, v in zip(gap_dates, _SHIFT_VALUES, strict=True)
    ]
    baseline_options = BaseChartOptions(value_field="incident", date_field="period", baseline=baseline)
    result_baseline = spc.chart(df, kind="t", options=baseline_options)
    assert all(p.point_type == "common_cause" for p in result_baseline.points)
    assert [p.mean for p in result_baseline.points] == pytest.approx([float(v) for v in _SHIFT_VALUES])


def test_g_baseline_matching_a_genuine_shift_reads_as_common_cause():
    dates = _dates(len(_SHIFT_VALUES))
    df = pd.DataFrame({"period": dates, "cnt": _SHIFT_VALUES})
    plain_options = BaseChartOptions(value_field="cnt", date_field="period")
    result_plain = spc.chart(df, kind="g", options=plain_options)
    assert all(p.point_type != "common_cause" for p in result_plain.points)

    baseline = [
        ExternalBaseline(x=d, expected_value=float(v), source="test") for d, v in zip(dates, _SHIFT_VALUES, strict=True)
    ]
    baseline_options = BaseChartOptions(value_field="cnt", date_field="period", baseline=baseline)
    result_baseline = spc.chart(df, kind="g", options=baseline_options)
    assert all(p.point_type == "common_cause" for p in result_baseline.points)
    assert [p.mean for p in result_baseline.points] == pytest.approx([float(v) for v in _SHIFT_VALUES])


def test_rare_event_baseline_expected_value_cannot_be_negative():
    df, gap_dates = _t_shift_df()
    baseline = [ExternalBaseline(x=gap_dates[0], expected_value=-1.0, source="test")]
    options = BaseChartOptions(value_field="incident", date_field="period", baseline=baseline)
    with pytest.raises(ValueError, match="can't be negative"):
        spc.chart(df, kind="t", options=options)


def test_rare_event_baseline_missing_coverage_raises():
    df, gap_dates = _t_shift_df()
    baseline = [ExternalBaseline(x=gap_dates[0], expected_value=5.0, source="test")]  # only 1 of 10 covered
    options = BaseChartOptions(value_field="incident", date_field="period", baseline=baseline)
    with pytest.raises(ValueError, match="no entry for facet"):
        spc.chart(df, kind="t", options=options)


# --- facet_field / rebase / target ------------------------------------------


def _event_dates(
        gaps,
        start
):
    out = [start]
    for g in gaps:
        out.append(out[-1] + timedelta(days=g))
    return out


_GAPS_A = [5, 10, 3, 20, 8, 7, 9, 6]
_GAPS_B = [30, 25, 40, 28, 33, 90, 31, 29, 35]


def _faceted_t_df():
    a = _event_dates(_GAPS_A, date(2024, 1, 1))
    b = _event_dates(_GAPS_B, date(2024, 1, 3))
    return pd.DataFrame({"incident": a + b, "unit": ["A"] * len(a) + ["B"] * len(b)})


def _series(
        result,
        facet=None
):
    pts = [p for p in result.points if p.facet == facet]
    return [(p.x, p.y, p.mean, p.ucl, p.lcl) for p in pts]


def test_t_chart_facets_equal_separate_runs_per_group():
    df = _faceted_t_df()
    faceted = spc.chart(
        df, kind="t", options=BaseChartOptions(value_field="incident", date_field="incident", facet_field="unit")
    )
    for unit in ("A", "B"):
        sub = df[df["unit"] == unit][["incident"]]
        alone = spc.chart(sub, kind="t", options=BaseChartOptions(value_field="incident", date_field="incident"))
        assert_rows_approx(_series(faceted, unit), _series(alone))
    # a facet's first event yields no point: n events -> n-1 gaps per facet
    assert len(_series(faceted, "A")) == len(_GAPS_A)
    assert len(_series(faceted, "B")) == len(_GAPS_B)


def test_g_chart_facets_equal_separate_runs_per_group():
    dates = _dates(6)
    df = pd.DataFrame(
        {
            "period": dates + dates,
            "cnt": [12, 25, 8, 40, 15, 22, 100, 90, 120, 95, 110, 105],
            "unit": ["A"] * 6 + ["B"] * 6,
        }
    )
    faceted = spc.chart(
        df, kind="g", options=BaseChartOptions(value_field="cnt", date_field="period", facet_field="unit")
    )
    for unit in ("A", "B"):
        sub = df[df["unit"] == unit][["period", "cnt"]]
        alone = spc.chart(sub, kind="g", options=BaseChartOptions(value_field="cnt", date_field="period"))
        assert_rows_approx(_series(faceted, unit), _series(alone))


def test_t_chart_rebase_splits_at_the_later_event_of_the_straddling_gap():
    events = _event_dates([5, 10, 3, 20, 8, 7, 9, 6], date(2024, 1, 1))
    df = pd.DataFrame({"incident": events})
    rebase_at = events[5]  # the gap (events[4] -> events[5]) straddles the change
    result = spc.chart(
        df, kind="t", options=BaseChartOptions(value_field="incident", date_field="incident", rebase=[rebase_at])
    )
    assert sorted({p.segment_id for p in result.points}) == [0, 1]
    old = [p for p in result.points if p.segment_id == 0]
    new = [p for p in result.points if p.segment_id == 1]
    # segment 0: gaps ending before the change; segment 1: the straddling gap onward
    assert [p.x for p in old] == events[1:5]
    assert [p.x for p in new] == events[5:]

    def alone(
            subset
    ):
        return spc.chart(
            pd.DataFrame({"incident": subset}),
            kind="t",
            options=BaseChartOptions(value_field="incident", date_field="incident"),
        )

    assert_rows_approx(
        [(p.y, p.mean, p.ucl, p.lcl) for p in old], [(p.y, p.mean, p.ucl, p.lcl) for p in alone(events[:5]).points]
    )
    assert_rows_approx(
        [(p.y, p.mean, p.ucl, p.lcl) for p in new], [(p.y, p.mean, p.ucl, p.lcl) for p in alone(events[4:]).points]
    )


def test_g_chart_rebase_equals_two_separate_runs():
    dates = _dates(10)
    counts = [12, 25, 8, 40, 15, 130, 110, 150, 120, 140]
    df = pd.DataFrame({"period": dates, "cnt": counts})
    result = spc.chart(
        df, kind="g", options=BaseChartOptions(value_field="cnt", date_field="period", rebase=[dates[5]])
    )
    assert sorted({p.segment_id for p in result.points}) == [0, 1]
    for seg_id, sl in ((0, slice(0, 5)), (1, slice(5, 10))):
        alone = spc.chart(
            pd.DataFrame({"period": dates[sl], "cnt": counts[sl]}),
            kind="g",
            options=BaseChartOptions(value_field="cnt", date_field="period"),
        )
        got = [(p.x, p.y, p.mean, p.ucl, p.lcl) for p in result.points if p.segment_id == seg_id]
        assert_rows_approx(got, [(p.x, p.y, p.mean, p.ucl, p.lcl) for p in alone.points])


def test_t_chart_per_facet_baseline_applies_to_its_own_facet_only():
    df = _faceted_t_df()
    a_gap_dates = _event_dates(_GAPS_A, date(2024, 1, 1))[1:]
    b_gap_dates = _event_dates(_GAPS_B, date(2024, 1, 3))[1:]
    baseline = [
        ExternalBaseline(facet="A", x=d, expected_value=float(v), source="test")
        for d, v in zip(a_gap_dates, _GAPS_A, strict=True)
    ] + [
        ExternalBaseline(facet="B", x=d, expected_value=float(v), source="test")
        for d, v in zip(b_gap_dates, _GAPS_B, strict=True)
    ]
    result = spc.chart(
        df,
        kind="t",
        options=BaseChartOptions(value_field="incident", date_field="incident", facet_field="unit", baseline=baseline),
    )
    assert all(p.point_type == "common_cause" for p in result.points)


@pytest.mark.parametrize("kind", ["t", "g"])
def test_rare_event_matching_baseline_uses_the_same_power_calculation(
        kind,
        monkeypatch
):
    from spc.core import rare_event_segments

    original_transform = rare_event_segments._transform

    def rounded_vector_transform(
            values
    ):
        transformed = original_transform(values)
        # Emulate a platform's vector power rounding one value differently
        # from Python's scalar power. An exact baseline must still cancel.
        at_33 = values == 33.0
        transformed.loc[at_33] = np.nextafter(transformed.loc[at_33], np.inf)
        return transformed

    monkeypatch.setattr(rare_event_segments, "_transform", rounded_vector_transform)
    a_dates = _event_dates(_GAPS_A, date(2024, 1, 1))[1:]
    b_dates = _event_dates(_GAPS_B, date(2024, 1, 3))[1:]
    baseline = [
        ExternalBaseline(facet=facet, x=d, expected_value=float(v), source="test")
        for facet, dates, values in (("A", a_dates, _GAPS_A), ("B", b_dates, _GAPS_B))
        for d, v in zip(dates, values, strict=True)
    ]
    if kind == "t":
        df = _faceted_t_df()
        value_field, date_field = "incident", "incident"
    else:
        df = pd.DataFrame(
            {
                "period": a_dates + b_dates,
                "count": _GAPS_A + _GAPS_B,
                "unit": ["A"] * len(a_dates) + ["B"] * len(b_dates),
            }
        )
        value_field, date_field = "count", "period"
    result = spc.chart(
        df,
        kind=kind,
        options=BaseChartOptions(value_field=value_field, date_field=date_field, facet_field="unit", baseline=baseline),
    )
    assert all(p.point_type == "common_cause" for p in result.points)
    assert [p.mean for p in result.points] == pytest.approx([p.y for p in result.points])


def test_t_chart_per_facet_target_gives_per_facet_assurance():
    df = _faceted_t_df()
    result = spc.chart(
        df,
        kind="t",
        options=BaseChartOptions(
            value_field="incident", date_field="incident", facet_field="unit", target={"A": 2.0, "B": 1000.0}
        ),
    )
    assert isinstance(result.assurance_type, dict)
    assert set(result.assurance_type) == {"A", "B"}


def test_rare_event_builder_rejects_other_kinds():
    from spc.core.rare_event_segments import build_rare_event_segment_arrays

    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "period": _dates(3)})
    opts = BaseChartOptions(value_field="y", date_field="period")
    for bad in ("xmr", "p", "", "T"):
        with pytest.raises(ValueError, match="'t' or 'g' only"):
            build_rare_event_segment_arrays(df, opts, kind=bad)

from datetime import date, timedelta

import pandas as pd
import pytest

import spc
from spc.core.models import BaseChartOptions, LimitMode

plotly = pytest.importorskip("plotly")


def _dates(
        n,
        start=date(2024, 1, 1)
):
    return [start + timedelta(weeks=i) for i in range(n)]


def test_plotly_renderer_produces_a_figure_with_expected_traces():
    dates = _dates(12)
    df = pd.DataFrame({"period": dates, "y": [10, 12, 11, 13, 9, 14, 10, 11, 12, 13, 10, 11]})
    options = BaseChartOptions(value_field="y", date_field="period", target=12.0)
    result = spc.chart(df, kind="xmr", options=options)

    fig = spc.render.plotly(result, title="Test chart")
    trace_names = {t.name for t in fig.data}
    assert "Baseline" in trace_names
    assert "UCL or LCL" in trace_names
    assert "XMR" in trace_names
    assert fig.layout.title.text == "Test chart"
    # target line is drawn as a shape (add_hline), not a trace
    assert len(fig.layout.shapes) == 1


def test_plotly_renderer_value_trace_hides_its_own_legend_entry():
    # marker.color on this trace is a per-point array -- Plotly's legend
    # can't represent that with one swatch, so it's suppressed in favour of
    # the point_type proxy legend entries checked below.
    dates = _dates(12)
    df = pd.DataFrame({"period": dates, "y": [10, 12, 11, 13, 9, 14, 10, 11, 12, 13, 10, 11]})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)

    fig = spc.render.plotly(result)
    value_trace = next(t for t in fig.data if t.name == "XMR")
    assert value_trace.showlegend is False


def test_plotly_renderer_legend_explains_point_colours():
    from spc.render.plotly_renderer import POINT_COLOURS

    dates = _dates(10)
    # A stable series with one big upward spike, baseline fixed to the
    # pre-spike points: common_cause throughout except the spike itself,
    # which reads as special_cause_improvement (default
    # improvement_direction="increase" -- higher is "better" here).
    df = pd.DataFrame({"period": dates, "y": [10, 11, 9, 10, 11, 9, 10, 11, 50, 10]})
    options = BaseChartOptions(
        value_field="y",
        date_field="period",
        limit_mode=LimitMode.FIXED_N,
        fix_after_n_points=8,
    )
    result = spc.chart(df, kind="xmr", options=options)

    fig = spc.render.plotly(result)
    legend_traces = [t for t in fig.data if t.showlegend and t.marker and t.marker.color in POINT_COLOURS.values()]
    legend_by_name = {t.name: t.marker.color for t in legend_traces}
    assert legend_by_name == {
        "Common cause": POINT_COLOURS["common_cause"],
        "Special cause (improvement)": POINT_COLOURS["special_cause_improvement"],
    }
    # Only point_types actually present get a legend entry -- no
    # "concern"/"neutral" clutter for a chart that never produced one.
    assert "Special cause (concern)" not in {t.name for t in fig.data}


def test_plotly_renderer_neutral_high_and_low_share_one_legend_entry():
    # special_cause_neutral_high/low are visually identical (same colour,
    # per POINT_COLOURS) -- direction is still readable from each point's
    # position relative to the baseline line, so the legend collapses them to
    # one "Special cause (neutral)" entry rather than two indistinguishable
    # rows.
    dates = _dates(10)
    df = pd.DataFrame({"period": dates, "y": [10, 11, 9, 10, 11, 9, 10, 11, 50, 1]})
    options = BaseChartOptions(
        value_field="y",
        date_field="period",
        improvement_direction="neutral",
        limit_mode=LimitMode.FIXED_N,
        fix_after_n_points=8,
    )
    result = spc.chart(df, kind="xmr", options=options)
    assert {p.point_type for p in result.points} >= {"special_cause_neutral_high", "special_cause_neutral_low"}

    fig = spc.render.plotly(result)
    neutral_traces = [t for t in fig.data if t.name == "Special cause (neutral)"]
    assert len(neutral_traces) == 1


def test_plotly_renderer_does_not_shade_the_ucl_lcl_band_by_default():
    dates = _dates(12)
    df = pd.DataFrame({"period": dates, "y": [10, 12, 11, 13, 9, 14, 10, 11, 12, 13, 10, 11]})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)

    fig = spc.render.plotly(result)
    assert not any(t.name == "UCL or LCL" and t.fill == "tonexty" for t in fig.data)


def test_plotly_renderer_shades_the_ucl_lcl_band_when_requested():
    dates = _dates(12)
    df = pd.DataFrame({"period": dates, "y": [10, 12, 11, 13, 9, 14, 10, 11, 12, 13, 10, 11]})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)

    fig = spc.render.plotly(result, shade_limits=True)
    lcl_traces = [t for t in fig.data if t.name == "UCL or LCL" and t.fill == "tonexty"]
    assert len(lcl_traces) == 1  # one segment here -- one shaded LCL trace
    assert lcl_traces[0].fillcolor == "rgba(166, 166, 166, 0.15)"  # default limit_colour (#a6a6a6) at low opacity
    # The UCL trace itself carries no fill -- only the LCL trace fills
    # "tonexty" back to it, so the band isn't shaded twice.
    ucl_traces = [t for t in fig.data if t.name == "UCL or LCL" and t.fill != "tonexty"]
    assert len(ucl_traces) == 1


def test_plotly_renderer_baseline_and_limit_colour_are_overridable():
    dates = _dates(12)
    df = pd.DataFrame({"period": dates, "y": [10, 12, 11, 13, 9, 14, 10, 11, 12, 13, 10, 11]})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)

    fig = spc.render.plotly(result, baseline_colour="#123456", limit_colour="#abcdef", shade_limits=True)
    assert next(t for t in fig.data if t.name == "Baseline").line.color == "#123456"
    limit_traces = [t for t in fig.data if t.name == "UCL or LCL"]
    assert all(t.line.color == "#abcdef" for t in limit_traces)
    shaded = next(t for t in limit_traces if t.fill == "tonexty")
    assert shaded.fillcolor == "rgba(171, 205, 239, 0.15)"  # #abcdef at low opacity


def test_plotly_renderer_defaults_to_nhs_black_and_grey():
    dates = _dates(12)
    df = pd.DataFrame({"period": dates, "y": [10, 12, 11, 13, 9, 14, 10, 11, 12, 13, 10, 11]})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)

    fig = spc.render.plotly(result)
    assert next(t for t in fig.data if t.name == "Baseline").line.color == "#000000"
    assert next(t for t in fig.data if t.name == "UCL or LCL").line.color == "#a6a6a6"


def test_plotly_renderer_uses_two_separate_legend_boxes():
    # Lines (Baseline/UCL-LCL) stay in the default legend; point-related
    # entries (colour meaning) go in their own box (legend2) -- a genuinely
    # separate legend, not just a grouped header within one.
    dates = _dates(12)
    df = pd.DataFrame({"period": dates, "y": [10, 12, 11, 13, 9, 14, 10, 11, 12, 13, 10, 11]})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)

    fig = spc.render.plotly(result)
    line_legend = {t.legend for t in fig.data if t.name in ("Baseline", "UCL or LCL")}
    point_legend = {t.legend for t in fig.data if t.name == "Common cause"}
    assert line_legend in ({None}, {"legend"})  # default legend, however Plotly reports the unset case
    assert point_legend == {"legend2"}
    assert fig.layout.legend.title.text == "Limits & lines"
    assert fig.layout.legend2.title.text == "Data points"


def test_plotly_renderer_spec_breach_gets_a_legend_entry_only_when_present():
    dates = _dates(5)
    df_breach = pd.DataFrame({"period": dates, "y": [1.0, 2.0, 3.0, 4.0, 20.0]})
    options_breach = BaseChartOptions(value_field="y", date_field="period", spec_upper=6.0)
    result_breach = spc.chart(df_breach, kind="xmr", options=options_breach)
    fig_breach = spc.render.plotly(result_breach)
    assert "Breaches spec limit" in {t.name for t in fig_breach.data}

    df_clean = pd.DataFrame({"period": dates, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    options_clean = BaseChartOptions(value_field="y", date_field="period", spec_upper=6.0)
    result_clean = spc.chart(df_clean, kind="xmr", options=options_clean)
    fig_clean = spc.render.plotly(result_clean)
    assert "Breaches spec limit" not in {t.name for t in fig_clean.data}


def test_plotly_renderer_uses_nhs_making_data_count_colours():
    from spc.render.plotly_renderer import POINT_COLOURS

    assert POINT_COLOURS["common_cause"] == "#a6a6a6"
    assert POINT_COLOURS["special_cause_improvement"] == "#00b0f0"
    assert POINT_COLOURS["special_cause_concern"] == "#e46c0a"
    assert POINT_COLOURS["special_cause_neutral_high"] == "#490092"
    assert POINT_COLOURS["special_cause_neutral_low"] == "#490092"


def test_plotly_renderer_breaks_baseline_line_at_rebase_segments():
    dates = _dates(10)
    df = pd.DataFrame({"period": dates, "y": list(range(10))})
    options = BaseChartOptions(value_field="y", date_field="period", rebase=[dates[5]])
    result = spc.chart(df, kind="xmr", options=options)

    fig = spc.render.plotly(result)
    baseline_traces = [t for t in fig.data if t.name == "Baseline"]
    # one baseline-line trace per segment (2 segments here), not one continuous line
    assert len(baseline_traces) == 2


# --- Axis range / future_space --------------------------------------------


def test_plotly_renderer_default_future_space_pads_right_edge_only():
    dates = _dates(10)  # weekly, so a 9-week span from first to last point
    df = pd.DataFrame({"period": dates, "y": list(range(10))})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)

    fig = spc.render.plotly(result)
    range_start, range_end = fig.layout.xaxis.range
    assert pd.Timestamp(range_start).date() == dates[0]  # left edge pinned tight
    assert pd.Timestamp(range_end).date() > dates[-1]  # right edge padded past the last point


def test_plotly_renderer_future_space_zero_is_tight_on_both_ends():
    dates = _dates(10)
    df = pd.DataFrame({"period": dates, "y": list(range(10))})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)

    fig = spc.render.plotly(result, future_space=0)
    range_start, range_end = fig.layout.xaxis.range
    assert pd.Timestamp(range_start).date() == dates[0]
    assert pd.Timestamp(range_end).date() == dates[-1]


def test_plotly_renderer_future_space_scales_the_right_padding():
    dates = _dates(10)
    df = pd.DataFrame({"period": dates, "y": list(range(10))})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)

    span_days = (dates[-1] - dates[0]).days
    fig_small = spc.render.plotly(result, future_space=0.10)
    fig_large = spc.render.plotly(result, future_space=0.50)
    _start_s, end_small = fig_small.layout.xaxis.range
    _start_l, end_large = fig_large.layout.xaxis.range
    padding_small = (pd.Timestamp(end_small).date() - dates[-1]).days
    padding_large = (pd.Timestamp(end_large).date() - dates[-1]).days
    assert padding_small == pytest.approx(span_days * 0.10, abs=1)
    assert padding_large == pytest.approx(span_days * 0.50, abs=1)


def test_plotly_renderer_negative_future_space_raises():
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "y": [1, 2, 3, 4, 5]})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)
    with pytest.raises(ValueError, match="future_space"):
        spc.render.plotly(result, future_space=-0.1)


# --- Axis labels ------------------------------------------------------------


def test_plotly_renderer_axis_labels_are_capitalised_and_field_specific():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "defects": [1, 2, 1], "n": [10, 10, 10]})
    from spc.core.models import ProportionChartOptions

    options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    result = spc.chart(df, kind="p", options=options)

    fig = spc.render.plotly(result)
    assert fig.layout.xaxis.title.text == "Period"
    assert fig.layout.yaxis.title.text == "Defects / n"  # proportion, not the raw count


def test_plotly_renderer_p_prime_uses_same_y_axis_label_as_p():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "defects": [1, 2, 1], "n": [10, 10, 10]})
    from spc.core.models import ProportionChartOptions

    options = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    result = spc.chart(df, kind="p_prime", options=options)

    fig = spc.render.plotly(result)
    assert fig.layout.yaxis.title.text == "Defects / n"


# --- facet small multiples ---------------------------------------------------


def _faceted_result(
        two_d=False,
        **opt
):
    rows = []
    for r in ("A", "B"):
        for c in ("x", "y"):
            for i, d in enumerate(_dates(8)):
                rows.append(
                    {
                        "period": d,
                        "y": 10 + i % 3 + (5 if r == "B" else 0) + (20 if c == "y" else 0),
                        "row": r,
                        "col": c,
                    }
                )
    df = pd.DataFrame(rows)
    if not two_d:
        df = df[df["col"] == "x"]
    options = BaseChartOptions(
        value_field="y",
        date_field="period",
        facet_field="row",
        column_facet_field="col" if two_d else None,
        **opt,
    )
    return spc.chart(df, kind="xmr", options=options)


def test_one_variable_facets_render_one_panel_each_with_independent_y_axes():
    fig = spc.render.plotly(_faceted_result(), facet_columns=2)
    assert len(fig.layout.annotations) == 2  # one subplot title per facet
    assert {a.text for a in fig.layout.annotations} == {"A", "B"}
    # each panel has its own y-axis: yaxis, yaxis2
    assert fig.layout.yaxis.matches is None and fig.layout.yaxis2.matches is None


def test_two_variable_facets_render_a_row_by_column_grid():
    fig = spc.render.plotly(_faceted_result(two_d=True))
    titles = [a.text for a in fig.layout.annotations]
    assert sorted(titles) == sorted(["A | x", "A | y", "B | x", "B | y"])
    assert len(fig.data) > 4


def test_facet_legend_lists_each_line_once():
    fig = spc.render.plotly(_faceted_result(two_d=True))
    legend_names = [t.name for t in fig.data if t.showlegend and t.legend in (None, "legend")]
    assert legend_names.count("Baseline") == 1
    assert legend_names.count("UCL or LCL") == 1


def test_per_facet_target_dict_is_drawn_in_each_panel():
    fig = spc.render.plotly(_faceted_result(target={"A": 12.0, "B": 18.0}))
    ys = sorted(s.y0 for s in fig.layout.shapes)
    assert ys == [12.0, 18.0]


def test_facet_columns_must_be_positive():
    with pytest.raises(ValueError, match="facet_columns"):
        spc.render.plotly(_faceted_result(), facet_columns=0)

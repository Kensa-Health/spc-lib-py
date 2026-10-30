"""spec_upper/spec_lower (specification/capability limits) tests --
externally-imposed requirements, deliberately independent of the
statistical control-limit machinery. Verified via hand computation and
explicit independence checks against point_type/outside_limits.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

import spc
from spc.core.models import BaseChartOptions


def _dates(
        n,
        start=date(2024, 1, 1)
):
    return [start + timedelta(weeks=i) for i in range(n)]


# --- Model validation --------------------------------------------------


def test_spec_upper_must_exceed_spec_lower():
    with pytest.raises(ValueError, match="spec_upper"):
        BaseChartOptions(value_field="y", date_field="period", spec_upper=1.0, spec_lower=2.0)


def test_spec_upper_equal_to_spec_lower_raises():
    with pytest.raises(ValueError, match="spec_upper"):
        BaseChartOptions(value_field="y", date_field="period", spec_upper=5.0, spec_lower=5.0)


def test_spec_limits_independently_optional():
    # Neither raises alone -- a one-sided spec (e.g. a rare-event minimum
    # gap) is a legitimate, common shape, not an incomplete pair.
    BaseChartOptions(value_field="y", date_field="period", spec_upper=5.0)
    BaseChartOptions(value_field="y", date_field="period", spec_lower=1.0)


def test_spec_limits_per_facet_dict_validated_per_facet():
    with pytest.raises(ValueError, match="spec_upper"):
        BaseChartOptions(
            value_field="y",
            date_field="period",
            facet_field="site",
            spec_upper={"a": 5.0, "b": 3.0},
            spec_lower={"a": 1.0, "b": 4.0},  # b: lower (4.0) > upper (3.0)
        )


# --- spec_breach computation --------------------------------------------


def test_spec_breach_flags_points_above_upper_and_below_lower():
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "y": [1.0, 5.0, 3.0, 0.5, 3.0]})
    options = BaseChartOptions(value_field="y", date_field="period", spec_upper=4.0, spec_lower=1.0)
    result = spc.chart(df, kind="xmr", options=options)

    breaches = [p.spec_breach for p in result.points]
    assert breaches == [False, True, False, True, False]


def test_spec_breach_false_everywhere_when_no_spec_limits_set():
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "y": [1.0, 100.0, 3.0, -50.0, 3.0]})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)
    assert not any(p.spec_breach for p in result.points)


def test_spec_breach_is_independent_of_point_type_and_outside_limits():
    # A point can breach the spec while the process itself is behaving
    # completely normally (common_cause, not outside_limits) -- the whole
    # point of keeping these two questions separate. Stable process +
    # unmet requirement is a real, distinct state that must be visible.
    dates = _dates(10)
    df = pd.DataFrame({"period": dates, "y": [4.1, 4.3, 4.0, 4.5, 5.2, 4.4, 4.6, 4.3, 4.1, 4.9]})
    options = BaseChartOptions(value_field="y", date_field="period", spec_upper=4.8)
    result = spc.chart(df, kind="xmr", options=options)

    breached = [p for p in result.points if p.spec_breach]
    assert len(breached) == 2
    for p in breached:
        assert p.point_type == "common_cause"
        assert p.outside_limits is False
        assert p.special_cause_rule is None


def test_spec_breach_one_sided_upper_only():
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "y": [1.0, -10.0, 3.0, 20.0, 3.0]})
    options = BaseChartOptions(value_field="y", date_field="period", spec_upper=5.0)
    result = spc.chart(df, kind="xmr", options=options)
    breaches = [p.spec_breach for p in result.points]
    assert breaches == [False, False, False, True, False]  # -10 never flagged: no spec_lower


def test_spec_breach_one_sided_lower_only():
    # e.g. a rare-event chart requiring a minimum gap between adverse events.
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "y": [10.0, 2.0, 15.0, 25.0, 8.0]})
    options = BaseChartOptions(value_field="y", date_field="period", spec_lower=5.0)
    result = spc.chart(df, kind="xmr", options=options)
    breaches = [p.spec_breach for p in result.points]
    assert breaches == [False, True, False, False, False]


def test_spec_breach_resolved_per_facet():
    dates = _dates(4)
    df = pd.DataFrame(
        {
            "period": dates * 2,
            "site": ["a"] * 4 + ["b"] * 4,
            "y": [1.0, 6.0, 2.0, 3.0, 1.0, 6.0, 2.0, 3.0],
        }
    )
    options = BaseChartOptions(
        value_field="y", date_field="period", facet_field="site", spec_upper={"a": 5.0, "b": 100.0}
    )
    result = spc.chart(df, kind="xmr", options=options)

    site_a = [p for p in result.points if p.facet == "a"]
    site_b = [p for p in result.points if p.facet == "b"]
    assert any(p.spec_breach for p in site_a)  # a's y=6.0 > a's spec_upper=5.0
    assert not any(p.spec_breach for p in site_b)  # b's y=6.0 < b's spec_upper=100.0


# --- Renderer ------------------------------------------------------------


def test_renderer_draws_spec_limit_lines_when_scalar():
    pytest.importorskip("plotly")
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    options = BaseChartOptions(value_field="y", date_field="period", spec_upper=6.0, spec_lower=0.0)
    result = spc.chart(df, kind="xmr", options=options)

    fig = spc.render.plotly(result)
    # Drawn as Scatter traces (not add_hline shapes) specifically so they
    # appear in the legend -- see plotly_renderer.py.
    spec_traces = [t for t in fig.data if t.name == "Spec limit"]
    assert len(spec_traces) == 2  # spec_upper + spec_lower
    assert [t.y[0] for t in spec_traces] == [6.0, 0.0]


def test_renderer_spec_limit_legend_uses_custom_label_and_shares_one_entry():
    pytest.importorskip("plotly")
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    options = BaseChartOptions(
        value_field="y",
        date_field="period",
        spec_upper=6.0,
        spec_lower=0.0,
        spec_label="Spec limit (78% target)",
    )
    result = spc.chart(df, kind="xmr", options=options)

    fig = spc.render.plotly(result)
    spec_traces = [t for t in fig.data if t.name == "Spec limit (78% target)"]
    assert len(spec_traces) == 2
    # Only the first of the pair shows a legend entry -- the pair reads as
    # one legend item, not two, like the existing Mean/UCL-LCL pattern.
    assert [t.showlegend for t in spec_traces] == [True, False]
    assert {t.legendgroup for t in spec_traces} == {"spec"}


def test_renderer_spec_limit_label_defaults_when_not_supplied():
    pytest.importorskip("plotly")
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    options = BaseChartOptions(value_field="y", date_field="period", spec_upper=6.0)
    result = spc.chart(df, kind="xmr", options=options)

    fig = spc.render.plotly(result)
    assert any(t.name == "Spec limit" for t in fig.data)


def test_renderer_breached_points_get_a_marker_outline():
    pytest.importorskip("plotly")
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "y": [1.0, 2.0, 3.0, 4.0, 20.0]})
    options = BaseChartOptions(value_field="y", date_field="period", spec_upper=6.0)
    result = spc.chart(df, kind="xmr", options=options)

    fig = spc.render.plotly(result)
    value_trace = next(t for t in fig.data if t.name == "XMR")
    outline_widths = list(value_trace.marker.line.width)
    assert outline_widths[-1] > 0  # the breaching point (y=20) has a visible outline
    assert all(w == 0 for w in outline_widths[:-1])  # the rest don't

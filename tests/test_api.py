from datetime import date, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError

import spc
from spc.core.models import BaseChartOptions, LimitMode


def _dates(
        n,
        start=date(2024, 1, 1)
):
    return [start + timedelta(weeks=i) for i in range(n)]


def test_chart_xmr_end_to_end():
    dates = _dates(15)
    df = pd.DataFrame({"period": dates, "y": [10, 12, 11, 13, 9, 14, 10, 11, 12, 13, 10, 11, 9, 12, 11]})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)
    assert result.kind == "xmr"
    assert len(result.points) == 15
    assert result.schema_version == "1.0"


def test_chart_auto_dispatches_to_xmr_for_generic_data():
    dates = _dates(10)
    df = pd.DataFrame({"period": dates, "y": list(range(10))})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="auto", options=options)
    assert result.kind == "xmr"
    assert result.suggested_kind is None  # auto already picked xmr, nothing further to suggest


def test_chart_unknown_kind_raises_value_error():
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "y": [1, 2, 1, 2, 1]})
    options = BaseChartOptions(value_field="y", date_field="period")
    with pytest.raises(ValueError, match="unknown chart kind"):
        spc.chart(df, kind="not_a_real_kind", options=options)


def test_fixed_n_without_fix_after_n_points_fails_fast_at_construction():
    with pytest.raises(ValidationError, match="requires fix_after_n_points"):
        BaseChartOptions(value_field="y", date_field="period", limit_mode=LimitMode.FIXED_N)


def test_run_rule_length_below_three_fails_fast_at_construction():
    with pytest.raises(ValidationError, match="run_rule_length must be >= 3"):
        BaseChartOptions(value_field="y", date_field="period", run_rule_length=2)


def test_run_rule_length_custom_value_flags_a_shorter_shift_than_the_default():
    # A 5-point non-monotonic plateau above the mean -- not a big enough
    # run to trip the default (7) shift rule, but is at a configured 5.
    dates = _dates(20)
    y = [
        10,
        9,
        11,
        10,
        9,
        11,
        10,
        9,
        12,
        11,
        13,
        11,
        12,
        10,
        9,
        11,
        10,
        9,
        11,
        10,
    ]
    df = pd.DataFrame({"period": dates, "y": y})

    default_options = BaseChartOptions(value_field="y", date_field="period")
    result_default = spc.chart(df, kind="xmr", options=default_options)
    assert all(p.special_cause_rule is None for p in result_default.points)

    short_run_options = BaseChartOptions(value_field="y", date_field="period", run_rule_length=5)
    result_short_run = spc.chart(df, kind="xmr", options=short_run_options)
    flagged = [p for p in result_short_run.points if p.special_cause_rule is not None]
    assert len(flagged) == 5
    assert all(p.special_cause_rule == "seven_point_shift" for p in flagged)
    assert all(
        p.point_type == "special_cause_improvement" for p in flagged
    )  # above mean, improvement_direction="increase"


def test_run_rule_length_proportional_end_to_end():
    # 24 points -> proportional run length resolves to 8 (see
    # test_rules.py's _resolve_run_length coverage), one more than the
    # fixed default of 7.
    dates = _dates(24)
    y = [10, 9, 11, 10, 9, 11, 10, 9] + [12, 11, 13, 11, 12, 11, 13, 11] + [10, 9, 11, 10, 9, 11, 10, 9]
    df = pd.DataFrame({"period": dates, "y": y})
    options = BaseChartOptions(value_field="y", date_field="period", run_rule_length="proportional")
    result = spc.chart(df, kind="xmr", options=options)
    flagged = [p for p in result.points if p.special_cause_rule is not None]
    assert flagged  # the elevated plateau is long enough to trip the proportional threshold too
    assert all(p.special_cause_rule == "seven_point_shift" for p in flagged)


def test_assurance_type_single_facet_is_a_plain_scalar():
    dates = _dates(15)
    df = pd.DataFrame({"period": dates, "y": [10, 12, 11, 13, 9, 14, 10, 11, 12, 13, 10, 11, 9, 12, 11]})
    options = BaseChartOptions(value_field="y", date_field="period", target=30.0, improvement_direction="decrease")
    result = spc.chart(df, kind="xmr", options=options)
    assert result.assurance_type in ("consistent_pass", "consistent_fail", "inconsistent")


def test_assurance_type_multi_facet_is_a_dict_per_facet():
    # Each facet here is set up so its own limits clearly land on one
    # side of the shared target.
    dates = _dates(15)
    low_facet = [10, 12, 11, 13, 9, 14, 10, 11, 12, 13, 10, 11, 9, 12, 11]  # centred ~11, well under target
    high_facet = [50, 52, 51, 53, 49, 54, 50, 51, 52, 53, 50, 51, 49, 52, 51]  # centred ~51, well over target
    df = pd.DataFrame(
        {
            "period": dates * 2,
            "y": low_facet + high_facet,
            "org": ["low"] * 15 + ["high"] * 15,
        }
    )
    options = BaseChartOptions(
        value_field="y", date_field="period", facet_field="org", target=30.0, improvement_direction="decrease"
    )
    result = spc.chart(df, kind="xmr", options=options)

    assert isinstance(result.assurance_type, dict)
    assert result.assurance_type["low"] == "consistent_pass"  # low facet's limits sit entirely under the target
    assert result.assurance_type["high"] == "consistent_fail"  # high facet's limits sit entirely over the target


def test_assurance_type_dict_target_resolves_per_facet():
    dates = _dates(15)
    values = [10, 12, 11, 13, 9, 14, 10, 11, 12, 13, 10, 11, 9, 12, 11]
    df = pd.DataFrame({"period": dates * 2, "y": values * 2, "org": ["a"] * 15 + ["b"] * 15})
    options = BaseChartOptions(
        value_field="y",
        date_field="period",
        facet_field="org",
        target={"a": 30.0},  # only "a" has a target -- "b" should be omitted from the result, not crash
        improvement_direction="decrease",
    )
    result = spc.chart(df, kind="xmr", options=options)
    assert isinstance(result.assurance_type, dict)
    assert "a" in result.assurance_type
    assert "b" not in result.assurance_type


def test_undefined_limits_serialise_as_null_not_nan():
    # A moving range needs >= 2 points -- a 1-point segment has genuinely
    # undefined limits. This must come through as JSON `null` (and a
    # nullable PointResult.ucl/lcl field), not a NaN that violates the
    # model's own generated JSON Schema (which declares ucl/lcl as
    # required, non-nullable numbers if typed as plain `float`).
    df = pd.DataFrame({"period": [date(2024, 1, 1)], "y": [10.0]})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)

    assert result.points[0].ucl is None
    assert result.points[0].lcl is None
    assert result.segments[0].ucl is None
    assert result.segments[0].lcl is None

    import json

    parsed = json.loads(result.model_dump_json())
    assert parsed["points"][0]["ucl"] is None

    schema = type(result.points[0]).model_json_schema()
    ucl_types = {t["type"] for t in schema["properties"]["ucl"]["anyOf"]}
    assert ucl_types == {"number", "null"}  # not a bare non-nullable "number"


def test_result_json_round_trips_through_schema_export():
    dates = _dates(10)
    df = pd.DataFrame({"period": dates, "y": list(range(10))})
    options = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=options)

    js = spc.render.to_json(result)
    d = spc.render.to_dict(result)
    assert isinstance(js, str)
    assert d["kind"] == "xmr"
    assert len(d["points"]) == 10

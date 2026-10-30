"""Input validation tests.

Each test asserts both the exception type AND that the message actually
names the offending column/value -- a bare pytest.raises(ValueError) isn't
enough evidence the right thing was validated for the right reason.
"""

import math
from datetime import date, timedelta

import pandas as pd
import pytest

import spc
from spc.core.models import BaseChartOptions, ProportionChartOptions


def _dates(
        n,
        start=date(2024, 1, 1)
):
    return [start + timedelta(weeks=i) for i in range(n)]


# --- Non-finite values / denominator validity -------------------------------


def test_zero_denominator_raises_instead_of_producing_a_non_finite_point():
    # Confirms the bug (result.points[1].y silently becoming None) is
    # closed, not relocated.
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "defects": [1, 2, 1, 2, 1], "n": [10, 0, 10, 10, 10]})
    opts = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    with pytest.raises(ValueError, match="'n' must be strictly positive"):
        spc.chart(df, kind="p", options=opts)


def test_negative_denominator_raises():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "defects": [1, 2, 1], "n": [10, -5, 10]})
    opts = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    with pytest.raises(ValueError, match="strictly positive"):
        spc.chart(df, kind="p", options=opts)


def test_no_reachable_none_y_or_mean_for_any_valid_chart():
    # Regression guard: for input that passes validation, y/mean must
    # never be None for any chart kind -- confirms the fix is closing the
    # contract violation, not just this one reproduction of it.
    dates = _dates(6)
    df = pd.DataFrame({"period": dates, "defects": [1, 2, 1, 2, 1, 2], "n": [10, 12, 9, 11, 10, 13]})
    opts = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    result = spc.chart(df, kind="p", options=opts)
    for p in result.points:
        assert p.y is not None and not math.isnan(p.y) and not math.isinf(p.y)
        assert p.mean is not None and not math.isnan(p.mean)
    for s in result.segments:
        assert s.mean is not None and not math.isnan(s.mean)


def test_value_exceeding_denominator_raises_for_p():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "defects": [1, 15, 1], "n": [10, 10, 10]})  # 15 > 10, impossible
    opts = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    with pytest.raises(ValueError, match="exceeds"):
        spc.chart(df, kind="p", options=opts)


def test_value_exceeding_denominator_raises_for_np():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "defects": [1, 15, 1], "n": [10, 10, 10]})
    opts = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    with pytest.raises(ValueError, match="exceeds"):
        spc.chart(df, kind="np", options=opts)


def test_value_exceeding_denominator_does_not_raise_for_u():
    # u has no such bound (a rate can exceed 1 per unit of opportunity) --
    # confirms the bounded-numerator check is correctly kind-scoped, not
    # blanket-applied to every chart with a denominator_field.
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "events": [1, 15, 1], "opportunity": [10, 10, 10]})
    opts = ProportionChartOptions(value_field="events", date_field="period", denominator_field="opportunity")
    result = spc.chart(df, kind="u", options=opts)
    assert len(result.points) == 3


# --- Column existence --------------------------------------------------


def test_missing_value_field_raises_with_available_columns_listed():
    df = pd.DataFrame({"period": _dates(3), "y": [1, 2, 3]})
    opts = BaseChartOptions(value_field="typo_field", date_field="period")
    with pytest.raises(ValueError, match="typo_field") as exc_info:
        spc.chart(df, kind="xmr", options=opts)
    assert "Available columns" in str(exc_info.value)
    assert "'period'" in str(exc_info.value) or "period" in str(exc_info.value)


def test_missing_date_field_raises():
    df = pd.DataFrame({"period": _dates(3), "y": [1, 2, 3]})
    opts = BaseChartOptions(value_field="y", date_field="not_a_column")
    with pytest.raises(ValueError, match="not_a_column"):
        spc.chart(df, kind="xmr", options=opts)


def test_missing_denominator_field_raises():
    df = pd.DataFrame({"period": _dates(3), "defects": [1, 2, 3]})
    opts = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="missing_n")
    with pytest.raises(ValueError, match="missing_n"):
        spc.chart(df, kind="p", options=opts)


def test_missing_value_field_raises_for_t():
    df = pd.DataFrame({"period": _dates(3), "y": [1, 2, 3]})
    opts = BaseChartOptions(value_field="missing_event", date_field="period")
    with pytest.raises(ValueError, match="missing_event"):
        spc.chart(df, kind="t", options=opts)


def test_missing_facet_field_raises():
    df = pd.DataFrame({"period": _dates(3), "y": [1, 2, 3]})
    opts = BaseChartOptions(value_field="y", date_field="period", facet_field="missing_facet")
    with pytest.raises(ValueError, match="missing_facet"):
        spc.chart(df, kind="xmr", options=opts)


def test_column_validation_protects_the_auto_kind_path_too():
    # suggest_chart_type() itself indexes df by value_field -- must not
    # raise a raw, unrelated KeyError before validation gets a chance to run.
    df = pd.DataFrame({"period": _dates(3), "y": [1, 2, 3]})
    opts = BaseChartOptions(value_field="typo_field", date_field="period")
    with pytest.raises(ValueError, match="typo_field"):
        spc.chart(df, kind="auto", options=opts)


# --- Empty dataframe -----------------------------------------------------


def test_empty_dataframe_raises():
    df = pd.DataFrame({"period": pd.Series([], dtype="datetime64[ns]"), "y": pd.Series([], dtype=float)})
    opts = BaseChartOptions(value_field="y", date_field="period")
    with pytest.raises(ValueError, match="no rows"):
        spc.chart(df, kind="xmr", options=opts)


# --- NaN in the primary value column -------------------------------------


def test_nan_in_value_field_raises_with_dates():
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "y": [1, 2, float("nan"), 4, 5]})
    opts = BaseChartOptions(value_field="y", date_field="period")
    with pytest.raises(ValueError, match="y.*missing") as exc_info:
        spc.chart(df, kind="xmr", options=opts)
    assert str(dates[2]) in str(exc_info.value)


def test_nan_in_value_field_raises_for_g():
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "count": [5, 10, float("nan"), 8, 6]})
    opts = BaseChartOptions(value_field="count", date_field="period")
    with pytest.raises(ValueError, match="missing"):
        spc.chart(df, kind="g", options=opts)


def test_nan_in_denominator_field_raises():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "defects": [1, 2, 1], "n": [10, float("nan"), 10]})
    opts = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    with pytest.raises(ValueError, match="'n' has missing"):
        spc.chart(df, kind="p", options=opts)


# --- Count fields (p/np/c/u/g) must be non-negative integers -- a defect
# or occurrence count can't be negative or fractional, so this fails fast
# rather than letting a nonsensical value silently flow into a formula. ---


def test_negative_count_raises_for_p():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "defects": [1, -2, 1], "n": [10, 10, 10]})
    opts = ProportionChartOptions(value_field="defects", date_field="period", denominator_field="n")
    with pytest.raises(ValueError, match="non-negative integer"):
        spc.chart(df, kind="p", options=opts)


def test_fractional_count_raises_for_c():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "falls": [3, 4.5, 2]})
    opts = BaseChartOptions(value_field="falls", date_field="period")
    with pytest.raises(ValueError, match="non-negative integer") as exc_info:
        spc.chart(df, kind="c", options=opts)
    assert str(dates[1]) in str(exc_info.value)


def test_negative_value_field_raises_for_g():
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "count": [5, 10, -3, 8, 6]})
    opts = BaseChartOptions(value_field="count", date_field="period")
    with pytest.raises(ValueError, match="non-negative integer"):
        spc.chart(df, kind="g", options=opts)


def test_negative_value_does_not_raise_for_xmr():
    # xmr's value_field is a continuous measurement, not a count -- a
    # negative value (e.g. a deviation) is completely legitimate and must
    # not trip the count-field check.
    dates = _dates(5)
    df = pd.DataFrame({"period": dates, "y": [1.5, -2.3, 0.0, -1.1, 3.2]})
    opts = BaseChartOptions(value_field="y", date_field="period")
    result = spc.chart(df, kind="xmr", options=opts)
    assert len(result.points) == 5


def test_fractional_value_does_not_raise_for_xbar_r():
    # xbar_r's value_field is also a continuous measurement -- fractional
    # values are the normal case, not an error.
    dates = _dates(4)
    rows = [
        {"period": d, "measurement": v}
        for d, vals in zip(
            dates, [[10.1, 12.3, 11.7], [9.4, 10.2, 8.8], [13.1, 14.6, 12.2], [11.3, 10.9, 12.4]], strict=True
        )
        for v in vals
    ]
    df = pd.DataFrame(rows)
    opts = BaseChartOptions(value_field="measurement", date_field="period")
    result = spc.chart(df, kind="xbar_r", options=opts)
    assert len(result.points) == 4


def test_count_check_does_not_apply_to_t():
    # t's value_field holds timestamps, not counts -- the count-field check
    # must not apply to it at all (there's no numeric "count" to validate).
    dates = _dates(5)
    df = pd.DataFrame({"period": dates})
    opts = BaseChartOptions(value_field="period", date_field="period")
    result = spc.chart(df, kind="t", options=opts)
    assert len(result.points) == 4


# --- Internal invariants no longer silently disappear under -O -----------


def test_default_rules_name_consistency_is_a_real_check_not_an_assert():
    from spc.core.rules import DEFAULT_RULE_NAMES, DEFAULT_RULES

    # The module already imported successfully (this test file imports
    # spc, which imports spc.core.rules) -- if this line executed, the
    # if-raise at module scope already ran and passed. Just confirm the
    # invariant it checks is what we think it is.
    assert {r.name for r in DEFAULT_RULES} == set(DEFAULT_RULE_NAMES)

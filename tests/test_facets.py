"""Two-variable facets (facet_field as row + column_facet_field): the
combined key, facet_parts, per-facet options, and equivalence with
running each cell on its own.
"""

from datetime import date, timedelta

import pandas as pd
import pytest
from approx_rows import assert_rows_approx

import spc
from spc.core.facets import FACET_SEPARATOR
from spc.core.models import BaseChartOptions, ExternalBaseline


def _dates(
        n,
        start=date(2024, 1, 1)
):
    return [start + timedelta(weeks=i) for i in range(n)]


def _grid_df():
    rows = []
    values = {
        ("A", "x"): [10, 12, 11, 13, 9, 12],
        ("A", "y"): [30, 28, 33, 31, 29, 32],
        ("B", "x"): [50, 55, 52, 49, 51, 53],
        ("B", "y"): [5, 6, 4, 7, 5, 6],
    }
    for (r, c), vals in values.items():
        for d, v in zip(_dates(6), vals, strict=True):
            rows.append({"period": d, "y": v, "row": r, "col": c})
    return pd.DataFrame(rows)


def _opts(
        **kw
):
    return BaseChartOptions(value_field="y", date_field="period", facet_field="row", column_facet_field="col", **kw)


def test_two_variable_facets_equal_running_each_cell_alone():
    df = _grid_df()
    result = spc.chart(df, kind="xmr", options=_opts())
    for (r, c), cell in df.groupby(["row", "col"]):
        alone = spc.chart(
            cell[["period", "y"]], kind="xmr", options=BaseChartOptions(value_field="y", date_field="period")
        )
        key = f"{r}{FACET_SEPARATOR}{c}"
        got = [(p.x, p.y, p.mean, p.ucl, p.lcl) for p in result.points if p.facet == key]
        assert_rows_approx(got, [(p.x, p.y, p.mean, p.ucl, p.lcl) for p in alone.points])


def test_facet_parts_recover_row_and_column():
    result = spc.chart(_grid_df(), kind="xmr", options=_opts())
    assert {tuple(p.facet_parts) for p in result.points} == {("A", "x"), ("A", "y"), ("B", "x"), ("B", "y")}
    assert all(s.facet_parts == s.facet.split(FACET_SEPARATOR) for s in result.segments)


def test_facet_parts_none_for_one_variable_and_no_facet():
    df = _grid_df()
    one = spc.chart(
        df[df["col"] == "x"],
        kind="xmr",
        options=BaseChartOptions(value_field="y", date_field="period", facet_field="row"),
    )
    assert all(p.facet_parts is None for p in one.points)
    none = spc.chart(
        df[(df["row"] == "A") & (df["col"] == "x")],
        kind="xmr",
        options=BaseChartOptions(value_field="y", date_field="period"),
    )
    assert all(p.facet is None and p.facet_parts is None for p in none.points)


def test_column_facet_field_requires_facet_field():
    with pytest.raises(ValueError, match="requires facet_field"):
        BaseChartOptions(value_field="y", date_field="period", column_facet_field="col")


def test_column_facet_field_must_differ_from_facet_field():
    with pytest.raises(ValueError, match="must differ"):
        BaseChartOptions(value_field="y", date_field="period", facet_field="row", column_facet_field="row")


def test_missing_column_facet_field_column_raises():
    opts = BaseChartOptions(value_field="y", date_field="period", facet_field="row", column_facet_field="nope")
    with pytest.raises(ValueError, match="nope"):
        spc.chart(_grid_df(), kind="xmr", options=opts)


def test_separator_in_a_facet_value_raises():
    df = _grid_df()
    df.loc[0, "row"] = f"A{FACET_SEPARATOR}x"
    with pytest.raises(ValueError, match="reserved"):
        spc.chart(df, kind="xmr", options=_opts())


def test_per_facet_target_keyed_by_combined_label():
    key_a = f"A{FACET_SEPARATOR}x"
    result = spc.chart(
        _grid_df(), kind="xmr", options=_opts(target={f"{r}{FACET_SEPARATOR}{c}": 1000.0 for r in "AB" for c in "xy"})
    )
    assert isinstance(result.assurance_type, dict)
    assert key_a in result.assurance_type


def test_per_facet_rebase_keyed_by_combined_label():
    df = _grid_df()
    key = f"A{FACET_SEPARATOR}x"
    result = spc.chart(df, kind="xmr", options=_opts(rebase={key: [_dates(6)[3]]}))
    segs_a = {p.segment_id for p in result.points if p.facet == key}
    segs_b = {p.segment_id for p in result.points if p.facet == f"B{FACET_SEPARATOR}x"}
    assert len(segs_a) == 2 and len(segs_b) == 1


def test_per_facet_baseline_keyed_by_combined_label():
    df = _grid_df()
    baseline = [
        ExternalBaseline(facet=f"{r}{FACET_SEPARATOR}{c}", x=d, expected_value=float(v), source="test")
        for r, c, d, v in zip(df["row"], df["col"], df["period"], df["y"], strict=True)
    ]
    result = spc.chart(df, kind="xmr", options=_opts(baseline=baseline))
    assert all(p.point_type == "common_cause" for p in result.points)


def test_two_variable_facets_work_for_subgroup_attribute_and_rare_event_kinds():
    dates = _dates(6)
    # xbar_r: 3 raw measurements per (row, col, date)
    rows = [
        {"period": d, "m": 10 + i + j, "row": r, "col": c}
        for r in "AB"
        for c in "xy"
        for i, d in enumerate(dates)
        for j in (0.1, 0.2, 0.3)
    ]
    xr = spc.chart(
        pd.DataFrame(rows),
        kind="xbar_r",
        options=BaseChartOptions(value_field="m", date_field="period", facet_field="row", column_facet_field="col"),
    )
    assert len({p.facet for p in xr.points}) == 4
    # t: event timestamps within each cell
    ev = [
        {
            "t": date(2024, 1, 1) + timedelta(days=k * (3 + "AB".index(r) + 2 * "xy".index(c)) + k * k % 5),
            "row": r,
            "col": c,
        }
        for r in "AB"
        for c in "xy"
        for k in range(8)
    ]
    t = spc.chart(
        pd.DataFrame(ev),
        kind="t",
        options=BaseChartOptions(value_field="t", date_field="t", facet_field="row", column_facet_field="col"),
    )
    assert len({p.facet for p in t.points}) == 4


@pytest.mark.parametrize("two_variable", [False, True])
def test_missing_facet_values_raise(
        two_variable
):
    df = _grid_df()
    df.loc[3, "row"] = None
    opts = _opts() if two_variable else BaseChartOptions(value_field="y", date_field="period", facet_field="row")
    with pytest.raises(ValueError, match="missing value"):
        spc.chart(df, kind="xmr", options=opts)

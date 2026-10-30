import pandas as pd
import pytest

import spc
from spc.pareto import ParetoOptions, pareto


def _df():
    return pd.DataFrame(
        {"cause": ["a", "b", "a", "c", "b", "a", "d", "a", "b", "e"], "minutes": [5, 3, 5, 1, 2, 5, 1, 5, 4, 2]}
    )


def test_counts_rows_when_no_count_field():
    r = pareto(_df(), ParetoOptions(category_field="cause"))
    assert [(x.category, x.count) for x in r.rows] == [("a", 4.0), ("b", 3.0), ("c", 1.0), ("d", 1.0), ("e", 1.0)]
    assert r.total == 10.0


def test_weights_by_count_field_and_shares_are_hand_computed():
    r = pareto(_df(), ParetoOptions(category_field="cause", count_field="minutes"))
    assert [(x.category, x.count) for x in r.rows] == [("a", 20.0), ("b", 9.0), ("e", 2.0), ("c", 1.0), ("d", 1.0)]
    assert r.total == 33.0
    assert r.rows[0].share == pytest.approx(20 / 33)
    assert r.rows[1].cumulative_share == pytest.approx(29 / 33)
    assert r.rows[-1].cumulative_share == 1.0
    assert [x.rank for x in r.rows] == [1, 2, 3, 4, 5]


def test_ties_break_by_category_name_for_a_deterministic_order():
    df = pd.DataFrame({"c": ["z", "y", "x", "x", "y", "z"]})
    assert [x.category for x in pareto(df, ParetoOptions(category_field="c")).rows] == ["x", "y", "z"]


def test_max_categories_collapses_the_tail_into_other_last():
    r = pareto(_df(), ParetoOptions(category_field="cause", count_field="minutes", max_categories=2))
    assert [(x.category, x.count, x.is_other) for x in r.rows] == [
        ("a", 20.0, False),
        ("b", 9.0, False),
        ("Other", 4.0, True),
    ]
    assert r.rows[-1].cumulative_share == 1.0
    assert sum(x.share for x in r.rows) == pytest.approx(1.0)


def test_other_is_last_even_when_larger_than_the_kept_categories():
    df = pd.DataFrame({"c": ["a"] * 3 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2 + ["e"] * 2})
    r = pareto(df, ParetoOptions(category_field="c", max_categories=2, other_label="Rest"))
    assert [x.category for x in r.rows] == ["a", "b", "Rest"] and r.rows[-1].count == 6.0 and r.rows[-1].is_other


def test_no_collapse_when_categories_do_not_exceed_max():
    r = pareto(_df(), ParetoOptions(category_field="cause", max_categories=5))
    assert not any(x.is_other for x in r.rows)


@pytest.mark.parametrize(
    "df, opts, match",
    [
        (pd.DataFrame({"c": []}), ParetoOptions(category_field="c"), "no rows"),
        (pd.DataFrame({"c": ["a"]}), ParetoOptions(category_field="nope"), "nope"),
        (pd.DataFrame({"c": ["a", None]}), ParetoOptions(category_field="c"), "missing value"),
        (pd.DataFrame({"c": ["a"], "n": [-1]}), ParetoOptions(category_field="c", count_field="n"), "non-negative"),
        (pd.DataFrame({"c": ["a"], "n": ["x"]}), ParetoOptions(category_field="c", count_field="n"), "numeric"),
        (pd.DataFrame({"c": ["a"], "n": [0]}), ParetoOptions(category_field="c", count_field="n"), "zero"),
    ],
)
def test_invalid_inputs_raise(
        df,
        opts,
        match
):
    with pytest.raises(ValueError, match=match):
        pareto(df, opts)


def test_max_categories_must_be_positive():
    with pytest.raises(ValueError, match="max_categories"):
        ParetoOptions(category_field="c", max_categories=0)


def test_result_is_json_serialisable():
    assert '"rows"' in pareto(_df(), ParetoOptions(category_field="cause")).model_dump_json()


def test_plotly_pareto_renders_bars_and_cumulative_line():
    pytest.importorskip("plotly")
    r = pareto(_df(), ParetoOptions(category_field="cause", count_field="minutes", max_categories=2))
    fig = spc.render.plotly_pareto(r)
    assert [t.type for t in fig.data] == ["bar", "scatter"]
    assert list(fig.data[0].x) == ["a", "b", "Other"]
    assert fig.data[0].marker.color[-1] == "#a6a6a6"
    assert list(fig.data[1].y)[-1] == pytest.approx(100.0)

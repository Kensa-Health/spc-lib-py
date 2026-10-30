"""Input-handling tests: missing or infinite values, option keys that match
nothing, ambiguous ordering, facet dtypes and CUSUM numerical limits. Each must
raise (or warn) clearly rather than pass silently or fail cryptically.
"""

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from approx_rows import assert_rows_approx

import spc
from spc.core.models import BaseChartOptions, ProportionChartOptions
from spc.cusum import CusumOptions, cusum, cusum_arl, cusum_threshold, estimate_event_rate
from spc.pareto import ParetoOptions, pareto


def _dates(
        n,
        start=date(2024, 1, 1)
):
    return [start + timedelta(weeks=i) for i in range(n)]


def _xmr(
        df,
        **kw
):
    return spc.chart(df, kind="xmr", options=BaseChartOptions(value_field="y", date_field="p", **kw))


def _base(
        n=10
):
    return pd.DataFrame({"p": _dates(n), "y": np.arange(float(n)) % 3 + 10, "g": ["a"] * n})


# --- missing facets / dates ---------------------------------------------------


def test_missing_facet_values_raise_for_the_subgroup_kinds_too():
    rows = [
        {"p": d, "m": 10.0 + i + j, "g": "a" if k < 2 else None}
        for k in range(3)
        for i, d in enumerate(_dates(4))
        for j in (0.1, 0.2)
    ]
    with pytest.raises(ValueError, match="missing value"):
        spc.chart(
            pd.DataFrame(rows),
            kind="xbar_r",
            options=BaseChartOptions(value_field="m", date_field="p", facet_field="g"),
        )


@pytest.mark.parametrize("kind", ["xmr", "c", "g"])
def test_missing_dates_raise_a_clear_error(
        kind
):
    df = pd.DataFrame({"p": [_dates(1)[0], None, _dates(3)[2], _dates(4)[3]], "y": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="missing date"):
        spc.chart(df, kind=kind, options=BaseChartOptions(value_field="y", date_field="p"))


def test_unparseable_dates_raise_a_clear_error():
    df = pd.DataFrame({"p": ["2024-01-01", "not a date", "2024-01-15"], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="must hold dates"):
        _xmr(df)


def test_cusum_and_event_rate_raise_on_missing_dates():
    df = pd.DataFrame({"d": [_dates(1)[0], None, _dates(3)[2]], "y": [0, 1, 0]})
    with pytest.raises(ValueError, match="missing date"):
        cusum(df, CusumOptions(value_field="y", date_field="d", p0=0.2, h=3.0))
    big = pd.DataFrame({"d": [_dates(1)[0], None] + [_dates(2)[1]] * 20, "y": [1, 0] * 11})
    with pytest.raises(ValueError, match="missing date"):
        estimate_event_rate(big, "y", "d", until=date(2030, 1, 1))


# --- infinite values ----------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, df",
    [
        ("xmr", pd.DataFrame({"p": _dates(5), "y": [1, 2, np.inf, 4, 5]})),
        ("c", pd.DataFrame({"p": _dates(5), "y": [1, 2, -np.inf, 4, 5]})),
        ("g", pd.DataFrame({"p": _dates(5), "y": [1, 2, np.inf, 4, 5]})),
        ("xbar_r", pd.DataFrame({"p": [_dates(1)[0]] * 3 + [_dates(2)[1]] * 3, "y": [1, 2, np.inf, 4, 5, 6]})),
    ],
)
def test_infinite_values_raise(
        kind,
        df
):
    with pytest.raises(ValueError, match="infinite"):
        spc.chart(df, kind=kind, options=BaseChartOptions(value_field="y", date_field="p"))


def test_infinite_denominator_raises():
    df = pd.DataFrame({"p": _dates(5), "y": [1, 2, 3, 4, 5], "n": [10, 10, np.inf, 10, 10]})
    with pytest.raises(ValueError, match="infinite"):
        spc.chart(df, kind="p", options=ProportionChartOptions(value_field="y", date_field="p", denominator_field="n"))


def test_pareto_rejects_infinite_weights():
    df = pd.DataFrame({"c": ["a", "b"], "n": [np.inf, 1.0]})
    with pytest.raises(ValueError, match="finite"):
        pareto(df, ParetoOptions(category_field="c", count_field="n"))


# --- options that match nothing -----------------------------------------------------


def test_rebase_date_matching_no_row_raises():
    with pytest.raises(ValueError, match="no entry in rebase matches"):
        _xmr(_base(), rebase=[date(2024, 1, 2)])


def test_rebase_date_matching_a_row_still_works():
    r = _xmr(_base(), rebase=[_dates(10)[5]])
    assert sorted({p.segment_id for p in r.points}) == [0, 1]


def test_rebase_dict_with_an_unknown_facet_or_a_wrong_date_raises():
    with pytest.raises(ValueError, match="no entry in rebase matches"):
        _xmr(_base(), facet_field="g", rebase={"zzz": [_dates(10)[5]]})
    with pytest.raises(ValueError, match="no entry in rebase matches"):
        _xmr(_base(), facet_field="g", rebase={"a": [date(2000, 1, 1)]})


def test_rebase_dict_without_a_facet_field_raises():
    with pytest.raises(ValueError, match="no entry in rebase matches"):
        _xmr(_base(), rebase={"a": [_dates(10)[5]]})


@pytest.mark.parametrize("name", ["target", "spec_upper", "spec_lower"])
def test_per_facet_dicts_with_unknown_keys_raise(
        name
):
    with pytest.raises(ValueError, match=f"no entry in {name} matches"):
        _xmr(_base(), facet_field="g", **{name: {"zzz": 5.0}})


def test_t_chart_rebase_is_checked_against_event_timestamps():
    events = [date(2024, 1, 1) + timedelta(days=d) for d in (0, 5, 15, 18, 30, 33, 50)]
    df = pd.DataFrame({"t": events})
    opts = BaseChartOptions(value_field="t", date_field="t", rebase=[date(2024, 1, 2)])
    with pytest.raises(ValueError, match="no entry in rebase matches"):
        spc.chart(df, kind="t", options=opts)
    ok = spc.chart(df, kind="t", options=opts.model_copy(update={"rebase": [events[4]]}))
    assert sorted({p.segment_id for p in ok.points}) == [0, 1]


def test_a_baseline_wider_than_the_data_is_still_allowed():
    from spc.core.models import ExternalBaseline

    df = _base(6)
    baseline = [ExternalBaseline(x=d, expected_value=10.0, source="t") for d in _dates(12)]
    assert len(_xmr(df, baseline=baseline).points) == 6


# --- unknown rule names ----------------------------------------------------------


def test_unknown_rule_name_raises_a_value_error():
    with pytest.raises(ValueError, match="unknown rule name"):
        _xmr(_base(), rules=["outside_limits", "nonsense_rule"])


# --- ambiguous order ------------------------------------------------------------------


def test_duplicate_dates_within_a_facet_warn():
    df = pd.DataFrame({"p": [_dates(1)[0], _dates(1)[0], _dates(2)[1], _dates(3)[2]], "y": [1.0, 2.0, 3.0, 4.0]})
    with pytest.warns(UserWarning, match="share a date"):
        _xmr(df)


def test_same_date_in_different_facets_is_not_a_tie():
    df = pd.DataFrame({"p": _dates(4) * 2, "y": np.arange(8.0), "g": list("aaaabbbb")})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _xmr(df, facet_field="g")


def test_subgroup_kinds_may_repeat_dates_without_warning():
    rows = [{"p": d, "m": 10.0 + i + j} for i, d in enumerate(_dates(4)) for j in (0.1, 0.2, 0.3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spc.chart(pd.DataFrame(rows), kind="xbar_r", options=BaseChartOptions(value_field="m", date_field="p"))


# --- facet dtypes ---------------------------------------------------------------------


def test_integer_facet_values_work_and_become_text_keys():
    df = pd.DataFrame({"p": _dates(5) * 2, "y": np.arange(10.0), "g": [1] * 5 + [2] * 5})
    r = _xmr(df, facet_field="g")
    assert {p.facet for p in r.points} == {"1", "2"}


def test_categorical_facet_with_unused_categories_works_for_every_path():
    cats = ["a", "b", "unused"]
    df = pd.DataFrame(
        {"p": _dates(6) * 2, "y": np.arange(12.0), "g": pd.Categorical(["a"] * 6 + ["b"] * 6, categories=cats)}
    )
    assert {p.facet for p in _xmr(df, facet_field="g").points} == {"a", "b"}
    rows = [
        {"p": d, "m": float(i + j), "g": g}
        for g in ("a", "b")
        for i, d in enumerate(_dates(4))
        for j in (0.1, 0.2, 0.3)
    ]
    sub = pd.DataFrame(rows)
    sub["g"] = pd.Categorical(sub["g"], categories=cats)
    xb = spc.chart(sub, kind="xbar_r", options=BaseChartOptions(value_field="m", date_field="p", facet_field="g"))
    assert {p.facet for p in xb.points} == {"a", "b"}
    cu = cusum(
        pd.DataFrame(
            {"d": _dates(6) * 2, "y": [0, 1] * 6, "g": pd.Categorical(["a"] * 6 + ["b"] * 6, categories=cats)}
        ),
        CusumOptions(value_field="y", date_field="d", p0={"a": 0.3, "b": 0.4}, h=3.0, facet_field="g"),
    )
    assert [s.facet for s in cu.series] == ["a", "b"]


def test_t_chart_warns_about_a_facet_with_a_single_event():
    ev = pd.DataFrame(
        {"t": [date(2024, 1, 1), date(2024, 1, 5), date(2024, 1, 9), date(2024, 2, 1)], "f": ["a", "a", "a", "b"]}
    )
    with pytest.warns(UserWarning, match=r"\['b'\].*fewer than 2 events"):
        spc.chart(ev, kind="t", options=BaseChartOptions(value_field="t", date_field="t", facet_field="f"))


# --- pareto label clash ------------------------------------------------------------------


def test_pareto_other_label_clash_raises_only_when_a_collapse_happens():
    df = pd.DataFrame({"c": ["Other"] * 5 + ["b"] * 3 + ["c"] * 2 + ["d"]})
    with pytest.raises(ValueError, match="already named 'Other'"):
        pareto(df, ParetoOptions(category_field="c", max_categories=2))
    assert not any(r.is_other for r in pareto(df, ParetoOptions(category_field="c")).rows)
    assert (
        pareto(df, ParetoOptions(category_field="c", max_categories=2, other_label="Rest")).rows[-1].category == "Rest"
    )


# --- CUSUM numerics --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: cusum_arl(0.1, 2.0, float("inf")),
        lambda: cusum_arl(0.1, 2.0, float("nan")),
        lambda: cusum_arl(0.1, float("inf"), 3.0),
        lambda: cusum_arl(1e-300, 2.0, 3.0),
        lambda: cusum_arl(0.1, 1 + 1e-12, 3.0),
        lambda: cusum_arl(0.3, 2.0, 40.0),  # ARL far beyond what the solve can resolve
        lambda: cusum_threshold(0.3, 2.0, 1e15),
        lambda: cusum_threshold(0.3, 2.0, float("inf")),
    ],
)
def test_cusum_run_length_refuses_inputs_it_cannot_compute_reliably(
        call
):
    with pytest.raises(ValueError):
        call()


def test_cusum_run_length_is_still_available_up_to_the_reliable_limit():
    # near 1e12 the default lattice is a few percent off, and the threshold says so
    with pytest.warns(UserWarning, match="finer lattice"):
        h = cusum_threshold(0.3, 2.0, 1e12)
    assert cusum_arl(0.3, 2.0, h) == pytest.approx(1e12, rel=0.08)


@pytest.mark.parametrize(
    "kw", [{"h": float("inf")}, {"arl0": float("inf")}, {"odds_ratio": float("inf")}, {"arl0": 1e14}]
)
def test_cusum_options_reject_non_finite_or_uncalibratable_values(
        kw
):
    with pytest.raises(ValueError):
        CusumOptions(value_field="y", date_field="d", p0=0.1, **kw)


def test_cusum_arl0_limit_halves_for_two_sided_calibration():
    with pytest.raises(ValueError, match="too large"):
        CusumOptions(value_field="y", date_field="d", p0=0.1, arl0=8e12, detect="both")


# --- unknown option names are rejected --------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: BaseChartOptions(value_field="y", date_field="p", facet_feild="g"),
        lambda: BaseChartOptions(value_field="y", date_field="p", rebase_dates=["2024-01-08"]),
        lambda: ProportionChartOptions(value_field="y", date_field="p", denominator_field="n", denominator="n"),
        lambda: CusumOptions(value_field="y", date_field="d", p0=0.1, odds_ratoi=3),
        lambda: ParetoOptions(category_field="c", max_categorys=5),
        lambda: spc.ExternalBaseline(x="2024-01-01", expected_value=1.0, source="s", expected=3),
    ],
)
def test_misspelled_option_names_raise(
        build
):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        build()


# --- the result keeps and restores the options it was run with --------------------------------


def test_result_json_keeps_the_denominator_field_and_round_trips_the_options_class():
    from spc.core.models import ChartResult

    df = pd.DataFrame({"p": _dates(8), "y": [3, 4, 5, 2, 6, 4, 3, 5], "n": [100] * 8})
    result = spc.chart(
        df, kind="p", options=ProportionChartOptions(value_field="y", date_field="p", denominator_field="n")
    )
    assert result.model_dump(mode="json")["options"]["denominator_field"] == "n"
    back = ChartResult.model_validate_json(result.model_dump_json())
    assert isinstance(back.options, ProportionChartOptions) and back.options.denominator_field == "n"
    plain = spc.chart(_base(), kind="xmr", options=BaseChartOptions(value_field="y", date_field="p"))
    assert type(ChartResult.model_validate_json(plain.model_dump_json()).options) is BaseChartOptions


# --- non-finite numbers in options -------------------------------------------------------------


@pytest.mark.parametrize("name", ["target", "spec_upper", "spec_lower"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_target_and_spec_limits_raise(
        name,
        bad
):
    with pytest.raises(ValueError):
        BaseChartOptions(value_field="y", date_field="p", **{name: bad})
    with pytest.raises(ValueError):
        BaseChartOptions(value_field="y", date_field="p", facet_field="g", **{name: {"a": bad}})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_baseline_value_raises(
        bad
):
    with pytest.raises(ValueError):
        spc.ExternalBaseline(x=date(2024, 1, 1), expected_value=bad, source="s")


def test_plotly_rejects_a_non_finite_future_space():
    pytest.importorskip("plotly")
    result = spc.chart(_base(), kind="xmr", options=BaseChartOptions(value_field="y", date_field="p"))
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="future_space"):
            spc.render.plotly(result, future_space=bad)


# --- Laney charts with a zero centre line -------------------------------------------------------


@pytest.mark.parametrize(("laney", "classical"), [("u_prime", "u"), ("p_prime", "p")])
def test_laney_charts_with_a_zero_baseline_window_flag_like_the_classical_chart(
        laney,
        classical
):
    y = [0, 0, 0, 0, 0, 3, 5, 2, 4, 6, 3, 5]
    df = pd.DataFrame({"p": _dates(12), "y": y, "n": [100] * 12})

    def run(
            kind
    ):
        opts = ProportionChartOptions(
            value_field="y", date_field="p", denominator_field="n", limit_mode="fixed_n", fix_after_n_points=5
        )
        return sorted(spc.chart(df, kind=kind, options=opts).points, key=lambda pt: pt.x)

    got, want = run(laney), run(classical)
    assert all(pt.ucl is not None for pt in got)
    assert [pt.ucl for pt in got] == pytest.approx([pt.ucl for pt in want])
    assert [pt.outside_limits for pt in got] == [pt.outside_limits for pt in want]
    assert sum(pt.outside_limits for pt in got) == 7


@pytest.mark.parametrize("kind", ["u_prime", "p_prime"])
def test_laney_chart_of_an_all_zero_series_matches_the_classical_chart(
        kind
):
    df = pd.DataFrame({"p": _dates(10), "y": [0] * 10, "n": [100] * 10})
    opts = ProportionChartOptions(value_field="y", date_field="p", denominator_field="n")
    result = spc.chart(df, kind=kind, options=opts)
    assert {pt.ucl for pt in result.points} == {0.0}


def test_laney_sigma_z_ignores_points_with_zero_standard_error():
    from spc.core.limits import laney_sigma_z

    y = np.array([0.0, 0.0, 4.0, 6.0, 5.0])
    centre = np.array([0.0, 0.0, 5.0, 5.0, 5.0])
    se = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    # z is undefined for the first two points and is -1, 1, 0 for the rest, so the
    # usable moving ranges are |1 - -1| = 2 and |0 - 1| = 1: sigma_z = 1.5 / 1.128
    assert laney_sigma_z(y, centre, se) == pytest.approx(1.5 / 1.128)
    assert laney_sigma_z(np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 1.0
    assert np.isnan(laney_sigma_z(np.array([1.0]), np.array([1.0]), np.array([1.0])))


# --- shared configs that cover more than the data ---------------------------------------------------


def _three_facets():
    return pd.DataFrame({"p": _dates(10) * 3, "y": np.arange(30.0) % 4 + 10, "g": list("A" * 10 + "B" * 10 + "C" * 10)})


@pytest.mark.parametrize("name", ["target", "spec_upper", "spec_lower"])
def test_dict_options_covering_extra_facets_warn_but_still_run(
        name
):
    df = _three_facets()
    df = df[df["g"] != "C"]
    with pytest.warns(UserWarning, match=rf"1 of 3 entries in {name} match nothing"):
        result = _xmr(df, facet_field="g", **{name: {"A": 12.0, "B": 12.0, "C": 12.0}})
    assert {p.facet for p in result.points} == {"A", "B"}


def test_rebase_dict_with_extra_facets_or_dates_warns_but_still_rebases():
    df = _three_facets()
    df = df[df["g"] == "A"]
    with pytest.warns(UserWarning, match="match nothing"):
        result = _xmr(df, facet_field="g", rebase={"A": [_dates(10)[5]], "B": [_dates(10)[5]]})
    assert sorted({p.segment_id for p in result.points}) == [0, 1]
    with pytest.warns(UserWarning, match="1 of 2 entries in rebase"):
        result = _xmr(_base(), rebase=[date(2000, 1, 1), _dates(10)[5]])
    assert sorted({p.segment_id for p in result.points}) == [0, 1]


def test_fully_matching_options_are_quiet():
    df = _three_facets()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _xmr(df, facet_field="g", target={"A": 1.0, "B": 2.0, "C": 3.0}, rebase={"A": [_dates(10)[4]]})


# --- sequence_field: a tie-break for the Shewhart kinds ---------------------------------------------


def _tied_rows():
    # five events on each of two dates, listed out of their true order
    d = [date(2024, 1, 1)] * 5 + [date(2024, 1, 2)] * 5
    return pd.DataFrame({"p": d, "y": [30, 10, 50, 20, 40, 15, 45, 25, 35, 5], "t": [5, 1, 3, 2, 4, 2, 5, 1, 4, 3]})


@pytest.mark.parametrize("kind", ["xmr", "g"])
def test_sequence_field_orders_tied_rows_like_a_presorted_frame(
        kind
):
    df = _tied_rows()
    ordered = df.sort_values(["p", "t"]).reset_index(drop=True)
    plain = BaseChartOptions(value_field="y", date_field="p")
    by_sequence = BaseChartOptions(value_field="y", date_field="p", sequence_field="t")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no tie warning once a tie-break is given
        got = spc.chart(df, kind=kind, options=by_sequence)
        want = spc.chart(ordered, kind=kind, options=by_sequence)
    assert_rows_approx(
        [(p.y, p.mean, p.ucl, p.lcl) for p in got.points], [(p.y, p.mean, p.ucl, p.lcl) for p in want.points]
    )
    assert [p.y for p in got.points] == [float(v) for v in ordered["y"]]
    with pytest.warns(UserWarning, match="sequence_field"):
        spc.chart(df, kind=kind, options=plain)


def test_sequence_field_works_with_facets_and_attribute_charts():
    df = pd.concat([_tied_rows().assign(g="a"), _tied_rows().assign(g="b")], ignore_index=True)
    df["n"] = 100
    opts = ProportionChartOptions(
        value_field="y", date_field="p", denominator_field="n", sequence_field="t", facet_field="g"
    )
    result = spc.chart(df.assign(y=df["y"] % 40), kind="p", options=opts)
    assert len(result.points) == 20 and {p.facet for p in result.points} == {"a", "b"}


def test_rows_still_tied_on_date_and_sequence_raise():
    df = _tied_rows().assign(t=[1, 1, 2, 3, 4, 1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="still ambiguous"):
        spc.chart(df, kind="xmr", options=BaseChartOptions(value_field="y", date_field="p", sequence_field="t"))


def test_sequence_field_must_be_complete_and_of_one_type():
    df = _tied_rows()
    with pytest.raises(ValueError, match="missing values"):
        spc.chart(
            df.assign(t=[1.0, None, 3, 4, 5, 1, 2, 3, 4, 5]),
            kind="xmr",
            options=BaseChartOptions(value_field="y", date_field="p", sequence_field="t"),
        )
    with pytest.raises(ValueError, match="mixes value types"):
        spc.chart(
            df.assign(t=[1, "b", 3, 4, 5, 1, 2, 3, 4, 5]),
            kind="xmr",
            options=BaseChartOptions(value_field="y", date_field="p", sequence_field="t"),
        )
    with pytest.raises(ValueError, match="nope"):
        spc.chart(df, kind="xmr", options=BaseChartOptions(value_field="y", date_field="p", sequence_field="nope"))


@pytest.mark.parametrize("kind", ["t", "xbar_r", "xbar_s"])
def test_kinds_that_cannot_use_a_tie_break_reject_it(
        kind
):
    df = (
        _tied_rows().assign(y=lambda x: pd.to_datetime("2024-01-01") + pd.to_timedelta(x["y"], unit="D"))
        if kind == "t"
        else _tied_rows()
    )
    with pytest.raises(ValueError, match="sequence_field isn't used by kind"):
        spc.chart(df, kind=kind, options=BaseChartOptions(value_field="y", date_field="p", sequence_field="t"))

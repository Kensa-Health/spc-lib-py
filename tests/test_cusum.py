import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import spc
from spc.cusum import (
    CusumOptions,
    _alternative_rate,
    _weights,
    cusum,
    cusum_arl,
    cusum_threshold,
    estimate_event_rate,
)


def _dates(
        n,
        start=date(2024, 1, 1)
):
    return [start + timedelta(days=i) for i in range(n)]


def _df(
        outcomes,
        **extra
):
    return pd.DataFrame({"d": _dates(len(outcomes)), "y": outcomes, **extra})


# --- scoring ---------------------------------------------------------------


def test_alternative_rate_matches_the_odds_ratio():
    p1 = _alternative_rate(0.05, 2.0, "increase")
    assert (p1 / (1 - p1)) / (0.05 / 0.95) == pytest.approx(2.0)
    p1d = _alternative_rate(0.05, 2.0, "decrease")
    assert (p1d / (1 - p1d)) / (0.05 / 0.95) == pytest.approx(0.5)


def test_scores_hand_computed():
    p0, p1 = 0.2, _alternative_rate(0.2, 2.0, "increase")
    w1, w0 = _weights(p0, p1)
    assert w1 == pytest.approx(math.log(p1 / p0)) and w0 == pytest.approx(math.log((1 - p1) / (1 - p0)))
    outcomes = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    opts = CusumOptions(value_field="y", date_field="d", p0=p0, h=10.0)
    pts = cusum(_df(outcomes), opts).series[0].points
    expected, s = [], 0.0
    for y in outcomes:
        s = max(0.0, s + (w1 if y else w0))
        expected.append(s)
    assert [p.score for p in pts] == pytest.approx(expected)
    assert expected[0] == 0.0  # a non-event at zero stays floored
    assert all(not p.signal for p in pts)


def test_signal_flag_and_reset():
    opts = CusumOptions(value_field="y", date_field="d", p0=0.1, h=1.0)
    r = cusum(_df([1, 1, 1, 1, 1, 1]), opts).series[0]
    w1 = math.log(_alternative_rate(0.1, 2.0, "increase") / 0.1)  # ~0.6 per event
    # signals every ceil(1.0/w1) = 2 events, and the score restarts after each
    assert r.signal_indices == [1, 3, 5]
    assert r.points[1].score == pytest.approx(2 * w1) and r.points[2].score == pytest.approx(w1)


def test_without_reset_the_score_keeps_growing_and_one_signal_is_reported():
    opts = CusumOptions(value_field="y", date_field="d", p0=0.1, h=1.0, reset_on_signal=False)
    r = cusum(_df([1, 1, 1, 1, 1, 1]), opts).series[0]
    assert r.signal_indices == [1]
    assert [p.signal for p in r.points] == [False, True, True, True, True, True]
    assert r.points[-1].score == pytest.approx(6 * math.log(_alternative_rate(0.1, 2.0, "increase") / 0.1))


def test_decrease_direction_accumulates_on_fewer_events():
    opts = CusumOptions(value_field="y", date_field="d", p0=0.5, detect="decrease", h=1.5)
    r = cusum(_df([0] * 10), opts).series[0]
    assert r.direction == "decrease" and r.odds_ratio == pytest.approx(0.5)
    assert r.signal_indices and r.points[0].score > 0
    # an increase-only chart on the same data never accumulates
    up = cusum(_df([0] * 10), CusumOptions(value_field="y", date_field="d", p0=0.5, h=1.5)).series[0]
    assert all(p.score == 0.0 for p in up.points)


def test_detect_both_gives_two_series_per_facet():
    opts = CusumOptions(value_field="y", date_field="d", p0=0.3, detect="both", h=2.0)
    r = cusum(_df([0, 1, 0, 1, 1, 0]), opts)
    assert [s.direction for s in r.series] == ["increase", "decrease"]
    assert r.series[0].p1 > 0.3 > r.series[1].p1


def test_cases_are_ordered_by_date():
    df = pd.DataFrame({"d": [date(2024, 1, 3), date(2024, 1, 1), date(2024, 1, 2)], "y": [1, 0, 1]})
    r = cusum(df, CusumOptions(value_field="y", date_field="d", p0=0.2, h=10.0)).series[0]
    assert [p.x for p in r.points] == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    assert [p.y for p in r.points] == [0, 1, 1]


def _tied_df():
    # four cases on one day, listed in the wrong order for their true sequence
    return pd.DataFrame({"d": [date(2024, 1, 1)] * 4, "y": [0, 1, 1, 0], "t": [4, 1, 2, 3]})


def test_cases_sharing_a_date_warn_when_no_sequence_field_is_given():
    with pytest.warns(UserWarning, match="share a date/time"):
        cusum(_tied_df(), CusumOptions(value_field="y", date_field="d", p0=0.2, h=10.0))


def test_sequence_field_breaks_ties_and_silences_the_warning():
    import warnings

    opts = CusumOptions(value_field="y", date_field="d", p0=0.2, h=10.0, sequence_field="t")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = cusum(_tied_df(), opts).series[0]
    assert [p.y for p in r.points] == [1, 1, 0, 0]  # ordered by t = 1, 2, 3, 4


def test_order_changes_the_signal_so_the_tie_break_matters():
    # threshold of exactly two events' worth of score: two adjacent events
    # reach it, but an event, a non-event and an event do not (the
    # non-event subtracts)
    df = pd.DataFrame({"d": [date(2024, 1, 1)] * 3, "y": [1, 0, 1]})
    w1 = math.log(_alternative_rate(0.2, 2.0, "increase") / 0.2)
    h = 2 * w1 - 1e-9

    def run(
            sequence
    ):
        opts = CusumOptions(value_field="y", date_field="d", p0=0.2, h=h, sequence_field="t")
        return cusum(df.assign(t=sequence), opts).series[0].signal_indices

    assert run([1, 3, 2]) == [1]  # event, event, non-event
    assert run([1, 2, 3]) == []  # event, non-event, event


def test_ambiguous_order_still_raises_when_the_sequence_field_has_ties():
    df = pd.DataFrame({"d": [date(2024, 1, 1)] * 3, "y": [0, 1, 1], "t": [1, 2, 2]})
    with pytest.raises(ValueError, match="still ambiguous"):
        cusum(df, CusumOptions(value_field="y", date_field="d", p0=0.2, h=10.0, sequence_field="t"))


def test_missing_sequence_values_raise_and_the_column_must_exist():
    df = pd.DataFrame({"d": _dates(3), "y": [0, 1, 0], "t": [1.0, None, 3.0]})
    with pytest.raises(ValueError, match="missing values"):
        cusum(df, CusumOptions(value_field="y", date_field="d", p0=0.2, h=10.0, sequence_field="t"))
    with pytest.raises(ValueError, match="not found"):
        cusum(df, CusumOptions(value_field="y", date_field="d", p0=0.2, h=10.0, sequence_field="nope"))


def test_ties_are_judged_within_each_facet():
    import warnings

    df = pd.DataFrame({"d": [date(2024, 1, 1)] * 2, "y": [0, 1], "g": ["a", "b"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cusum(df, CusumOptions(value_field="y", date_field="d", p0=0.2, h=10.0, facet_field="g"))


def test_facets_run_independent_cusums_and_support_per_facet_p0():
    df = pd.DataFrame({"d": _dates(8) * 2, "y": [1, 1, 0, 0, 1, 0, 0, 0] + [0] * 8, "g": ["a"] * 8 + ["b"] * 8})
    r = cusum(df, CusumOptions(value_field="y", date_field="d", p0={"a": 0.1, "b": 0.3}, facet_field="g", h=1.0))
    by = {s.facet: s for s in r.series}
    assert by["a"].p0 == 0.1 and by["b"].p0 == 0.3
    assert by["a"].signal_indices and not by["b"].signal_indices
    alone = cusum(df[df["g"] == "a"], CusumOptions(value_field="y", date_field="d", p0=0.1, h=1.0)).series[0]
    assert [p.score for p in by["a"].points] == pytest.approx([p.score for p in alone.points])


def test_two_variable_facets_set_facet_parts():
    df = pd.DataFrame({"d": _dates(4) * 2, "y": [0, 1, 0, 1] * 2, "r": ["A"] * 4 + ["B"] * 4, "c": ["x"] * 8})
    r = cusum(df, CusumOptions(value_field="y", date_field="d", p0=0.2, h=2.0, facet_field="r", column_facet_field="c"))
    assert sorted(tuple(s.facet_parts) for s in r.series) == [("A", "x"), ("B", "x")]


# --- validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"p0": 0.0}, "strictly between"),
        ({"p0": 1.0}, "strictly between"),
        ({"p0": 0.1, "odds_ratio": 1.0}, "odds_ratio"),
        ({"p0": 0.1, "odds_ratio": 0.5}, "odds_ratio"),
        ({"p0": 0.1, "h": 0.0}, "h must be"),
        ({"p0": 0.1, "arl0": 1.0}, "arl0"),
        ({"p0": {"a": 0.1}}, "facet_field"),
        ({"p0": 0.1, "column_facet_field": "c"}, "requires facet_field"),
    ],
)
def test_invalid_options_raise(
        kwargs,
        match
):
    with pytest.raises(ValueError, match=match):
        CusumOptions(value_field="y", date_field="d", **kwargs)


@pytest.mark.parametrize(
    "df, match",
    [
        (pd.DataFrame({"d": [], "y": []}), "no rows"),
        (pd.DataFrame({"d": _dates(2), "y": [0, 2]}), "0/1"),
        (pd.DataFrame({"d": _dates(2), "y": [0, float("nan")]}), "missing"),
        (pd.DataFrame({"d": _dates(2), "y": ["a", "b"]}), "0/1"),
        (pd.DataFrame({"d": _dates(2)}), "not found"),
    ],
)
def test_invalid_data_raises(
        df,
        match
):
    with pytest.raises(ValueError, match=match):
        cusum(df, CusumOptions(value_field="y", date_field="d", p0=0.1, h=1.0))


def test_missing_facet_in_p0_dict_raises():
    df = pd.DataFrame({"d": _dates(2), "y": [0, 1], "g": ["a", "b"]})
    with pytest.raises(ValueError, match="no p0 supplied for facet 'b'"):
        cusum(df, CusumOptions(value_field="y", date_field="d", p0={"a": 0.1}, facet_field="g", h=1.0))


def test_boolean_outcomes_are_accepted():
    r = cusum(_df([True, False, True]), CusumOptions(value_field="y", date_field="d", p0=0.2, h=5.0))
    assert [p.y for p in r.series[0].points] == [1, 0, 1]


# --- run lengths ---------------------------------------------------------------


def test_arl_exact_when_the_score_steps_are_equal_and_opposite():
    # p0=0.4, p1=0.6 (odds ratio 2.25) -> steps of +/-log(1.5). A threshold of
    # one step signals on the first event: ARL = 1/p. Two steps, with the
    # score floored at 0, solves to (1+p)/p^2 by hand.
    step = math.log(1.5)
    assert cusum_arl(0.4, 2.25, step) == pytest.approx(1 / 0.4, rel=1e-9)
    assert cusum_arl(0.4, 2.25, 2 * step) == pytest.approx((1 + 0.4) / 0.4**2, rel=1e-9)
    # true rate 0.6 (the alternative): p=0.6
    assert cusum_arl(0.4, 2.25, 2 * step, p_true=0.6) == pytest.approx((1 + 0.6) / 0.6**2, rel=1e-9)


def _simulate_arl(
        p0,
        odds_ratio,
        h,
        p_true,
        direction="increase",
        runs=10000,
        seed=7
):
    p1 = _alternative_rate(p0, odds_ratio, direction)
    w1, w0 = _weights(p0, p1)
    rng = np.random.default_rng(seed)
    score = np.zeros(runs)
    alive = np.ones(runs, bool)
    length = np.zeros(runs)
    t = 0
    while alive.any():
        t += 1
        step = np.where(rng.random(runs) < p_true, w1, w0)
        score = np.where(alive, np.maximum(0.0, score + step), score)
        done = alive & (score >= h)
        length[done] = t
        alive &= ~done
    return length.mean()


@pytest.mark.parametrize(
    "p0, ratio, h, p_true, direction",
    [
        (0.05, 2.0, 2.0, 0.05, "increase"),
        (0.05, 2.0, 2.0, _alternative_rate(0.05, 2.0, "increase"), "increase"),  # at the alternative
        (0.2, 2.0, 2.0, 0.2, "decrease"),
        (0.2, 2.0, 2.0, _alternative_rate(0.2, 2.0, "decrease"), "decrease"),  # at the alternative
    ],
)
def test_arl_matches_a_monte_carlo_simulation(
        p0,
        ratio,
        h,
        p_true,
        direction
):
    expected = _simulate_arl(p0, ratio, h, p_true, direction)
    assert cusum_arl(p0, ratio, h, p_true=p_true, direction=direction) == pytest.approx(expected, rel=0.04)


def test_arl_converges_as_the_lattice_is_refined():
    coarse = cusum_arl(0.05, 2.0, 4.0, n_states=200)
    medium = cusum_arl(0.05, 2.0, 4.0, n_states=800)
    fine = cusum_arl(0.05, 2.0, 4.0, n_states=3200)
    assert abs(medium - fine) < abs(coarse - fine)
    assert medium == pytest.approx(fine, rel=0.01)


def test_arl_increases_with_h_and_is_much_shorter_out_of_control():
    in_control = [cusum_arl(0.05, 2.0, h) for h in (1.0, 2.0, 3.0)]
    assert in_control == sorted(in_control) and len(set(in_control)) == 3
    assert cusum_arl(0.05, 2.0, 3.0, p_true=0.0952) < in_control[-1] / 5


def test_threshold_round_trips_to_the_target_arl_and_is_conservative():
    for p0, ratio, arl0, d in [(0.05, 2.0, 500, "increase"), (0.3, 1.5, 200, "increase"), (0.3, 2.0, 300, "decrease")]:
        h = cusum_threshold(p0, ratio, arl0, direction=d)
        got = cusum_arl(p0, ratio, h, direction=d)
        assert got >= arl0 * 0.999 and got == pytest.approx(arl0, rel=0.01)


def test_threshold_validation():
    with pytest.raises(ValueError, match="arl0"):
        cusum_threshold(0.1, 2.0, 1.0)
    with pytest.raises(ValueError, match="strictly between"):
        cusum_threshold(0.0, 2.0, 100)


def test_cusum_computes_h_from_arl0_when_not_given():
    r = cusum(_df([0] * 5), CusumOptions(value_field="y", date_field="d", p0=0.1, arl0=200)).series[0]
    assert r.h == pytest.approx(cusum_threshold(0.1, 2.0, 200))


# --- helpers / output ---------------------------------------------------------


def test_estimate_event_rate_from_a_reference_window():
    outcomes = [1, 0, 0, 0] * 10 + [1] * 20  # first 40 rows: 10 events, 30 non-events
    df = _df(outcomes)
    assert estimate_event_rate(df, "y", "d", until=_dates(len(outcomes))[39]) == pytest.approx(0.25)


def test_estimate_event_rate_requires_enough_events_and_non_events():
    df = _df([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])  # 2 events, 8 non-events
    until = _dates(10)[-1]
    with pytest.raises(ValueError, match="2 event"):
        estimate_event_rate(df, "y", "d", until=until)
    assert estimate_event_rate(df, "y", "d", until=until, min_events=2) == pytest.approx(0.2)


@pytest.mark.parametrize("outcomes", [[0] * 12, [1] * 12])
def test_estimate_event_rate_rejects_rates_of_exactly_0_or_1(
        outcomes
):
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        estimate_event_rate(_df(outcomes), "y", "d", until=_dates(12)[-1], min_events=1)


def test_estimate_event_rate_empty_window_raises():
    with pytest.raises(ValueError, match="empty"):
        estimate_event_rate(_df([1, 0]), "y", "d", until=date(2000, 1, 1))


def test_result_is_json_serialisable_and_exported():
    r = cusum(_df([0, 1, 0, 1]), CusumOptions(value_field="y", date_field="d", p0=0.2, h=2.0))
    assert '"series"' in r.model_dump_json()
    assert "series" in spc.render.to_dict(r)


def test_plotly_cusum_draws_score_threshold_and_signals():
    pytest.importorskip("plotly")
    df = pd.DataFrame({"d": _dates(6) * 2, "y": [1, 1, 1, 1, 1, 1] + [0] * 6, "g": ["a"] * 6 + ["b"] * 6})
    r = cusum(df, CusumOptions(value_field="y", date_field="d", p0=0.1, h=1.0, facet_field="g", detect="both"))
    fig = spc.render.plotly_cusum(r)
    names = [t.name for t in fig.data]
    assert "Threshold h" in names and "Signal" in names
    assert sum(1 for t in fig.data if t.name == "Threshold h" and t.showlegend) == 1
    # the decrease series is mirrored below zero
    assert any(min(t.y) < 0 for t in fig.data if t.name and t.name.startswith("CUSUM score (rate decrease)"))


def test_unknown_direction_raises():
    with pytest.raises(ValueError, match="direction"):
        cusum_arl(0.1, 2.0, 2.0, direction="both")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="direction"):
        _alternative_rate(0.1, 2.0, "sideways")  # type: ignore[arg-type]


def test_detect_both_calibrates_arl0_as_the_joint_either_side_run_length():
    # Each one-sided chart is set to 2*arl0, so a signal on EITHER side comes
    # every ~arl0 observations. (Calibrating each side to arl0 would give
    # ~arl0/2 jointly, as the simulation below confirms.)
    arl0 = 100
    r = cusum(_df([0] * 5), CusumOptions(value_field="y", date_field="d", p0=0.2, detect="both", arl0=arl0))
    up, down = r.series
    assert up.h == pytest.approx(cusum_threshold(0.2, 2.0, 2 * arl0, direction="increase"))
    assert down.h == pytest.approx(cusum_threshold(0.2, 2.0, 2 * arl0, direction="decrease"))

    runs = 6000
    rng = np.random.default_rng(11)
    sides = []
    for direction, h in (("increase", up.h), ("decrease", down.h)):
        w1, w0 = _weights(0.2, _alternative_rate(0.2, 2.0, direction))
        sides.append((w1, w0, h))
    s = np.zeros((2, runs))
    alive = np.ones(runs, bool)
    length = np.zeros(runs)
    t = 0
    while alive.any():
        t += 1
        x = rng.random(runs) < 0.2
        for k, (w1, w0, _h) in enumerate(sides):
            s[k] = np.where(alive, np.maximum(0.0, s[k] + np.where(x, w1, w0)), s[k])
        done = alive & ((s[0] >= sides[0][2]) | (s[1] >= sides[1][2]))
        length[done] = t
        alive &= ~done
    assert length.mean() == pytest.approx(arl0, rel=0.05)


def test_explicit_h_is_used_as_given_on_both_sides():
    r = cusum(_df([0] * 5), CusumOptions(value_field="y", date_field="d", p0=0.2, detect="both", h=1.7))
    assert [s.h for s in r.series] == [1.7, 1.7]


def test_threshold_warns_when_the_lattice_has_not_converged():
    # a large odds ratio with a very short run length: the default lattice
    # is visibly off here, and the function says so instead of staying quiet
    with pytest.warns(UserWarning, match="finer lattice"):
        cusum_threshold(0.1, 20.0, 8.5, direction="decrease")


def test_threshold_is_quiet_in_the_ordinary_regime():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cusum_threshold(0.05, 2.0, 500)


def test_time_of_day_orders_same_day_cases_and_is_not_a_tie():
    import warnings

    ts = pd.to_datetime(["2024-01-01 15:00", "2024-01-01 09:00", "2024-01-01 12:00"])
    df = pd.DataFrame({"t": ts, "y": [1, 0, 1]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = cusum(df, CusumOptions(value_field="y", date_field="t", p0=0.2, h=10.0)).series[0]
    assert [p.y for p in r.points] == [0, 1, 1]  # 09:00, 12:00, 15:00
    assert all(p.x == date(2024, 1, 1) for p in r.points)


def test_mixed_type_sequence_field_raises():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01"] * 3), "y": [0, 1, 1], "s": [1, "b", 3]})
    with pytest.raises(ValueError, match="mixes value types"):
        cusum(df, CusumOptions(value_field="y", date_field="d", p0=0.2, h=10.0, sequence_field="s"))

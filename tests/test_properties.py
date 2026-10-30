"""Property checks that hold for every chart kind, over seeded random inputs:
point counts, limit ordering, flag consistency, segment bookkeeping, JSON round
trip, determinism, the input frame left untouched, and a faceted chart equal to
one chart per facet.
"""

import math
from collections import Counter
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import spc
from spc.core.models import BaseChartOptions, ChartResult, LimitMode, ProportionChartOptions

KINDS = ["xmr", "p", "p_prime", "np", "u", "u_prime", "c", "xbar_r", "xbar_s", "t", "g"]
BASE = date(2024, 1, 1)
SEEDS = 60


def _facet_layout(
        rng
):
    mode = ["none", "one", "two"][int(rng.integers(0, 3))]
    n_row = 1 if mode == "none" else int(rng.integers(1, 4))
    n_col = 2 if mode == "two" else 1
    return mode, n_row, n_col


def make_case(
        kind,
        seed
):
    """A random frame and matching options for one chart kind."""
    rng = np.random.default_rng(seed * 7919 + KINDS.index(kind))
    n = int(rng.choice([1, 2, 3, 5, 8, 13, 20, 40]))
    facet_mode, n_row, n_col = _facet_layout(rng)
    rows = []
    for row in range(n_row):
        for col in range(n_col):
            level = rng.uniform(5, 50)
            for i in range(n):
                record = {"p": BASE + timedelta(weeks=i), "r": f"F{row}", "c": f"C{col}"}
                if kind == "xmr":
                    record["y"] = float(rng.normal(level, 3))
                elif kind in ("p", "p_prime", "np"):
                    denominator = 200 if kind == "np" else int(rng.integers(50, 400))
                    record["y"] = int(rng.binomial(denominator, rng.uniform(0.05, 0.5)))
                    record["n"] = denominator
                elif kind in ("u", "u_prime"):
                    denominator = int(rng.integers(50, 400))
                    record["y"] = int(rng.poisson(denominator * 0.05))
                    record["n"] = denominator
                elif kind == "c":
                    record["y"] = int(rng.poisson(level))
                elif kind in ("xbar_r", "xbar_s"):
                    for _ in range(4 if kind == "xbar_r" else 12):
                        rows.append({**record, "y": float(rng.normal(level, 2))})
                    continue
                elif kind == "t":
                    record["y"] = pd.Timestamp(BASE) + pd.Timedelta(days=int(i * rng.integers(1, 15) + i))
                else:  # g
                    record["y"] = int(rng.integers(1, 200))
                rows.append(record)
    df = pd.DataFrame(rows)
    limit_mode = list(LimitMode)[int(rng.integers(0, 3))]
    options = {
        "value_field": "y",
        "date_field": "p",
        "limit_mode": limit_mode,
        "fix_after_n_points": int(rng.integers(2, 8)) if limit_mode != LimitMode.SEGMENT else None,
        "screen_outliers": ["off", "single_pass", "iterative"][int(rng.integers(0, 3))],
        "run_rule_length": [7, 4, "proportional"][int(rng.integers(0, 3))],
        "improvement_direction": ["increase", "decrease", "neutral"][int(rng.integers(0, 3))],
    }
    if facet_mode != "none":
        options["facet_field"] = "r"
    if facet_mode == "two":
        options["column_facet_field"] = "c"
    if rng.random() < 0.3:
        options["target"] = float(rng.uniform(1, 50))
    if rng.random() < 0.3:
        options["spec_upper"], options["spec_lower"] = float(rng.uniform(50, 99)), float(rng.uniform(0, 4))
    if kind == "xmr" and rng.random() < 0.3:
        options["autocorrelation_adjustment"] = True
    if rng.random() < 0.3 and n > 3 and kind != "t":
        options["rebase"] = [BASE + timedelta(weeks=int(rng.integers(1, n - 1)))]
    if kind in ("p", "p_prime", "np", "u", "u_prime"):
        return df, ProportionChartOptions(denominator_field="n", **options)
    return df, BaseChartOptions(**options)


def expected_points(
        kind,
        df,
        options
):
    facets = [c for c in (options.facet_field, options.column_facet_field) if c]
    if kind == "t":
        return sum(max(len(g) - 1, 0) for _, g in df.groupby(facets)) if facets else max(len(df) - 1, 0)
    if kind in ("xbar_r", "xbar_s"):
        return len(df.groupby(["p", *facets]))
    return len(df)


def finite_or_none(
        value
):
    return value is None or (isinstance(value, (int, float)) and math.isfinite(value))


def signature(
        pts
):
    def r(
            v
    ):
        return None if v is None else round(v, 9)

    return [(p.x, r(p.y), r(p.mean), r(p.ucl), r(p.lcl), p.point_type) for p in pts]


def check_invariants(
        kind,
        df,
        options,
        seed
):
    before = df.copy(deep=True)
    result = spc.chart(df, kind=kind, options=options)
    where = f"{kind} seed {seed}"
    assert df.equals(before), f"input modified: {where}"
    assert len(result.points) == expected_points(kind, df, options), f"point count: {where}"
    for p in result.points:
        assert math.isfinite(p.y) and math.isfinite(p.mean), f"non-finite value: {where}"
        assert finite_or_none(p.ucl) and finite_or_none(p.lcl), f"non-finite limit: {where}"
        if p.ucl is not None and p.lcl is not None:
            assert p.lcl - 1e-9 <= p.mean <= p.ucl + 1e-9, f"mean outside limits: {where}"
            outside = p.y > p.ucl + 1e-12 or p.y < p.lcl - 1e-12
            on_limit = abs(p.y - p.ucl) < 1e-9 or abs(p.y - p.lcl) < 1e-9
            assert outside == p.outside_limits or on_limit, f"outside_limits flag: {where}"
    counts = Counter((p.facet, p.segment_id) for p in result.points)
    assert all(counts[(s.facet, s.segment_id)] == s.n for s in result.segments), f"segment n: {where}"
    if options.column_facet_field:
        assert all(p.facet_parts and " | ".join(p.facet_parts) == p.facet for p in result.points), where
    back = ChartResult.model_validate_json(result.model_dump_json())
    assert back.model_dump(exclude={"generated_at"}) == result.model_dump(exclude={"generated_at"}), (
        f"round trip: {where}"
    )
    again = spc.chart(df, kind=kind, options=options)
    assert [p.model_dump() for p in again.points] == [p.model_dump() for p in result.points], f"determinism: {where}"
    return result


def check_facets_equal_single_runs(
        kind,
        df,
        options,
        result,
        seed
):
    facet_fields = [c for c in (options.facet_field, options.column_facet_field) if c]
    if not facet_fields or options.rebase is not None or isinstance(options.target, dict):
        return
    alone_options = options.model_copy(update={"facet_field": None, "column_facet_field": None})
    for key, group in df.groupby(facet_fields):
        values = key if isinstance(key, tuple) else (key,)
        label = " | ".join(map(str, values))
        try:
            alone = spc.chart(group, kind=kind, options=alone_options)
        except ValueError:
            continue
        got = signature([p for p in result.points if p.facet == label])
        assert got == signature(alone.points), f"facet {label} != run alone: {kind} seed {seed}"


def check_row_order_does_not_matter(
        kind,
        df,
        options,
        result,
        seed
):
    """Shuffling the input rows (no two rows share a date within a facet, apart from
    xbar subgroups, which are order-free) gives the same chart.
    """
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    again = spc.chart(shuffled, kind=kind, options=options)
    key = lambda p: (p.facet or "", p.x)  # noqa: E731
    got = signature(sorted(again.points, key=key))
    assert got == signature(sorted(result.points, key=key)), f"row order changed the chart: {kind} seed {seed}"


@pytest.mark.parametrize("kind", KINDS)
def test_invariants_hold_over_random_inputs(
        kind
):
    ran = 0
    for seed in range(SEEDS):
        df, options = make_case(kind, seed)
        try:
            result = check_invariants(kind, df, options, seed)
        except ValueError:
            continue  # e.g. a t chart with fewer than two events, or a rebase date that is no event
        check_facets_equal_single_runs(kind, df, options, result, seed)
        check_row_order_does_not_matter(kind, df, options, result, seed)
        ran += 1
    assert ran >= SEEDS * 0.6, f"only {ran} of {SEEDS} generated cases were valid for {kind}"

"""Segment building for the subgrouped-continuous-data chart family
(xbar_r, xbar_s). A genuinely different data shape from every other chart
kind: multiple raw rows collapse into one plotted point per (facet, date)
subgroup, rather than one row = one point.

Supports an external baseline the same way core/segments.py does for xmr:
the subgroup mean is residualised against the supplied expected value,
the grand mean is computed on the residual, then the baseline is added
back. Valid here specifically because the control-limit width (A2*R-bar
or A3*s-bar) is computed from subgroup spread alone, independent of the
subgroup means -- unlike p/np/c/u, where the width depends on the level
itself and this same approach would not carry over unmodified.

Scope, stated explicitly rather than left implicit:
  - Subgroup size (n) must be constant within a segment. Real Xbar charts
    can in principle handle varying n (each point using its own A2(n)/
    A3(n)), the same way P/U charts already do here -- but that adds real
    complexity (weighted grand-mean conventions aren't as settled as the
    equal-n case), so a clear error is raised if n varies instead, same
    pattern as np_chart's constant-denominator requirement.
  - Only the means (Xbar) chart is returned as a ChartResult. Classic
    practice always reviews the companion range/S chart alongside it (the
    means chart's validity depends on the spread being in control) -- but
    ChartResult has no field for a second, companion chart. This is a
    known gap: xbar_r_constants()/xbar_s_constants() in core/limits.py
    already compute D3/D4 and B3/B4 for exactly this, ready to wire up if
    ChartResult grows room for it.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from spc.core.attribute_segments import _windowed_mean
from spc.core.facets import facet_columns
from spc.core.limits import compute_point_flags, xbar_r_constants, xbar_s_constants
from spc.core.models import BaseChartOptions
from spc.core.segments import NO_FACET_SENTINEL, SegmentArrays, _build_baseline_lookup, _expected_value_series

_SUBGROUP_MEAN = "_subgroup_mean"
_SUBGROUP_SPREAD = "_subgroup_spread"  # range for xbar_r, sample std (ddof=1) for xbar_s
_SUBGROUP_N = "_subgroup_n"


def aggregate_subgroups(
        df: pd.DataFrame,
        options: BaseChartOptions,
        spread: str
) -> pd.DataFrame:
    """One row per (facet, date) subgroup: mean, range-or-std, and size of
    that subgroup's raw measurements. Keeps date_field/facet_field column
    names unchanged so the result can be passed straight into
    assign_segments() -- that function only reads those two columns (plus
    `rebase`), never value_field, so it works on this aggregated frame
    exactly as it does on a raw one-row-per-point frame.
    """
    group_cols = [options.date_field, *facet_columns(options)]
    # observed=True: a categorical facet with unused categories must not create empty groups
    grouped = df.groupby(group_cols, observed=True)[options.value_field]

    spread_fn = (lambda s: s.max() - s.min()) if spread == "range" else (lambda s: s.std(ddof=1))

    # Explicitly typed `dict[str, Any]` rather than an inline dict literal
    # or a narrower union -- pandas-stubs' only SeriesGroupBy.agg overload
    # that accepts **kwargs-style named aggregation (mixing string method
    # names with a callable) types those kwargs as `Any`; a narrower
    # value type doesn't match any overload via **unpacking.
    agg_spec: dict[str, Any] = {
        _SUBGROUP_MEAN: "mean",
        _SUBGROUP_SPREAD: spread_fn,
        _SUBGROUP_N: "count",
    }
    agg = grouped.agg(**agg_spec).reset_index()

    if (agg[_SUBGROUP_N] < 2).any():
        raise ValueError(
            "every (facet, date) subgroup needs at least 2 raw measurements -- "
            "found a subgroup with only 1 row, which has no range or standard deviation"
        )
    return cast(pd.DataFrame, agg)


def build_subgroup_segment_arrays(
        df_with_segments: pd.DataFrame,
        options: BaseChartOptions,
        kind: str
) -> list[SegmentArrays]:
    """Runs the Xbar-R/Xbar-S means-chart calculation independently within
    each (facet, segment_id) group, on a frame already produced by
    aggregate_subgroups() + assign_segments().
    """
    if kind not in ("xbar_r", "xbar_s"):
        raise ValueError(f"subgroup-chart kind must be 'xbar_r' or 'xbar_s', got {kind!r}")
    baseline_lookup = _build_baseline_lookup(options)

    results: list[SegmentArrays] = []
    for segment_id, seg in df_with_segments.groupby("segment_id", sort=True):
        seg = seg.sort_values(options.date_field, kind="stable").reset_index(drop=True)

        n_values = seg[_SUBGROUP_N].unique()
        if len(n_values) != 1:
            raise ValueError(
                f"kind='{kind}' requires a constant subgroup size within each segment; "
                f"segment {segment_id} has subgroup sizes {sorted(n_values)} -- "
                "split into separate rebase segments, or ensure every period has the same sample count"
            )
        n = int(n_values[0])

        subgroup_mean = seg[_SUBGROUP_MEAN].astype(float)
        subgroup_spread = seg[_SUBGROUP_SPREAD].astype(float)

        expected_value = _expected_value_series(seg, options, baseline_lookup)
        centre_component = _windowed_mean(subgroup_mean - expected_value, options)
        grand_mean = expected_value + centre_component
        mean_spread = _windowed_mean(subgroup_spread, options)

        if kind == "xbar_r":
            a, _d3, _d4 = xbar_r_constants(n)
        elif kind == "xbar_s":
            a, _b3, _b4 = xbar_s_constants(n)
        else:  # unreachable: checked at the top of this function
            raise ValueError(f"unhandled subgroup-chart kind {kind!r}")

        width = a * mean_spread
        ucl = grand_mean + width
        lcl = grand_mean - width
        # The near-limit zone is 2/3 of the way to a 3-sigma limit (see
        # limits.py's _ATTRIBUTE_NEAR_SIGMA_MULTIPLIER comment); A2/A3
        # already encode the full 3-sigma width, so 2/3 of that width
        # gives the same ratio without needing a separate "near" constant.
        near_width = width * (2.0 / 3.0)
        near_ucl = grand_mean + near_width
        near_lcl = grand_mean - near_width

        y = subgroup_mean  # the plotted value IS the subgroup mean
        outside_limits, relative_to_mean, close_to_limits = compute_point_flags(
            y, grand_mean, ucl, lcl, near_ucl, near_lcl
        )

        facet_value = seg["facet"].iloc[0]
        results.append(
            SegmentArrays(
                facet=None if facet_value == NO_FACET_SENTINEL else facet_value,
                # See core/segments.py's build_segment_arrays() for why this cast is safe.
                segment_id=int(cast(int, segment_id)),
                x=seg[options.date_field],
                y=y,
                centre_line=grand_mean,
                ucl=ucl,
                lcl=lcl,
                amr=pd.Series(float("nan"), index=seg.index),  # not applicable
                outside_limits=outside_limits,
                relative_to_mean=relative_to_mean,
                close_to_limits=close_to_limits,
                low_count_warning=pd.Series(False, index=seg.index),  # not applicable to xbar_r/xbar_s
                autocorrelation=pd.Series(float("nan"), index=seg.index),  # not applicable to xbar_r/xbar_s
            )
        )
    return results

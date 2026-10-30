"""Facet/rebase segmentation and limit-mode dispatch.

Groups input data into (facet, rebase_group) segments, then computes
per-point centre line / limits within each segment according to LimitMode
and, if supplied, an external baseline.

The baseline seam is unified into the same code path as the plain-mean case
rather than kept as a separate branch: every point's "expected value" is
either the supplied baseline or 0 (when none is supplied), and the
mean/moving-range machinery always runs against `y - expected_value`. When
expected_value is 0 everywhere, this reduces exactly to the original XmR
calculation -- there is no separate flat-mean code path to keep in sync.

Two statistical caveats worth being explicit about:

- **Phase I/II conflation in SEGMENT mode.** SEGMENT computes limits from
  the same data it then screens for special-cause signals -- if that
  segment contains a genuine shift, the shift's own points inflate the
  mean/moving range used to judge them, which can both mute the signal and
  scatter false positives through the pre-shift data (see the
  shift-detection tests in tests/test_rebase.py). Classic SPC practice
  separates a clean Phase I baseline (used only to *set* limits) from
  Phase II monitoring (new data judged against those fixed limits).
  FIXED_N / EXPANDING with a deliberately-chosen `fix_after_n_points`
  covering only known-stable history is how to get that separation here
  -- SEGMENT is a reasonable default for retrospective review, not for
  maximum sensitivity to an ongoing shift.
- **No gap detection.** Input must be one row per period with no missing
  periods within a facet -- this is a caller responsibility, not something
  assign_segments validates. A missing period is silently treated as
  adjacent to its neighbour when computing the moving range.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from spc.core.facets import facet_series
from spc.core.limits import (
    _MAX_EXPANDING_WINDOW,
    LIMIT,
    LIMIT_CLOSE,
    compute_limit_stats,
    compute_point_flags,
    widen_amr_for_autocorrelation,
)
from spc.core.models import BaseChartOptions, LimitMode

# Sentinel used internally when no facet_field is configured. Converted back
# to None at the PointResult/SegmentSummary boundary (see charts/xmr.py) --
# this stays internal so segments.py never needs an Optional[str] facet
# column, which would complicate the groupby/sort logic for no real benefit.
# (facet, date) -> the externally supplied expected value for that point
BaselineLookup = dict[tuple[str | None, date], float]

NO_FACET_SENTINEL = "__all__"


@dataclass
class SegmentArrays:
    """Per-segment computed data -- the input rules.py detectors run against.

    Plain dataclass, not pydantic: this is internal working state (numpy/
    pandas arrays), not a boundary contract.
    """

    facet: str | None
    segment_id: int
    x: pd.Series
    y: pd.Series
    centre_line: pd.Series
    ucl: pd.Series
    lcl: pd.Series
    amr: pd.Series
    outside_limits: pd.Series
    relative_to_mean: pd.Series
    close_to_limits: pd.Series
    # True where the chart's underlying statistical approximation is
    # questionable for that point -- currently only meaningful for p/np
    # (binomial_approximation_low_count, core/limits.py); every other chart
    # kind fills this with all-False, the same "not applicable" convention
    # already used for `amr` on non-XmR chart kinds.
    low_count_warning: pd.Series
    # Lag-1 sample autocorrelation of the segment's detrended residuals
    # (xmr only, core/limits.py::estimate_autocorrelation) -- NaN-filled,
    # same "not applicable" convention as `amr`, for every other chart kind.
    autocorrelation: pd.Series


def _rebase_dates_for(
        rebase: list[date] | dict[str, list[date]] | None,
        facet: str
) -> set[date]:
    """Resolves the rebase-date list for one facet, from either a flat
    list (applies to every facet) or a per-facet dict.
    """
    if rebase is None:
        return set()
    if isinstance(rebase, dict):
        return set(rebase.get(facet, []))
    return set(rebase)


def order_columns(
        options: BaseChartOptions
) -> list[str]:
    """The columns rows are ordered by within a facet: the date, then the
    tie-break `sequence_field` when one is set.
    """
    return [options.date_field, *([options.sequence_field] if options.sequence_field else [])]


def assign_segments(
        df: pd.DataFrame,
        options: BaseChartOptions
) -> pd.DataFrame:
    """Sorts by (facet, date) and assigns an integer `segment_id` per
    (facet, rebase_group) combination. Returns a new dataframe; does not
    mutate the input.

    Normalises `date_field` to plain `datetime.date` regardless of input
    dtype (str, `pd.Timestamp`/datetime64, or already `date`). This matters
    beyond cosmetics: `rebase` and `ExternalBaseline.x` are validated by
    pydantic into `datetime.date`, and `date != Timestamp` even for the
    same calendar date in Python -- without normalising here, both
    `rebase` matching (below) and the baseline lookup in
    `_expected_value_series` would silently match nothing against a
    dataframe loaded the ordinary way, via
    `pd.read_csv(..., parse_dates=[...])`.
    """
    out = df.copy()
    out[options.date_field] = pd.to_datetime(out[options.date_field]).dt.date
    facets = facet_series(out, options)
    out["facet"] = facets if facets is not None else NO_FACET_SENTINEL
    out = out.sort_values(["facet", *order_columns(options)], kind="stable").reset_index(drop=True)

    out["is_rebase"] = [
        dt in _rebase_dates_for(options.rebase, facet)
        for facet, dt in zip(out["facet"], out[options.date_field], strict=True)
    ]
    out["rebase_group"] = out.groupby("facet")["is_rebase"].cumsum().astype(int)

    seg_keys = out[["facet", "rebase_group"]].drop_duplicates().reset_index(drop=True)
    seg_keys["segment_id"] = seg_keys.index
    return out.merge(seg_keys, on=["facet", "rebase_group"], how="left")


def _expected_value_series(
        seg: pd.DataFrame,
        options: BaseChartOptions,
        baseline_lookup: BaselineLookup
) -> pd.Series:
    """Per-point expected value for this segment: the supplied baseline
    where one exists, otherwise flat 0.
    """
    if not options.baseline:
        return pd.Series(0.0, index=seg.index)

    facet = seg["facet"].iloc[0]
    real_facet = None if facet == NO_FACET_SENTINEL else facet
    values = []
    for dt in seg[options.date_field]:
        key = (real_facet, dt)
        if key not in baseline_lookup:
            raise ValueError(
                f"baseline was supplied but has no entry for facet={real_facet!r}, date={dt!r} -- "
                "a supplied baseline must cover every point in every segment it applies to; "
                "partial coverage would silently mix two different centre-line concepts within one segment"
            )
        values.append(baseline_lookup[key])
    return pd.Series(values, index=seg.index, dtype=float)


def _flat_series(
        stats_mean: float,
        width_amr: float,
        autocorrelation: float,
        index: pd.Index
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Broadcasts a single mean/amr/autocorrelation triple across every index position."""
    return (
        pd.Series(stats_mean, index=index),
        pd.Series(width_amr, index=index),
        pd.Series(autocorrelation, index=index),
    )


def _limit_mode_series(
        y_for_calc: pd.Series,
        options: BaseChartOptions
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per-point (centre_component, amr, autocorrelation) for the given
    (possibly residual) value series, dispatching on options.limit_mode.
    All three are constant across the segment for SEGMENT/FIXED_N; all
    three vary during EXPANDING's warm-up phase, then lock.

    The returned `amr` is already autocorrelation-adjusted where
    applicable (see limits.py::widen_amr_for_autocorrelation) -- this is
    the same series build_segment_arrays() multiplies by LIMIT/LIMIT_CLOSE
    below to get ucl/lcl width, so the adjustment takes effect there
    without a second copy of the widening formula.
    """
    n = len(y_for_calc)

    if options.limit_mode == LimitMode.SEGMENT:
        stats = compute_limit_stats(
            y_for_calc, options.screen_outliers, autocorrelation_adjustment=options.autocorrelation_adjustment
        )
        width_amr = widen_amr_for_autocorrelation(stats.amr, stats.autocorrelation, options.autocorrelation_adjustment)
        return _flat_series(stats.mean, width_amr, stats.autocorrelation, y_for_calc.index)

    # FIXED_N and EXPANDING both require fix_after_n_points -- enforced by
    # BaseChartOptions.check_fix_after_n_points at construction time; this
    # local narrows the Optional[int] for the type checker (and doubles as
    # a defensive check, in case options was ever mutated post-construction).
    fix_after_n_points = options.fix_after_n_points
    assert fix_after_n_points is not None

    if options.limit_mode == LimitMode.FIXED_N:
        window = y_for_calc.iloc[:fix_after_n_points]
        stats = compute_limit_stats(
            window, options.screen_outliers, autocorrelation_adjustment=options.autocorrelation_adjustment
        )
        width_amr = widen_amr_for_autocorrelation(stats.amr, stats.autocorrelation, options.autocorrelation_adjustment)
        return _flat_series(stats.mean, width_amr, stats.autocorrelation, y_for_calc.index)

    # EXPANDING
    n_lock = min(fix_after_n_points, n)
    if n_lock > _MAX_EXPANDING_WINDOW:
        raise ValueError(
            f"fix_after_n_points={options.fix_after_n_points} exceeds the expanding-mode safety "
            f"cap of {_MAX_EXPANDING_WINDOW} points (O(N^2) recompute cost) -- use limit_mode='fixed_n' "
            "instead for large warm-up windows, since expanding's point-by-point recalculation isn't "
            "needed once N is this large anyway"
        )
    centre = np.full(n, np.nan)
    amr_arr = np.full(n, np.nan)
    rho_arr = np.full(n, np.nan)
    for i in range(n_lock):
        window = y_for_calc.iloc[: i + 1]
        if len(window) < 2:
            centre[i] = float(window.iloc[-1])
            amr_arr[i] = np.nan
        else:
            stats = compute_limit_stats(
                window, options.screen_outliers, autocorrelation_adjustment=options.autocorrelation_adjustment
            )
            centre[i] = stats.mean
            amr_arr[i] = widen_amr_for_autocorrelation(
                stats.amr, stats.autocorrelation, options.autocorrelation_adjustment
            )
            rho_arr[i] = stats.autocorrelation
    if n > n_lock:
        locked = compute_limit_stats(
            y_for_calc.iloc[:n_lock],
            options.screen_outliers,
            autocorrelation_adjustment=options.autocorrelation_adjustment,
        )
        centre[n_lock:] = locked.mean
        amr_arr[n_lock:] = widen_amr_for_autocorrelation(
            locked.amr, locked.autocorrelation, options.autocorrelation_adjustment
        )
        rho_arr[n_lock:] = locked.autocorrelation
    return (
        pd.Series(centre, index=y_for_calc.index),
        pd.Series(amr_arr, index=y_for_calc.index),
        pd.Series(rho_arr, index=y_for_calc.index),
    )


def _build_baseline_lookup(
        options: BaseChartOptions
) -> BaselineLookup:
    """(facet, date) -> expected_value, from options.baseline."""
    lookup: BaselineLookup = {}
    if options.baseline:
        for point in options.baseline:
            lookup[(point.facet, point.x)] = point.expected_value
    return lookup


def build_segment_arrays(
        df_with_segments: pd.DataFrame,
        options: BaseChartOptions
) -> list[SegmentArrays]:
    """Runs the limit-mode/baseline calculation independently within each
    (facet, segment_id) group and returns one SegmentArrays per segment,
    in segment_id order.
    """
    baseline_lookup = _build_baseline_lookup(options)

    results: list[SegmentArrays] = []
    for segment_id, seg in df_with_segments.groupby("segment_id", sort=True):
        seg = seg.sort_values(order_columns(options), kind="stable").reset_index(drop=True)
        y = seg[options.value_field].astype(float)
        expected_value = _expected_value_series(seg, options, baseline_lookup)
        y_for_calc = y - expected_value

        centre_component, amr, autocorrelation = _limit_mode_series(y_for_calc, options)
        centre_line = expected_value + centre_component
        limit_width = LIMIT * amr
        near_width = LIMIT_CLOSE * amr
        ucl = centre_line + limit_width
        lcl = centre_line - limit_width
        near_ucl = centre_line + near_width
        near_lcl = centre_line - near_width

        outside_limits, relative_to_mean, close_to_limits = compute_point_flags(
            y, centre_line, ucl, lcl, near_ucl, near_lcl
        )

        facet_value = seg["facet"].iloc[0]
        results.append(
            SegmentArrays(
                facet=None if facet_value == NO_FACET_SENTINEL else facet_value,
                # groupby("segment_id") types the loop key broadly (any
                # hashable), but the column is always the int index
                # assigned in assign_segments() above -- cast reflects
                # that known runtime type rather than changing it.
                segment_id=int(cast(int, segment_id)),
                x=seg[options.date_field],
                y=y,
                centre_line=centre_line,
                ucl=ucl,
                lcl=lcl,
                amr=amr,
                outside_limits=outside_limits,
                relative_to_mean=relative_to_mean,
                close_to_limits=close_to_limits,
                low_count_warning=pd.Series(False, index=seg.index),  # not applicable to xmr
                autocorrelation=autocorrelation,
            )
        )
    return results

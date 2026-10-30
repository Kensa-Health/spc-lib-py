"""Segment building for the rare-event chart family (t, g). Both apply the
Nelson (1994) power transform y = t^(1/3.6) to approximate normality, run
the ordinary XmR calculation on the transformed values, then
back-transform the centre line and limits for display.

Both kinds take the shared `BaseChartOptions`; `value_field` holds the
raw data, whose meaning differs by kind:
  - t: `value_field` holds a timestamp per rare event; elapsed time between
    consecutive events is computed here, not supplied pre-computed, and
    the later event of each consecutive pair supplies the x-position.
    `date_field` is required by the shared contract but not read for t --
    there's no natural single date for a *gap between* two events other
    than one of its endpoints, which the timestamp itself already gives.
  - g: each row already has both a count (`value_field`, opportunities/
    cases since the previous rare event) and its own date (`date_field`)
    -- simpler than t, since nothing needs to be derived across rows.

Supports an external baseline: `event_dates` (the synthetic frame's date
column) are real calendar dates drawn from the original data -- the later
event of each pair for t, the row's own date for g -- so the same
date-keyed lookup core/segments.py uses works here too, no separate
mechanism needed. The one real adjustment is that the synthetic
calculation runs in Nelson-power-transformed space, so a baseline
supplied in raw elapsed-time/count units is re-expressed in that space
before being handed to the residual machinery (`_baseline_for_synthetic`).

facet_field and rebase are supported: T gaps are computed within each
facet (each facet's first event yields no point), and a rebase date
starts a new segment at the point whose x-position equals it. For t that
x-position is the *later* event of a gap, so the gap straddling a process
change belongs to the new segment (consistent with the gap being plotted
at its later event), and the rebase date must equal an event timestamp.
`target` (scalar or per-facet) is applied by the shared assurance code
against the displayed, untransformed limits.
"""

from __future__ import annotations

import warnings

import pandas as pd

from spc.core.facets import facet_series
from spc.core.models import BaseChartOptions, ExternalBaseline
from spc.core.segments import SegmentArrays, assign_segments, build_segment_arrays, order_columns

NELSON_POWER = 1 / 3.6  # Nelson (1994) power transform, y = t^(1/3.6)


def _transform(
        y: pd.Series
) -> pd.Series:
    """Applies the Nelson power transform to approximate normality."""
    return y**NELSON_POWER


def _baseline_for_synthetic(
        options: BaseChartOptions
) -> list[ExternalBaseline] | None:
    """Re-expresses a supplied baseline in the power-transformed space the
    synthetic xmr calculation runs in -- expected_value arrives in raw
    elapsed-time/count units, and Nelson's transform has to be applied to
    it exactly as it is to the observed data it's compared against, or the
    two wouldn't be on the same scale.

    Each entry keeps its own facet, matching the facet values carried in
    the synthetic frame's `_facet` column.
    """
    if not options.baseline:
        return None
    for point in options.baseline:
        if point.expected_value < 0:
            raise ValueError(
                f"baseline expected_value can't be negative for kind='t'/'g' "
                f"(got {point.expected_value} for x={point.x!r}) -- elapsed time and counts can't be "
                "negative, and a negative base raised to a non-integer power is not a real number"
            )
    # Use the same vector power calculation as observations. Scalar ** can
    # round differently on some platforms; with an exact baseline, even a
    # one-ULP residual can create false signals against zero-width limits.
    transformed = _transform(pd.Series([point.expected_value for point in options.baseline], dtype=float))
    return [
        ExternalBaseline(facet=point.facet, x=point.x, expected_value=float(expected), source=point.source)
        for point, expected in zip(options.baseline, transformed, strict=True)
    ]


def _inverse_transform(
        y: pd.Series
) -> pd.Series:
    """Reverses the Nelson power transform for display.

    Elapsed time/counts can't be negative, but XmR's moving-range-based
    limits aren't bounded below zero -- a transformed-space LCL can go
    negative, and a negative base raised to a non-integer power (3.6) is
    not a real number. Clip at zero first, same reasoning as the p/np/c/u
    lower-bound clip in core/limits.py.
    """
    return y.clip(lower=0) ** (1 / NELSON_POWER)


def build_rare_event_segment_arrays(
        df: pd.DataFrame,
        options: BaseChartOptions,
        kind: str
) -> list[SegmentArrays]:
    """Runs the T/G rare-event calculation: transform, run the ordinary
    XmR calculation on a synthetic frame, then back-transform the result
    for display.

    The synthetic frame carries each point's original (untransformed)
    value in `_orig_y` and, when `facet_field` is set, its facet in
    `_facet` (the combined key when there are two facet variables); `rebase` and
    `baseline` are forwarded to the ordinary segmentation. Original values are recovered per segment from that
    frame (not by position across the whole series), so any number of
    facets/rebase groups realign correctly.
    """
    if kind not in ("t", "g"):
        raise ValueError(f"build_rare_event_segment_arrays handles kind 't' or 'g' only, got {kind!r}")
    facet_values = facet_series(df, options)
    has_facets = facet_values is not None
    if kind == "t":
        events = pd.DataFrame({"_t": pd.to_datetime(df[options.value_field])})
        if has_facets:
            events["_facet"] = facet_values
        events = events.sort_values((["_facet"] if has_facets else []) + ["_t"], kind="stable").reset_index(drop=True)
        # Gaps are computed within each facet: a gap spanning two facets'
        # events would be meaningless. Each facet's first event therefore
        # yields no point.
        gap = events.groupby("_facet")["_t"].diff() if has_facets else events["_t"].diff()
        keep = gap.notna()
        if has_facets:
            sizes = events.groupby("_facet").size()
            dropped = sorted(sizes[sizes < 2].index)
            if dropped and keep.any():
                warnings.warn(
                    f"kind='t': facet(s) {dropped} have fewer than 2 events, so they produce no points",
                    UserWarning,
                    stacklevel=4,
                )
        if not keep.any():
            raise ValueError(
                "kind='t' needs at least 2 events to compute an elapsed time between them"
                + (" (within at least one facet)" if has_facets else "")
            )
        original_y = (gap[keep].dt.total_seconds() / 86400.0).reset_index(drop=True)
        event_dates = events.loc[keep, "_t"].dt.date.reset_index(drop=True)
        facets = events.loc[keep, "_facet"].reset_index(drop=True) if has_facets else None
    elif kind == "g":
        if len(df) < 1:
            raise ValueError("kind='g' needs at least 1 event")
        original_y = df[options.value_field].astype(float).reset_index(drop=True)
        event_dates = pd.to_datetime(df[options.date_field]).dt.date.reset_index(drop=True)
        facets = facet_values.reset_index(drop=True) if facet_values is not None else None

    synthetic_df = pd.DataFrame({"_value": _transform(original_y), "_date": event_dates, "_orig_y": original_y})
    if facets is not None:
        synthetic_df["_facet"] = facets
    if options.sequence_field:  # only reachable for g: validation rejects it for t
        synthetic_df["_seq"] = df[options.sequence_field].reset_index(drop=True)

    synthetic_options = BaseChartOptions(
        value_field="_value",
        date_field="_date",
        facet_field="_facet" if has_facets else None,
        sequence_field="_seq" if options.sequence_field else None,
        rebase=options.rebase,
        limit_mode=options.limit_mode,
        fix_after_n_points=options.fix_after_n_points,
        screen_outliers=options.screen_outliers,
        improvement_direction=options.improvement_direction,
        rules=options.rules,
        short_segment_threshold=options.short_segment_threshold,
        baseline=_baseline_for_synthetic(options),
        # target isn't forwarded: assurance is computed later against the
        # original options, in display (untransformed) units.
    )
    segmented = assign_segments(synthetic_df, synthetic_options)
    transformed_segments = build_segment_arrays(segmented, synthetic_options)

    results: list[SegmentArrays] = []
    for seg in transformed_segments:
        # Flags (outside_limits/relative_to_mean/close_to_limits) were
        # already computed correctly in transformed space -- t^(1/3.6) is
        # monotonically increasing for t >= 0, so those comparisons hold
        # identically once both sides are consistently transformed. Only
        # the *displayed* values need converting back.
        #
        # Original values for this segment: the same selection and stable
        # date sort build_segment_arrays applied to produce `seg`, so the
        # two line up row for row (T-chart gap-end dates can collide, so a
        # date-keyed lookup would be ambiguous; this isn't one).
        original_for_segment = (
            segmented.loc[segmented["segment_id"] == seg.segment_id]
            .sort_values(order_columns(synthetic_options), kind="stable")["_orig_y"]
            .reset_index(drop=True)
        )
        results.append(
            SegmentArrays(
                facet=seg.facet,
                segment_id=seg.segment_id,
                x=seg.x,
                y=original_for_segment,
                centre_line=_inverse_transform(seg.centre_line),
                ucl=_inverse_transform(seg.ucl),
                lcl=_inverse_transform(seg.lcl),
                amr=seg.amr,  # stays in transformed units -- not part of the output contract
                outside_limits=seg.outside_limits,
                relative_to_mean=seg.relative_to_mean,
                close_to_limits=seg.close_to_limits,
                low_count_warning=seg.low_count_warning,  # not applicable to t/g -- already all-False
                autocorrelation=seg.autocorrelation,  # not applicable to t/g -- already all-NaN
            )
        )
    return results

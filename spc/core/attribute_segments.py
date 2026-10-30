"""Segment building for the attribute chart family (p, p_prime, np, c, u,
u_prime). Separate from core/segments.py's XmR-specific dispatch (which is
built around a moving-range estimate that these charts don't have) but
reuses everything chart-agnostic from it: assign_segments() (facet/rebase
grouping), compute_point_flags() (outside_limits/relative_to_mean/
close_to_limits -- works on any centre/ucl/lcl arrays, not XmR-specific
despite living in the same module), NO_FACET_SENTINEL, and the
SegmentArrays contract itself, which rules.py's detectors consume
identically regardless of which chart kind produced it.

Supports an external baseline, but not via the same subtract/residual/
add-back mechanism core/segments.py uses for xmr -- that mechanism relies
on pooling a residual mean, which doesn't compose with p/u's *pooled
ratio* (sum(numerator)/sum(denominator)) the way it does with xmr's plain
mean. Instead, when a baseline is supplied, it replaces the pooled/
windowed centre line outright: each point's expected value comes directly
from the baseline, and the usual per-point variance formula
(p_chart_limits etc.) is evaluated at that baseline value and the point's
own n. No separate bias-correction layer is added on top: the baseline is
taken as the expected value, so there is nothing left to correct. `limit_mode`/`fix_after_n_points` have nothing left to
estimate once a baseline supplies the centre directly, so they're not
applied in that case.

`screen_outliers` is also ignored here: it's specifically about screening
an outlying *moving range*, a concept these charts don't have.
`autocorrelation_adjustment` is ignored for the same reason -- it
corrects a bias specific to a moving-range-based sigma estimate.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd

from spc.core.limits import (
    _MAX_EXPANDING_WINDOW,
    AttributeLimitStats,
    binomial_approximation_low_count,
    c_chart_limits,
    compute_point_flags,
    laney_sigma_z,
    np_chart_limits,
    p_chart_limits,
    p_prime_chart_limits,
    u_chart_limits,
    u_prime_chart_limits,
)
from spc.core.models import BaseChartOptions, LimitMode
from spc.core.segments import (
    NO_FACET_SENTINEL,
    BaselineLookup,
    SegmentArrays,
    _build_baseline_lookup,
    _expected_value_series,
    order_columns,
)

# p_prime/u_prime (Laney, 2002) are overdispersion-corrected siblings of
# p/u -- same pooled centre line and plotted value, wider limits when the
# data shows more variation than pure binomial/Poisson sampling noise
# predicts. Opt-in, never suggested by suggest_chart_type() -- classical
# p/u stay the default.
AttributeKind = Literal["p", "np", "c", "u", "p_prime", "u_prime"]
_LANEY_KINDS = ("p_prime", "u_prime")
_ATTRIBUTE_KINDS = ("p", "np", "c", "u", "p_prime", "u_prime")


def _check_kind(
        kind: str
) -> None:
    """Every kind-dispatch below names its kinds explicitly; an unexpected
    one must raise rather than fall through to whichever branch is last.
    """
    if kind not in _ATTRIBUTE_KINDS:
        raise ValueError(f"attribute-chart kind must be one of {list(_ATTRIBUTE_KINDS)}, got {kind!r}")


def _windowed_mean(
        y: pd.Series,
        options: BaseChartOptions
) -> pd.Series:
    """Per-point centre-line estimate for np/c (a plain mean of the window),
    dispatching on limit_mode -- same SEGMENT/FIXED_N/EXPANDING semantics as
    XmR, independently implemented here since np/c have no moving-range
    component to carry along the way XmR's dispatch does.
    """
    n = len(y)
    if options.limit_mode == LimitMode.SEGMENT:
        return pd.Series(y.mean(), index=y.index)

    fix_after_n_points = options.fix_after_n_points
    assert fix_after_n_points is not None

    if options.limit_mode == LimitMode.FIXED_N:
        return pd.Series(y.iloc[:fix_after_n_points].mean(), index=y.index)

    # EXPANDING
    n_lock = min(fix_after_n_points, n)
    if n_lock > _MAX_EXPANDING_WINDOW:
        raise ValueError(
            f"fix_after_n_points={fix_after_n_points} exceeds the expanding-mode safety cap of "
            f"{_MAX_EXPANDING_WINDOW} points -- use limit_mode='fixed_n' instead for large warm-up windows"
        )
    cum_mean = y.expanding().mean().to_numpy()
    out = np.full(n, np.nan)
    out[:n_lock] = cum_mean[:n_lock]
    if n > n_lock:
        out[n_lock:] = cum_mean[n_lock - 1]
    return pd.Series(out, index=y.index)


def _windowed_pooled_ratio(
        numerator: pd.Series,
        denominator: pd.Series,
        options: BaseChartOptions
) -> pd.Series:
    """Per-point centre-line estimate for p/u: p_bar = Sum(defects)/Sum(n),
    pooled -- not a mean of per-point ratios, which would under-weight
    high-n periods -- dispatching on limit_mode.
    """
    n = len(numerator)
    if options.limit_mode == LimitMode.SEGMENT:
        ratio = numerator.sum() / denominator.sum()
        return pd.Series(ratio, index=numerator.index)

    fix_after_n_points = options.fix_after_n_points
    assert fix_after_n_points is not None

    if options.limit_mode == LimitMode.FIXED_N:
        window_num = numerator.iloc[:fix_after_n_points]
        window_den = denominator.iloc[:fix_after_n_points]
        ratio = window_num.sum() / window_den.sum()
        return pd.Series(ratio, index=numerator.index)

    # EXPANDING
    n_lock = min(fix_after_n_points, n)
    if n_lock > _MAX_EXPANDING_WINDOW:
        raise ValueError(
            f"fix_after_n_points={fix_after_n_points} exceeds the expanding-mode safety cap of "
            f"{_MAX_EXPANDING_WINDOW} points -- use limit_mode='fixed_n' instead for large warm-up windows"
        )
    cum_num = numerator.cumsum().to_numpy()
    cum_den = denominator.cumsum().to_numpy()
    ratio_arr = np.full(n, np.nan)
    ratio_arr[:n_lock] = cum_num[:n_lock] / cum_den[:n_lock]
    if n > n_lock:
        ratio_arr[n_lock:] = cum_num[n_lock - 1] / cum_den[n_lock - 1]
    return pd.Series(ratio_arr, index=numerator.index)


def _centre_line_for(
        kind: AttributeKind,
        raw: pd.Series,
        denom: pd.Series | None,
        seg: pd.DataFrame,
        options: BaseChartOptions,
        baseline_lookup: BaselineLookup
) -> pd.Series:
    """Per-point centre line: the supplied baseline directly where one
    exists (replacing pooling/windowing entirely -- see module docstring),
    otherwise the existing pooled-ratio (p/u family) or windowed-mean
    (np/c) estimate.
    """
    _check_kind(kind)
    if options.baseline:
        return _expected_value_series(seg, options, baseline_lookup)
    if kind in ("p", "u") + _LANEY_KINDS:
        assert denom is not None
        return _windowed_pooled_ratio(raw, denom, options)
    # np, c: windowed mean of the raw counts
    return _windowed_mean(raw, options)


def _stats_for(
        kind: AttributeKind,
        centre: pd.Series,
        denom: pd.Series | None,
        *,
        sigma_z: float | None = None
) -> AttributeLimitStats:
    """Dispatches to the correct control-limit formula for this chart kind."""
    _check_kind(kind)
    centre_arr = centre.to_numpy()
    if kind == "p":
        assert denom is not None
        return p_chart_limits(centre_arr, denom.to_numpy())
    if kind == "u":
        assert denom is not None
        return u_chart_limits(centre_arr, denom.to_numpy())
    if kind == "p_prime":
        assert denom is not None and sigma_z is not None
        return p_prime_chart_limits(centre_arr, denom.to_numpy(), sigma_z)
    if kind == "u_prime":
        assert denom is not None and sigma_z is not None
        return u_prime_chart_limits(centre_arr, denom.to_numpy(), sigma_z)
    if kind == "np":
        assert denom is not None
        return np_chart_limits(centre_arr, denom.to_numpy())
    # the only kind left after _check_kind and the branches above
    return c_chart_limits(centre_arr)


def build_attribute_segment_arrays(
        df_with_segments: pd.DataFrame,
        options: BaseChartOptions,
        kind: AttributeKind
) -> list[SegmentArrays]:
    """Runs the p/np/c/u calculation independently within each (facet,
    segment_id) group. Mirrors core/segments.py's build_segment_arrays()
    in shape and contract, but with attribute-chart formulas in place of
    XmR's moving-range one.
    """
    _check_kind(kind)
    denominator_field = getattr(options, "denominator_field", None)
    baseline_lookup = _build_baseline_lookup(options)

    results: list[SegmentArrays] = []
    for segment_id, seg in df_with_segments.groupby("segment_id", sort=True):
        seg = seg.sort_values(order_columns(options), kind="stable").reset_index(drop=True)
        raw = seg[options.value_field].astype(float)

        if kind in ("p", "u") + _LANEY_KINDS:
            assert denominator_field is not None  # enforced by ProportionChartOptions
            denom = seg[denominator_field].astype(float)
            y = raw / denom  # plotted value is the per-point proportion/rate, not the raw count
        elif kind == "np":
            assert denominator_field is not None
            denom = seg[denominator_field].astype(float)
            if denom.nunique() != 1:
                raise ValueError(
                    f"kind='np' requires a constant denominator within each segment; "
                    f"'{denominator_field}' varies in segment {segment_id} -- use kind='p' instead"
                )
            y = raw  # plotted value is the raw count -- n is constant, no per-point normalisation needed
        elif kind == "c":
            denom = None
            y = raw
        else:  # unreachable: _check_kind ran at the top of this function
            raise ValueError(f"unhandled attribute-chart kind {kind!r}")

        centre_line = _centre_line_for(kind, raw, denom, seg, options, baseline_lookup)

        if kind in ("p", "p_prime"):
            assert denom is not None
            low_count_warning = pd.Series(
                binomial_approximation_low_count(centre_line.to_numpy(), denom.to_numpy()), index=seg.index
            )
        elif kind == "np":
            assert denom is not None
            # centre_line here is np_bar (mean count), not a proportion --
            # binomial_approximation_low_count needs p_bar, so divide by
            # the (constant, already-validated) denominator first.
            p_bar = centre_line / denom
            low_count_warning = pd.Series(
                binomial_approximation_low_count(p_bar.to_numpy(), denom.to_numpy()), index=seg.index
            )
        elif kind in ("c", "u", "u_prime"):  # Poisson-based, not the binomial approximation this diagnostic checks
            low_count_warning = pd.Series(False, index=seg.index)
        else:  # unreachable: _check_kind ran at the top of this function
            raise ValueError(f"unhandled attribute-chart kind {kind!r}")

        sigma_z = None
        if kind in _LANEY_KINDS:
            assert denom is not None
            centre_arr, denom_arr, y_arr = centre_line.to_numpy(), denom.to_numpy(), y.to_numpy()
            se = (
                np.sqrt(centre_arr * (1 - centre_arr) / denom_arr)
                if kind == "p_prime"
                else np.sqrt(centre_arr / denom_arr)
            )
            # One sigma_z per segment, from the whole segment's Z-series --
            # not re-estimated per limit_mode window the way centre_line
            # is. Laney's own method treats it as a single dispersion
            # estimate characterising the segment/process, not something
            # that should itself vary point-to-point the way a Phase I/II
            # centre line legitimately needs to.
            sigma_z = laney_sigma_z(y_arr, centre_arr, se)

        stats = _stats_for(kind, centre_line, denom, sigma_z=sigma_z)
        ucl = pd.Series(stats.ucl, index=seg.index)
        lcl = pd.Series(stats.lcl, index=seg.index)
        near_ucl = pd.Series(stats.near_ucl, index=seg.index)
        near_lcl = pd.Series(stats.near_lcl, index=seg.index)

        outside_limits, relative_to_mean, close_to_limits = compute_point_flags(
            y, centre_line, ucl, lcl, near_ucl, near_lcl
        )

        facet_value = seg["facet"].iloc[0]
        results.append(
            SegmentArrays(
                facet=None if facet_value == NO_FACET_SENTINEL else facet_value,
                # See core/segments.py's build_segment_arrays() for why this cast is safe.
                segment_id=int(cast(int, segment_id)),
                x=seg[options.date_field],
                y=y,
                centre_line=centre_line,
                ucl=ucl,
                lcl=lcl,
                amr=pd.Series(np.nan, index=seg.index),  # not applicable to attribute charts
                outside_limits=outside_limits,
                relative_to_mean=relative_to_mean,
                close_to_limits=close_to_limits,
                low_count_warning=low_count_warning,
                autocorrelation=pd.Series(np.nan, index=seg.index),  # not applicable to attribute charts
            )
        )
    return results

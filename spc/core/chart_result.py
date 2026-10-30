"""Shared, chart-kind-agnostic ChartResult assembly: SegmentArrays ->
PointResult/SegmentSummary, and the per-facet assurance-type computation.

Only reads SegmentArrays' generic fields (x, y, centre_line, ucl, lcl,
outside_limits) plus the rule engine's output, all of which every chart
kind (XmR, p/np/c/u, ...) produces in the same shape -- so every chart
kind shares this single assembly path rather than duplicating it.
"""

from __future__ import annotations

import math

from spc.core.facets import split_facet
from spc.core.limits import HIGH_AUTOCORRELATION_THRESHOLD, assurance_type
from spc.core.models import AssuranceType, BaseChartOptions, PointResult, SegmentSummary
from spc.core.rules import evaluate_rules, point_type
from spc.core.segments import SegmentArrays


def _or_none(
        x: float
) -> float | None:
    """NaN -> None at the ChartResult boundary -- see PointResult.ucl's
    docstring in core/models.py for why some segments genuinely have no
    defined limits (not just XmR's < 2-point case: an attribute chart's
    np/c formula is undefined for np_bar/c_bar producing a negative
    variance term, for instance).
    """
    return None if math.isnan(x) else float(x)


def _resolve_per_facet(
        value: float | dict[str, float] | None,
        facet: str | None
) -> float | None:
    """Scalar-or-per-facet-dict resolution, the shape shared by `target`
    and `spec_upper`/`spec_lower` alike.
    """
    if isinstance(value, dict):
        return value.get(facet) if facet is not None else None
    return value


def _is_spec_breach(
        y: float,
        spec_upper: float | None,
        spec_lower: float | None
) -> bool:
    """Whether y falls outside an externally-imposed requirement --
    entirely independent of outside_limits/point_type. See
    PointResult.spec_breach's docstring for why these are deliberately
    never merged.
    """
    return bool((spec_upper is not None and y > spec_upper) or (spec_lower is not None and y < spec_lower))


def _high_autocorrelation_warning(
        rho: float,
        autocorrelation_adjustment: bool
) -> bool:
    """True when the adjustment is on and the segment's own (unclamped)
    rho estimate is high enough that the correction is least reliable --
    see limits.py::HIGH_AUTOCORRELATION_THRESHOLD. Never True with the
    adjustment off, regardless of what rho would have been.
    """
    return bool(autocorrelation_adjustment and not math.isnan(rho) and rho >= HIGH_AUTOCORRELATION_THRESHOLD)


def segment_to_results(
        seg: SegmentArrays,
        options: BaseChartOptions
) -> tuple[list[PointResult], SegmentSummary]:
    """Runs the rule engine and assembles one PointResult per point plus a
    SegmentSummary for the segment as a whole.
    """
    winners, labels = evaluate_rules(seg, options.rules, run_rule_length=options.run_rule_length)
    n = len(seg.y)
    spec_upper = _resolve_per_facet(options.spec_upper, seg.facet)
    spec_lower = _resolve_per_facet(options.spec_lower, seg.facet)
    facet_parts = split_facet(seg.facet, options)

    points = [
        PointResult(
            x=seg.x.iloc[i],
            y=float(seg.y.iloc[i]),
            facet=seg.facet,
            facet_parts=facet_parts,
            segment_id=seg.segment_id,
            mean=float(seg.centre_line.iloc[i]),
            ucl=_or_none(seg.ucl.iloc[i]),
            lcl=_or_none(seg.lcl.iloc[i]),
            outside_limits=bool(seg.outside_limits.iloc[i]),
            special_cause_rule=winners[i],
            point_type=point_type(winners[i], labels[i], options.improvement_direction),
            low_count_warning=bool(seg.low_count_warning.iloc[i]),
            spec_breach=_is_spec_breach(float(seg.y.iloc[i]), spec_upper, spec_lower),
        )
        for i in range(n)
    ]

    summary = SegmentSummary(
        segment_id=seg.segment_id,
        facet=seg.facet,
        facet_parts=facet_parts,
        n=n,
        mean=float(seg.centre_line.iloc[-1]),
        ucl=_or_none(seg.ucl.iloc[-1]),
        lcl=_or_none(seg.lcl.iloc[-1]),
        short_segment_warning=n < options.short_segment_threshold,
        autocorrelation_estimate=_or_none(seg.autocorrelation.iloc[-1]),
        high_autocorrelation_warning=_high_autocorrelation_warning(
            seg.autocorrelation.iloc[-1], options.autocorrelation_adjustment
        ),
    )
    return points, summary


def compute_assurance(
        summaries: list[SegmentSummary],
        options: BaseChartOptions
) -> AssuranceType | dict[str, AssuranceType] | None:
    """Each facet's own *latest* segment against its own target, grouped
    by facet natively. A single value when there's one facet (or none,
    i.e. facet_field unset); a dict keyed by facet name when there's more
    than one.
    """
    if options.target is None or options.improvement_direction == "neutral" or not summaries:
        return None

    latest_by_facet: dict[str | None, SegmentSummary] = {}
    for s in summaries:
        current = latest_by_facet.get(s.facet)
        if current is None or s.segment_id > current.segment_id:
            latest_by_facet[s.facet] = s

    def _target_for(
            facet: str | None
    ) -> float | None:
        if isinstance(options.target, dict):
            return options.target.get(facet) if facet is not None else None
        return options.target

    per_facet: dict[str | None, AssuranceType] = {}
    for facet, latest in latest_by_facet.items():
        if latest.ucl is None or latest.lcl is None:
            continue  # latest segment's limits are undefined -- nothing to assess against
        target = _target_for(facet)
        if target is None:
            continue
        value = assurance_type(target, latest.ucl, latest.lcl, options.improvement_direction)
        assert value is not None  # guaranteed: improvement_direction != "neutral", checked above
        per_facet[facet] = value

    if not per_facet:
        return None
    if len(latest_by_facet) == 1:
        return next(iter(per_facet.values()))
    # More than one facet means every key here is a real facet name -- None
    # only ever appears as the sole key when facet_field wasn't set at all,
    # which is the len == 1 branch above.
    return {str(facet): value for facet, value in per_facet.items()}

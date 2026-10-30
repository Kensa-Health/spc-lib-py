"""U' chart (Laney overdispersion-corrected rate) -- an opt-in sibling of
`u`, never suggested automatically by suggest_chart_type(). See
core/limits.py's laney_sigma_z()/u_prime_chart_limits() for the method.
"""

from __future__ import annotations

import pandas as pd

from spc.core.attribute_segments import build_attribute_segment_arrays
from spc.core.chart_result import compute_assurance, segment_to_results
from spc.core.models import BaseChartOptions, ChartResult, PointResult, ProportionChartOptions, SegmentSummary
from spc.core.segments import assign_segments
from spc.core.validation import suggest_chart_type


class UPrimeChartCalculator:
    """Implements the ChartCalculator protocol for the U' chart."""

    kind = "u_prime"

    def fit(
            self,
            df: pd.DataFrame,
            options: BaseChartOptions
    ) -> ChartResult:
        """Computes a U' chart from one row per period."""
        if not isinstance(options, ProportionChartOptions):
            raise TypeError(
                "kind='u_prime' requires ProportionChartOptions (needs denominator_field, used as opportunity)"
            )

        segmented = assign_segments(df, options)
        segment_arrays = build_attribute_segment_arrays(segmented, options, kind="u_prime")

        points: list[PointResult] = []
        summaries: list[SegmentSummary] = []
        for seg in segment_arrays:
            seg_points, seg_summary = segment_to_results(seg, options)
            points.extend(seg_points)
            summaries.append(seg_summary)

        suggestion = suggest_chart_type(
            df, options.value_field, denominator_field=options.denominator_field, date_field=options.date_field
        )
        # See PPrimeChartCalculator's identical comment -- suggest_chart_type()
        # has no concept of u_prime, so only surface a suggestion pointing
        # outside the u/u_prime family entirely.
        suggested_kind = suggestion.kind if suggestion.kind not in ("u", self.kind) else None

        return ChartResult(
            kind=self.kind,
            options=options,
            points=points,
            segments=summaries,
            assurance_type=compute_assurance(summaries, options),
            suggested_kind=suggested_kind,
        )

"""P chart (proportion, varying denominator)."""

from __future__ import annotations

import pandas as pd

from spc.core.attribute_segments import build_attribute_segment_arrays
from spc.core.chart_result import compute_assurance, segment_to_results
from spc.core.models import BaseChartOptions, ChartResult, PointResult, ProportionChartOptions, SegmentSummary
from spc.core.segments import assign_segments
from spc.core.validation import suggest_chart_type


class PChartCalculator:
    """Implements the ChartCalculator protocol for the P chart."""

    kind = "p"

    def fit(
            self,
            df: pd.DataFrame,
            options: BaseChartOptions
    ) -> ChartResult:
        """Computes a P chart from one row per period."""
        if not isinstance(options, ProportionChartOptions):
            raise TypeError("kind='p' requires ProportionChartOptions (needs denominator_field)")

        segmented = assign_segments(df, options)
        segment_arrays = build_attribute_segment_arrays(segmented, options, kind="p")

        points: list[PointResult] = []
        summaries: list[SegmentSummary] = []
        for seg in segment_arrays:
            seg_points, seg_summary = segment_to_results(seg, options)
            points.extend(seg_points)
            summaries.append(seg_summary)

        suggestion = suggest_chart_type(
            df, options.value_field, denominator_field=options.denominator_field, date_field=options.date_field
        )
        suggested_kind = suggestion.kind if suggestion.kind != self.kind else None

        return ChartResult(
            kind=self.kind,
            options=options,
            points=points,
            segments=summaries,
            assurance_type=compute_assurance(summaries, options),
            suggested_kind=suggested_kind,
        )

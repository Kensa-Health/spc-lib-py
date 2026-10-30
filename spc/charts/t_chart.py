"""T chart (time elapsed between individual rare events). See
core/rare_event_segments.py's module docstring for the Nelson transform
and the value_field/date_field scoping decisions.
"""

from __future__ import annotations

import pandas as pd

from spc.core.chart_result import compute_assurance, segment_to_results
from spc.core.models import BaseChartOptions, ChartResult, PointResult, SegmentSummary
from spc.core.rare_event_segments import build_rare_event_segment_arrays
from spc.core.validation import suggest_chart_type


class TChartCalculator:
    """Implements the ChartCalculator protocol for the T chart."""

    kind = "t"

    def fit(
            self,
            df: pd.DataFrame,
            options: BaseChartOptions
    ) -> ChartResult:
        """Computes a T chart from one row per rare event."""
        segment_arrays = build_rare_event_segment_arrays(df, options, kind="t")

        points: list[PointResult] = []
        summaries: list[SegmentSummary] = []
        for seg in segment_arrays:
            seg_points, seg_summary = segment_to_results(seg, options)
            points.extend(seg_points)
            summaries.append(seg_summary)

        suggestion = suggest_chart_type(df, options.value_field)
        suggested_kind = suggestion.kind if suggestion.kind != self.kind else None

        return ChartResult(
            kind=self.kind,
            options=options,
            points=points,
            segments=summaries,
            assurance_type=compute_assurance(summaries, options),
            suggested_kind=suggested_kind,
        )

"""G chart (opportunities/cases between individual rare events, the
discrete analogue of T). See core/rare_event_segments.py's module
docstring for the Nelson transform and the
value_field/date_field scoping decisions.
"""

from __future__ import annotations

import pandas as pd

from spc.core.chart_result import compute_assurance, segment_to_results
from spc.core.models import BaseChartOptions, ChartResult, PointResult, SegmentSummary
from spc.core.rare_event_segments import build_rare_event_segment_arrays


class GChartCalculator:
    """Implements the ChartCalculator protocol for the G chart."""

    kind = "g"

    def fit(
            self,
            df: pd.DataFrame,
            options: BaseChartOptions
    ) -> ChartResult:
        """Computes a G chart from one row per rare event."""
        segment_arrays = build_rare_event_segment_arrays(df, options, kind="g")

        points: list[PointResult] = []
        summaries: list[SegmentSummary] = []
        for seg in segment_arrays:
            seg_points, seg_summary = segment_to_results(seg, options)
            points.extend(seg_points)
            summaries.append(seg_summary)

        # No suggested_kind: suggest_chart_type() never proposes 'g' (integer
        # counts between events look like c/u counts), so any suggestion it
        # made here would just be noise against a deliberately chosen kind.
        return ChartResult(
            kind=self.kind,
            options=options,
            points=points,
            segments=summaries,
            assurance_type=compute_assurance(summaries, options),
        )

"""Xbar-S chart (subgrouped continuous data, subgroup size > 10). Means
chart only -- see core/subgroup_segments.py's module docstring for scope.
"""

from __future__ import annotations

import pandas as pd

from spc.core.chart_result import compute_assurance, segment_to_results
from spc.core.models import BaseChartOptions, ChartResult, PointResult, SegmentSummary
from spc.core.segments import assign_segments
from spc.core.subgroup_segments import aggregate_subgroups, build_subgroup_segment_arrays
from spc.core.validation import suggest_chart_type


class XbarSCalculator:
    """Implements the ChartCalculator protocol for the Xbar-S chart."""

    kind = "xbar_s"

    def fit(
            self,
            df: pd.DataFrame,
            options: BaseChartOptions
    ) -> ChartResult:
        """Computes an Xbar-S chart from one row per raw measurement,
        aggregated into (facet, date) subgroups.
        """
        aggregated = aggregate_subgroups(df, options, spread="std")
        segmented = assign_segments(aggregated, options)
        segment_arrays = build_subgroup_segment_arrays(segmented, options, kind="xbar_s")

        points: list[PointResult] = []
        summaries: list[SegmentSummary] = []
        for seg in segment_arrays:
            seg_points, seg_summary = segment_to_results(seg, options)
            points.extend(seg_points)
            summaries.append(seg_summary)

        suggestion = suggest_chart_type(df, options.value_field, date_field=options.date_field)
        suggested_kind = suggestion.kind if suggestion.kind != self.kind else None

        return ChartResult(
            kind=self.kind,
            options=options,
            points=points,
            segments=summaries,
            assurance_type=compute_assurance(summaries, options),
            suggested_kind=suggested_kind,
        )

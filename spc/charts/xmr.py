"""XmR (individuals & moving range) chart calculator."""

from __future__ import annotations

import pandas as pd

from spc.core.chart_result import compute_assurance, segment_to_results
from spc.core.models import BaseChartOptions, ChartResult, PointResult, SegmentSummary
from spc.core.segments import assign_segments, build_segment_arrays
from spc.core.validation import suggest_chart_type


class XmRCalculator:
    """Implements the ChartCalculator protocol for the XmR chart."""

    kind = "xmr"

    def fit(
            self,
            df: pd.DataFrame,
            options: BaseChartOptions
    ) -> ChartResult:
        """Computes an XmR chart from one row per period."""
        segmented = assign_segments(df, options)
        segment_arrays = build_segment_arrays(segmented, options)

        points: list[PointResult] = []
        summaries: list[SegmentSummary] = []
        for seg in segment_arrays:
            seg_points, seg_summary = segment_to_results(seg, options)
            points.extend(seg_points)
            summaries.append(seg_summary)

        suggestion = suggest_chart_type(df, options.value_field, date_field=options.date_field)
        suggested_kind = suggestion.kind if suggestion.kind != self.kind else None

        # Non-fatal diagnostic, not a validation error or a silent rules
        # override -- see BaseChartOptions.autocorrelation_adjustment's
        # docstring. Names the run-length rules still calibrated assuming
        # independence even though ucl/lcl are now correctly widened;
        # never mutates options.rules itself.
        autocorrelation_uncorrected_rules = (
            [r for r in options.rules if r != "outside_limits"] if options.autocorrelation_adjustment else []
        )

        return ChartResult(
            kind=self.kind,
            options=options,
            points=points,
            segments=summaries,
            assurance_type=compute_assurance(summaries, options),
            suggested_kind=suggested_kind,
            autocorrelation_uncorrected_rules=autocorrelation_uncorrected_rules,
        )

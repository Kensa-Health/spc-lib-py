"""spc -- SPC (statistical process control) calculation library.

Owns calculation, validation, and data contracts only -- no UI opinions.
Implements individuals/moving-range (xmr), attribute (p, p_prime, u,
u_prime, np, c), subgrouped (xbar_r, xbar_s), and rare-event (t, g) chart
kinds, plus a Bernoulli CUSUM (cusum) and a Pareto analysis (pareto). See
the README's "Design choices and known limitations" for scope limits, and the
module docstrings of core/segments.py, core/attribute_segments.py,
core/subgroup_segments.py and core/rare_event_segments.py for kind-specific notes.

    import spc

    result = spc.chart(df, kind="xmr", options=spc.BaseChartOptions(
        value_field="breaches", date_field="period",
    ))
    result.model_dump_json()
    spc.render.plotly(result, title="Breaches")   # optional; needs the 'render' extra

    p_result = spc.chart(df, kind="p", options=spc.ProportionChartOptions(
        value_field="breaches", date_field="period", denominator_field="attendances",
    ))
"""

from __future__ import annotations

from typing import Any

from spc.api import chart
from spc.core.models import (
    BaseChartOptions,
    ChartResult,
    ChartSuggestion,
    ExternalBaseline,
    LimitMode,
    PointResult,
    ProportionChartOptions,
    RebaseSuggestion,
    SegmentSummary,
)
from spc.core.rebase import suggest_rebase
from spc.core.validation import suggest_chart_type
from spc.cusum import (
    CusumOptions,
    CusumPoint,
    CusumResult,
    CusumSeries,
    cusum,
    cusum_arl,
    cusum_threshold,
    estimate_event_rate,
)
from spc.pareto import ParetoOptions, ParetoResult, ParetoRow, pareto

__version__ = "0.1.0"

__all__ = [
    "chart",
    "suggest_chart_type",
    "suggest_rebase",
    "cusum",
    "cusum_arl",
    "cusum_threshold",
    "estimate_event_rate",
    "CusumOptions",
    "CusumPoint",
    "CusumSeries",
    "CusumResult",
    "pareto",
    "ParetoOptions",
    "ParetoResult",
    "ParetoRow",
    "BaseChartOptions",
    "ProportionChartOptions",
    "LimitMode",
    "ExternalBaseline",
    "PointResult",
    "SegmentSummary",
    "ChartResult",
    "ChartSuggestion",
    "RebaseSuggestion",
    "__version__",
]


def __getattr__(
        name: str
) -> Any:
    """Lazily imports spc.render, so `import spc` never requires the
    optional 'render' extra (plotly) to be installed -- only actually
    touching spc.render does. Uses importlib directly rather than
    `from spc import render`, which would call getattr(spc, "render")
    again mid-resolution and recurse.
    """
    if name == "render":
        import importlib

        module = importlib.import_module("spc.render")
        globals()["render"] = module  # cache so subsequent access skips __getattr__ entirely
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

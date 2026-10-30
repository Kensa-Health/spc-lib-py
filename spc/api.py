"""Public entry point."""

from __future__ import annotations

import pandas as pd

from spc.charts import ChartCalculator
from spc.charts.c_chart import CChartCalculator
from spc.charts.g_chart import GChartCalculator
from spc.charts.np_chart import NpChartCalculator
from spc.charts.p_chart import PChartCalculator
from spc.charts.p_prime_chart import PPrimeChartCalculator
from spc.charts.t_chart import TChartCalculator
from spc.charts.u_chart import UChartCalculator
from spc.charts.u_prime_chart import UPrimeChartCalculator
from spc.charts.xbar_r import XbarRCalculator
from spc.charts.xbar_s import XbarSCalculator
from spc.charts.xmr import XmRCalculator
from spc.core.models import BaseChartOptions, ChartResult
from spc.core.validation import suggest_chart_type, validate_columns_and_shape, validate_values

_CALCULATORS: dict[str, ChartCalculator] = {
    "xmr": XmRCalculator(),
    "p": PChartCalculator(),
    "p_prime": PPrimeChartCalculator(),
    "u": UChartCalculator(),
    "u_prime": UPrimeChartCalculator(),
    "np": NpChartCalculator(),
    "c": CChartCalculator(),
    "xbar_r": XbarRCalculator(),
    "xbar_s": XbarSCalculator(),
    "t": TChartCalculator(),
    "g": GChartCalculator(),
}


def chart(
        df: pd.DataFrame,
        kind: str,
        options: BaseChartOptions
) -> ChartResult:
    """Calculate an SPC chart.

    `kind="auto"` defers to suggest_chart_type(); if that suggests a kind
    this library doesn't implement, this raises rather than silently
    falling back to xmr, since that fallback would silently produce a
    statistically-wrong chart for data suggest_chart_type flagged as a
    poor fit for XmR in the first place.

    Validates the input at this single choke point rather than inside
    each chart kind's own fit() -- every kind gets these checks, including
    any added later, without needing to remember to call them.
    Column-existence/emptiness (kind-independent) run before kind="auto"
    is resolved, since suggest_chart_type() itself indexes df by these
    columns and would otherwise raise its own unrelated KeyError first;
    value/denominator checks (kind-dependent) run after, once a real
    calculator is confirmed to exist for the resolved kind.
    """
    validate_columns_and_shape(df, options)

    if kind == "auto":
        suggestion = suggest_chart_type(
            df,
            options.value_field,
            denominator_field=getattr(options, "denominator_field", None),
            date_field=options.date_field,
        )
        kind = suggestion.kind

    calculator = _CALCULATORS.get(kind)
    if calculator is None:
        raise ValueError(f"unknown chart kind: '{kind}' -- available: {sorted(_CALCULATORS)}")

    validate_values(df, options, kind)

    return calculator.fit(df, options)

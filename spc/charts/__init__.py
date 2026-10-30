"""Chart-kind calculators, each implementing ChartCalculator:
`fit(df, options) -> ChartResult`.

Implemented kinds: xmr, p, p_prime, u, u_prime, np, c, xbar_r, xbar_s, t, g.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from spc.core.models import BaseChartOptions, ChartResult


class ChartCalculator(Protocol):
    """Every chart kind implements this -- api.py dispatches on `kind`
    without ever needing chart-specific branching beyond that.
    """

    kind: str

    def fit(
            self,
            df: pd.DataFrame,
            options: BaseChartOptions
    ) -> ChartResult:
        """Computes this chart kind's control limits and point classifications."""
        ...

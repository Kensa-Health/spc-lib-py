"""Pareto analysis: categories ranked by frequency (or weight) with a
cumulative share, for finding the few causes behind most of a problem.

Adjacent to SPC rather than a control chart -- there is no time axis, no
control limits and no rule engine -- so it has its own options and result
types instead of `ChartResult`. Calculation only; see
`spc.render.plotly_pareto` for the optional chart.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ParetoOptions(BaseModel):
    """Inputs for `pareto()`.

    `count_field`: a numeric weight per row (defects, cost, minutes); when
    omitted, each row counts as 1. `max_categories`: keep only the largest
    N categories and collapse the rest into one `other_label` row, always
    placed last whatever its size.
    """

    model_config = ConfigDict(extra="forbid")  # a misspelled option must raise, not be ignored

    category_field: str
    count_field: str | None = None
    max_categories: int | None = None
    other_label: str = "Other"

    @model_validator(mode="after")
    def check_max_categories(self) -> ParetoOptions:
        """max_categories, if given, must be at least 1."""
        if self.max_categories is not None and self.max_categories < 1:
            raise ValueError("max_categories must be >= 1")
        return self


class ParetoRow(BaseModel):
    """One ranked category."""

    rank: int
    category: str
    count: float
    share: float  # this category's fraction of the total, 0..1
    cumulative_share: float  # running fraction through this row, 0..1
    is_other: bool = False


class ParetoResult(BaseModel):
    """Output of `pareto()`: ranked rows and the grand total."""

    schema_version: str = "1.0"
    options: ParetoOptions
    rows: list[ParetoRow]
    total: float
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def pareto(
        df: pd.DataFrame,
        options: ParetoOptions
) -> ParetoResult:
    """Ranks categories by total count/weight, descending (ties broken by
    category name, ascending, so the order is deterministic).

    Raises ValueError on missing columns, missing category values, a
    non-numeric/negative/NaN weight, an empty frame, or a zero total (the
    shares would be undefined).
    """
    missing = [c for c in (options.category_field, options.count_field) if c is not None and c not in df.columns]
    if missing:
        raise ValueError(
            f"column(s) not found in the input dataframe: {missing}. Available columns: {list(df.columns)}"
        )
    if len(df) == 0:
        raise ValueError("input dataframe has no rows -- nothing to rank")

    categories = df[options.category_field]
    if categories.isna().any():
        raise ValueError(
            f"'{options.category_field}' has {int(categories.isna().sum())} missing value(s) -- "
            "fill or drop them first (a missing category can't be ranked)"
        )

    if options.count_field is None:
        weights = pd.Series(1.0, index=df.index)
    else:
        raw = df[options.count_field]
        weights = pd.to_numeric(raw, errors="coerce")
        if weights.isna().any():
            raise ValueError(f"'{options.count_field}' must be numeric with no missing values")
        if not np.isfinite(weights.to_numpy(dtype=float)).all():
            raise ValueError(f"'{options.count_field}' must be finite (no infinite values)")
        if (weights < 0).any():
            raise ValueError(f"'{options.count_field}' must be non-negative")
        weights = weights.astype(float)

    totals = weights.groupby(categories.astype(str)).sum()
    # largest first; ties by category name so the ranking is reproducible
    ranked = totals.reset_index()
    ranked.columns = ["category", "count"]
    ranked = ranked.sort_values(["count", "category"], ascending=[False, True], kind="stable").reset_index(drop=True)

    total = float(ranked["count"].sum())
    if total == 0:
        raise ValueError("total count is zero -- shares are undefined")

    is_other = [False] * len(ranked)
    if options.max_categories is not None and len(ranked) > options.max_categories:
        if options.other_label in set(ranked["category"]):
            raise ValueError(
                f"a category is already named {options.other_label!r}, which would clash with the collapsed "
                "bucket -- set other_label to something else"
            )
        head = ranked.iloc[: options.max_categories]
        tail_total = float(ranked.iloc[options.max_categories :]["count"].sum())
        ranked = pd.concat(
            [head, pd.DataFrame({"category": [options.other_label], "count": [tail_total]})], ignore_index=True
        )
        is_other = [False] * options.max_categories + [True]

    counts = ranked["count"].to_numpy(dtype=float)
    cumulative = np.cumsum(counts) / total
    cumulative[-1] = 1.0  # exact, not 0.9999999999999999
    rows = [
        ParetoRow(
            rank=i + 1,
            category=str(ranked["category"].iloc[i]),
            count=float(counts[i]),
            share=float(counts[i] / total),
            cumulative_share=float(cumulative[i]),
            is_other=is_other[i],
        )
        for i in range(len(ranked))
    ]
    return ParetoResult(options=options, rows=rows, total=total)

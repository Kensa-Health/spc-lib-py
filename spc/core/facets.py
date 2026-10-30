"""Facet keys. A chart may be split by one variable (`facet_field`) or two
(`facet_field` as the row variable plus `column_facet_field`). Calculation
treats every facet as one string key; for two variables that key is the
two values joined by `FACET_SEPARATOR`, and the same combined string keys
the per-facet options (`target`, `rebase`, `baseline`, `spec_upper`/
`spec_lower`). `PointResult.facet_parts` recovers the pieces.
"""

from __future__ import annotations

import pandas as pd

from spc.core.models import BaseChartOptions

FACET_SEPARATOR = " | "


def facet_columns(
        options: BaseChartOptions
) -> list[str]:
    """The dataframe columns that define a facet: none, one, or two."""
    cols = [options.facet_field] if options.facet_field else []
    if options.column_facet_field:
        cols.append(options.column_facet_field)
    return cols


def raise_if_facet_missing(
        df: pd.DataFrame,
        fields: list[str]
) -> None:
    """Raises if any facet column has missing values: a row without a facet
    can't be assigned to a chart, and grouping would drop it silently.
    """
    for field in fields:
        if df[field].isna().any():
            raise ValueError(
                f"'{field}' has {int(df[field].isna().sum())} missing value(s) -- a row without a facet "
                "can't be assigned to a chart; fill or drop them first"
            )


def facet_key_series(
        df: pd.DataFrame,
        facet_field: str | None,
        column_facet_field: str | None
) -> pd.Series | None:
    """One facet key per row of `df`, or None when no facet is configured.

    Keys are text. With two facet variables the key is `"<row><SEP><column>"`. A value
    containing the separator would make the key ambiguous to split back
    apart, so it raises instead.
    """
    if not facet_field:
        return None
    raise_if_facet_missing(df, [f for f in (facet_field, column_facet_field) if f])
    row_text = df[facet_field].astype(str)  # facet keys are always text, whatever the column's dtype
    if not column_facet_field:
        return row_text
    col_text = df[column_facet_field].astype(str)
    for field, text in ((facet_field, row_text), (column_facet_field, col_text)):
        if text.str.contains(FACET_SEPARATOR, regex=False).any():
            raise ValueError(
                f"'{field}' has values containing {FACET_SEPARATOR!r}, which is reserved for joining "
                "the row and column facet values -- rename those values"
            )
    return row_text + FACET_SEPARATOR + col_text


def facet_series(
        df: pd.DataFrame,
        options: BaseChartOptions
) -> pd.Series | None:
    """`facet_key_series` for a chart's options."""
    return facet_key_series(df, options.facet_field, options.column_facet_field)


def split_facet(
        facet: str | None,
        options: BaseChartOptions
) -> list[str] | None:
    """[row, column] for a two-variable facet key, else None."""
    if facet is None or not options.column_facet_field:
        return None
    return facet.split(FACET_SEPARATOR)

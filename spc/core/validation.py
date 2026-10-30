"""Data-shape validation and suggest_chart_type().

A guardrail, not a gate: this never blocks a user's chosen chart kind, it
only proposes one with a reason, so a consuming app can show a "did you
mean...?" prompt (`ChartResult.suggested_kind`).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from spc.core.facets import facet_columns, facet_series, raise_if_facet_missing
from spc.core.models import BaseChartOptions, ChartSuggestion

_XBAR_S_THRESHOLD = 10  # subgroup size above which xbar_s replaces xbar_r

# Kinds where the numerator can never legitimately exceed its denominator
# ("defects out of a sample") -- u/c have no such bound (a rate can
# exceed 1 per opportunity). p_prime is the same
# proportion data as p (Laney's correction only changes the limit width,
# not what's a physically valid count), so it's bound the same way.
_BOUNDED_NUMERATOR_KINDS = ("p", "np", "p_prime")
# Kinds whose primary field is a count (defects, occurrences, opportunities)
# rather than a continuous measurement -- a negative or fractional value is
# structurally meaningless, not just statistically unusual. xmr/xbar_r/xbar_s
# are excluded (a continuous measurement can legitimately be negative, e.g.
# a deviation), and t is excluded (its value_field holds event timestamps,
# not counts -- see core/rare_event_segments.py's module docstring).
_COUNT_FIELD_KINDS = ("p", "np", "c", "u", "g", "p_prime", "u_prime")


def _looks_like_count(
        series: pd.Series
) -> bool:
    """True if every value is a non-negative, whole number."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return False
    return bool((numeric >= 0).all() and np.allclose(numeric, np.round(numeric)))


def suggest_chart_type(
        df: pd.DataFrame,
        value_field: str,
        *,
        denominator_field: str | None = None,
        date_field: str | None = None
) -> ChartSuggestion:
    """Heuristics, in priority order:

      1. denominator_field supplied, value_field looks like a count, and
         every value_field <= its paired denominator_field (so it can be
         read as "defects out of a sample") -> 'p', or 'np' if the
         denominator is constant across all rows.
      2. value_field datetime-typed -> 't' (it holds the timestamps of
         individual rare events). Checked first, before any numeric
         heuristic, since a timestamp column can't be a count.
      3. denominator_field supplied but value_field is *not* bounded by it
         (a rate against a varying opportunity measure, not a proportion of
         a sample) -> 'u', or 'c' if the denominator is constant.
      4. date_field supplied and multiple rows share the same date
         (subgroups) -> 'xbar_r' (subgroup size <= 10) or 'xbar_s' (> 10).
      5. otherwise -> 'xmr'.

    'g' is never suggested: a column of integer counts between rare
    events is indistinguishable from the counts a c/u chart uses, so the
    data alone can't justify it. Choose kind='g' explicitly.

    Never a hard block -- always returns a suggestion with a confidence
    level and a one-line reason; the caller decides whether to use it.

    `date_field` is optional since not every caller needs heuristic 4
    (subgroup detection), which is the only one that requires it.
    """
    value = df[value_field]

    if pd.api.types.is_datetime64_any_dtype(value):
        return ChartSuggestion(
            kind="t", confidence="high", reason=f"'{value_field}' holds timestamps of individual rare events"
        )

    if denominator_field is not None and _looks_like_count(value):
        denominator = df[denominator_field]
        if (value <= denominator).all():
            if denominator.nunique() == 1:
                return ChartSuggestion(
                    kind="np",
                    confidence="high",
                    reason=f"'{value_field}' is a count bounded by a constant '{denominator_field}'",
                )
            return ChartSuggestion(
                kind="p",
                confidence="high",
                reason=f"'{value_field}' is a count bounded by a varying '{denominator_field}' (proportion)",
            )

    if denominator_field is not None and _looks_like_count(value):
        denominator = df[denominator_field]
        if denominator.nunique() == 1:
            return ChartSuggestion(
                kind="c",
                confidence="medium",
                reason=f"'{value_field}' is a count against a constant opportunity '{denominator_field}'",
            )
        return ChartSuggestion(
            kind="u",
            confidence="medium",
            reason=(
                f"'{value_field}' is a count against a varying opportunity '{denominator_field}' (rate, not bounded)"
            ),
        )

    if date_field is not None:
        counts = df.groupby(date_field)[value_field].size()
        if (counts > 1).any():
            subgroup_size = int(counts[counts > 1].median())
            kind = "xbar_s" if subgroup_size > _XBAR_S_THRESHOLD else "xbar_r"
            return ChartSuggestion(
                kind=kind,
                confidence="medium",
                reason=f"multiple rows share the same '{date_field}' (median subgroup size {subgroup_size})",
            )

    return ChartSuggestion(
        kind="xmr",
        confidence="medium",
        reason="no denominator/event/subgroup signal detected; individuals (XmR) chart is the safe default",
    )


# --- Input validation -----------------------------------------------------
# A guardrail with teeth, unlike suggest_chart_type() above: these DO
# raise. The distinction is deliberate -- suggest_chart_type() proposes a
# statistical framing (a judgement call, never provably "wrong"); the
# functions below check structural/mathematical preconditions a chart
# calculation cannot proceed without (a column existing, a denominator
# being positive), where silently continuing would produce a corrupted
# result rather than a debatable one.


def validate_columns_and_shape(
        df: pd.DataFrame,
        options: BaseChartOptions
) -> None:
    """Column-existence and non-emptiness only -- kind-independent, so
    this runs before kind="auto" is resolved (suggest_chart_type() itself
    indexes df by these same columns and would otherwise raise its own
    unrelated KeyError first).
    """
    candidates = {
        "value_field": options.value_field,
        "date_field": options.date_field,
        "facet_field": options.facet_field,
        "column_facet_field": options.column_facet_field,
        "sequence_field": options.sequence_field,
        "denominator_field": getattr(options, "denominator_field", None),
    }
    missing = {attr: col for attr, col in candidates.items() if col is not None and col not in df.columns}
    if missing:
        lines = [f"  options.{attr} = {col!r}" for attr, col in missing.items()]
        raise ValueError(
            "column(s) referenced in options were not found in the input dataframe:\n"
            + "\n".join(lines)
            + f"\nAvailable columns: {list(df.columns)}"
        )

    if len(df) == 0:
        raise ValueError("input dataframe has no rows -- nothing to chart")

    # Every kind needs a facet for every row. Checked here, once, because the
    # subgroup path groups by facet before segmentation and would otherwise
    # drop rows with a missing facet without a word.
    raise_if_facet_missing(df, facet_columns(options))


def _format_dates(
        dates: pd.Series
) -> list[str]:
    """ISO-formatted strings, not raw date-like objects -- embedding a list
    of datetime.date in an f-string invokes repr() on each element (e.g.
    "datetime.date(2024, 1, 15)"), not the readable str() form.
    """
    return [timestamp.strftime("%Y-%m-%d") for timestamp in pd.to_datetime(dates)]


def _raise_if_na(
        df: pd.DataFrame,
        date_field: str,
        field: str
) -> None:
    """Raises with the offending dates listed if `field` has any NaN values."""
    col = df[field]
    if col.isna().any():
        bad_dates = _format_dates(df.loc[col.isna(), date_field])
        raise ValueError(f"'{field}' has missing (NaN) values at: {bad_dates}")


def _raise_if_not_nonneg_integer(
        df: pd.DataFrame,
        date_field: str,
        field: str,
        kind: str
) -> None:
    """A count (defects, occurrences, opportunities) can't be negative or
    fractional -- structurally meaningless input, not just an unusual
    value, so this raises rather than letting it silently flow into
    p_chart_limits/c_chart_limits etc. and produce a nonsensical (but
    non-crashing) result, e.g. a negative p_bar.
    """
    numeric = pd.to_numeric(df[field], errors="coerce")
    bad = (numeric < 0) | ~np.isclose(numeric, np.round(numeric))
    if bad.any():
        bad_dates = _format_dates(df.loc[bad, date_field])
        raise ValueError(
            f"'{field}' must be a non-negative integer count for a {kind!r} chart "
            f"(negative or fractional defect/occurrence counts aren't meaningful) -- "
            f"found invalid values at: {bad_dates}"
        )


def _raise_if_not_timestamps(
        df: pd.DataFrame,
        field: str
) -> None:
    """A T chart's value_field holds one timestamp per rare event. A
    numeric column would be silently misread as nanoseconds-since-epoch
    by pd.to_datetime, giving plausible-looking but meaningless gaps, so
    it raises instead.
    """
    col = df[field]
    if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
        raise ValueError(
            f"'{field}' must hold event timestamps (datetime values) for a 't' chart, "
            f"not numbers (got dtype {col.dtype}) -- pass one timestamp per rare event; the "
            f"elapsed time between consecutive events is computed for you"
        )
    try:
        pd.to_datetime(col)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"'{field}' must hold event timestamps (datetime values) for a 't' chart: {exc}") from exc


def _raise_if_bad_dates(
        df: pd.DataFrame,
        field: str
) -> None:
    """Raises if `field` has missing or unparseable dates, naming the rows."""
    try:
        parsed = pd.to_datetime(df[field])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"'{field}' must hold dates: {exc}") from exc
    missing = parsed.isna()
    if missing.any():
        rows = [str(i) for i in parsed.index[missing][:5]]
        more = "" if missing.sum() <= 5 else f" (and {int(missing.sum()) - 5} more)"
        raise ValueError(f"'{field}' has {int(missing.sum())} missing date(s), at row(s) {', '.join(rows)}{more}")


def _raise_if_infinite(
        df: pd.DataFrame,
        date_field: str,
        field: str
) -> None:
    """Raises if a numeric column holds +/-infinity: it would give infinite
    control limits with no error, the same way NaN would give NaN ones.
    """
    numeric = pd.to_numeric(df[field], errors="coerce")
    bad = np.isinf(numeric)
    if bad.any():
        bad_dates = _format_dates(df.loc[bad, date_field])
        raise ValueError(f"'{field}' has infinite values at: {bad_dates}")


def _report_unmatched(
        option: str,
        unmatched: list[str],
        total: int,
        available: str
) -> None:
    """Raises when none of an option's entries match the data (almost certainly
    a typo or the wrong frame) and warns when only some do (a config shared
    across filtered frames can legitimately cover more than the data).
    """
    if not unmatched:
        return
    if len(unmatched) == total:
        raise ValueError(f"no entry in {option} matches the data ({available}): {unmatched}")
    warnings.warn(
        f"{len(unmatched)} of {total} entries in {option} match nothing in the data and are ignored: {unmatched}",
        UserWarning,
        stacklevel=5,
    )


def _check_option_keys(
        df: pd.DataFrame,
        options: BaseChartOptions,
        kind: str
) -> None:
    """A rebase date or a per-facet dict key that matches nothing in the data
    changes nothing, so a typo would pass unnoticed: the chart looks fine and
    simply isn't rebased (or its target never applies). Raise when an option
    matches nothing at all; warn when it matches only in part.

    Rebase dates are matched against the dates the chart actually plots: the
    event timestamps for `t`, `date_field` for every other kind. Baseline
    entries are not checked: a baseline may legitimately cover more dates or
    facets than the data.
    """
    facets = facet_series(df, options)
    labels = set(facets) if facets is not None else set()
    date_col = options.value_field if kind == "t" else options.date_field
    dates = pd.to_datetime(df[date_col]).dt.date
    available_facets = f"facets: {sorted(labels)}"

    if options.rebase is not None:
        unmatched: list[str] = []
        total = 0
        if isinstance(options.rebase, dict):
            for facet, wanted in options.rebase.items():
                total += max(len(wanted), 1)
                if facet not in labels:
                    unmatched.append(f"facet {facet!r} is not in the data")
                    continue
                present = set(dates[facets == facet]) if facets is not None else set()
                unmatched.extend(f"{facet!r}: {d}" for d in wanted if d not in present)
            _report_unmatched("rebase", unmatched, total, available_facets)
        else:
            present = set(dates)
            total = len(options.rebase)
            unmatched = [str(d) for d in options.rebase if d not in present]
            _report_unmatched("rebase", unmatched, total, "no row has that date")

    for name in ("target", "spec_upper", "spec_lower"):
        value = getattr(options, name)
        if isinstance(value, dict):
            unmatched = [repr(key) for key in value if key not in labels]
            _report_unmatched(name, unmatched, len(value), available_facets)


def raise_if_bad_sequence(
        df: pd.DataFrame,
        field: str
) -> None:
    """A tie-break column needs a position for every row, of one comparable type."""
    column = df[field]
    if column.isna().any():
        raise ValueError(f"'{field}' has missing values -- every row needs a position in the sequence")
    if pd.api.types.infer_dtype(column).startswith("mixed"):
        raise ValueError(
            f"'{field}' mixes value types (e.g. numbers and text), which have no meaningful order -- use one type"
        )


def _check_row_order(
        df: pd.DataFrame,
        options: BaseChartOptions,
        kind: str
) -> None:
    """Rows sharing a date within a facet have no defined order, and order
    changes the moving range and the run rules. xbar kinds treat them as a
    subgroup and `t` orders by its own timestamps, so neither accepts a
    `sequence_field`. For the other kinds: without a `sequence_field`, ties
    warn; with one, rows still tied on (date, sequence) raise.
    """
    if kind in ("xbar_r", "xbar_s", "t"):
        if options.sequence_field:
            raise ValueError(
                f"sequence_field isn't used by kind {kind!r}: "
                + (
                    "rows sharing a date are one subgroup"
                    if kind != "t"
                    else "events are ordered by their own timestamps"
                )
            )
        return
    keys = [options.date_field, *facet_columns(options)]
    if options.sequence_field:
        raise_if_bad_sequence(df, options.sequence_field)
        tied = int(df.duplicated(subset=[*keys, options.sequence_field], keep=False).sum())
        if tied:
            raise ValueError(
                f"{tied} row(s) share a date and a '{options.sequence_field}' value with another row in the "
                f"same facet, so their order is still ambiguous"
            )
        return
    tied = int(df.duplicated(subset=keys, keep=False).sum())
    if tied:
        warnings.warn(
            f"{tied} row(s) share a date with another row in the same facet, so their order is taken from the "
            f"input and affects the moving range and run rules. Set sequence_field to order them, aggregate to "
            f"one row per period, or use xbar_r/xbar_s for subgrouped data.",
            UserWarning,
            stacklevel=4,
        )


def validate_values(
        df: pd.DataFrame,
        options: BaseChartOptions,
        kind: str
) -> None:
    """Kind-dependent value checks (NaN, non-negative counts, denominator
    positivity/boundedness). Runs after kind is resolved (whether given
    explicitly or via "auto") and confirmed to have a calculator. Assumes
    validate_columns_and_shape() already ran -- does not re-check column
    existence or emptiness.
    """
    if kind != "t":  # t reads its timestamps from value_field and never uses date_field
        _raise_if_bad_dates(df, options.date_field)
    _raise_if_na(df, options.date_field, options.value_field)
    if kind == "t":
        _raise_if_not_timestamps(df, options.value_field)
    else:
        _raise_if_infinite(df, options.date_field, options.value_field)
        if kind in _COUNT_FIELD_KINDS:
            _raise_if_not_nonneg_integer(df, options.date_field, options.value_field, kind)
    _check_option_keys(df, options, kind)
    _check_row_order(df, options, kind)

    denominator_field = getattr(options, "denominator_field", None)
    if denominator_field is None:
        return

    _raise_if_na(df, options.date_field, denominator_field)
    _raise_if_infinite(df, options.date_field, denominator_field)

    denominator = df[denominator_field]
    if (denominator <= 0).any():
        bad_dates = _format_dates(df.loc[denominator <= 0, options.date_field])
        raise ValueError(
            f"'{denominator_field}' must be strictly positive everywhere -- found values <= 0 "
            f"at: {bad_dates} (a zero or negative sample size/opportunity is not meaningful "
            f"for a p/u/np chart, and would divide by zero)"
        )

    if kind in _BOUNDED_NUMERATOR_KINDS:
        value = df[options.value_field]
        violates = value > denominator
        if violates.any():
            bad_dates = _format_dates(df.loc[violates, options.date_field])
            raise ValueError(
                f"'{options.value_field}' exceeds '{denominator_field}' at: {bad_dates} -- "
                f"a {kind!r} chart's numerator can't exceed its denominator "
                f"(defects can't exceed sample size)"
            )

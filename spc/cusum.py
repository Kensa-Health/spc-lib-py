"""Bernoulli CUSUM: sequential monitoring of a binary outcome (an adverse
event, a failed procedure) for a shift in its rate.

Adjacent to SPC rather than a Shewhart control chart -- it works on one
0/1 observation per case, accumulates evidence instead of comparing each
point to limits, and has no rule engine -- so it has its own options and
result types instead of `ChartResult`. Calculation only; see
`spc.render.plotly_cusum` for the optional chart.

The scheme (the log-likelihood-ratio CUSUM for a Bernoulli rate): with
in-control rate p0 and an alternative p1 defined by an odds ratio, each
observation adds
    log(p1/p0)              if the event occurred,
    log((1-p1)/(1-p0))      if it did not,
and the running score S is floored at 0 (S = max(0, S + w)). A signal is
raised when S reaches the decision threshold h. `cusum_arl` and
`cusum_threshold` compute the average run length and a threshold for a
target in-control run length from a discretised Markov chain. The approximation
was checked against hand-solved cases, Monte Carlo simulation and a finer
lattice; the README's caveats give its accuracy range.
"""

from __future__ import annotations

import math
import warnings
from datetime import UTC, date, datetime
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from spc.core.facets import FACET_SEPARATOR, facet_key_series
from spc.core.validation import raise_if_bad_sequence

CusumDirection = Literal["increase", "decrease"]

_DEFAULT_N_STATES = 800
# cusum_threshold re-checks its answer on a lattice this many times finer and
# warns when the two disagree by more than this fraction.
_REFINEMENT_FACTOR = 2
_CONVERGENCE_WARN_REL = 0.02
_MIN_REFERENCE_EVENTS = 5
# The dense solve loses accuracy as the run length grows (the score matrix
# becomes numerically singular); beyond this the answer is not trustworthy,
# so it is refused rather than returned. Checked: results track the expected
# exponential growth in h up to ~2e14 and drift well below it from ~1e15.
_MAX_RELIABLE_ARL = 1e13
_THRESHOLD_GROWTH = 1.25  # step when bracketing h; small, so the search never jumps far past the target
_MAX_THRESHOLD_STEPS = 200


def _check_direction(
        direction: str
) -> None:
    """Raises ValueError unless `direction` is "increase" or "decrease"."""
    if direction not in ("increase", "decrease"):
        raise ValueError(f"direction must be 'increase' or 'decrease', got {direction!r}")


def _alternative_rate(
        p0: float,
        odds_ratio: float,
        direction: CusumDirection
) -> float:
    """p1 whose odds are `odds_ratio` times p0's ("increase") or 1/odds_ratio
    times ("decrease").
    """
    _check_direction(direction)
    odds0 = p0 / (1.0 - p0)
    odds1 = odds0 * odds_ratio if direction == "increase" else odds0 / odds_ratio
    return odds1 / (1.0 + odds1)


def _weights(
        p0: float,
        p1: float
) -> tuple[float, float]:
    """(score for an event, score for a non-event)."""
    return math.log(p1 / p0), math.log((1.0 - p1) / (1.0 - p0))


def _check_rate(
        name: str,
        value: float
) -> None:
    """Raises ValueError unless the rate is strictly between 0 and 1."""
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value}")


def _check_odds_ratio(
        odds_ratio: float
) -> None:
    """Raises ValueError unless the odds ratio is finite and exceeds 1."""
    if not (math.isfinite(odds_ratio) and odds_ratio > 1.0):
        raise ValueError(
            "odds_ratio must be finite and > 1 (the size of the shift to detect, e.g. 2.0 for a doubling "
            f"of the odds), got {odds_ratio}"
        )


def _check_positive_finite(
        name: str,
        value: float,
        *,
        minimum: float = 0.0
) -> None:
    """Raises ValueError unless the value is finite and greater than `minimum`."""
    if not (math.isfinite(value) and value > minimum):
        raise ValueError(f"{name} must be finite and > {minimum:g}, got {value}")


def _arl(
        p0: float,
        odds_ratio: float,
        h: float,
        p_true: float | None,
        direction: CusumDirection,
        n_states: int
) -> float:
    """The Markov-chain ARL with input checks but no range check on the answer."""
    _check_rate("p0", p0)
    _check_odds_ratio(odds_ratio)
    _check_positive_finite("h", h)
    if n_states < 2:
        raise ValueError("n_states must be >= 2")
    p_true = p0 if p_true is None else p_true
    _check_rate("p_true", p_true)

    p1 = _alternative_rate(p0, odds_ratio, direction)
    w_event, w_nonevent = _weights(p0, p1)
    # One outcome moves the score up, the other down.
    if w_event > 0:
        p_up, step_up, step_down = p_true, w_event, -w_nonevent
    else:
        p_up, step_up, step_down = 1.0 - p_true, w_nonevent, -w_event
    p_down = 1.0 - p_up

    m = n_states
    delta = h / m

    def _split(
            step: float
    ) -> tuple[int, float]:
        """Whole lattice units and the fractional remainder of a score step."""
        units = round(step / delta, 9)
        whole = math.floor(units)
        return whole, units - whole

    up_whole, up_frac = _split(step_up)
    down_whole, down_frac = _split(step_down)

    a = np.eye(m)
    s = np.arange(m)
    for jump, weight in ((up_whole, p_up * (1.0 - up_frac)), (up_whole + 1, p_up * up_frac)):
        if weight == 0.0:
            continue
        target = s + jump
        inside = target < m  # reaching m or beyond is the absorbing "signal" state
        a[s[inside], target[inside]] -= weight
    for drop, weight in ((down_whole, p_down * (1.0 - down_frac)), (down_whole + 1, p_down * down_frac)):
        if weight == 0.0:
            continue
        a[s, np.maximum(0, s - drop)] -= weight
    try:
        expected = np.linalg.solve(a, np.ones(m))
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "the run length is too long to compute (the score matrix is numerically singular) -- "
            "use a smaller h, or an odds ratio further from 1"
        ) from exc
    result = float(expected[0])
    if not math.isfinite(result) or result < 1.0 - 1e-9:
        raise ValueError(f"the run-length calculation did not converge (got {result}) -- use a smaller h")
    return result


def cusum_arl(
        p0: float,
        odds_ratio: float,
        h: float,
        *,
        p_true: float | None = None,
        direction: CusumDirection = "increase",
        n_states: int = _DEFAULT_N_STATES
) -> float:
    """Average run length (observations until the first signal) from a
    fresh start, when the true event rate is `p_true` (default `p0`, giving
    the in-control ARL0; pass the alternative rate for the out-of-control
    ARL1).

    Approximates the continuous score by a lattice of `n_states` points up
    to h. Both score steps are rounded onto the lattice at random so that
    each step's mean is preserved exactly, and the resulting finite Markov
    chain is solved exactly. Error shrinks as n_states grows; the default
    is a few tenths of a percent in the cases checked (see the README's caveats).

    Raises ValueError for invalid inputs, and when the run length exceeds
    1e13, beyond which the calculation is not reliable.
    """
    result = _arl(p0, odds_ratio, h, p_true, direction, n_states)
    if result > _MAX_RELIABLE_ARL:
        raise ValueError(
            f"the run length ({result:.3g}) exceeds {_MAX_RELIABLE_ARL:.0e}, beyond which it cannot be "
            "computed reliably -- use a smaller h, or an odds ratio further from 1"
        )
    return result


def cusum_threshold(
        p0: float,
        odds_ratio: float,
        arl0: float,
        *,
        direction: CusumDirection = "increase",
        n_states: int = _DEFAULT_N_STATES,
        rel_tol: float = 1e-3
) -> float:
    """The decision threshold h whose in-control ARL is `arl0`: the
    smallest h (to within `rel_tol`, rounded up) with ARL0 >= arl0, so the
    false-alarm rate is no worse than asked for.
    """
    _check_rate("p0", p0)
    _check_odds_ratio(odds_ratio)
    _check_positive_finite("arl0", arl0, minimum=1.0)
    if arl0 > _MAX_RELIABLE_ARL:
        raise ValueError(
            f"arl0 ({arl0:g}) exceeds {_MAX_RELIABLE_ARL:.0e}, beyond which run lengths cannot be computed reliably"
        )

    def arl(
            h: float
    ) -> float:
        """In-control ARL at threshold h."""
        return _arl(p0, odds_ratio, h, None, direction, n_states)

    lo, hi = 1e-6, 1.0
    for _ in range(_MAX_THRESHOLD_STEPS):
        if arl(hi) >= arl0:
            break
        lo, hi = hi, hi * _THRESHOLD_GROWTH
    else:
        raise ValueError(f"no threshold found for arl0={arl0}")
    while (hi - lo) > rel_tol * hi:
        mid = 0.5 * (lo + hi)
        if arl(mid) >= arl0:
            hi = mid
        else:
            lo = mid
    # The lattice error is first-order in 1/n_states and grows with the run
    # length and with extreme odds ratios, so check the answer on a finer
    # lattice rather than trusting the default silently.
    refined = _arl(p0, odds_ratio, hi, None, direction, n_states * _REFINEMENT_FACTOR)
    if abs(refined - arl0) > _CONVERGENCE_WARN_REL * arl0:
        warnings.warn(
            f"cusum_threshold: h={hi:.4g} gives an in-control ARL of {refined:.4g} on a finer lattice "
            f"(n_states={n_states * _REFINEMENT_FACTOR}), more than {_CONVERGENCE_WARN_REL:.0%} from the "
            f"target {arl0:g}. The lattice approximation is least accurate for very long run lengths and "
            f"for odds ratios near 1 or very large; raise n_states, or check the threshold by simulation.",
            UserWarning,
            stacklevel=2,
        )
    return hi


def _timestamps(
        df: pd.DataFrame,
        field: str
) -> pd.Series:
    """`df[field]` as timestamps, raising a clear error for unparseable or
    missing dates (which would otherwise surface as an unrelated TypeError).
    """
    try:
        parsed = pd.to_datetime(df[field])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"'{field}' must hold dates: {exc}") from exc
    missing = parsed.isna()
    if missing.any():
        rows = [str(i) for i in parsed.index[missing][:5]]
        more = "" if missing.sum() <= 5 else f" (and {int(missing.sum()) - 5} more)"
        raise ValueError(f"'{field}' has {int(missing.sum())} missing date(s), at row(s) {', '.join(rows)}{more}")
    return parsed


def estimate_event_rate(
        df: pd.DataFrame,
        value_field: str,
        date_field: str,
        *,
        until: date,
        min_events: int = _MIN_REFERENCE_EVENTS
) -> float:
    """Event rate over a reference window: the share of rows dated on or
    before `until` whose outcome is 1. A convenience for choosing p0 from
    historical data.

    Raises if the window is empty, or has fewer than `min_events` events or
    non-events (default 5, the same rule of thumb as the p-chart's
    n*p >= 5 check): a rate from so few events is too noisy to be a
    baseline, and a p0 estimated too low produces false alarms. This also
    rejects a rate of exactly 0 or 1, which can never be a CUSUM baseline.
    Pass `min_events=1` to override. Estimation error in p0 is not
    propagated into the run-length calculations.
    """
    for col in (value_field, date_field):
        if col not in df.columns:
            raise ValueError(f"column {col!r} not found in the input dataframe")
    window = df[_timestamps(df, date_field).dt.date <= until]
    if len(window) == 0:
        raise ValueError(f"no rows dated on or before {until} -- the reference window is empty")
    _check_outcomes(window[value_field], value_field)
    outcomes = window[value_field].astype(float)
    events = int(outcomes.sum())
    non_events = len(outcomes) - events
    if min(events, non_events) < max(min_events, 1):
        raise ValueError(
            f"the reference window has {events} event(s) and {non_events} non-event(s) in {len(window)} rows -- "
            f"at least {max(min_events, 1)} of each are needed for a usable baseline rate (a rate of exactly 0 "
            "or 1 can never be one); supply p0 explicitly or use a longer window"
        )
    return events / len(outcomes)


def _check_outcomes(
        values: pd.Series,
        field: str
) -> None:
    """Raises ValueError for missing values or anything other than 0/1."""
    if values.isna().any():
        raise ValueError(f"'{field}' has missing (NaN) values -- every case needs a 0/1 outcome")
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise ValueError(f"'{field}' must hold 0/1 outcomes (1 = the event occurred)")


class CusumOptions(BaseModel):
    """Inputs for `cusum()`.

    `value_field`: a 0/1 outcome per case (1 = the event happened).
    `date_field`: orders the cases (a time of day, if present, is used for
    ordering; the reported x-position is the date). A CUSUM depends on the
    order of the cases, so cases sharing a timestamp need a tie-break: `sequence_field` (any
    sortable column, e.g. a time of day or a case number) breaks ties, and
    must make (date, sequence) unique within a facet. Without it, tied
    cases keep their input order and `cusum()` warns.
    `p0`: the in-control event rate, supplied rather than estimated from
    the monitored data (use `estimate_event_rate` on a reference window);
    a scalar, or a dict keyed by facet.
    `odds_ratio`: the size of shift to detect, > 1 (default 2.0: a
    doubling of the odds). `detect` says which direction of change in the
    event *rate* to detect -- "increase" (default, for an adverse event),
    "decrease", or "both" (two one-sided charts). Unlike the Shewhart
    charts' `improvement_direction` this names the rate itself, so there is
    no default that silently watches the wrong side of an adverse event.
    `h`: the decision threshold, or None to compute it from `arl0`, the
    target average run length between false alarms (default 1000). For
    `detect="both"`, `arl0` is the run length for a signal on *either*
    side: each side is calibrated to 2*arl0, because the two one-sided
    charts' false-alarm rates add (1/ARL = 1/ARL_up + 1/ARL_down, checked
    by simulation). An explicit `h` is used as given on both sides, so the
    joint ARL0 is then roughly half a single side's.
    `reset_on_signal`: restart the score at 0 after each signal (default).
    `facet_field`/`column_facet_field`: an independent CUSUM per group.
    """

    model_config = ConfigDict(extra="forbid")  # a misspelled option must raise, not be ignored

    value_field: str
    date_field: str
    p0: float | dict[str, float]
    odds_ratio: float = 2.0
    detect: Literal["increase", "decrease", "both"] = "increase"
    h: float | None = None
    arl0: float = 1000.0
    reset_on_signal: bool = True
    sequence_field: str | None = None
    facet_field: str | None = None
    column_facet_field: str | None = None

    @model_validator(mode="after")
    def check_values(self) -> CusumOptions:
        """Validates the rates, odds ratio, threshold, target run length and facet settings."""
        rates = self.p0.values() if isinstance(self.p0, dict) else [self.p0]
        for rate in rates:
            _check_rate("p0", rate)
        _check_odds_ratio(self.odds_ratio)
        if self.h is not None:
            _check_positive_finite("h", self.h)
        _check_positive_finite("arl0", self.arl0, minimum=1.0)
        if self.arl0 > _MAX_RELIABLE_ARL / (2.0 if self.detect == "both" else 1.0):
            raise ValueError(
                f"arl0 ({self.arl0:g}) is too large to calibrate reliably "
                f"(limit {_MAX_RELIABLE_ARL / (2.0 if self.detect == 'both' else 1.0):.0e} for detect={self.detect!r})"
            )
        if self.column_facet_field and not self.facet_field:
            raise ValueError("column_facet_field requires facet_field (the row facet) to be set")
        if isinstance(self.p0, dict) and not self.facet_field:
            raise ValueError("a per-facet p0 dict needs facet_field to be set")
        return self


class CusumPoint(BaseModel):
    """One case's outcome and the CUSUM score after it."""

    x: date
    y: int  # the outcome, 0 or 1
    score: float  # the CUSUM score after this case (the pre-reset value on a signal)
    signal: bool  # score >= h


class CusumSeries(BaseModel):
    """One CUSUM run: a facet and a direction, with its parameters and per-case points."""

    facet: str | None
    facet_parts: list[str] | None = None
    direction: CusumDirection
    p0: float
    p1: float
    odds_ratio: float  # > 1 for "increase", < 1 for "decrease" (the shift being tested for)
    h: float
    points: list[CusumPoint]
    signal_indices: list[int]  # positions in `points` where a signal starts


class CusumResult(BaseModel):
    """Output of `cusum()`: one series per facet and direction."""

    schema_version: str = "1.0"
    options: CusumOptions
    series: list[CusumSeries]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _score(
        outcomes: list[int],
        w_event: float,
        w_nonevent: float,
        h: float,
        reset: bool
) -> tuple[list[float], list[bool]]:
    """The CUSUM recursion: running scores and a signal flag per case."""
    scores: list[float] = []
    flags: list[bool] = []
    s = 0.0
    for y in outcomes:
        s = max(0.0, s + (w_event if y else w_nonevent))
        flagged = s >= h
        scores.append(s)
        flags.append(flagged)
        if flagged and reset:
            s = 0.0
    return scores, flags


def cusum(
        df: pd.DataFrame,
        options: CusumOptions
) -> CusumResult:
    """Runs the Bernoulli CUSUM, independently per facet and per direction.

    Raises ValueError on missing columns, an empty frame, NaN or non-0/1
    outcomes, or a facet without a p0.
    """
    needed = [
        options.value_field,
        options.date_field,
        *([options.sequence_field] if options.sequence_field else []),
        *([options.facet_field] if options.facet_field else []),
        *([options.column_facet_field] if options.column_facet_field else []),
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"column(s) not found in the input dataframe: {missing}. Available columns: {list(df.columns)}"
        )
    if len(df) == 0:
        raise ValueError("input dataframe has no rows -- nothing to monitor")
    _check_outcomes(df[options.value_field], options.value_field)

    # Order by the full timestamp, so a time of day orders same-day cases;
    # only the reported x-position is reduced to a date.
    timestamps = _timestamps(df, options.date_field)
    frame = pd.DataFrame(
        {
            "ts": timestamps,
            "x": timestamps.dt.date,
            "y": pd.to_numeric(df[options.value_field]).astype(int),
        }
    )
    if options.sequence_field:
        raise_if_bad_sequence(df, options.sequence_field)
        sequence = df[options.sequence_field]
        frame["seq"] = sequence
    facets = facet_key_series(df, options.facet_field, options.column_facet_field)
    frame["facet"] = facets if facets is not None else None
    directions: list[CusumDirection] = ["increase", "decrease"] if options.detect == "both" else [options.detect]

    # The calibrated threshold depends only on (p0, direction), not on the
    # facet's data, so compute each once however many facets share it.
    side_arl0 = options.arl0 * 2.0 if options.detect == "both" else options.arl0
    thresholds: dict[tuple[float, str], float] = {}

    def threshold_for(
            p0: float,
            direction: CusumDirection
    ) -> float:
        """The explicit h, or the calibrated one (computed once per p0 and direction)."""
        if options.h is not None:
            return options.h
        if (p0, direction) not in thresholds:
            thresholds[(p0, direction)] = cusum_threshold(p0, options.odds_ratio, side_arl0, direction=direction)
        return thresholds[(p0, direction)]

    series: list[CusumSeries] = []
    groups = (
        [(None, frame)] if facets is None else sorted(frame.groupby("facet", observed=True), key=lambda kv: str(kv[0]))
    )
    for facet, group in groups:
        facet_key = None if facet is None else str(facet)
        if isinstance(options.p0, dict):
            # CusumOptions guarantees facet_field is set when p0 is a dict,
            # so every group has a facet key here.
            if facet_key is None or facet_key not in options.p0:
                raise ValueError(f"no p0 supplied for facet {facet_key!r}")
            p0 = options.p0[facet_key]
        else:
            p0 = options.p0
        order = ["ts", "seq"] if options.sequence_field else ["ts"]
        group = group.sort_values(order, kind="stable")
        tied = group.duplicated(subset=order, keep=False)
        if tied.any() and options.sequence_field:
            raise ValueError(
                f"{int(tied.sum())} case(s) in facet {facet_key!r} share both a timestamp and a "
                f"'{options.sequence_field}' value with another case, so their order is still ambiguous"
            )
        if tied.any():
            warnings.warn(
                f"{int(tied.sum())} case(s) in facet {facet_key!r} share a date/time with another case; their "
                "order is taken from the input and can change when or whether the CUSUM signals. "
                "Supply sequence_field to order them.",
                UserWarning,
                stacklevel=2,
            )
        outcomes = group["y"].tolist()
        xs = group["x"].tolist()
        for direction in directions:
            p1 = _alternative_rate(p0, options.odds_ratio, direction)
            w_event, w_nonevent = _weights(p0, p1)
            h = threshold_for(p0, direction)
            scores, flags = _score(outcomes, w_event, w_nonevent, h, options.reset_on_signal)
            signal_indices = [i for i, f in enumerate(flags) if f and (i == 0 or not flags[i - 1])]
            parts = None
            if options.column_facet_field and facet_key is not None:
                parts = facet_key.split(FACET_SEPARATOR)
            series.append(
                CusumSeries(
                    facet=facet_key,
                    facet_parts=parts,
                    direction=direction,
                    p0=p0,
                    p1=p1,
                    odds_ratio=options.odds_ratio if direction == "increase" else 1.0 / options.odds_ratio,
                    h=h,
                    points=[
                        CusumPoint(x=x, y=y, score=s, signal=f)
                        for x, y, s, f in zip(xs, outcomes, scores, flags, strict=True)
                    ],
                    signal_indices=signal_indices,
                )
            )
    return CusumResult(options=options, series=series)

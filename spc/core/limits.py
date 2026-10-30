"""Chart-type-specific control limit formulas.

Covers XmR's moving-range-based formula and the attribute-chart formulas
(p/np/c/u, binomial/Poisson-based). The baseline/residual seam and the
target-assurance check are both written generically enough to be reused
by every chart kind, not just XmR -- though the baseline seam is not
currently wired up for the attribute charts; see
core/attribute_segments.py's module docstring. The XmR limit calculation and the
assurance check follow NHSRplotthedots (see THIRD_PARTY_NOTICES.md).

Statistical caveat -- not something this module corrects for, only states:
LIMIT (2.66) is a ~3-sigma-equivalent width derived under an approximate-
normality assumption for the underlying process. For skewed data (routine
for count/rate/behavioural metrics), the *actual* false-alarm rate of a
raw XmR chart will differ from the nominal one implied by that constant --
sometimes substantially, depending on skew direction and magnitude. This
library does not detect skew or transform the input before applying LIMIT.
The T/G chart kinds apply the Nelson power transform for exactly this
reason -- consider transforming skewed data yourself before calling
`chart(kind="xmr")` if false-alarm rate matters for your use case, or use
T/G if the data is genuinely a rare-event series.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from spc.core.models import AssuranceType, ImprovementDirection, ScreenOutliers

LIMIT = 2.66
LIMIT_CLOSE = 2 * (LIMIT / 3)  # ~1.7733 -- defines the near-limit zone for the 2-in-3 rule
OUTLIER_SCREEN_MULTIPLIER = 3.267  # D4 constant for n=2, used to screen outlier moving ranges
# Caps how much an autocorrelation correction can widen the limits, so a
# noisy small-n rho estimate near 1 can't blow 1/sqrt(1-rho) up
# arbitrarily. No corresponding floor -- a genuinely negative rho
# narrows the limits, which is correct, not something to clamp away.
_AUTOCORRELATION_CLAMP_MAX = 0.95
# Threshold for SegmentSummary.high_autocorrelation_warning: the
# correction is least reliable exactly as rho approaches 1, so this
# flags a segment's estimate as worth a second look without changing
# anything it produces.
HIGH_AUTOCORRELATION_THRESHOLD = 0.8

# EXPANDING mode recomputes limits from scratch for each of the first N
# points -- O(N^2) in the warm-up window. Fine for realistic SPC series
# (weekly/monthly data, N typically 12-20); this guards against a
# pathologically large fix_after_n_points silently doing unbounded work.
_MAX_EXPANDING_WINDOW = 500

# Safety cap on iterative outlier-screening re-passes (see compute_limit_stats)
# -- real data converges in a handful of iterations; this just prevents an
# unbounded loop if some pathological input never stabilises.
_MAX_SCREEN_ITERATIONS = 50


@dataclass
class LimitStats:
    """Scalar mean/amr/limits computed from one value series (a whole
    segment, a fixed first N, or one point of an expanding window -- the
    caller decides which points to pass in; this function doesn't know).
    """

    mean: float
    amr: float
    ucl: float
    lcl: float
    near_ucl: float
    near_lcl: float
    # Lag-1 sample autocorrelation of fix_y (see estimate_autocorrelation).
    # Always computed and populated regardless of autocorrelation_
    # adjustment -- visible for review even when not acted on, same split
    # as low_count_warning. NaN, not None: composes directly with this
    # dataclass's other float fields; None only appears at the
    # ChartResult/SegmentSummary boundary (core/chart_result.py's
    # _or_none()), same as ucl/lcl.
    autocorrelation: float


def widen_amr_for_autocorrelation(
        amr: float,
        rho: float,
        autocorrelation_adjustment: bool
) -> float:
    """The AMR that should actually drive control-limit width: `amr`
    itself when the adjustment is off or `rho` is NaN (no-op, so the
    default reproduces today's numbers exactly), otherwise `amr` divided
    by `sqrt(1 - min(rho, _AUTOCORRELATION_CLAMP_MAX))`.

    A separate function, not inlined into compute_limit_stats, because
    core/segments.py needs the identical formula when it recomputes
    control-limit width itself from a segment's own (possibly baseline-
    adjusted) centre line -- one formula, not two copies to keep in sync.
    """
    if not autocorrelation_adjustment or math.isnan(rho):
        return amr
    rho_clamped = min(rho, _AUTOCORRELATION_CLAMP_MAX)  # no floor -- see estimate_autocorrelation's docstring
    return amr / math.sqrt(1 - rho_clamped)


def estimate_autocorrelation(
        residuals: pd.Series
) -> float:
    """Lag-1 sample autocorrelation of a detrended residual series. The
    standard Box-Jenkins/Yule-Walker lag-1 estimator, sharing one mean
    across both the lagged and current term.

    NaN entries in `residuals` mark points to exclude (e.g. screened
    outliers) -- pairwise deletion then applies: only lag-1 pairs where
    both points are valid contribute, each half of the pair (leading vs.
    lagging) using its own mean over just the valid pairs, since the two
    halves aren't the same set of points once any are excluded.

    NaN when fewer than 2 valid points remain, or the valid series (or
    either half of a pairwise-deleted one) has zero variance -- nothing
    to correlate. Matches laney_sigma_z's own NaN-for-too-few-points
    return (this module) rather than inventing a new sentinel.
    """
    valid = residuals.notna()
    x = residuals.to_numpy(dtype=float)

    if valid.all():
        if valid.sum() < 2:
            return float("nan")
        xbar = x.mean()
        denominator = float(np.sum((x - xbar) ** 2))
        if denominator == 0:
            return float("nan")
        numerator = float(np.sum((x[1:] - xbar) * (x[:-1] - xbar)))
        return numerator / denominator

    pair_mask = valid.to_numpy()[1:] & valid.to_numpy()[:-1]
    if pair_mask.sum() < 2:
        return float("nan")
    x_lag, x_cur = x[:-1][pair_mask], x[1:][pair_mask]
    xbar_lag, xbar_cur = x_lag.mean(), x_cur.mean()
    v_lag = float(np.sum((x_lag - xbar_lag) ** 2))
    v_cur = float(np.sum((x_cur - xbar_cur) ** 2))
    if v_lag == 0 or v_cur == 0:
        return float("nan")
    covariance = float(np.sum((x_cur - xbar_cur) * (x_lag - xbar_lag)))
    return covariance / math.sqrt(v_lag * v_cur)


def compute_limit_stats(
        fix_y: pd.Series,
        screen_outliers: ScreenOutliers,
        *,
        autocorrelation_adjustment: bool = False
) -> LimitStats:
    """Core XmR mean/moving-range/limit calculation.

    `fix_y` is whichever subset of a segment's values should drive the
    mean/limits calculation -- this function is agnostic to LimitMode
    (segment / fixed_n / expanding); see core/segments.py for that dispatch.
    When used against a *residual* series (y - expected_value) rather than
    raw y, `mean` here is the bias term of the baseline seam, not a final
    centre line -- see compute_limit_stats_with_baseline.
    """
    mean_col = float(fix_y.mean())
    mr = fix_y.diff().abs()
    amr = float(mr.mean())
    excluded_mr = pd.Series(False, index=mr.index)

    if screen_outliers == "single_pass":
        threshold = OUTLIER_SCREEN_MULTIPLIER * amr
        excluded_mr = (mr >= threshold).fillna(False)
        screened = mr.where(~excluded_mr, other=np.nan)
        amr = float(screened.mean())
    elif screen_outliers == "iterative":
        amr, excluded_mr = _screen_iteratively(mr, amr)
    # "off": amr stays as computed from the raw, unscreened moving range;
    # excluded_mr stays all-False, so autocorrelation is estimated from
    # fix_y unmodified.

    # A screened-out moving range implicates the jump between its two
    # endpoints, not one attributable point -- both get excluded from the
    # residual series rho is estimated from, the conservative reading
    # rather than guessing which endpoint was "the" outlier.
    excluded_point = excluded_mr | excluded_mr.shift(-1, fill_value=False)
    residuals_for_rho = (fix_y - mean_col).where(~excluded_point, other=np.nan)
    rho = estimate_autocorrelation(residuals_for_rho)

    amr_for_width = widen_amr_for_autocorrelation(amr, rho, autocorrelation_adjustment)

    limit_width = LIMIT * amr_for_width
    near_width = LIMIT_CLOSE * amr_for_width
    return LimitStats(
        mean=mean_col,
        amr=amr,
        ucl=mean_col + limit_width,
        lcl=mean_col - limit_width,
        near_ucl=mean_col + near_width,
        near_lcl=mean_col - near_width,
        autocorrelation=rho,
    )


def _screen_iteratively(
        mr: pd.Series,
        initial_amr: float
) -> tuple[float, pd.Series]:
    """Re-screen moving ranges against a repeatedly-recomputed threshold
    until the set of excluded points stops changing. Returns the
    converged AMR and the final exclusion mask (the mask isn't otherwise
    recoverable from the AMR alone -- single_pass's threshold is fixed,
    but this one moves with every iteration).

    Always re-applies the current threshold to the *original* moving-range
    series, not to an already-screened one -- otherwise exclusions would
    compound across iterations instead of the threshold simply moving.
    """
    current_amr = initial_amr
    prev_excluded = pd.Series(False, index=mr.index)
    for _ in range(_MAX_SCREEN_ITERATIONS):
        threshold = OUTLIER_SCREEN_MULTIPLIER * current_amr
        excluded = (mr >= threshold).fillna(False)
        if excluded.equals(prev_excluded):
            break
        screened = mr.where(~excluded, other=np.nan)
        current_amr = float(screened.mean())
        prev_excluded = excluded
    return current_amr, prev_excluded


@dataclass
class SeriesLimitStats:
    """Per-point mean/amr/limits -- the general case that also covers a
    flat scalar (SEGMENT/FIXED_N modes, broadcast to every point) or a
    genuinely varying series (EXPANDING mode's warm-up phase, and/or a
    supplied baseline). See core/segments.py, which is the only caller.
    """

    centre_line: pd.Series
    amr: pd.Series
    ucl: pd.Series
    lcl: pd.Series
    near_ucl: pd.Series
    near_lcl: pd.Series
    autocorrelation: pd.Series


def compute_point_flags(
        y: pd.Series,
        centre_line: pd.Series,
        ucl: pd.Series,
        lcl: pd.Series,
        near_ucl: pd.Series,
        near_lcl: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per-point outside_limits / relative_to_mean / close_to_limits flags.
    Works whether centre_line/ucl/lcl/near_ucl/near_lcl are flat scalars
    broadcast to every point or a genuinely per-point series -- pandas
    alignment handles both identically.
    """
    outside_limits = (y > ucl) | (y < lcl)
    # np.sign() on a pandas Series returns a Series at runtime (via
    # __array_ufunc__), but numpy's stubs type it as ndarray -- cast tells
    # mypy what's actually true rather than changing behaviour.
    relative_to_mean = cast(pd.Series, np.sign(y - centre_line))
    close_to_limits = (~outside_limits) & ((y < near_lcl) | (y > near_ucl))
    return outside_limits, relative_to_mean, close_to_limits


@dataclass
class AttributeLimitStats:
    """Per-point ucl/lcl/near_ucl/near_lcl for an attribute (p/np/c/u)
    chart. Unlike XmR's LimitStats, there's no `amr` here -- these charts'
    limit width comes from a closed-form variance formula (binomial for p/
    np, Poisson for c/u), not a moving-range estimate. `ucl`/etc. are
    np.ndarray, not scalars, because P and U charts have a per-point-varying
    width (driven by that point's own sample size / opportunity) even when
    the centre line itself is flat -- see core/attribute_segments.py.
    """

    ucl: np.ndarray
    lcl: np.ndarray
    near_ucl: np.ndarray
    near_lcl: np.ndarray


# The near-limit zone for every 3-sigma-style Shewhart chart sits 2/3 of
# the way to the limit (the same ratio XmR's LIMIT_CLOSE embodies relative
# to LIMIT) -- a property of the 2-in-3 rule generically, per Montgomery,
# not something specific to XmR's moving-range formula. Applied here as
# NEAR_LIMIT_RATIO * 3 = 2 standard errors, mirroring
# LIMIT_CLOSE = (2/3) * LIMIT.
_ATTRIBUTE_SIGMA_MULTIPLIER = 3.0
_ATTRIBUTE_NEAR_SIGMA_MULTIPLIER = 2.0


def p_chart_limits(
        p_bar: np.ndarray | float,
        n: np.ndarray
) -> AttributeLimitStats:
    """p +/- 3*sqrt(p(1-p)/n), per point since n can vary. Lower bound
    clipped at zero -- a naive implementation can otherwise produce
    nonsensical negative lower limits on small-count data. Upper bound is
    NOT clipped at 1 -- a raw p-chart UCL can mathematically exceed 1 for
    small p_bar/small n, a known, textbook limitation of the normal
    approximation this formula relies on. Accepts scalar or per-point
    array input for p_bar, same reason as np_chart_limits: a genuinely
    varying p_bar arises whenever centre is an externally-supplied
    baseline rather than a flat pooled/windowed mean.
    """
    p_bar = np.asarray(p_bar, dtype=float)
    n = np.asarray(n, dtype=float)
    se = np.sqrt(p_bar * (1 - p_bar) / n)
    return AttributeLimitStats(
        ucl=p_bar + _ATTRIBUTE_SIGMA_MULTIPLIER * se,
        lcl=np.maximum(p_bar - _ATTRIBUTE_SIGMA_MULTIPLIER * se, 0.0),
        near_ucl=p_bar + _ATTRIBUTE_NEAR_SIGMA_MULTIPLIER * se,
        near_lcl=np.maximum(p_bar - _ATTRIBUTE_NEAR_SIGMA_MULTIPLIER * se, 0.0),
    )


def binomial_approximation_low_count(
        p_bar: np.ndarray | float,
        n: np.ndarray | float
) -> np.ndarray:
    """True where a p/np control limit's normal-approximation-to-binomial
    basis is questionable: the standard rule of thumb is that the
    approximation is trustworthy only when both n*p_bar >= 5 and
    n*(1-p_bar) >= 5 -- below that, the true binomial distribution is
    skewed enough that the nominal ~3-sigma false-alarm rate the limits
    imply may not hold. Purely a diagnostic: does not change the computed
    limits (p_chart_limits/np_chart_limits still return the same formula
    unconditionally), it only flags where they deserve a caveat -- see
    PointResult.low_count_warning.
    """
    p_bar = np.asarray(p_bar, dtype=float)
    n = np.asarray(n, dtype=float)
    return np.asarray((n * p_bar < 5) | (n * (1 - p_bar) < 5))


def np_chart_limits(
        np_bar: np.ndarray | float,
        n: np.ndarray | float
) -> AttributeLimitStats:
    """np +/- 3*sqrt(np_bar*(1-np_bar/n)), per point. Accepts scalar or
    per-point array input for np_bar/n alike -- a genuinely varying
    np_bar arises whenever centre is an externally-supplied baseline
    rather than a flat pooled/windowed mean.
    """
    np_bar = np.asarray(np_bar, dtype=float)
    n = np.asarray(n, dtype=float)
    se = np.sqrt(np_bar * (1 - np_bar / n))
    return AttributeLimitStats(
        ucl=np_bar + _ATTRIBUTE_SIGMA_MULTIPLIER * se,
        lcl=np.maximum(np_bar - _ATTRIBUTE_SIGMA_MULTIPLIER * se, 0.0),
        near_ucl=np_bar + _ATTRIBUTE_NEAR_SIGMA_MULTIPLIER * se,
        near_lcl=np.maximum(np_bar - _ATTRIBUTE_NEAR_SIGMA_MULTIPLIER * se, 0.0),
    )


def c_chart_limits(
        c_bar: np.ndarray | float
) -> AttributeLimitStats:
    """c +/- 3*sqrt(c_bar), per point. No denominator field at all: the
    "area of opportunity" is assumed constant and isn't tracked as data,
    since the formula doesn't need its value, only that it doesn't change.
    Accepts scalar or per-point array input, same reason as np_chart_limits.
    """
    c_bar = np.asarray(c_bar, dtype=float)
    se = np.sqrt(c_bar)
    return AttributeLimitStats(
        ucl=c_bar + _ATTRIBUTE_SIGMA_MULTIPLIER * se,
        lcl=np.maximum(c_bar - _ATTRIBUTE_SIGMA_MULTIPLIER * se, 0.0),
        near_ucl=c_bar + _ATTRIBUTE_NEAR_SIGMA_MULTIPLIER * se,
        near_lcl=np.maximum(c_bar - _ATTRIBUTE_NEAR_SIGMA_MULTIPLIER * se, 0.0),
    )


def u_chart_limits(
        u_bar: np.ndarray | float,
        opportunity: np.ndarray
) -> AttributeLimitStats:
    """u +/- 3*sqrt(u/opportunity), per point since opportunity can vary --
    the Poisson-rate analogue of p_chart_limits. Lower bound clipped at
    zero, same reasoning as p. Accepts scalar or per-point array input for
    u_bar, same reason as p_chart_limits.
    """
    u_bar = np.asarray(u_bar, dtype=float)
    opportunity = np.asarray(opportunity, dtype=float)
    se = np.sqrt(u_bar / opportunity)
    return AttributeLimitStats(
        ucl=u_bar + _ATTRIBUTE_SIGMA_MULTIPLIER * se,
        lcl=np.maximum(u_bar - _ATTRIBUTE_SIGMA_MULTIPLIER * se, 0.0),
        near_ucl=u_bar + _ATTRIBUTE_NEAR_SIGMA_MULTIPLIER * se,
        near_lcl=np.maximum(u_bar - _ATTRIBUTE_NEAR_SIGMA_MULTIPLIER * se, 0.0),
    )


# d2(2), the same rounded constant XmR's LIMIT (2.66 = 3/d2(2)) is built
# from -- used here for the same reason: an average-moving-range estimate
# of a series' spread is scaled by dividing by d2(2) to get back to a
# sigma. Kept as its own named constant since laney_sigma_z uses it
# directly rather than through LIMIT's 3x-scaled form.
_D2_TWO = 1.128


def laney_sigma_z(
        y: np.ndarray,
        centre: np.ndarray,
        se: np.ndarray
) -> float:
    """The dispersion-inflation factor behind Laney's p'/u' charts (Laney,
    "Improving the Robustness of Control Charts for Proportions and
    Rates", Quality Engineering, 2002) -- corrects classical p/u limits
    for overdispersion: real process variation beyond what pure binomial/
    Poisson sampling noise predicts, which becomes common with large
    sample sizes and makes the classical formula's limits unrealistically
    tight.

    Method: standardise each point, Z_i = (y_i - centre_i) / se_i, using
    the same classical standard error p_chart_limits/u_chart_limits
    already compute (binomial or Poisson). If the data genuinely follows
    that model with no extra process variation, Z behaves like an i.i.d.
    standard-normal sequence, and treating it as an XmR "individuals"
    series -- average moving range, divided by d2(2) -- converges to 1
    (no correction needed). Genuine overdispersion inflates Z's own
    moving range, giving sigma_z > 1 and correspondingly wider final
    limits (p_prime_chart_limits/u_prime_chart_limits multiply the
    classical standard error by this value).

    Floored at 1: this correction only ever widens limits for real
    overdispersion, never narrows them for apparent *under*-dispersion,
    which in a real series is more often small-sample noise in the Z
    estimate than a genuine "more consistent than sampling theory
    predicts" signal worth acting on.

    NaN (not floored to 1) when there are fewer than 2 points to form a
    moving range from -- propagates to NaN/None limits at the ChartResult
    boundary exactly like XmR's own single-point segments, rather than
    silently assuming "no correction needed" for data that couldn't
    actually establish that.

    A point whose standard error is zero (a centre line of 0, or of 1 for a
    proportion) has no defined Z. It is left out of the moving ranges, and the
    classical limit it gets is already zero-width, so a non-zero value there
    is flagged exactly as on the classical chart. If no pair of consecutive
    points has a defined Z, there is no overdispersion to estimate and the
    factor is 1 (the classical limits), not NaN: the data does not call the
    classical chart's own limits into question.
    """
    y = np.asarray(y, dtype=float)
    centre = np.asarray(centre, dtype=float)
    se = np.asarray(se, dtype=float)
    if len(y) < 2:
        return float("nan")
    z = np.full(len(y), np.nan)
    defined = se > 0
    z[defined] = (y[defined] - centre[defined]) / se[defined]
    mr_z = np.abs(np.diff(z))
    usable = mr_z[np.isfinite(mr_z)]
    if len(usable) == 0:
        return 1.0
    sigma_z = float(usable.mean()) / _D2_TWO
    return max(sigma_z, 1.0)


def p_prime_chart_limits(
        p_bar: np.ndarray,
        n: np.ndarray,
        sigma_z: float
) -> AttributeLimitStats:
    """Laney's p' chart -- identical to p_chart_limits except the standard
    error is inflated by sigma_z (laney_sigma_z) before scaling by the
    usual 3-sigma-equivalent multiplier. sigma_z=1 (no overdispersion)
    reduces this exactly to p_chart_limits.
    """
    n = np.asarray(n, dtype=float)
    se = np.sqrt(p_bar * (1 - p_bar) / n)
    width = _ATTRIBUTE_SIGMA_MULTIPLIER * sigma_z * se
    near_width = _ATTRIBUTE_NEAR_SIGMA_MULTIPLIER * sigma_z * se
    return AttributeLimitStats(
        ucl=p_bar + width,
        lcl=np.maximum(p_bar - width, 0.0),
        near_ucl=p_bar + near_width,
        near_lcl=np.maximum(p_bar - near_width, 0.0),
    )


def u_prime_chart_limits(
        u_bar: np.ndarray,
        opportunity: np.ndarray,
        sigma_z: float
) -> AttributeLimitStats:
    """Laney's u' chart -- see p_prime_chart_limits/laney_sigma_z."""
    opportunity = np.asarray(opportunity, dtype=float)
    se = np.sqrt(u_bar / opportunity)
    width = _ATTRIBUTE_SIGMA_MULTIPLIER * sigma_z * se
    near_width = _ATTRIBUTE_NEAR_SIGMA_MULTIPLIER * sigma_z * se
    return AttributeLimitStats(
        ucl=u_bar + width,
        lcl=np.maximum(u_bar - width, 0.0),
        near_ucl=u_bar + near_width,
        near_lcl=np.maximum(u_bar - near_width, 0.0),
    )


# Xbar-R constants (n: A2, D3, D4): published values (Montgomery). Cross-
# checked against NIST's e-Handbook of Statistical Methods
# (itl.nist.gov/div898/handbook/pmc/section3/pmc321.htm, "Factors for
# Calculating Limits for Xbar and R Charts") -- matches to 3 decimal
# places (NIST's D4 for n=3 is 2.575 vs this table's 2.574, a last-digit
# rounding difference between published sources). Kept as a table for
# n=2..10 rather than always computing (see _xbar_r_constants_computed
# below) since it's exact, instant, and needs no optional dependency for
# the common case.
XBAR_R_CONSTANTS: dict[int, tuple[float, float, float]] = {
    2: (1.880, 0.000, 3.267),
    3: (1.023, 0.000, 2.574),
    4: (0.729, 0.000, 2.282),
    5: (0.577, 0.000, 2.114),
    6: (0.483, 0.000, 2.004),
    7: (0.419, 0.076, 1.924),
    8: (0.373, 0.136, 1.864),
    9: (0.337, 0.184, 1.816),
    10: (0.308, 0.223, 1.777),
}

# +/- this many standard deviations for the numerical integration below --
# the standard normal density is ~1e-23 out there, so this is effectively
# -inf..inf for any n this library will realistically see.
_XBAR_R_INTEGRATION_BOUND = 10.0


def _relative_range_moments(
        n: int
) -> tuple[float, float]:
    """(d2, d3): mean and standard deviation of the relative range
    W = R/sigma for n iid N(0,1) samples, via numerical integration --
    used to extend xbar_r_constants() past XBAR_R_CONSTANTS' n=10 cap.

    d2/d3 have an exact formula as integrals over the order statistics of
    the standard normal -- the same "exact but not elementary" category
    of technique this module already uses for c4(n)'s gamma-function
    formula, just requiring numerical rather than symbolic integration.

    d2(n) = E[range] = E[max] - E[min], which reduces (via the general
    E[X] = integral of P(X>x) identity for any real-valued X) to the
    single integral:
        d2(n) = integral_{-inf}^{inf} [1 - Phi(x)^n - (1-Phi(x))^n] dx
    d3(n) = sqrt(Var(range)) = sqrt(E[range^2] - d2(n)^2), where E[range^2]
    needs the *joint* density of (min, max) -- n(n-1)[Phi(y)-Phi(x)]^(n-2)
    phi(x)phi(y) for x<y -- hence a double integral, not a second easy
    single one.

    Verified (not just derived): for n=2..10, both d2 and d3 computed this
    way reproduce XBAR_R_CONSTANTS' A2/D3/D4 to 3-4 decimal places -- see
    tests/test_subgroup_charts.py's cross-check test.

    Lazily imports scipy, not a core dependency of this library (see
    pyproject.toml's `extended` optional-dependency group) -- only needed
    for xbar_r subgroup sizes above the tabulated n=10, an uncommon case.
    """
    try:
        from scipy import integrate
        from scipy.stats import norm
    except ImportError as exc:
        raise ImportError(
            f"Xbar-R constants for subgroup size {n} (>10) require numerical "
            "integration, which needs scipy -- not a core dependency of this "
            "library since it's only needed for this uncommon case. Install "
            "with: pip install 'spc-lib-py[extended]'"
        ) from exc

    bound = _XBAR_R_INTEGRATION_BOUND

    def _d2_integrand(
            x: float
    ) -> float:
        f = norm.cdf(x)
        return 1.0 - f**n - (1.0 - f) ** n

    d2, _ = integrate.quad(_d2_integrand, -bound, bound)

    def _e_range_sq_integrand(
            y: float,
            x: float
    ) -> float:
        # dblquad's integrand signature is (inner_var, outer_var) -- y
        # (the max, inner/dependent bound) first, x (the min, outer bound)
        # second. See the dblquad call below: x ranges over a fixed
        # [-bound, bound], y ranges over [x, bound] for each x.
        return (y - x) ** 2 * n * (n - 1) * (norm.cdf(y) - norm.cdf(x)) ** (n - 2) * norm.pdf(x) * norm.pdf(y)

    e_range_sq, _ = integrate.dblquad(_e_range_sq_integrand, -bound, bound, lambda x: x, lambda x: bound)

    # Rounding error in the two independent integrals can leave this
    # fractionally negative for large n, where the true variance is small
    # -- clip rather than let a negative land inside sqrt() below.
    variance = max(e_range_sq - d2**2, 0.0)
    d3 = math.sqrt(variance)
    return d2, d3


def _xbar_r_constants_computed(
        n: int
) -> tuple[float, float, float]:
    """(A2, D3, D4) for n>10, from d2(n)/d3(n) via the same shape of
    formula xbar_s_constants() uses for A3/B3/B4 (c4 there, d2 here):
    A2 = 3/(d2*sqrt(n)), D3/D4 = 1 -/+ 3*d3/d2.
    """
    d2, d3 = _relative_range_moments(n)
    spread = 3 * d3 / d2
    a2 = 3 / (d2 * math.sqrt(n))
    return a2, max(0.0, 1 - spread), 1 + spread


def xbar_r_constants(
        n: int
) -> tuple[float, float, float]:
    """(A2, D3, D4) for subgroup size n. n=2..10 come from the tabulated,
    NIST-cross-checked XBAR_R_CONSTANTS -- exact, instant, no extra
    dependency. n>10 is computed via numerical integration
    (_xbar_r_constants_computed), which needs scipy -- an optional extra,
    lazily imported, so it's only pulled in by the uncommon large-subgroup
    case rather than becoming a hard dependency for everyone.
    """
    if n in XBAR_R_CONSTANTS:
        return XBAR_R_CONSTANTS[n]
    if n < 2:
        raise ValueError(f"subgroup size must be >= 2 for an Xbar-R chart, got {n}")
    return _xbar_r_constants_computed(n)


def c4(
        n: int
) -> float:
    """The unbiasing constant relating a sample standard deviation's
    expected value to the true sigma: E[s] = c4 * sigma. Computed from its
    exact closed form (NIST e-Handbook, pmc321.htm: "c4 factor"), not
    looked up from a table -- verified against a known reference value
    (c4(10) = 0.9727, stated explicitly on that page). Uses the Gamma
    function for the non-integer factorials in the original formula
    ((n/2 - 1)! = Gamma(n/2)).
    """
    return math.sqrt(2 / (n - 1)) * math.gamma(n / 2) / math.gamma((n - 1) / 2)


def xbar_s_constants(
        n: int
) -> tuple[float, float, float]:
    """(A3, B3, B4) for subgroup size n, derived from c4(n) via NIST's
    confirmed Xbar/S formulas (pmc321.htm):
        UCL_xbar = xbar_bar + 3*s_bar/(c4*sqrt(n))       -> A3 = 3/(c4*sqrt(n))
        UCL_s    = s_bar + 3*(s_bar/c4)*sqrt(1 - c4^2)   -> B4 = 1 + 3*sqrt(1-c4^2)/c4
        LCL_s    = s_bar - 3*(s_bar/c4)*sqrt(1 - c4^2)   -> B3 = max(0, 1 - 3*sqrt(1-c4^2)/c4)
    Computed rather than tabulated, unlike xbar_r_constants -- no closed
    form exists for A2/D3/D4 that's this clean, but this one does, and
    computing it directly means no artificial cap at n=10 the way a fixed
    lookup table would need (matches commonly published Montgomery values:
    A3(5)=1.427, B4(10)=1.716, to 3 d.p.).
    """
    c = c4(n)
    a3 = 3 / (c * math.sqrt(n))
    spread = 3 * math.sqrt(1 - c**2) / c
    return a3, max(0.0, 1 - spread), 1 + spread


def assurance_type(
        target: float,
        ucl: float,
        lcl: float,
        improvement_direction: ImprovementDirection
) -> AssuranceType | None:
    """Whether the most recent segment's limits sit consistently on the
    "good" or "bad" side of a target. Undefined (None) when
    improvement_direction is "neutral" -- there's no good/bad side to be
    consistent about.
    """
    if improvement_direction == "neutral":
        return None
    is_increasing = improvement_direction == "increase"
    if target > ucl:
        return "consistent_fail" if is_increasing else "consistent_pass"
    if target < lcl:
        return "consistent_pass" if is_increasing else "consistent_fail"
    return "inconsistent"

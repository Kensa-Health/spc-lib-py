"""Deliberately naive reference implementations, written from the definitions
rather than from the library's code, for the differential tests.

They are slow and plain on purpose: a plain loop is easy to check by eye, and
agreement with the library is only worth something if the two were written
independently.
"""

import math

import numpy as np

# d2 and c4 for subgroup sizes 2-10, as published in standard SPC tables (3-4 d.p.)
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
C4 = {2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400, 6: 0.9515, 7: 0.9594, 8: 0.9650, 9: 0.9693, 10: 0.9727}

XMR_LIMIT = 2.66
OUTLIER_MULTIPLIER = 3.267


def windows(
        n,
        mode,
        fix_n
):
    """The index range whose values estimate the centre line at each point."""
    out = []
    for i in range(n):
        if mode == "segment":
            out.append(range(n))
        elif mode == "fixed_n":
            out.append(range(min(fix_n, n)))
        elif i < min(fix_n, n):  # expanding, still warming up
            out.append(range(i + 1))
        else:
            out.append(range(min(fix_n, n)))
    return out


def xmr_window_stats(
        values,
        screen
):
    """(mean, average moving range) of one window, with the chosen outlier screen."""
    w = np.asarray(values, float)
    mean = w.mean()
    if len(w) < 2:
        return mean, math.nan
    mr = np.abs(np.diff(w))
    amr = mr.mean()
    if screen == "single_pass":
        keep = mr < OUTLIER_MULTIPLIER * amr
        amr = mr[keep].mean() if keep.any() else math.nan
    elif screen == "iterative":
        current = amr
        previous = np.zeros(len(mr), bool)
        for _ in range(50):
            excluded = mr >= OUTLIER_MULTIPLIER * current
            if (excluded == previous).all():
                break
            current = mr[~excluded].mean() if (~excluded).any() else math.nan
            previous = excluded
        amr = current
    return mean, amr


def xmr_limits(
        y,
        mode,
        fix_n,
        screen
):
    """Per-point (mean, ucl, lcl) for an XmR chart."""
    n = len(y)
    mean = np.full(n, np.nan)
    amr = np.full(n, np.nan)
    if mode == "segment":
        mean[:], amr[:] = xmr_window_stats(y, screen)
    elif mode == "fixed_n":
        mean[:], amr[:] = xmr_window_stats(y[:fix_n], screen)
    else:
        lock = min(fix_n, n)
        for i in range(lock):
            mean[i], amr[i] = xmr_window_stats(y[: i + 1], screen)
            if i == 0:
                amr[i] = math.nan
        if n > lock:
            mean[lock:], amr[lock:] = xmr_window_stats(y[:lock], screen)
    return mean, mean + XMR_LIMIT * amr, mean - XMR_LIMIT * amr


def run_length(
        option,
        n
):
    """The run length a `run_rule_length` option resolves to for a segment of n points."""
    return round(math.log2(max(n, 1))) + 3 if option == "proportional" else option


def rule_winners(
        y,
        mean,
        ucl,
        lcl,
        run_len
):
    """The winning rule per point, from the rule definitions (priority: limits,
    trend, 2-in-3 near the limit, shift).
    """
    n = len(y)
    near_upper = mean + (2 / 3) * (ucl - mean)
    near_lower = mean - (2 / 3) * (mean - lcl)
    limits_defined = ~(np.isnan(ucl) | np.isnan(lcl))
    with np.errstate(invalid="ignore"):
        outside = ((y > ucl) | (y < lcl)) & limits_defined
        side = np.sign(y - mean)
        near = (~outside) & ((y < near_lower) | (y > near_upper)) & limits_defined
    trend = np.zeros(n, bool)
    shift = np.zeros(n, bool)
    two_in_three = np.zeros(n, bool)
    for start in range(n - run_len + 1):
        window = slice(start, start + run_len)
        steps = np.sign(np.diff(y[window]))
        if (steps == 1).all() or (steps == -1).all():
            trend[window] = True
        sides = side[window]
        if sides[0] != 0 and (sides == sides[0]).all():
            shift[window] = True
    for start in range(n - 2):
        idx = [start, start + 1, start + 2]
        if near[idx].sum() >= 2 and abs(side[idx].sum()) == 3:
            for i in idx:
                if near[i]:
                    two_in_three[i] = True
    winners = []
    for i in range(n):
        if outside[i]:
            winners.append("outside_limits")
        elif trend[i]:
            winners.append("seven_point_trend")
        elif two_in_three[i]:
            winners.append("two_in_three_near_limit")
        elif shift[i]:
            winners.append("seven_point_shift")
        else:
            winners.append(None)
    return winners


def attribute_limits(
        kind,
        y,
        denominator,
        mode,
        fix_n
):
    """Per-point (plotted value, centre, ucl, lcl) for p, np, u, c, p' and u'."""
    n = len(y)
    pooled = kind in ("p", "u", "p_prime", "u_prime")
    centre = np.zeros(n)
    for i, w in enumerate(windows(n, mode, fix_n)):
        idx = list(w)
        centre[i] = y[idx].sum() / denominator[idx].sum() if pooled else y[idx].mean()
    if kind in ("p", "p_prime"):
        se = np.sqrt(centre * (1 - centre) / denominator)
        plotted = y / denominator
    elif kind in ("u", "u_prime"):
        se = np.sqrt(centre / denominator)
        plotted = y / denominator
    elif kind == "np":
        se = np.sqrt(centre * (1 - centre / denominator))
        plotted = y
    else:  # c
        se = np.sqrt(centre)
        plotted = y
    factor = 1.0
    if kind in ("p_prime", "u_prime"):
        z = np.full(n, np.nan)
        defined = se > 0
        z[defined] = (plotted[defined] - centre[defined]) / se[defined]
        moving = np.abs(np.diff(z))
        moving = moving[np.isfinite(moving)]
        factor = max(moving.mean() / 1.128, 1.0) if len(moving) else 1.0
    return plotted, centre, centre + 3 * factor * se, np.maximum(centre - 3 * factor * se, 0.0)


def subgroup_limits(
        kind,
        subgroup_size,
        means,
        spreads,
        mode,
        fix_n
):
    """Per-subgroup (centre, half-width) for an Xbar-R or Xbar-S chart."""
    k = len(means)
    centre = np.zeros(k)
    spread = np.zeros(k)
    for i, w in enumerate(windows(k, mode, fix_n)):
        idx = list(w)
        centre[i] = means[idx].mean()
        spread[i] = spreads[idx].mean()
    factor = 3 / (D2[subgroup_size] if kind == "xbar_r" else C4[subgroup_size]) / math.sqrt(subgroup_size)
    return centre, factor * spread


def lag1_autocorrelation(
        y
):
    """The lag-1 sample autocorrelation, centred on the mean."""
    y = np.asarray(y, float)
    mean = y.mean()
    denominator = ((y - mean) ** 2).sum()
    return ((y[1:] - mean) * (y[:-1] - mean)).sum() / denominator if denominator > 0 else math.nan

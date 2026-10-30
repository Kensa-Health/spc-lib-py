"""Deterministically (re)generates the CSV fixtures in this directory.

Run from the repository root: `python data/generate.py`

Each series is built to exercise one specific, well-understood SPC
behaviour -- not just "plausible-looking noise". Every generator below was
tuned by actually running it through spc.chart() and checking the result
matches its docstring's claim (see data/README.md's "Verified" column) --
none of these are just "looks about right"; the seeds were picked precisely
because they produce a clean, single-signal demonstration free of the
incidental extra flags a naive random draw usually produces (a real risk --
see core/segments.py's module docstring on SEGMENT-mode Phase I/II
conflation, which is exactly what causes that noise if you don't control
for it).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent
START = date(2024, 1, 1)  # a Monday; all series are weekly


def _weeks(
        n: int,
        start: date = START
) -> list[date]:
    return [start + timedelta(weeks=i) for i in range(n)]


def make_stable(
        n: int = 26,
        mean: float = 40.0,
        sd: float = 4.0,
        seed: int = 4
) -> pd.DataFrame:
    """Pure common-cause variation, no injected special causes. Verified:
    0 points flagged under default options (SEGMENT mode).
    """
    rng = np.random.default_rng(seed)
    values = rng.normal(mean, sd, n).round(1)
    return pd.DataFrame({"period": _weeks(n), "value": values})


def make_outlier(
        n: int = 15,
        mean: float = 40.0,
        sd: float = 4.0,
        spike_at: int = 9,
        seed: int = 2
) -> pd.DataFrame:
    """Stable series with one isolated extreme spike. Verified: with
    screen_outliers="single_pass" (the default), the spike's own moving
    ranges inflate the *unscreened* average moving range enough that a
    single pass doesn't fully exclude them (amr ~= 4.61); with "iterative",
    re-screening against the shrinking threshold converges lower (amr ~=
    3.48) -- a real, reproducible difference between the two modes, not
    just a hypothetical one (see core/limits.py's screening functions).
    """
    rng = np.random.default_rng(seed)
    values = rng.normal(mean, sd, n).round(1)
    values[spike_at] = mean + 8 * sd
    return pd.DataFrame({"period": _weeks(n), "value": values})


def make_shift(
        n_baseline: int = 10,
        n_shifted: int = 10,
        baseline_mean: float = 40.0,
        shift: float = 6.0,
        sd: float = 4.0,
        seed: int = 32
) -> pd.DataFrame:
    """A stable baseline, then a genuine sustained shift that stays within
    the *baseline's own* control limits -- only catchable via the
    seven_point_shift rule, not outside_limits. Verified with
    limit_mode="fixed_n", fix_after_n_points=10: 0 outside_limits points,
    and suggest_rebase() proposes exactly one candidate, at the true shift
    boundary (2024-03-11), with points_since=10 -- i.e. this seed was
    chosen specifically because the baseline's own last point doesn't
    coincidentally extend the shift run backward (see core/rebase.py's
    module docstring for why that's a real, not hypothetical, risk with an
    arbitrary seed). Use SEGMENT (the default) instead and the pre-shift
    period gets contaminated with spurious flags -- that contrast is the
    point of this fixture; see data/README.md.
    """
    rng = np.random.default_rng(seed)
    baseline = rng.normal(baseline_mean, sd, n_baseline)
    shifted = rng.normal(baseline_mean + shift, sd, n_shifted)
    values = np.concatenate([baseline, shifted]).round(1)
    return pd.DataFrame({"period": _weeks(n_baseline + n_shifted), "value": values})


def make_trend(
        n_before: int = 8,
        n_trend: int = 7,
        n_after: int = 5,
        before_mean: float = 40.0,
        seed: int = 56
) -> pd.DataFrame:
    """A monotonically increasing run long enough to trip seven_point_trend,
    embedded in stable noise before and after it. Verified with
    limit_mode="fixed_n", fix_after_n_points=8 (locking limits from the
    "before" period only): every flagged point is seven_point_trend, no
    other rule fires -- the trend stays within the baseline-derived limits
    throughout, so this isolates the trend rule cleanly rather than mixing
    it with outside_limits/shift signals the way a wider or steeper trend
    (or SEGMENT-mode limits) would.
    """
    rng = np.random.default_rng(seed)
    before = rng.normal(before_mean, 3.0, n_before)
    trend = (before_mean + 1) + np.linspace(0, 6, n_trend) + rng.normal(0, 0.5, n_trend)
    after = rng.normal(before_mean + 7, 3.0, n_after)
    values = np.concatenate([before, trend, after]).round(1)
    return pd.DataFrame({"period": _weeks(n_before + n_trend + n_after), "value": values})


def make_two_facets(
        n: int = 20,
        north_mean: float = 50.0,
        sd: float = 5.0,
        shift: float = 10.0,
        seed_north: int = 1,
        seed_south: int = 3
) -> pd.DataFrame:
    """Two facets (teams/orgs) in one long-format file, sharing the same
    date range and fix_after_n_points=10 baseline -- demonstrates
    facet_field, and that each facet's limits are computed independently.
    Verified: "north" (stable throughout) gets 0 flagged points; "south"
    (shifts upward halfway through) gets 11 flagged points, all
    seven_point_shift, none outside_limits.
    """
    rng_n = np.random.default_rng(seed_north)
    north = rng_n.normal(north_mean, sd, n).round(1)

    rng_s = np.random.default_rng(seed_south)
    s_base = rng_s.normal(north_mean, sd, n // 2)
    s_shifted = rng_s.normal(north_mean + shift, sd, n - n // 2)
    south = np.concatenate([s_base, s_shifted]).round(1)

    return pd.DataFrame(
        {
            "period": _weeks(n) * 2,
            "value": np.concatenate([north, south]),
            "org": ["north"] * n + ["south"] * n,
        }
    )


def make_rebase(
        n_before: int = 12,
        n_after: int = 12,
        seed: int = 6
) -> pd.DataFrame:
    """A single series with a real, known process change at a specific
    date -- demonstrates the `rebase` option itself (as opposed to
    suggest_rebase(), which *proposes* a date like this one from the
    data). Verified: without `rebase` set, SEGMENT mode pools both regimes
    into one calculation and floods the series with spurious flags (23 of
    24 points); with `rebase=[<the known date>]`, the two regimes are
    judged independently and only 1 of 24 points is flagged -- a direct,
    reproducible illustration of core/segments.py's Phase I/II caveat.
    """
    rng = np.random.default_rng(seed)
    before = rng.normal(30.0, 3.0, n_before)
    after = rng.normal(18.0, 2.0, n_after)  # a real improvement after a known intervention
    values = np.concatenate([before, after]).round(1)
    df = pd.DataFrame({"period": _weeks(n_before + n_after), "value": values})
    df.attrs["rebase_date"] = _weeks(n_before + n_after)[n_before].isoformat()
    return df


def main() -> None:
    make_stable().to_csv(OUT_DIR / "xmr_stable.csv", index=False)
    make_outlier().to_csv(OUT_DIR / "xmr_outlier.csv", index=False)
    make_shift().to_csv(OUT_DIR / "xmr_shift.csv", index=False)
    make_trend().to_csv(OUT_DIR / "xmr_trend.csv", index=False)
    make_two_facets().to_csv(OUT_DIR / "xmr_two_facets.csv", index=False)

    rebase_df = make_rebase()
    rebase_date = rebase_df.attrs["rebase_date"]
    rebase_df.to_csv(OUT_DIR / "xmr_rebase.csv", index=False)
    print(f"xmr_rebase.csv's known rebase date: {rebase_date} (also recorded in data/README.md)")

    print(f"Wrote 6 fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()

"""Runnable demo: one chart per SPC chart kind (xmr, p, u, np, c, xbar_r,
xbar_s, t, g) implemented in this library.

Run from the repository root:
    python scripts/demo_ae_admissions.py

Writes one HTML file per chart kind plus an index.html linking all of
them, opens the index in your browser, and prints a short summary to the
console for each. Needs the 'render' extra:
    pip install -e ".[render]"

Real vs. synthetic data, by design, not convenience: xmr, p, xbar_r, and
xbar_s use the real pinned NHS A&E dataset (tests/golden/nhsrdatasets/,
see scripts/fetch_nhsrdatasets.py). u, np, c, t, and g do NOT -- the
pinned dataset's only numeric columns (attendances/breaches/admissions)
are all mutually *bounded* (breaches and admissions can't exceed
attendances), so any ratio among them is honestly a p-chart relationship,
not u/c/np. Forcing them into those shapes just to use "real data
everywhere" would be exactly the chart-type misapplication
suggest_chart_type() exists to catch. Those five use small, clearly
labelled, hand-constructed illustrative series instead -- realistic in
shape (u's mirrors a classic "falls per 1,000 bed-days" example), but
invented, not measured.
"""

from __future__ import annotations

import webbrowser
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import spc
from spc.core.models import BaseChartOptions, LimitMode, ProportionChartOptions

DATA_PATH = Path(__file__).parent.parent / "tests" / "golden" / "nhsrdatasets" / "ae_attendances.parquet"
OUTPUT_DIR = Path(__file__).parent / "demo_output"

# Trust RJZ, type "1" (major A&E) -- a complete, gap-free 36-month series
# (Apr 2016-Mar 2019), verified in tests/test_nhs_fixtures.py. 130 of the
# dataset's 274 org_codes have equally complete type-"1" data; a few more:
# R1F, RA2, RAL, RBK, RCB, RDU, RHM, RWA.
ORG_CODE = "RJZ"
AE_TYPE = "1"

# 5 trusts, all verified to report every one of the 36 months for type "1"
# with no missing values -- used as an Xbar-R subgroup (subgroup size <= 10).
XBAR_R_TRUSTS = ["RJZ", "R1F", "RA2", "RAL", "RBK"]

# 12 trusts, same verification, for Xbar-S (subgroup size > 10).
XBAR_S_TRUSTS = ["R1F", "R1H", "R1K", "RA2", "RA3", "RA4", "RA7", "RA9", "RAE", "RAJ", "RAL", "RAP"]


def _weeks(
        n: int,
        start: date = date(2024, 1, 1)
) -> list[date]:
    return [start + timedelta(weeks=i) for i in range(n)]


# Real-data charts (xmr, p, xbar_r, xbar_s) deliberately use the
# default limit_mode="segment" and flag heavily -- confirmed this is NOT
# a Phase I/II baseline artifact fixable with fixed_n (checked: still
# 33/36 for `p` with a 12-month fixed_n baseline). It's genuine
# overdispersion: tens of thousands of attendances/month makes binomial
# sampling error tiny, so real operational variation -- much larger than
# pure sampling noise -- blows through the limits regardless of baseline
# choice. An honest, real finding, not a bug -- printed once below rather
# than hidden.
_OVERDISPERSION_NOTE = (
    "Real administrative data at this volume typically flags heavily on p/xbar-style charts: "
    "sample sizes are large enough that binomial/subgroup sampling error is tiny, so genuine "
    "operational and seasonal variation (much bigger than sampling noise) reads as 'special cause' "
    "almost every period. Confirmed this isn't a baseline artifact (still true with a 12-month "
    "fixed_n baseline) -- it's a real property of applying tight statistical limits to high-volume data."
)


def demo_xmr(
        ae: pd.DataFrame
) -> tuple[spc.ChartResult, str]:
    subset = ae[(ae["org_code"] == ORG_CODE) & (ae["type"] == AE_TYPE)].sort_values("period").reset_index(drop=True)
    options = BaseChartOptions(value_field="admissions", date_field="period", improvement_direction="neutral")
    result = spc.chart(subset, kind="xmr", options=options)
    return result, f"Monthly admissions, trust {ORG_CODE} type {AE_TYPE} (real NHS data)"


def demo_p(
        ae: pd.DataFrame
) -> tuple[spc.ChartResult, str]:
    subset = ae[(ae["org_code"] == ORG_CODE) & (ae["type"] == AE_TYPE)].sort_values("period").reset_index(drop=True)
    options = ProportionChartOptions(
        value_field="breaches", date_field="period", denominator_field="attendances", improvement_direction="decrease"
    )
    result = spc.chart(subset, kind="p", options=options)
    return result, f"4-hour breach rate (breaches / attendances), trust {ORG_CODE} type {AE_TYPE} (real NHS data)"


def demo_xbar_r(
        ae: pd.DataFrame
) -> tuple[spc.ChartResult, str]:
    subset = ae[(ae["org_code"].isin(XBAR_R_TRUSTS)) & (ae["type"] == AE_TYPE)]
    options = BaseChartOptions(value_field="admissions", date_field="period", improvement_direction="neutral")
    result = spc.chart(subset, kind="xbar_r", options=options)
    return result, f"Admissions across {len(XBAR_R_TRUSTS)} trusts as a monthly subgroup (real NHS data)"


def demo_xbar_s(
        ae: pd.DataFrame
) -> tuple[spc.ChartResult, str]:
    subset = ae[(ae["org_code"].isin(XBAR_S_TRUSTS)) & (ae["type"] == AE_TYPE)]
    options = BaseChartOptions(value_field="admissions", date_field="period", improvement_direction="neutral")
    result = spc.chart(subset, kind="xbar_s", options=options)
    return result, f"Admissions across {len(XBAR_S_TRUSTS)} trusts as a monthly subgroup (real NHS data)"


# Synthetic demos (u, np, c, t, g) all use limit_mode="fixed_n" with a
# baseline window ending right before the one injected spike in each
# series -- confirmed this drops each from "most points flagged" (the
# segment-mode default pools the spike into its own baseline estimate,
# see core/segments.py's Phase I/II caveat) down to exactly the one
# intended signal. Verified per-chart before writing these in.


def demo_u() -> tuple[spc.ChartResult, str]:
    # A classic u-chart example: falls per 1,000 bed-days, bed-days
    # varying month to month. Illustrative, not measured -- see this
    # module's docstring.
    dates = _weeks(15)
    bed_days = [980, 1050, 1120, 990, 1200, 1080, 950, 1300, 1150, 1000, 1080, 1220, 990, 1100, 1050]
    falls = [3, 3, 4, 3, 4, 3, 2, 15, 4, 3, 3, 4, 3, 3, 3]  # week 8 (index 7): a real spike
    df = pd.DataFrame({"period": dates, "falls": falls, "bed_days": bed_days})
    options = ProportionChartOptions(
        value_field="falls",
        date_field="period",
        denominator_field="bed_days",
        improvement_direction="decrease",
        limit_mode=LimitMode.FIXED_N,
        fix_after_n_points=7,
    )
    result = spc.chart(df, kind="u", options=options)
    return result, "Falls per 1,000 bed-days, varying occupancy (illustrative, not real data)"


def demo_np() -> tuple[spc.ChartResult, str]:
    dates = _weeks(15)
    readmissions = [4, 6, 5, 7, 4, 6, 5, 13, 6, 5, 7, 4, 6, 5, 6]  # week 8: a real spike
    df = pd.DataFrame({"period": dates, "readmissions": readmissions, "discharges": [50] * 15})
    options = ProportionChartOptions(
        value_field="readmissions",
        date_field="period",
        denominator_field="discharges",
        improvement_direction="decrease",
        limit_mode=LimitMode.FIXED_N,
        fix_after_n_points=7,
    )
    result = spc.chart(df, kind="np", options=options)
    return result, "30-day readmissions out of a constant 50 discharges/week (illustrative, not real data)"


def demo_c() -> tuple[spc.ChartResult, str]:
    dates = _weeks(15)
    falls = [3, 5, 4, 6, 2, 4, 5, 3, 4, 14, 5, 3, 4, 5, 3]  # week 10: a real spike
    df = pd.DataFrame({"period": dates, "falls": falls})
    options = BaseChartOptions(
        value_field="falls",
        date_field="period",
        improvement_direction="decrease",
        limit_mode=LimitMode.FIXED_N,
        fix_after_n_points=9,
    )
    result = spc.chart(df, kind="c", options=options)
    return result, "Weekly ward falls, assuming roughly constant ward capacity (illustrative, not real data)"


def demo_t() -> tuple[spc.ChartResult, str]:
    gap_days = [45, 60, 38, 52, 70, 65, 58, 3, 61, 55]  # event 8: a real cluster (3 days, not ~55)
    event_dates = [date(2024, 1, 1)]
    for g in gap_days:
        event_dates.append(event_dates[-1] + timedelta(days=g))
    df = pd.DataFrame({"incident": event_dates, "y": range(len(event_dates)), "period": event_dates})
    options = BaseChartOptions(
        value_field="incident",
        date_field="incident",
        improvement_direction="increase",
        limit_mode=LimitMode.FIXED_N,
        fix_after_n_points=7,
    )
    result = spc.chart(df, kind="t", options=options)
    return result, "Days between never-events -- longer gaps are better (illustrative, not real data)"


def demo_g() -> tuple[spc.ChartResult, str]:
    dates = _weeks(10)
    patients_between = [120, 150, 95, 140, 110, 135, 105, 8, 145, 125]  # event 8: a real cluster
    df = pd.DataFrame({"period": dates, "patients_seen": patients_between})
    options = BaseChartOptions(
        value_field="patients_seen",
        date_field="period",
        improvement_direction="increase",
        limit_mode=LimitMode.FIXED_N,
        fix_after_n_points=7,
    )
    result = spc.chart(df, kind="g", options=options)
    return result, "Patients seen between never-events -- more is better (illustrative, not real data)"


def _print_summary(
        kind: str,
        result: spc.ChartResult
) -> None:
    flagged = [p for p in result.points if p.special_cause_rule]
    print(f"\n[{kind}] {len(flagged)} of {len(result.points)} points flagged:")
    for p in flagged:
        print(f"    {p.x}  {p.y:>10,.3f}  {p.special_cause_rule}  ({p.point_type})")
    suggestions = spc.suggest_rebase(result)
    for s in suggestions:
        print(f"    suggest_rebase: {s.candidate_date} ({s.rule}, {s.points_since} pts, {s.confidence})")


def main() -> None:
    ae = pd.read_parquet(DATA_PATH)

    print("Real-data charts (xmr, p, xbar_r, xbar_s) below flag heavily -- why:")
    print(f"  {_OVERDISPERSION_NOTE}\n")

    demos = {
        "xmr": demo_xmr(ae),
        "p": demo_p(ae),
        "xbar_r": demo_xbar_r(ae),
        "xbar_s": demo_xbar_s(ae),
        "u": demo_u(),
        "np": demo_np(),
        "c": demo_c(),
        "t": demo_t(),
        "g": demo_g(),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for kind, (result, description) in demos.items():
        _print_summary(kind, result)
        fig = spc.render.plotly(result, title=f"{kind.upper()} -- {description}")
        out_path = OUTPUT_DIR / f"chart_{kind}.html"
        fig.write_html(out_path, include_plotlyjs="cdn")
        index_rows.append((kind, description, out_path.name))

    index_html = ["<html><head><meta charset='utf-8'><title>spc demo</title></head><body>"]
    index_html.append("<h1>spc chart kind demo</h1><ul>")
    for kind, description, filename in index_rows:
        index_html.append(f"<li><a href='{filename}'>{kind}</a> -- {description}</li>")
    index_html.append("</ul></body></html>")
    index_path = OUTPUT_DIR / "index.html"
    index_path.write_text("\n".join(index_html))

    print(f"\n{len(demos)} charts written to {OUTPUT_DIR}")
    webbrowser.open(index_path.resolve().as_uri())


if __name__ == "__main__":
    main()

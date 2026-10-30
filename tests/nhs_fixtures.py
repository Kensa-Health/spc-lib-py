"""Typed accessors for the pinned NHSRdatasets snapshot in
tests/golden/nhsrdatasets/. Dev/test-only -- not imported by spc/core or
spc/charts, and not a runtime dependency of the package (see
scripts/fetch_nhsrdatasets.py's docstring and pyproject.toml's `dev` extra).

`ons_mortality.counts` is already float64 with pyreadr's own rda-to-Python
conversion (51 genuine NaNs, from source rows that held text rather than a
number) -- a `pd.to_numeric(..., errors="coerce")` coercion step isn't
needed here, since running it added no NaNs beyond what were already
there when checked.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

FIXTURE_DIR = Path(__file__).parent / "golden" / "nhsrdatasets"


@lru_cache
def load_ae_attendances() -> pd.DataFrame:
    """NHS England A&E attendances/admissions, monthly, Apr 2016-Mar 2019.
    Columns: period (date), org_code (str), type ("1"/"2"/"other"),
    attendances (int-valued float), breaches (int-valued float),
    admissions (int-valued float). No missing values in any of these three.
    """
    return pd.read_parquet(FIXTURE_DIR / "ae_attendances.parquet")


@lru_cache
def load_ons_mortality() -> pd.DataFrame:
    """Weekly deaths registered, England & Wales, from Jan 2010. Columns:
    category_1/category_2 (grouping; category_1 can be None), counts
    (float, 51 of 18803 rows are NaN -- rows that held a non-numeric value
    at source), date (date, always a Friday), week_no (int).
    """
    return pd.read_parquet(FIXTURE_DIR / "ons_mortality.parquet")


@lru_cache
def load_ons_uk_population_2023() -> pd.DataFrame:
    """ONS mid-2023 population estimate, single-year-of-age. Columns: sex
    ("females"/"males" -- plural, not "female"/"male"), Code, Name (7
    geography rollups: UK, Great Britain, England and Wales, England,
    Wales, Scotland, Northern Ireland -- not just "UNITED KINGDOM"),
    Geography ("Country" for every row in this snapshot), age ("0".."89",
    "90+"), count (int-valued float).

    Single snapshot year only -- not a multi-year series, so it exercises
    an indirect-standardisation calculation but can't demonstrate a real
    demographic trend.
    """
    return pd.read_parquet(FIXTURE_DIR / "ons_uk_population_2023.parquet")

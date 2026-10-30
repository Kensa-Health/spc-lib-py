"""Real-data sanity tests against the pinned NHSRdatasets snapshot
(tests/golden/nhsrdatasets/, via tests/nhs_fixtures.py). Complements the
purely synthetic fixtures in data/ -- this is "does the pipeline hold up on
messy real-world data with genuine seasonality," not a replacement for the
precise, hand-constructed unit tests elsewhere in this suite. No network or
pyreadr needed here: the parquet snapshots are already pinned and committed
(see scripts/fetch_nhsrdatasets.py, run once, separately).
"""

from __future__ import annotations

from nhs_fixtures import load_ae_attendances, load_ons_mortality, load_ons_uk_population_2023

import spc
from spc.core.models import BaseChartOptions


def _rjz_type1():
    df = load_ae_attendances()
    subset = df[(df["org_code"] == "RJZ") & (df["type"] == "1")].sort_values("period").reset_index(drop=True)
    assert len(subset) == 36  # Apr 2016 - Mar 2019, one row per month, no gaps
    return subset


def test_ae_attendances_loads_with_expected_shape():
    df = load_ae_attendances()
    assert list(df.columns) == ["period", "org_code", "type", "attendances", "breaches", "admissions"]
    assert df["attendances"].isna().sum() == 0
    assert df["breaches"].isna().sum() == 0
    assert set(df["type"].unique()) == {"1", "2", "other"}


def test_xmr_runs_cleanly_on_real_attendances_series():
    subset = _rjz_type1()
    options = BaseChartOptions(value_field="attendances", date_field="period", improvement_direction="neutral")
    result = spc.chart(subset, kind="xmr", options=options)

    assert len(result.points) == 36
    assert len(result.segments) == 1
    assert result.segments[0].short_segment_warning is False  # 36 >= the default 13-point threshold
    # Not asserting 0 flags as a hard invariant -- real operational data can
    # legitimately have special-cause points. What matters here is that the
    # pipeline runs end to end on real, non-synthetic data without error and
    # produces a valid, JSON-serialisable result.
    result.model_dump_json()


def test_suggest_chart_type_correctly_identifies_a_real_proportion_metric():
    # breaches is a genuine count-out-of-a-varying-denominator relationship
    # against attendances -- confirms suggest_chart_type()'s heuristic
    # (core/validation.py) holds on real data, not just constructed cases.
    subset = _rjz_type1()
    suggestion = spc.suggest_chart_type(subset, "breaches", denominator_field="attendances", date_field="period")
    assert suggestion.kind == "p"
    assert suggestion.confidence == "high"


def test_ons_mortality_loads_with_expected_shape():
    df = load_ons_mortality()
    assert {"category_1", "category_2", "counts", "date", "week_no"} <= set(df.columns)
    # 51 genuine NaNs at source (non-numeric rows) -- not a loader bug if
    # this count changes with a re-fetch, but it shouldn't silently become 0
    # or the whole column (which would suggest the coercion broke).
    assert 0 < df["counts"].isna().sum() < len(df)


def test_ons_uk_population_2023_loads_with_expected_shape():
    df = load_ons_uk_population_2023()
    assert {"sex", "Code", "Name", "Geography", "age", "count"} <= set(df.columns)
    assert set(df["sex"].unique()) == {"females", "males"}
    assert "90+" in set(df["age"].unique())

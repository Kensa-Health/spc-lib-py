from datetime import date, timedelta

import pandas as pd
import pytest

from spc.core.limits import compute_limit_stats
from spc.core.models import BaseChartOptions, ExternalBaseline, LimitMode
from spc.core.segments import assign_segments, build_segment_arrays


def _dates(
        n,
        start=date(2024, 1, 1)
):
    return [start + timedelta(weeks=i) for i in range(n)]


def test_assign_segments_no_facet_no_rebase():
    df = pd.DataFrame({"period": _dates(5), "y": [1, 2, 3, 4, 5]})
    options = BaseChartOptions(value_field="y", date_field="period")
    out = assign_segments(df, options)
    assert out["segment_id"].nunique() == 1
    assert len(out) == 5


def test_assign_segments_rebase_splits_into_two_segments():
    dates = _dates(6)
    df = pd.DataFrame({"period": dates, "y": [1, 2, 3, 4, 5, 6]})
    options = BaseChartOptions(value_field="y", date_field="period", rebase=[dates[3]])
    out = assign_segments(df, options)
    assert out["segment_id"].nunique() == 2
    # first 3 rows before the rebase date are segment 0, rest are segment 1
    seg0 = out[out["period"] < dates[3]]
    seg1 = out[out["period"] >= dates[3]]
    assert seg0["segment_id"].nunique() == 1
    assert seg1["segment_id"].nunique() == 1
    assert seg0["segment_id"].iloc[0] != seg1["segment_id"].iloc[0]


def test_assign_segments_rebase_matches_when_date_column_is_pandas_timestamp():
    # pd.Timestamp != datetime.date even for the same calendar date -- a
    # dataframe loaded the ordinary way, via pd.read_csv(parse_dates=[...]),
    # has a Timestamp-typed date column, while `rebase` is validated by
    # pydantic into plain `date`. Regression test for that mismatch.
    dates = _dates(6)
    df = pd.DataFrame({"period": pd.to_datetime(dates), "y": [1, 2, 3, 4, 5, 6]})
    assert pd.api.types.is_datetime64_any_dtype(df["period"])  # sanity: genuinely Timestamp-typed
    options = BaseChartOptions(value_field="y", date_field="period", rebase=[dates[3]])
    out = assign_segments(df, options)
    assert out["segment_id"].nunique() == 2


def test_baseline_lookup_matches_when_date_column_is_pandas_timestamp():
    dates = _dates(4)
    df = pd.DataFrame({"period": pd.to_datetime(dates), "y": [10.0, 11.0, 9.0, 12.0]})
    baseline = [ExternalBaseline(facet=None, x=d, expected_value=0.0, source="test") for d in dates]
    options = BaseChartOptions(value_field="y", date_field="period", baseline=baseline)
    segments = build_segment_arrays(assign_segments(df, options), options)
    assert len(segments) == 1
    assert segments[0].centre_line.notna().all()


def test_assign_segments_per_facet_rebase_dict():
    dates = _dates(4)
    df = pd.DataFrame(
        {
            "period": dates * 2,
            "y": [1, 2, 3, 4, 10, 20, 30, 40],
            "org": ["A"] * 4 + ["B"] * 4,
        }
    )
    options = BaseChartOptions(
        value_field="y", date_field="period", facet_field="org", rebase={"A": [dates[2]], "B": []}
    )
    out = assign_segments(df, options)
    a = out[out["org"] == "A"]
    b = out[out["org"] == "B"]
    assert a["segment_id"].nunique() == 2  # A was rebased
    assert b["segment_id"].nunique() == 1  # B was not


def test_build_segment_arrays_segment_mode_matches_whole_segment_mean():
    dates = _dates(10)
    values = [10, 12, 11, 13, 9, 14, 10, 11, 12, 13]
    df = pd.DataFrame({"period": dates, "y": values})
    options = BaseChartOptions(value_field="y", date_field="period", limit_mode=LimitMode.SEGMENT)
    segmented = assign_segments(df, options)
    [seg] = build_segment_arrays(segmented, options)

    expected = compute_limit_stats(pd.Series(values, dtype=float), "single_pass")
    assert seg.centre_line.iloc[0] == pytest.approx(expected.mean)
    assert seg.centre_line.nunique() == 1  # flat across the whole segment
    assert seg.ucl.iloc[0] == pytest.approx(expected.ucl)


def test_build_segment_arrays_fixed_n_locks_after_n_points():
    dates = _dates(10)
    values = [10, 12, 11, 13, 9, 100, 100, 100, 100, 100]  # last 5 wildly different
    df = pd.DataFrame({"period": dates, "y": values})
    options = BaseChartOptions(value_field="y", date_field="period", limit_mode=LimitMode.FIXED_N, fix_after_n_points=5)
    segmented = assign_segments(df, options)
    [seg] = build_segment_arrays(segmented, options)

    expected = compute_limit_stats(pd.Series(values[:5], dtype=float), "single_pass")
    assert seg.centre_line.iloc[0] == pytest.approx(expected.mean)
    # the wild last-5 values must NOT have pulled the mean toward them
    assert seg.centre_line.nunique() == 1
    assert seg.centre_line.iloc[-1] == pytest.approx(expected.mean)


def test_build_segment_arrays_expanding_matches_fixed_n_after_lock_point():
    dates = _dates(8)
    values = [10, 12, 11, 13, 9, 50, 50, 50]
    df = pd.DataFrame({"period": dates, "y": values})

    expanding_options = BaseChartOptions(
        value_field="y", date_field="period", limit_mode=LimitMode.EXPANDING, fix_after_n_points=5
    )
    fixed_options = BaseChartOptions(
        value_field="y", date_field="period", limit_mode=LimitMode.FIXED_N, fix_after_n_points=5
    )
    segmented_e = assign_segments(df, expanding_options)
    segmented_f = assign_segments(df, fixed_options)
    [seg_e] = build_segment_arrays(segmented_e, expanding_options)
    [seg_f] = build_segment_arrays(segmented_f, fixed_options)

    # After the lock point (index 4, 0-based), expanding behaves exactly like fixed_n.
    assert seg_e.centre_line.iloc[5:].tolist() == pytest.approx(seg_f.centre_line.iloc[5:].tolist())
    assert seg_e.ucl.iloc[5:].tolist() == pytest.approx(seg_f.ucl.iloc[5:].tolist())

    # During warm-up, the expanding mean at point i should equal the plain
    # mean of points[0:i+1] (single point -> mean equals itself, and no
    # limit width since amr is undefined for n=1).
    assert seg_e.centre_line.iloc[0] == pytest.approx(values[0])
    expected_mean_at_2 = pd.Series(values[:3], dtype=float).mean()
    assert seg_e.centre_line.iloc[2] == pytest.approx(expected_mean_at_2)


def test_baseline_seam_reduces_to_flat_mean_when_baseline_is_zero():
    dates = _dates(6)
    values = [10, 12, 11, 13, 9, 14]
    df = pd.DataFrame({"period": dates, "y": values})
    baseline = [ExternalBaseline(facet=None, x=d, expected_value=0.0, source="test") for d in dates]

    plain_options = BaseChartOptions(value_field="y", date_field="period")
    baseline_options = BaseChartOptions(value_field="y", date_field="period", baseline=baseline)

    seg_plain = build_segment_arrays(assign_segments(df, plain_options), plain_options)[0]
    seg_baseline = build_segment_arrays(assign_segments(df, baseline_options), baseline_options)[0]

    assert seg_plain.centre_line.tolist() == pytest.approx(seg_baseline.centre_line.tolist())
    assert seg_plain.ucl.tolist() == pytest.approx(seg_baseline.ucl.tolist())


def test_baseline_seam_centres_on_expected_value_plus_bias():
    dates = _dates(6)
    values = [110, 112, 111, 113, 109, 114]  # a flat +100 offset from a 10-14 pattern
    expected_values = [100, 100, 100, 100, 100, 100]
    df = pd.DataFrame({"period": dates, "y": values})
    baseline = [
        ExternalBaseline(facet=None, x=d, expected_value=ev, source="test")
        for d, ev in zip(dates, expected_values, strict=True)
    ]
    options = BaseChartOptions(value_field="y", date_field="period", baseline=baseline)
    seg = build_segment_arrays(assign_segments(df, options), options)[0]

    residual = pd.Series(values, dtype=float) - pd.Series(expected_values, dtype=float)
    bias = residual.mean()
    assert seg.centre_line.iloc[0] == pytest.approx(100 + bias)


def test_baseline_missing_coverage_raises():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates, "y": [1, 2, 3]})
    baseline = [ExternalBaseline(facet=None, x=dates[0], expected_value=0.0, source="test")]  # only 1 of 3 covered
    options = BaseChartOptions(value_field="y", date_field="period", baseline=baseline)
    segmented = assign_segments(df, options)
    with pytest.raises(ValueError, match="no entry for facet"):
        build_segment_arrays(segmented, options)

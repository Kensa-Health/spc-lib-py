from datetime import date, timedelta

import pandas as pd

from spc.core.validation import suggest_chart_type


def _dates(
        n,
        start=date(2024, 1, 1)
):
    return [start + timedelta(weeks=i) for i in range(n)]


def test_suggests_p_when_bounded_by_varying_denominator():
    df = pd.DataFrame({"defects": [2, 3, 1, 4], "n": [100, 120, 90, 150]})
    suggestion = suggest_chart_type(df, "defects", denominator_field="n")
    assert suggestion.kind == "p"


def test_suggests_np_when_denominator_constant():
    df = pd.DataFrame({"defects": [2, 3, 1, 4], "n": [100, 100, 100, 100]})
    suggestion = suggest_chart_type(df, "defects", denominator_field="n")
    assert suggestion.kind == "np"


def test_suggests_u_when_value_not_bounded_by_opportunity():
    # falls (numerator) can exceed bed-days count on a small unit -- not a
    # proportion-of-sample relationship, so this should NOT be read as 'p'.
    df = pd.DataFrame({"falls": [5, 8, 3, 9], "bed_days": [4, 6, 2, 7]})
    suggestion = suggest_chart_type(df, "falls", denominator_field="bed_days")
    assert suggestion.kind == "u"


def test_suggests_c_when_opportunity_constant_and_unbounded():
    df = pd.DataFrame({"falls": [5, 8, 3, 9], "beds": [4, 4, 4, 4]})
    suggestion = suggest_chart_type(df, "falls", denominator_field="beds")
    assert suggestion.kind == "c"


def test_suggests_t_for_datetime_value_field():
    df = pd.DataFrame({"incident_at": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])})
    suggestion = suggest_chart_type(df, "incident_at")
    assert suggestion.kind == "t"


def test_never_suggests_g_for_integer_counts_without_a_denominator():
    # Integer counts between rare events are indistinguishable from
    # c/u counts by dtype alone -- g must be chosen explicitly.
    df = pd.DataFrame({"cases_between": [10, 20, 15]})
    suggestion = suggest_chart_type(df, "cases_between")
    assert suggestion.kind != "g"


def test_suggests_xbar_r_for_small_subgroups():
    dates = _dates(3)
    df = pd.DataFrame({"period": dates * 3, "y": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    suggestion = suggest_chart_type(df, "y", date_field="period")
    assert suggestion.kind == "xbar_r"


def test_suggests_xbar_s_for_large_subgroups():
    dates = _dates(2)
    df = pd.DataFrame({"period": dates * 12, "y": list(range(24))})
    suggestion = suggest_chart_type(df, "y", date_field="period")
    assert suggestion.kind == "xbar_s"


def test_defaults_to_xmr():
    dates = _dates(10)
    df = pd.DataFrame({"period": dates, "y": list(range(10))})
    suggestion = suggest_chart_type(df, "y", date_field="period")
    assert suggestion.kind == "xmr"


def test_never_raises_only_suggests():
    # No denominator/event/date at all -- still returns a suggestion, not an error.
    df = pd.DataFrame({"y": [1, 2, 3]})
    suggestion = suggest_chart_type(df, "y")
    assert suggestion.kind == "xmr"
    assert suggestion.confidence in ("low", "medium", "high")

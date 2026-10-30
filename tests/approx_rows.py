"""Row-wise approximate comparison for the tests.

pytest 9 refuses ``pytest.approx`` on a list of tuples ("does not support nested data
structures"), so tests that compare ``(x, y, mean, ucl, lcl)`` rows use this instead.
Numbers are compared with ``pytest.approx`` (default tolerances, as before); dates, ``None``
and any other value must be equal exactly.
"""

import math
from collections.abc import Sequence
from typing import Any

import pytest


def _same(
        got: Any,
        want: Any
) -> bool:
    if got is None or want is None:
        return got is want
    if isinstance(got, (int, float)) and isinstance(want, (int, float)) and not isinstance(got, bool):
        if math.isnan(want):
            return math.isnan(got)
        return bool(got == pytest.approx(want))
    return bool(got == want)


def assert_rows_approx(
        got: Sequence[Sequence[Any]],
        want: Sequence[Sequence[Any]]
) -> None:
    assert len(got) == len(want), f"{len(got)} rows, expected {len(want)}"
    for i, (g_row, w_row) in enumerate(zip(got, want, strict=True)):
        assert len(g_row) == len(w_row), f"row {i}: {len(g_row)} fields, expected {len(w_row)}"
        for j, (g, w) in enumerate(zip(g_row, w_row, strict=True)):
            assert _same(g, w), f"row {i}, field {j}: {g!r} != {w!r}\n  got:  {g_row!r}\n  want: {w_row!r}"

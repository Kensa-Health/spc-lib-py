"""Result model (ChartResult, ParetoResult, CusumResult) -> plain JSON/dict -- for apps that don't import Python at
all (e.g. a JS frontend calling an API that wraps
this library). Pydantic already gives this for free; this module exists so
the seam has a stable, discoverable name rather than requiring every
consumer to know `.model_dump_json()`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def to_json(
        result: BaseModel,
        **kwargs: Any
) -> str:
    """Equivalent to `result.model_dump_json(**kwargs)`."""
    return result.model_dump_json(**kwargs)


def to_dict(
        result: BaseModel,
        **kwargs: Any
) -> dict[str, Any]:
    """Equivalent to `result.model_dump(mode="json", **kwargs)` -- a
    JSON-safe plain dict (dates/datetimes as ISO strings), not Python-native types.
    """
    return result.model_dump(mode="json", **kwargs)

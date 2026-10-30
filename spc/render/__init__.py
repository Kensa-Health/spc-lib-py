"""Optional convenience rendering layer.

Deliberately thin and separate from core/ and charts/, which have zero
plotting dependencies. Importing this (directly, or via the lazy
`spc.render` accessor in spc/__init__.py) requires the 'render' extra:
`pip install spc-lib-py[render]`.
"""

from __future__ import annotations

from spc.render.plotly_cusum import plotly_cusum
from spc.render.plotly_pareto import plotly_pareto
from spc.render.plotly_renderer import plotly
from spc.render.schema_export import to_dict, to_json

__all__ = ["plotly", "plotly_cusum", "plotly_pareto", "to_json", "to_dict"]

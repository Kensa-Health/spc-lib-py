"""Plotly rendering for `spc.pareto.ParetoResult` -- optional, reference-only,
like `plotly_renderer.plotly`. Needs the 'render' extra.
"""

from __future__ import annotations

from spc.pareto import ParetoResult

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
    raise ImportError(
        "spc.render.plotly_pareto requires the 'render' extra -- install with `pip install spc-lib-py[render]`"
    ) from exc

BAR_COLOUR = "#005eb8"
OTHER_BAR_COLOUR = "#a6a6a6"
CUMULATIVE_COLOUR = "#e46c0a"
AXIS_LINE_COLOUR = "#000000"


def plotly_pareto(
        result: ParetoResult,
        *,
        title: str | None = None
) -> go.Figure:
    """Bars of each category's count, largest first (an "Other" bucket, if
    any, last and greyed), with the cumulative share as a line on a 0-100%
    right-hand axis.
    """
    options = result.options
    categories = [r.category for r in result.rows]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=categories,
            y=[r.count for r in result.rows],
            marker={"color": [OTHER_BAR_COLOUR if r.is_other else BAR_COLOUR for r in result.rows]},
            name="Count" if options.count_field is None else options.count_field,
            customdata=[[r.share * 100] for r in result.rows],
            hovertemplate="%{x}<br>%{y:,.4g} (%{customdata[0]:.1f}%)<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=categories,
            y=[r.cumulative_share * 100 for r in result.rows],
            mode="lines+markers",
            line={"color": CUMULATIVE_COLOUR, "width": 2},
            marker={"size": 6},
            name="Cumulative %",
            hovertemplate="%{x}<br>%{y:.1f}% cumulative<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(title=title or f"Pareto of {options.category_field}", template="plotly_white", showlegend=True)
    fig.update_xaxes(title_text=options.category_field, categoryorder="array", categoryarray=categories)
    fig.update_yaxes(
        title_text="Count" if options.count_field is None else options.count_field,
        rangemode="tozero",
        secondary_y=False,
    )
    fig.update_yaxes(title_text="Cumulative %", range=[0, 100], ticksuffix="%", secondary_y=True, showgrid=False)
    fig.update_xaxes(showline=True, linewidth=1, linecolor=AXIS_LINE_COLOUR, mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor=AXIS_LINE_COLOUR, mirror=True)
    return fig

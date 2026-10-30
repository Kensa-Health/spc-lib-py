"""Plotly rendering for `spc.cusum.CusumResult` -- optional, reference-only.
Needs the 'render' extra.
"""

from __future__ import annotations

from spc.cusum import CusumResult

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
    raise ImportError(
        "spc.render.plotly_cusum requires the 'render' extra -- install with `pip install spc-lib-py[render]`"
    ) from exc

SCORE_COLOUR = "#005eb8"
THRESHOLD_COLOUR = "#de1b1b"
SIGNAL_COLOUR = "#e46c0a"
AXIS_LINE_COLOUR = "#000000"


def plotly_cusum(
        result: CusumResult,
        *,
        title: str | None = None,
        row_height: int = 320
) -> go.Figure:
    """One panel per facet. The score is drawn upwards with its threshold
    for an "increase" series; a "decrease" series (from `detect="both"` or
    `"decrease"`) is drawn mirrored below zero so the two directions read
    as one two-sided chart. Signals are marked.
    """
    facets = list(dict.fromkeys(s.facet for s in result.series))
    rows = len(facets)
    fig = make_subplots(
        rows=rows,
        cols=1,
        subplot_titles=[f or "" for f in facets] if facets != [None] else None,
        vertical_spacing=min(0.12, 0.5 / rows),
    )
    shown: set[str] = set()
    for row, facet in enumerate(facets, start=1):
        for s in (x for x in result.series if x.facet == facet):
            sign = 1.0 if s.direction == "increase" else -1.0
            label = "Rate increase" if s.direction == "increase" else "Rate decrease"
            xs = [p.x for p in s.points]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=[sign * p.score for p in s.points],
                    mode="lines",
                    line={"color": SCORE_COLOUR, "width": 1.5},
                    name=f"CUSUM score ({label.lower()})",
                    legendgroup=f"score-{s.direction}",
                    showlegend=f"score-{s.direction}" not in shown,
                ),
                row=row,
                col=1,
            )
            shown.add(f"score-{s.direction}")
            if xs:
                fig.add_trace(
                    go.Scatter(
                        x=[xs[0], xs[-1]],
                        y=[sign * s.h, sign * s.h],
                        mode="lines",
                        line={"color": THRESHOLD_COLOUR, "width": 1, "dash": "dash"},
                        name="Threshold h",
                        legendgroup="threshold",
                        showlegend="threshold" not in shown,
                    ),
                    row=row,
                    col=1,
                )
                shown.add("threshold")
            flagged = [p for p in s.points if p.signal]
            if flagged:
                fig.add_trace(
                    go.Scatter(
                        x=[p.x for p in flagged],
                        y=[sign * p.score for p in flagged],
                        mode="markers",
                        marker={"color": SIGNAL_COLOUR, "size": 9, "symbol": "diamond"},
                        name="Signal",
                        legendgroup="signal",
                        showlegend="signal" not in shown,
                    ),
                    row=row,
                    col=1,
                )
                shown.add("signal")
    fig.update_layout(
        title=title or "Bernoulli CUSUM", template="plotly_white", showlegend=True, height=row_height * rows + 120
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor=AXIS_LINE_COLOUR, mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor=AXIS_LINE_COLOUR, mirror=True, title_text="CUSUM score")
    fig.update_xaxes(title_text=result.options.date_field, row=rows, col=1)
    return fig

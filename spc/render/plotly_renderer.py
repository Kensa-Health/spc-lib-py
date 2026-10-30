"""Plotly rendering -- optional, reference-only.

Colour palette and line conventions follow the NHS "Making Data Count"
SPC convention exactly: the value of this chart type to NHS users is
partly that it's an already-recognised convention, so this renderer
doesn't take stylistic liberties with it.

Demonstration-quality: a thin reference implementation, not a themed or
production charting component. Most consuming apps are expected to render
`result.points` themselves.

Known limitations (acceptable for a demonstration renderer, worth revisiting
before any production use):
  - Multiple facets are drawn as small multiples (one panel per facet,
    a rows x columns grid for two-variable facets); see plotly().
  - `trajectory_field` is not rendered: PointResult has no field carrying
    trajectory values through the calculation, so there is nothing here to
    plot yet -- a gap in the current contract, not something this renderer
    works around.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

from spc.core.models import ChartResult, PointResult

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
    raise ImportError(
        "spc.render.plotly requires the 'render' extra -- install with `pip install spc-lib-py[render]`"
    ) from exc

# NHS "Making Data Count" standard colours.
POINT_COLOURS: dict[str, str] = {
    "common_cause": "#a6a6a6",
    "special_cause_improvement": "#00b0f0",
    "special_cause_concern": "#e46c0a",
    "special_cause_neutral_high": "#490092",
    "special_cause_neutral_low": "#490092",
}
# Plain-English legend labels for each point_type, in the fixed, meaningful
# order they should appear in the legend (normal, then good/bad, then the
# neutral variants) -- iterated over, not POINT_COLOURS itself, wherever
# that order matters. neutral_high/neutral_low deliberately share one label
# (as well as one colour, per POINT_COLOURS above) -- they're visually
# identical on the chart itself (direction is still readable from each
# point's position relative to the baseline line), so a legend with two
# rows for one indistinguishable colour would only be confusing, not
# informative.
POINT_TYPE_LABELS: dict[str, str] = {
    "common_cause": "Common cause",
    "special_cause_improvement": "Special cause (improvement)",
    "special_cause_concern": "Special cause (concern)",
    "special_cause_neutral_high": "Special cause (neutral)",
    "special_cause_neutral_low": "Special cause (neutral)",
}
# Chart border only -- kept separate from the Baseline/UCL-LCL defaults
# below so a caller overriding those via plotly()'s baseline_colour/
# limit_colour doesn't also silently recolour the axis box.
AXIS_LINE_COLOUR = "#000000"
# Defaults for plotly()'s baseline_colour/limit_colour parameters -- the
# NHS Making Data Count convention this renderer otherwise follows exactly
# (see the module docstring). Module-level constants so callers can still
# import and reference "the default", not just pass a bare literal.
BASELINE_COLOUR = "#000000"
LIMIT_COLOUR = "#a6a6a6"
TARGET_COLOUR = "#de1b1b"
# Deliberately distinct from TARGET_COLOUR -- a spec limit (an externally
# imposed requirement) and a target (a single desired value) answer
# different questions and shouldn't be visually conflated. Not part of the
# NHS Making Data Count palette (which has no spec-limit concept at all),
# so this is this renderer's own choice. Used both for the spec-limit
# lines themselves and for breached points' marker outline, so anything
# in this colour reads as "spec limit related".
SPEC_LIMIT_COLOUR = "#7030a0"


def _hex_to_rgba(
        hex_colour: str,
        alpha: float
) -> str:
    """`"#228b22"` -> `"rgba(34, 139, 34, 0.15)"` -- lets shade_limits'
    fill follow whatever limit_colour the caller passes, rather than being
    a fixed colour independent of it.
    """
    hex_colour = hex_colour.lstrip("#")
    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _capitalize_first(
        label: str
) -> str:
    """Upper-cases only the first character, leaving the rest of the
    string untouched -- unlike str.capitalize(), which would also
    lower-case the rest, mangling a field name like 'ICU_admissions'.
    """
    return label[0].upper() + label[1:] if label else label


def _x_axis_label(
        result: ChartResult
) -> str:
    """The date/period column a point's x-position comes from -- normally
    `options.date_field`, except `t` (rare-event elapsed-time), whose
    x-position is the *later* event's own timestamp from `value_field`,
    not `date_field` at all (see core/rare_event_segments.py).
    """
    if result.kind == "t":
        return result.options.value_field
    return result.options.date_field


def _y_axis_label(
        result: ChartResult
) -> str:
    """What `PointResult.y` actually is, which varies by chart kind -- not
    always `options.value_field` verbatim, so this can't be a single fixed
    label (see the per-kind `y=` assignment in core/attribute_segments.py
    and core/rare_event_segments.py, which this mirrors).
    """
    options = result.options
    kind = result.kind
    if kind in ("p", "u", "p_prime", "u_prime"):
        # y is value_field/denominator_field (a proportion or rate), not
        # the raw count -- see core/attribute_segments.py's `y = raw / denom`.
        # p_prime/u_prime plot the same y as p/u (Laney's correction only
        # widens the limits, not what's plotted).
        denominator_field = getattr(options, "denominator_field", None)
        return f"{options.value_field} / {denominator_field}" if denominator_field else options.value_field
    if kind in ("xbar_r", "xbar_s"):
        return f"Mean {options.value_field}"  # y is each subgroup's mean, not a raw row value
    if kind == "t":
        return f"Days between '{options.value_field}' events"
    # xmr, np, c, g: y is value_field verbatim (raw measurement or raw count).
    return options.value_field


def _per_facet(
        value: float | dict[str, float] | None,
        facet: str | None
) -> float | None:
    """Scalar-or-per-facet-dict resolution for `target`/`spec_upper`/`spec_lower`."""
    if isinstance(value, dict):
        return value.get(facet) if facet is not None else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _draw_panel(
        fig: go.Figure,
        panel_points: list[PointResult],
        result: ChartResult,
        *,
        facet: str | None,
        x_range: tuple[date, date],
        future_space: float,
        shade_limits: bool,
        baseline_colour: str,
        limit_colour: str,
        cell: tuple[int, int] | None,
        with_legend: bool
) -> None:
    """Draws one facet's Baseline/UCL/LCL lines, value series, target and
    spec lines. `cell` is the (row, col) of its subplot, or None for the
    single-axes case. `with_legend` is True for exactly one panel so the
    legend lists each line once however many facets there are.
    """
    kw: dict[str, int] = {"row": cell[0], "col": cell[1]} if cell else {}

    points_by_segment: dict[int, list[PointResult]] = {}
    for point in panel_points:
        points_by_segment.setdefault(point.segment_id, []).append(point)
    first_segment_id = min(points_by_segment) if points_by_segment else None

    for segment_id in sorted(points_by_segment):
        seg_points = sorted(points_by_segment[segment_id], key=lambda p: p.x)
        xs = [p.x for p in seg_points]
        is_first = segment_id == first_segment_id and with_legend

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[p.mean for p in seg_points],
                mode="lines",
                line={"color": baseline_colour, "width": 1.5},
                name="Baseline",
                legendgroup="baseline",
                showlegend=is_first,
            ),
            **kw,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[p.ucl for p in seg_points],
                mode="lines",
                line={"color": limit_colour, "width": 1, "dash": "dash"},
                name="UCL or LCL",
                legendgroup="limits",
                showlegend=is_first,
            ),
            **kw,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[p.lcl for p in seg_points],
                mode="lines",
                line={"color": limit_colour, "width": 1, "dash": "dash"},
                name="UCL or LCL",
                legendgroup="limits",
                showlegend=False,
                # Shades the common-cause band when shade_limits=True --
                # fills back to the immediately preceding trace, which is
                # this same segment's own UCL line added just above, not
                # any other segment's (each segment draws its own
                # Baseline/UCL/LCL triplet in sequence, so "tonexty"
                # always resolves to the right pair). Derived from
                # limit_colour itself (not a separate parameter) so the
                # fill always matches whichever limit_colour is in use.
                **({"fill": "tonexty", "fillcolor": _hex_to_rgba(limit_colour, 0.15)} if shade_limits else {}),
            ),
            **kw,
        )

    all_points = sorted(panel_points, key=lambda p: p.x)
    fig.add_trace(
        go.Scatter(
            x=[p.x for p in all_points],
            y=[p.y for p in all_points],
            mode="lines+markers",
            line={"color": POINT_COLOURS["common_cause"], "width": 2},
            marker={
                "color": [POINT_COLOURS[p.point_type] for p in all_points],
                "size": 7,
                # spec_breach is drawn as a marker outline, never by
                # changing the fill colour above -- point_type's fill
                # always keeps meaning "is this statistically unusual",
                # independent of whether it also breaches a spec limit
                # (see PointResult.spec_breach's docstring for why those
                # two questions are deliberately kept visually separable).
                "line": {
                    "color": [SPEC_LIMIT_COLOUR if p.spec_breach else "rgba(0,0,0,0)" for p in all_points],
                    "width": [2 if p.spec_breach else 0 for p in all_points],
                },
            },
            name=result.kind.upper(),
            # This trace's own legend swatch is suppressed, not shown: its
            # marker.color above is a per-point array, and Plotly's legend
            # can't represent that with one swatch -- it falls back to a
            # generic default colour that means nothing here. The dummy,
            # off-chart proxy traces added in plotly() take over the
            # legend instead, one per point_type actually present.
            showlegend=False,
        ),
        **kw,
    )

    target = _per_facet(result.options.target, facet)
    if target is not None:
        fig.add_hline(y=target, line={"color": TARGET_COLOUR, "width": 1.5}, annotation_text="Target", **kw)

    # spec_upper/spec_lower are drawn as Scatter traces, not add_hline shapes,
    # specifically so they show up in the legend under a shared, customisable
    # name (options.spec_label) -- an add_hline shape/annotation never
    # appears in the Plotly legend at all. Both share one legendgroup so a
    # spec_upper+spec_lower pair reads as a single legend entry, matching the
    # Baseline/UCL-LCL pattern above. Spans the same start..end+future_space
    # range as the x-axis itself, so it still reaches the right edge.
    if all_points:
        start, end = x_range
        right_padding = (end - start) * future_space
        spec_label = result.options.spec_label or "Spec limit"
        spec_xs = [start, end + right_padding]
        spec_first = with_legend
        for spec_value in (
            _per_facet(result.options.spec_upper, facet),
            _per_facet(result.options.spec_lower, facet),
        ):
            if spec_value is not None:
                fig.add_trace(
                    go.Scatter(
                        x=spec_xs,
                        y=[spec_value, spec_value],
                        mode="lines",
                        line={"color": SPEC_LIMIT_COLOUR, "width": 1.5, "dash": "dot"},
                        name=spec_label,
                        legendgroup="spec",
                        showlegend=spec_first,
                    ),
                    **kw,
                )
                spec_first = False


def _facet_layout(
        facets: list[str],
        result: ChartResult,
        facet_columns: int | None
) -> tuple[int, int, dict[str, tuple[int, int]], list[str]]:
    """(rows, cols, facet -> (row, col) 1-based, subplot titles in row-major order).

    A two-variable chart (facet_parts present) gets a true rows x columns
    grid, the row variable down the side and the column variable across
    the top; a one-variable chart wraps `facet_columns` per row.
    """
    parts = {p.facet: p.facet_parts for p in result.points if p.facet_parts}
    if parts and all(f in parts for f in facets):
        row_values = sorted({parts[f][0] for f in facets})
        col_values = sorted({parts[f][1] for f in facets})
        cells = {f: (row_values.index(parts[f][0]) + 1, col_values.index(parts[f][1]) + 1) for f in facets}
        titles = [""] * (len(row_values) * len(col_values))
        for f, (r, c) in cells.items():
            titles[(r - 1) * len(col_values) + (c - 1)] = f
        return len(row_values), len(col_values), cells, titles
    cols = facet_columns if facet_columns else min(len(facets), 3)
    rows = -(-len(facets) // cols)
    cells = {f: (i // cols + 1, i % cols + 1) for i, f in enumerate(facets)}
    titles = list(facets) + [""] * (rows * cols - len(facets))
    return rows, cols, cells, titles


def plotly(
        result: ChartResult,
        *,
        title: str | None = None,
        future_space: float = 0.20,
        shade_limits: bool = False,
        baseline_colour: str = BASELINE_COLOUR,
        limit_colour: str = LIMIT_COLOUR,
        facet_columns: int | None = None,
        row_height: int = 320
) -> go.Figure:
    """Reference Plotly renderer for a ChartResult.

    Baseline/UCL/LCL lines are broken at segment (rebase) boundaries, per NHS
    convention -- a rebase starts an entirely new, independent calculation,
    and joining the lines across that boundary would visually imply
    continuity that isn't there. The value series itself is drawn
    continuously across the whole chart, points coloured by `point_type`.

    `future_space`: fraction of the data's own date span left as blank
    room past the last point (default 0.20, i.e. 20%) -- this renderer
    doesn't draw anything there itself, but a forward-looking overlay (a
    baseline/trend projection, for instance) needs somewhere to go without
    forcing a re-plot. 0 disables it, giving a plot area pinned tight to
    the data on both ends.

    `shade_limits`: fill the common-cause band between UCL and LCL, in
    `limit_colour` at low opacity. Off by default -- the dashed UCL/LCL
    lines alone are the standard NHS Making Data Count convention this
    renderer otherwise follows exactly; the shaded band is an opt-in
    embellishment, not part of it.

    `baseline_colour`/`limit_colour`: override the Baseline/UCL-LCL line
    colours, e.g. to distinguish one chart from the rest in a notebook or
    report. Default to the NHS Making Data Count convention's own black/
    grey -- pass these explicitly only where a specific chart calls for
    standing out, not as a global style change.

    A result with more than one facet is drawn as small multiples, one
    panel per facet, each with its own y-axis (limits differ per facet, so
    a shared scale would flatten the small ones) and a common x-axis range.
    A two-variable chart (`column_facet_field`) is laid out as a rows x
    columns grid; a one-variable chart wraps `facet_columns` panels per row
    (default: up to 3). `row_height` is the pixel height of each panel row.
    `target`/`spec_upper`/`spec_lower` may be per-facet dicts and are drawn
    in each panel from that facet's own value.
    """
    if not (math.isfinite(future_space) and future_space >= 0):
        raise ValueError(f"future_space must be finite and >= 0, got {future_space}")
    if facet_columns is not None and facet_columns < 1:
        raise ValueError(f"facet_columns must be >= 1, got {facet_columns}")

    all_points = sorted(result.points, key=lambda p: p.x)
    facets = sorted({p.facet for p in all_points if p.facet is not None})
    multi = len(facets) > 1
    x_range = (all_points[0].x, all_points[-1].x) if all_points else (date.min, date.min)  # unused when no points

    if multi:
        from plotly.subplots import make_subplots

        rows, cols, cells, titles = _facet_layout(facets, result, facet_columns)
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=titles,
            shared_xaxes=False,
            shared_yaxes=False,
            vertical_spacing=min(0.12, 0.5 / rows),
            horizontal_spacing=0.06,
        )
        for i, panel_facet in enumerate(facets):
            _draw_panel(
                fig,
                [p for p in all_points if p.facet == panel_facet],
                result,
                facet=panel_facet,
                x_range=x_range,
                future_space=future_space,
                shade_limits=shade_limits,
                baseline_colour=baseline_colour,
                limit_colour=limit_colour,
                cell=cells[panel_facet],
                with_legend=i == 0,
            )
    else:
        rows = cols = 1
        fig = go.Figure()
        _draw_panel(
            fig,
            all_points,
            result,
            facet=facets[0] if facets else None,
            x_range=x_range,
            future_space=future_space,
            shade_limits=shade_limits,
            baseline_colour=baseline_colour,
            limit_colour=limit_colour,
            cell=None,
            with_legend=True,
        )

    # All point-related legend entries (colour meaning + the spec-breach
    # marker outline below) go in their own legend box (`legend2`), kept
    # visually separate from the Baseline/UCL-LCL/spec-limit *lines* in the
    # default legend -- two different kinds of thing ("what does this point
    # look like" vs. "what does this line mean"), not one flat list.
    present_point_types = {p.point_type for p in all_points}
    added_point_labels: set[str] = set()
    for point_type in POINT_TYPE_LABELS:  # fixed, meaningful order
        if point_type not in present_point_types:
            continue
        label = POINT_TYPE_LABELS[point_type]
        if label in added_point_labels:
            continue  # neutral_high/neutral_low share one legend entry -- see POINT_TYPE_LABELS above
        added_point_labels.add(label)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"color": POINT_COLOURS[point_type], "size": 7},
                name=label,
                legend="legend2",
                showlegend=True,
            )
        )
    if any(p.spec_breach for p in all_points):
        # Explains the coloured marker outline drawn on spec-breaching
        # points above -- a plain-coloured dummy marker wouldn't show it;
        # this mimics the same hollow-centre/coloured-ring look instead.
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"color": "rgba(0,0,0,0)", "size": 7, "line": {"color": SPEC_LIMIT_COLOUR, "width": 2}},
                name="Breaches spec limit",
                legend="legend2",
                showlegend=True,
            )
        )

    layout: dict[str, Any] = {
        "title": title or f"{result.kind.upper()} chart",
        "showlegend": True,
        # Two separate legend boxes (Plotly's `legend`/`legend2`), not one
        # flat list or grouped headers within a single box -- lines
        # (Baseline/UCL-LCL/spec limit) stay in the default legend, top-right;
        # point-related entries (colour meaning, spec-breach outline) go in
        # legend2, stacked underneath it, still on the right margin. The
        # fixed y=0.55 gap is a heuristic (this renderer can't know its own
        # rendered height ahead of time) sized for the default legend's
        # usual 2-3 entries -- a chart with an unusually tall first legend
        # could still overlap it.
        "legend": {"title": {"text": "Limits & lines"}, "y": 1, "yanchor": "top", "x": 1.02, "xanchor": "left"},
        "legend2": {"title": {"text": "Data points"}, "y": 0.55, "yanchor": "top", "x": 1.02, "xanchor": "left"},
        "template": "plotly_white",
    }
    if multi:
        layout["height"] = row_height * rows + 120
    fig.update_layout(**layout)
    x_title = _capitalize_first(_x_axis_label(result))
    y_title = _capitalize_first(_y_axis_label(result))
    if multi:
        fig.update_xaxes(title_text=x_title, row=rows)
        fig.update_yaxes(title_text=y_title, col=1)
    else:
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    # Draw a full box around each plot area (mirrored axis lines on all four
    # sides) so the data area's boundary is visually explicit, not just
    # implied by where the gridlines/ticks happen to stop.
    fig.update_xaxes(showline=True, linewidth=1, linecolor=AXIS_LINE_COLOUR, mirror=True)
    if all_points:
        # Left edge stays pinned exactly to the data's first date -- no
        # reason to pad space nothing will ever occupy. Right edge gets
        # `future_space` worth of the data's own date span as breathing
        # room (see the docstring above). One common range for every panel.
        start, end = x_range
        fig.update_xaxes(range=[start, end + (end - start) * future_space])
    fig.update_yaxes(showline=True, linewidth=1, linecolor=AXIS_LINE_COLOUR, mirror=True)
    return fig

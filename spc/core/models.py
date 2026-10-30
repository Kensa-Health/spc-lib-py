"""Pydantic data contracts and shared enums for the spc library.

Every boundary type -- what goes into a calculation and what comes out of
it -- lives here.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The four default rule names, in priority order (lower priority number =
# evaluated first, first match wins). Defined here rather than in
# core.rules so BaseChartOptions.rules's default factory doesn't create a
# models <-> rules import cycle. core.rules.DEFAULT_RULES' Rule.name
# values must match these exactly -- a test enforces that, not just
# convention (see tests/test_rules.py::test_default_rule_names_match_models).
DEFAULT_RULE_NAMES: tuple[str, ...] = (
    "outside_limits",
    "seven_point_trend",
    "two_in_three_near_limit",
    "seven_point_shift",
)


class LimitMode(str, Enum):
    """How control limits are computed within a segment: from the whole
    segment, from a fixed first N points, or from a growing window that
    locks after N points.
    """

    SEGMENT = "segment"
    FIXED_N = "fixed_n"
    EXPANDING = "expanding"


ScreenOutliers = Literal["off", "single_pass", "iterative"]
# Fixed run length (any int >= 3), or "proportional" for a sample-size-scaled
# bound -- see BaseChartOptions.run_rule_length's docstring.
RunRuleLength = int | Literal["proportional"]
ImprovementDirection = Literal["increase", "decrease", "neutral"]
PointType = Literal[
    "common_cause",
    "special_cause_improvement",
    "special_cause_concern",
    "special_cause_neutral_high",
    "special_cause_neutral_low",
]
# A float that rejects NaN and +/-infinity: either would flow through a control
# limit, target or specification comparison as a silently wrong answer.
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

AssuranceType = Literal["consistent_pass", "consistent_fail", "inconsistent"]


class ExternalBaseline(BaseModel):
    """One point of an externally-supplied expected-value series.

    Produced by a separate baseline/trend library (or any other source of a
    non-flat expected value); this library only consumes the shape, never
    interprets how it was produced.
    """

    model_config = ConfigDict(extra="forbid")  # a misspelled field must raise, not be ignored

    facet: str | None = None
    x: date
    expected_value: FiniteFloat
    source: str


class BaseChartOptions(BaseModel):
    """Common input fields shared by every chart kind.

    Subclassed per kind for kind-specific required fields -- see
    ProportionChartOptions below.
    """

    model_config = ConfigDict(extra="forbid")  # a misspelled option must raise, not be ignored

    value_field: str
    date_field: str
    # Split into one chart per value of this column. With
    # column_facet_field also set, this is the *row* variable of a
    # two-variable facet grid (see core/facets.py).
    facet_field: str | None = None
    column_facet_field: str | None = None
    # Tie-break for rows that share a date: order is by date_field, then this
    # column. Used by every kind except xbar_r/xbar_s (rows sharing a date are one
    # subgroup) and t (ordered by its own timestamps), which reject it. Without it,
    # tied rows keep their input order and spc.chart() warns.
    sequence_field: str | None = None
    rebase: list[date] | dict[str, list[date]] | None = None
    limit_mode: LimitMode = LimitMode.SEGMENT
    fix_after_n_points: int | None = None
    screen_outliers: ScreenOutliers = "single_pass"
    # Binary on/off (default: do not apply) -- not a graded set of methods
    # the way screen_outliers is, since this is a single yes/no decision
    # about whether the data is autocorrelated enough to need correcting,
    # not several competing techniques to choose between. Only xmr's
    # fit() acts on it -- ignored by every other chart kind (see
    # attribute_segments.py's module docstring for the same precedent
    # already established for screen_outliers).
    autocorrelation_adjustment: bool = False
    improvement_direction: ImprovementDirection = "increase"
    target: FiniteFloat | dict[str, FiniteFloat] | None = None
    trajectory_field: str | None = None
    rules: list[str] = Field(default_factory=lambda: list(DEFAULT_RULE_NAMES))
    # Window length the seven_point_trend/seven_point_shift detectors require
    # for a run to count as special-cause. A fixed count's false-alarm rate
    # isn't constant across segment lengths -- it under-triggers on short
    # segments and over-triggers on long ones -- so "proportional" is
    # available as an alternative philosophy, not a replacement default:
    # round(log2(n)) + 3 per segment (Schilling 2012's longest-expected-run
    # bound), the same formula a second, independently-developed SPC package
    # uses for its own sample-size-scaled signal detection. Rule *names*
    # stay "seven_point_trend"/"seven_point_shift" regardless of the
    # resolved length -- stable identifiers, not literal descriptions.
    run_rule_length: RunRuleLength = 7
    short_segment_threshold: int = 13
    baseline: list[ExternalBaseline] | None = None
    # Specification/capability limits (USL/LSL in classic SPC terminology)
    # -- an externally-imposed requirement ("voice of the customer"),
    # independent of the process's own statistical variation ("voice of
    # the process", ucl/lcl). Deliberately two independent optionals, not
    # a forced pair: a rare-event chart's spec limit is naturally one-sided
    # (a minimum acceptable gap between adverse events, no upper bound),
    # same asymmetry already used for p-chart's lower-only clip. Same
    # scalar-or-per-facet-dict shape as `target`. Never used to reshape
    # ucl/lcl -- see PointResult.spec_breach's docstring for why keeping
    # these fully separate from statistical control is a deliberate,
    # not incidental, design choice.
    spec_upper: FiniteFloat | dict[str, FiniteFloat] | None = None
    spec_lower: FiniteFloat | dict[str, FiniteFloat] | None = None
    # Legend label for the spec_upper/spec_lower line(s) in spc.render.plotly
    # -- e.g. "Spec limit (78% target)" for a named external requirement.
    # Purely cosmetic (rendering-only, like spec_upper/spec_lower's own
    # scalar-vs-dict rendering limitation above); falls back to "Spec limit"
    # when not supplied. Has no effect on spec_breach or any calculation.
    spec_label: str | None = None

    @model_validator(mode="after")
    def check_column_facet_field(self) -> BaseChartOptions:
        """column_facet_field is the second facet variable, so it needs
        facet_field (the row variable) to exist and differ from it.
        """
        if self.column_facet_field is not None:
            if not self.facet_field:
                raise ValueError("column_facet_field requires facet_field (the row facet) to be set")
            if self.column_facet_field == self.facet_field:
                raise ValueError("column_facet_field must differ from facet_field")
        return self

    @model_validator(mode="after")
    def check_fix_after_n_points(self) -> BaseChartOptions:
        """fix_after_n_points is required for FIXED_N/EXPANDING modes and
        must be at least 2.
        """
        if self.limit_mode in (LimitMode.FIXED_N, LimitMode.EXPANDING) and not self.fix_after_n_points:
            raise ValueError(f"limit_mode={self.limit_mode!r} requires fix_after_n_points to be set")
        if self.fix_after_n_points is not None and self.fix_after_n_points < 2:
            raise ValueError("fix_after_n_points must be >= 2 (a moving range needs at least 2 points)")
        return self

    @model_validator(mode="after")
    def check_run_rule_length(self) -> BaseChartOptions:
        """A fixed run_rule_length must be >= 3 -- below that the
        run-length detectors' own windowing is degenerate (a run of 1 or 2
        points isn't a meaningful trend/shift signal). "proportional"
        always resolves to >= 3 by construction, so it needs no check here.
        """
        if isinstance(self.run_rule_length, int) and self.run_rule_length < 3:
            raise ValueError("run_rule_length must be >= 3, or 'proportional'")
        return self

    @model_validator(mode="after")
    def check_spec_limits(self) -> BaseChartOptions:
        """spec_upper must exceed spec_lower wherever both are supplied
        for the same facet.
        """

        def _pairs() -> list[tuple[float, float]]:
            """Every (upper, lower) pair actually supplied together, resolved per facet."""
            if isinstance(self.spec_upper, dict) or isinstance(self.spec_lower, dict):
                facets = set()
                if isinstance(self.spec_upper, dict):
                    facets |= set(self.spec_upper)
                if isinstance(self.spec_lower, dict):
                    facets |= set(self.spec_lower)
                out = []
                for facet in facets:
                    upper = self.spec_upper.get(facet) if isinstance(self.spec_upper, dict) else self.spec_upper
                    lower = self.spec_lower.get(facet) if isinstance(self.spec_lower, dict) else self.spec_lower
                    if upper is not None and lower is not None:
                        out.append((upper, lower))
                return out
            if self.spec_upper is not None and self.spec_lower is not None:
                return [(self.spec_upper, self.spec_lower)]
            return []

        for upper, lower in _pairs():
            if upper <= lower:
                raise ValueError(f"spec_upper ({upper}) must be greater than spec_lower ({lower})")
        return self


class ProportionChartOptions(BaseChartOptions):
    """Options for the 'p', 'p_prime', 'np', and 'u'/'u_prime' chart kinds."""

    denominator_field: str


class PointResult(BaseModel):
    """One plotted point with its limits, signal and classification."""

    x: date
    y: float
    facet: str | None
    # [row, column] values when the chart is split by two variables
    # (column_facet_field set), else None. `facet` itself is then the two
    # joined by core/facets.py::FACET_SEPARATOR.
    facet_parts: list[str] | None = None
    segment_id: int
    mean: float
    # Optional, not a loose default: a moving range needs >= 2 points, so
    # the very first point of an EXPANDING segment -- and any SEGMENT/
    # FIXED_N segment with only 1 point -- has genuinely undefined limits.
    # Typing these as plain `float` would let NaN leak through as a
    # required-but-null field, silently violating this model's own
    # generated JSON Schema for real, reachable input.
    ucl: float | None
    lcl: float | None
    outside_limits: bool
    special_cause_rule: str | None
    point_type: PointType
    # True when this point's control limits rest on a normal-approximation-
    # to-binomial calculation (p/np) that the standard n*p_bar>=5 and
    # n*(1-p_bar)>=5 rule of thumb flags as questionable -- a diagnostic
    # caveat, not an error: the limits are still computed and usable, just
    # with less statistical confidence than the nominal ~3-sigma false-
    # alarm rate implies. Always False for chart kinds this doesn't apply
    # to (everything except p/np).
    low_count_warning: bool
    # True when this point's own value (y) falls outside options.spec_upper/
    # spec_lower -- an externally-imposed requirement ("voice of the
    # customer"), computed and reported entirely independently of
    # outside_limits/special_cause_rule/point_type ("voice of the
    # process"). Deliberately never merged into or allowed to influence
    # those fields: a point can be common-cause (the process is behaving
    # exactly as it normally does) and still spec_breach=True (that normal
    # behaviour doesn't meet the requirement) -- collapsing the two into
    # one signal would hide that distinction, which is itself the useful
    # information a capability-style question is actually asking. Always
    # False when no spec limit is supplied for this point's facet.
    spec_breach: bool


class SegmentSummary(BaseModel):
    """Summary of one segment (a facet between rebase points)."""

    segment_id: int
    facet: str | None
    facet_parts: list[str] | None = None
    n: int
    mean: float
    ucl: float | None  # None iff every point in the segment has n < 2 -- see PointResult.ucl
    lcl: float | None
    short_segment_warning: bool
    # Lag-1 sample autocorrelation of this segment's detrended residuals
    # (xmr only -- see core/limits.py::estimate_autocorrelation). Always
    # computed and exposed regardless of autocorrelation_adjustment, same
    # split as low_count_warning: visible for review even when not acted
    # on. None when too few points to estimate (same convention as ucl/lcl).
    autocorrelation_estimate: float | None
    # True when autocorrelation_adjustment is on and the (unclamped)
    # estimate above is >= 0.8 -- the correction is least reliable exactly
    # where rho approaches 1, so this flags "the widened limits here
    # deserve a second look" without changing them. Non-fatal diagnostic,
    # same philosophy as low_count_warning.
    high_autocorrelation_warning: bool


class ChartResult(BaseModel):
    """Output of `spc.chart()`: points, segment summaries and diagnostics."""

    schema_version: str = "1.0"
    kind: str
    # The options the chart was run with, including a subclass's own fields
    # (ProportionChartOptions.denominator_field). Declared as a union so those
    # fields are serialised and parse back into the right class; "left to
    # right" tries the more specific class first, since BaseChartOptions
    # rejects the extra field.
    options: Annotated[ProportionChartOptions | BaseChartOptions, Field(union_mode="left_to_right")]
    points: list[PointResult]
    segments: list[SegmentSummary]
    # A single value for one facet (or none); a dict keyed by facet name
    # when there's more than one -- mirrors BaseChartOptions.target's own
    # `float | dict[str, float]` pattern, computed per facet natively.
    assurance_type: AssuranceType | dict[str, AssuranceType] | None = None
    suggested_kind: str | None = None
    # Non-fatal diagnostic, not a validation error -- same philosophy as
    # PointResult.low_count_warning. Populated only when
    # options.autocorrelation_adjustment is True: the rule names in
    # options.rules other than "outside_limits", i.e. the ones still
    # calibrated assuming independence even though ucl/lcl are now
    # correctly widened. A list, not a bare bool, so a caller/UI can see
    # which rules are still at risk. Never mutates options.rules itself.
    autocorrelation_uncorrected_rules: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChartSuggestion(BaseModel):
    """Output of suggest_chart_type().

    Never a hard block -- the caller always decides whether to follow it.
    """

    kind: str
    confidence: Literal["low", "medium", "high"]
    reason: str


class RebaseSuggestion(BaseModel):
    """One candidate rebase point surfaced by suggest_rebase()."""

    facet: str | None
    candidate_date: date
    rule: str
    direction: Literal["improvement", "concern", "neutral"]
    points_since: int
    confidence: Literal["low", "medium", "high"]

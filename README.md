# spc

SPC (statistical process control) calculation library. Owns calculation,
validation, and data contracts only — no UI opinions. Built throughout
around the NHS "Making Data Count" SPC methodology and philosophy.

## Why this exists

SPC is one of the core tools of quality improvement: measuring
operational data over time and using it to tell genuine change in a
process apart from routine variation, so effort goes toward real
problems and real improvements rather than reacting to noise. It's
central to quality improvement generally, and used the same way across
healthcare and industry more broadly.

This library follows the guidelines set out by the NHS's "Making Data
Count" programme, because those guidelines get something right: they
favour methods that are simple to compute, explain, and act on, over
ones that only make sense to a professional statistician or data
scientist. This library exists to give Python users that same correct,
accessible implementation, rather than everyone re-deriving the formulas
or reaching for a full BI/dashboarding stack to get a control chart.

It is aimed at data scientists, analysts and developers working with NHS
data, or operational data more generally, who need correct SPC without
reimplementing it.

It is designed to be used two ways: embedded in a product, returning a typed,
JSON-serialisable `ChartResult` that any front end can render, or used directly
by an analyst in a notebook. That is why calculation, validation and data
contracts are kept strictly separate from UI.

## Status

11 chart kinds are implemented: `xmr`, `p`, `p_prime`, `u`, `u_prime`,
`np`, `c`, `xbar_r`, `xbar_s`, `t`, `g`. `suggest_chart_type()`,
`suggest_rebase()`, and the `ChartResult` contract are shared across all
of them.

The core formulas are verified against NIST's e-Handbook of Statistical
Methods and hand computation. See "Design choices and known limitations"
below before relying on `p`/`u`/`np`/`c`/`xbar_r`/`xbar_s`/`t`/`g` for
anything important — a few structural choices there were genuine
judgment calls, not settled facts.

Beyond the control-chart kinds there are two adjacent tools with their own
functions and result types (not `spc.chart()`): `spc.cusum()` (Bernoulli
CUSUM for a binary outcome) and `spc.pareto()`. See "Adjacent tools"
below.

XmR numeric-parity fixtures generated with a pinned NHSRplotthedots 0.2.2 release
verify limits and rule classifications on two real NHS series (72 points).
[tests/golden/README.md](tests/golden/README.md) records the scope and reproduction steps.
The suite also uses synthetic cases, public NHS/ONS data, property checks and independent
reference implementations for the broader API.

Requires Python 3.11 or later. The version is 0.1.0: the API may still change.

Not validated for clinical or safety-critical decisions. Outputs are statistical aids for
review, provided without warranty (see [LICENSE](LICENSE)). Read the
[statistical caveats](#statistical-caveats) and
[design choices and known limitations](#design-choices-and-known-limitations) before relying on a result.

## Chart kinds

| Chart | `kind` | What it plots | Concrete NHS example | Why this chart fits |
|---|---|---|---|---|
| XmR — Individuals & Moving Range | `xmr` | One continuous observation per time point, plus the moving range between consecutive observations | Daily median time from arrival in A&E to initial clinical assessment | There is one continuous summary value per day; the chart monitors the level and short-term movement |
| p — Proportion | `p` | Proportion meeting a binary definition, with a potentially varying denominator | Weekly proportion of A&E patients spending more than 12 hours in the department | Each patient is classified as meeting or not meeting the condition; the weekly denominator may vary |
| p′ — Laney P′ | `p_prime` | Proportion with additional between-subgroup variation beyond the binomial model | Monthly proportion of patients readmitted within 30 days across a group of hospitals, where case mix and service configuration create extra variation | A conventional p-chart produces excessive signals because the observed variation is greater than binomial sampling variation |
| np — Number defective | `np` | Number of "defective" or non-conforming units, with a constant subgroup size | Number of patients exceeding the four-hour A&E standard in a fixed sample of exactly 500 randomly selected attendances each week | The outcome is binary, but the sample size is constant; the chart plots a count rather than a proportion |
| c — Count | `c` | Count of events in a constant area of opportunity or observation period | Number of medication administration errors recorded per month on a ward with stable reporting coverage and a stable number of medication administrations | The chart monitors event counts where the exposure or opportunity is effectively constant |
| u — Rate | `u` | Count of events per unit of exposure, where exposure may vary | Number of falls per 1,000 occupied bed-days per month | The denominator — occupied bed-days — varies, so a rate is more appropriate than a raw count |
| u′ — Laney U′ | `u_prime` | Rate with additional between-subgroup variation beyond the Poisson model | Number of delayed transfers of care per 1,000 bed-days across hospitals with materially different operational environments | A conventional u-chart over-signals because rates vary more than the Poisson assumption predicts |
| X̄-R — Xbar-R | `xbar_r` | Mean of each subgroup, with limits estimated from subgroup ranges; generally `n ≤ 10`. The companion range chart is not returned | Each day, measure the time to triage for five consecutive randomly selected A&E patients; plot the daily mean using the range to estimate limits | Each subgroup contains a small number of continuous observations; R estimates within-subgroup variation |
| X̄-S — Xbar-S | `xbar_s` | Mean of each subgroup, with limits estimated from subgroup standard deviations; generally `n > 10`. The companion S chart is not returned | Each day, measure time from referral to first specialist review for 20 patients; plot the daily mean using the standard deviation to estimate limits | With larger subgroups, the standard deviation is a more efficient estimate of within-subgroup variation than the range |
| t — Time between rare events | `t` | Elapsed time between successive rare events | Hours or days between hospital-acquired bloodstream infections in an NHS trust | The event is sufficiently rare that plotting counts per ordinary period would produce mostly zeros |
| g — Count between rare events | `g` | Number of opportunities or non-events between successive rare events | Number of emergency admissions between serious medication-related harm incidents | The event is rare, but there is a meaningful count of opportunities or cases between occurrences |

## Adjacent tools

Two tools that sit next to SPC rather than being Shewhart control charts.
They take their own options and return their own result models
(`model_dump_json()` works on both; `spc.render.plotly_cusum` and
`spc.render.plotly_pareto` draw them).

| Tool | Function | What it does | Concrete NHS example |
|---|---|---|---|
| Bernoulli CUSUM | `spc.cusum()` | Accumulates a log-likelihood-ratio score over one 0/1 outcome per case and signals when it crosses a threshold `h`; detects a sustained shift in an event rate faster than a p-chart when the rate is low | Surgical-site infection (yes/no) for each operation, watching for a doubling of the odds against a known baseline rate |
| Pareto | `spc.pareto()` | Ranks categories by count (or a weight), largest first, with cumulative share and an optional "Other" bucket | Minutes lost per cause of delayed discharge, to see which few causes account for most of the delay |

```python
import spc

# p0 is the in-control event rate, supplied (see spc.estimate_event_rate for a reference window)
result = spc.cusum(df, spc.CusumOptions(value_field="infection", date_field="operated_on", p0=0.02))
result.series[0].signal_indices            # where the score crossed the threshold
spc.render.plotly_cusum(result)

pareto = spc.pareto(df, spc.ParetoOptions(category_field="cause", count_field="minutes", max_categories=8))
spc.render.plotly_pareto(pareto)
```

CUSUM defaults: detect a doubling of the odds (`odds_ratio=2.0`) in the rate
*increasing* (`detect="increase"`; use `"decrease"` or `"both"`), with the
threshold chosen so a false alarm comes on average every `arl0=1000` cases.
`spc.cusum_arl()` and `spc.cusum_threshold()` expose the run-length
calculation (refused above a run length of 1e13, where it stops being
reliable); see the caveats below.

## Test data

Two separate pools, see each directory's own README:

- `data/` — synthetic, purpose-built series (`data/README.md`), each tuned
  to isolate one specific SPC behaviour for demos and integration tests.
- `tests/golden/nhsrdatasets/` — real, pinned NHS/ONS snapshots (A&E
  attendances, weekly mortality, ONS population), fetched via
  `scripts/fetch_nhsrdatasets.py` (needs the `dev` extra) and loaded
  through `tests/nhs_fixtures.py`.

## Statistical caveats

Four things worth knowing before trusting a chart's output, all kept as
documented behaviour:

- **XmR assumes approximate normality.** `LIMIT = 2.66` is a ~3σ-equivalent
  width; skewed data (common for count/rate/behavioural metrics) will have
  a real false-alarm rate different from the nominal one. See
  `core/limits.py`'s module docstring.
- **`limit_mode="segment"` (the default) can mute the very shift it's
  meant to catch**, by computing limits from data that includes the shift
  itself (a Phase I/Phase II conflation — see `core/segments.py`'s module
  docstring). Use `fixed_n` or `expanding` with a `fix_after_n_points`
  covering only known-stable history for real sensitivity to a new shift.
- **No gap detection.** A missing period in the input is silently treated
  as adjacent to its neighbour when computing the moving range.
- **The default centre line is a flat mean, not a seasonal or
  demographic model.** Every `limit_mode` (`segment`, `fixed_n`,
  `expanding`) computes its centre line as a plain average over some
  window of points — it has no notion of a metric's seasonal pattern
  (e.g. winter pressure on A&E admissions) or of gradual demographic
  drift (population growth, an ageing catchment) changing what "normal"
  should mean over a long window. A flat mean will read real seasonal
  variation as special-cause noise. This library deliberately doesn't
  attempt that modelling — forecasting and seasonal/demographic
  adjustment are a different problem from control-limit calculation, out
  of scope by design. Instead, it accepts a caller-supplied
  `BaseChartOptions.baseline` (a list of `ExternalBaseline` entries, one
  `expected_value` per date/facet) and computes limits against that
  instead of a flat mean when present — building the better baseline
  (seasonal, demographically adjusted, or otherwise) is left to the
  caller. See the "External baseline" point under "Design choices and
  known limitations" below.

## Design choices and known limitations

- **Inputs that would otherwise be ignored raise or warn.** An unknown rule
  name, a missing or unparseable date, a missing facet value, and infinite
  values raise a `ValueError` naming the problem. For `rebase` (dates and facet
  keys) and the per-facet dicts `target`, `spec_upper` and `spec_lower`, an
  option that matches nothing in the data raises (almost certainly a typo or
  the wrong frame), while one that matches only in part warns and runs, so one
  config can serve several filtered frames. For `t`, rebase dates are matched
  against the event timestamps. Baseline entries may cover more dates or
  facets than the data.
- **Row order.** Rows sharing a date within a facet (outside `xbar_r`/`xbar_s`,
  which treat them as a subgroup) have no defined order, and order changes the
  moving range and run rules, so they warn. `sequence_field` is a tie-break
  column for every kind except `xbar_r`, `xbar_s` and `t` (which reject it);
  with it, rows still tied on date and sequence raise. `cusum()` has the same
  option.
- **`suggest_rebase()` is a prompt to look, not evidence.** It reuses the
  rules, so it inherits their false-alarm rate: on stationary normal data
  roughly one 40-point series in five gets a suggestion, nearly all at "low"
  confidence. The candidate is the first point where a rule fired, so it can
  trail a real change by up to a run length. A run starting at the first point
  of a segment is never suggested, and `min_points_since` must be at least 1.
- **Option models are closed.** A misspelled option name (`facet_feild=`,
  `targt=`) raises instead of being ignored, on every options type and on
  `ExternalBaseline`. `target`, `spec_upper`, `spec_lower` and a baseline's
  `expected_value` must be finite. `ChartResult.options` keeps the options
  class the chart was run with, so `denominator_field` is in the JSON and a
  stored result parses back into `ProportionChartOptions`; its JSON schema lists
  both options classes.
- **Laney p′/u′ with a zero centre line.** A window or baseline with no
  events gives a centre line of 0, whose standard error is 0. Those points are
  left out of the dispersion estimate and get the same zero-width limits as the
  classical chart, so any later non-zero value is flagged; if no pair of points
  has a defined Z the factor is 1.
- **Facets.** `facet_field` splits a chart by one variable;
  `column_facet_field` adds a second, treating `facet_field` as the row
  variable. Calculation uses one combined key, `"<row> | <column>"` (a
  value containing ` | ` raises), and `PointResult.facet_parts` gives the
  pieces back. Per-facet `target`, `rebase`, `baseline` and `spec_*` are
  keyed by that combined string. A row with a missing facet value raises.
  `spc.render.plotly` draws a multi-facet result as small multiples with
  independent y-axes (a rows x columns grid for two variables).
- **Bernoulli CUSUM assumes independent cases with a constant baseline
  rate.** It has no risk adjustment: if the case mix drifts (sicker
  patients, a changed referral pattern) the score will move without the
  care changing. `p0` must be supplied; estimating it from a short window
  adds uncertainty that the run-length figures do not include
  (`estimate_event_rate` refuses windows with fewer than 5 events or 5
  non-events). A CUSUM depends on case order: cases are ordered by
  timestamp, `sequence_field` breaks ties, and tied cases without one
  trigger a warning.
- **CUSUM run lengths are a numerical approximation.** `cusum_arl` solves a
  discretised Markov chain (800 states by default). It matches the
  hand-solved cases exactly and a Monte Carlo simulation to within noise
  (about 1%), but converges more slowly for very long run lengths and for
  odds ratios near 1 or very large; `cusum_threshold` warns when a finer
  lattice disagrees by more than 2%. With `detect="both"`, `arl0` is the
  run length for a signal on either side (each side is set to twice
  that); an explicit `h` is used as given on both sides.
- **Pareto ranks are descriptive.** There is no uncertainty on the counts:
  with small counts the order of near-tied categories is sampling noise.
  Category values are matched as text, so `1` and `"1"` merge and `"A"` and
  `"a"` do not.

A few structural choices were deliberate scope decisions, not settled
facts — worth knowing before depending on them:

- **Xbar-R / Xbar-S return only the means chart.** Standard practice
  reviews the means chart alongside its companion range (R) or
  standard-deviation (S) chart; only the means chart is exposed via
  `ChartResult` today. The underlying constants (`xbar_r_constants()`,
  `xbar_s_constants()`) are already computed internally and available if
  you need to build the companion chart yourself.
- **Subgroup/denominator size must stay constant within a segment** for
  `xbar_r`, `xbar_s`, and `np`. These charts' formulas assume a fixed n
  by definition — per-point varying sizes raise rather than silently
  produce misleading limits.
- **`t`/`g` (rare-event) charts use the plain `BaseChartOptions`.**
  `value_field` holds one timestamp per rare event for `t` (elapsed time
  is computed internally via diff; `date_field` is required by the shared
  options but not read) and a per-row count for `g` (with its own
  `date_field`). A numeric `value_field` for `t` raises. `suggest_chart_type()`
  proposes `t` for a datetime `value_field` but never `g`, since integer
  counts between events look like `c`/`u` counts; choose `kind="g"`
  explicitly. `facet_field`, `rebase`, `baseline` and `target` work as for
  other charts. For `t`, gaps are computed within each facet (a facet's
  first event yields no point), and the gap straddling a `rebase` date
  belongs to the new segment, so the rebase date must equal an event
  timestamp.
- **P-chart's upper control limit isn't clipped at 1.** A raw p-chart UCL
  can mathematically exceed 1 for small p̄ / small n — a known, documented
  limitation in SPC literature, not an oversight here.
- **Laney's overdispersion correction (`p_prime`/`u_prime`) is opt-in**,
  never a default or a `suggest_chart_type()` suggestion, and is only
  offered for `p`/`u` — not `np`/`c`, which assume a constant denominator
  and aren't the case Laney's method addresses. It earns its keep only
  when there's genuine overdispersion beyond what the binomial/Poisson
  model predicts; classical `p`/`u` remain simpler and equally valid
  otherwise.
- **`run_rule_length` offers two philosophies** for the seven-point trend
  and shift rules: a fixed length (default, backward compatible) or
  `"proportional"`, a sample-size-scaled bound. Neither is universally
  "more correct" — a fixed length under-triggers on short segments and
  over-triggers on long ones, a real statistical tradeoff either way.
- **External baseline support for `p`/`np`/`c`/`u` (and their Laney variants) recenters each point
  directly against the supplied baseline value**, rather than reusing the
  subtract/residual/add-back mechanism `xmr` uses — the two families'
  variance structures (heteroscedastic binomial/Poisson vs. constant-width)
  aren't compatible with the same technique.
- **`spec_upper`/`spec_lower` are legend-visible in the Plotly renderer,
  under a customisable name** (`BaseChartOptions.spec_label`, falling back
  to "Spec limit") — useful for naming what the requirement actually is,
  e.g. `"Spec limit (78% target)"`. Each may be a plain float or a per-facet
  dict, and is drawn in each facet's panel; `PointResult.spec_breach` is
  computed per facet either way.

## Install

Install a tagged release directly from GitHub (requires `git`):

```bash
python -m pip install "spc-lib-py @ git+https://github.com/Kensa-Health/spc-lib-py@v0.1.0"            # calculation only
python -m pip install "spc-lib-py[render] @ git+https://github.com/Kensa-Health/spc-lib-py@v0.1.0"   # optional Plotly charts
```

Keep the `@v0.1.0` tag: without it pip installs the current tip of `main`, which changes
with every merge. This package is not published on PyPI, so install it from Git only. Do not
install the unrelated `spc` distribution from PyPI: it shares the import name and should not
be installed in the same environment.

From a source checkout, use `python -m pip install .` or `python -m pip install ".[render]"`.
For editable development setup, see [CONTRIBUTING.md](CONTRIBUTING.md).

The distribution is named `spc-lib-py`; the import name is `spc` (`import spc`). The core
(numpy, pandas, pydantic) is all `import spc` itself needs. Optional extras:

- `render`: Plotly, for `spc.render.plotly`, `plotly_pareto` and `plotly_cusum`.
- `extended`: SciPy, only for an `xbar_r` chart with subgroups larger than 10, whose
  constants are computed by numerical integration.
- `test`: pytest, ruff, mypy, the pandas and SciPy type stubs, and pyarrow.
- `dev`: pyreadr and pyarrow, only to run `scripts/fetch_nhsrdatasets.py`.
- `notebook`: JupyterLab, pyarrow, Plotly and Kaleido for the demo notebook.
- `release`: build and twine for checking distribution archives.

The Plotly renderers are provided for demos, quick inspection and integration tests,
not as a production-grade UI component. Anybody building a real product surface will
need to render `ChartResult` themselves against their own design system.

## Usage

```python
from datetime import date

import pandas as pd
import spc
from spc import BaseChartOptions, ProportionChartOptions

# Small synthetic weekly series; no downloaded data required.
df = pd.DataFrame({
    "period": pd.date_range("2024-01-01", periods=20, freq="W-MON"),
    "admissions": [100, 110, 105, 95, 100] * 4,
    "breaches": [8, 10, 9, 7, 8] * 4,
    "attendances": [150, 160, 155, 145, 150] * 4,
    "hospital": ["example"] * 20,
})
options = BaseChartOptions(value_field="admissions", date_field="period")
result = spc.chart(df, kind="xmr", options=options)

result.model_dump_json()               # ChartResult contract, ready for an API/storage
# Optional, after installing the render extra:
# spc.render.plotly(result, title="Admissions").show()

# p chart: breaches out of attendances (a real proportion, varying denominator)
p_options = ProportionChartOptions(
    value_field="breaches", date_field="period", denominator_field="attendances"
)
p_result = spc.chart(df, kind="p", options=p_options)

# One chart per facet, a known process change, and a target; options are keyword arguments
# and a misspelled name raises rather than being ignored
options = BaseChartOptions(
    value_field="admissions",
    date_field="period",
    facet_field="hospital",
    rebase=[date(2024, 3, 25)],
    target=4500.0,
    improvement_direction="decrease",
)
```

`spc.suggest_chart_type(df, "admissions", denominator_field=..., date_field=...)` proposes a
kind from the shape of the data, and `spc.suggest_rebase(result)` proposes candidate rebase
dates from a result; both only suggest.

## Demo notebook

From a source checkout, install `python -m pip install ".[notebook]"`, then run
`python -m jupyterlab notebooks/demo.ipynb`. The notebook reads the bundled fixtures
relative to the `notebooks/` directory. Its static SVG output uses Kaleido, which needs
Chrome; if Chrome is not installed, run `plotly_get_chrome` to install it.
For interactive output without Kaleido/Chrome, change the first cell's renderer to
`pio.renderers.default = "plotly_mimetype"`.

## Tests

Run all local validation checks with Python 3.11 or later (no preinstalled project tools needed):

```bash
python scripts/check_local.py
```

The runner creates fresh virtual environments for tests, formatting/lint/type checks and
packaging. It builds and inspects a wheel/source archive, smoke-tests the core-only installed
wheel outside the checkout, and runs the full test suite from the unpacked source archive in
another environment. Dependency installation needs network access and `venv`/`ensurepip`.
It uses the current working tree, including uncommitted edits; avoid editing it during a run.
Dependency installs use pip's isolated mode. Local `PYTEST_*` overrides and `PYTHONOPTIMIZE`
are cleared so they cannot silently reduce verification.

To check several installed Python versions, repeat `--python` with an executable name or path:

```bash
python scripts/check_local.py --python python3.11 --python python3.12 --python python3.13 --python python3.14
python scripts/check_local.py --checks quality tests
```

On Windows, `py -3.12 scripts/check_local.py` uses that interpreter; `--python` accepts paths
to other `python.exe` installations. Every selected group runs for each requested interpreter.
Missing interpreters and failed checks produce a nonzero exit; other groups/interpreters still
run. Each command has a 20-minute timeout, configurable with `--timeout SECONDS`.

Logs, built archives and `report.json` are retained in a fresh directory under ignored
`build/local-checks/`. The report records checks, interpreter/OS/architecture, Git commit and
working-tree status. Temporary environments are removed at the end. This tests the current OS
only: run it inside Linux/Windows environments too for cross-platform coverage.

### Local validation limits

GitHub Actions CI is disabled; its previous definition is retained in
`.github/workflows/ci.yml.disabled`. Checks must be run manually before merging or releasing.

- The default run covers one Python interpreter on the host OS/architecture. The former matrix
  tested Python 3.11–3.14 on Linux and Python 3.12 on macOS/Windows. Reproduce that coverage by
  running the script in those environments; it does not install interpreters or create OS guests.
- Fresh virtual environments isolate Python packages but share the host OS and system tools.
  Linux/Windows behaviour and x86-64 behaviour cannot be inferred from an Apple Silicon Mac run.
- The script supplies no automatic push/PR results or GitHub status checks. Maintainers must
  review and retain the reports; `--checks` can produce a successful run with groups omitted.
  Remove any branch-protection requirements for the retired CI checks to avoid blocked merges.
- The script checks a mutable working tree, including uncommitted files. Git commit/status
  records do not freeze or hash its contents. Use a clean checkout of the reviewed commit for
  release validation and avoid editing it while checks run.
- Dependencies and build tools use current compatible releases rather than a lock file.
  Results may change between runs; this does not verify minimum dependency versions or scan
  dependencies for vulnerabilities. The former CI had the same dependency limitations.
- The installed wheel gets a core smoke test; full suites run in the checkout and unpacked
  source archive. Each command has a timeout, but there is no overall run time limit.

The runner has been verified on macOS arm64 with Python 3.12. Other Python versions and operating
systems still need recorded runs. See [RELEASING.md](RELEASING.md) for the release checklist.

For quick iteration in an existing development environment, individual checks are:

```bash
pip install -e ".[render,test]"
pytest                    # warnings are errors (see [tool.pytest.ini_options])
python scripts/format_code.py --check   # ruff format + the signature layout (see Conventions)
ruff check .
mypy                      # strict, with warn_unreachable (see [tool.mypy])
```

Besides the unit tests, `tests/test_properties.py` checks invariants that must hold
for every chart kind over seeded random inputs (point counts, limit ordering, JSON
round trip, a faceted chart equal to one chart per facet, row order not mattering),
and `tests/test_differential.py` compares the library with deliberately naive
reference implementations in `tests/reference.py` (XmR limits, the four rules,
attribute, Laney and Xbar limits, the autocorrelation adjustment, and a rebased
chart equal to the two charts either side of it).

## Conventions

Code here is written so that bad input fails loudly and wrong numbers do not pass silently:

- Checks on user data live in one place (`spc/core/validation.py`), run before any calculation.
  Missing, infinite and unparseable values raise a `ValueError` that names the problem.
- Options models reject unknown fields, and anything that selects a branch (a chart kind, a
  direction) names every case and raises on the rest.
- Function signatures with parameters are written one parameter per line, parameters indented
  eight spaces and the closing parenthesis under the `def`, with no trailing comma; options after
  a bare `*` are keyword-only. `ruff format` alone would undo that layout, so use
  `python scripts/format_code.py` to format and `--check` to verify.
- Tests run with warnings as errors; each fixed bug has a regression test, and statistics are
  compared with independent reference implementations.

## Contributing and support

Kensa Health maintains this project. Use [GitHub issues](https://github.com/Kensa-Health/spc-lib-py/issues)
for questions, reproducible bugs and feature proposals. See [CONTRIBUTING.md](CONTRIBUTING.md)
for development and PR guidance, [SECURITY.md](SECURITY.md) for private vulnerability reports,
and [CHANGELOG.md](CHANGELOG.md) for release notes. Support is on a best-effort basis.

## Licence and data

The code is released under the [MIT licence](LICENSE). The bundled test data in `tests/golden/nhsrdatasets/`
comes from the NHSRdatasets R package (CC0-1.0), with the A&E attendances, mortality and ONS
population series sourced from NHS England and the Office for National Statistics under the
Open Government Licence v3.0 (see `tests/golden/nhsrdatasets/PROVENANCE.md`).
Contains public sector information licensed under the Open Government Licence v3.0.

This is an independent project. It follows the published NHS "Making Data Count" SPC
methodology and is not affiliated with or endorsed by NHS England.

The XmR calculation, the special-cause rules and several option and field names follow the
MIT-licensed R package [NHSRplotthedots](https://github.com/nhs-r-community/NHSRplotthedots);
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the details and its licence text.

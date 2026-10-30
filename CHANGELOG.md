# Changelog

## 0.1.0 - 30-10-2026

Initial release.

- Eleven SPC chart kinds, external baselines, rebasing and one- or two-variable facets.
- Bernoulli CUSUM and Pareto analysis with typed, JSON-serialisable results.
- Optional Plotly rendering and extended Xbar-R constants through SciPy.
- Input validation, synthetic and public aggregate test data, and differential checks.
- XmR parity fixtures generated with pinned NHSRplotthedots 0.2.2 on two NHS series.
- T/G charts use the same power calculation for observations and external baselines,
  preventing rounding differences from creating false signals for an exact baseline.
- Release packaging, contributor documentation and a local validation runner
  (`scripts/check_local.py`). GitHub Actions CI is disabled; releases are validated manually
  (see [RELEASING.md](RELEASING.md)).

The API is provisional during 0.x. Breaking changes belong in a minor version; compatible
fixes belong in a patch version.

# Contributing

Report bugs through [GitHub issues](https://github.com/Kensa-Health/spc-lib-py/issues).
Ask questions in [Discussions Q&A](https://github.com/Kensa-Health/spc-lib-py/discussions/categories/q-a).
Propose new statistical methods, integrations or changes to the public API in
[Discussions Ideas](https://github.com/Kensa-Health/spc-lib-py/discussions/categories/ideas) first;
maintainers open an issue once there is agreement, before anyone implements it.
Kensa Health Virtual Care Ltd maintains the project; repository maintainers review pull requests.
For security reports, follow [SECURITY.md](SECURITY.md). Everyone taking part follows the
[code of conduct](CODE_OF_CONDUCT.md).

## Development setup

Use Python 3.11 or later in a virtual environment. From your clone:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[render,test,release]"
```

Tests use committed synthetic and public aggregate data. They need no network, R installation
or private datasets. The `dev` extra is only for refreshing the NHSRdatasets snapshots.
See [tests/golden/README.md](tests/golden/README.md) for reference-fixture provenance.

## Checks before a pull request

For a complete local run, including isolated installation and packaging checks:

```bash
python scripts/check_local.py
```

This creates fresh environments and downloads dependencies. Results are saved under
`build/local-checks/`; see the [README](README.md#tests) for selecting Python interpreters
or check groups. Run on each target OS when checking platform compatibility.
GitHub Actions CI is disabled. Include the tested commit, Python version, OS/architecture and
check groups in your PR, and provide the report/logs for maintainer review. A pass from selected
groups or one Mac interpreter does not establish full supported-platform coverage. See the
[local validation limits](README.md#local-validation-limits) for the remaining gaps.

For quick iteration using your development environment:

```bash
python scripts/format_code.py
python scripts/format_code.py --check
python -m ruff check .
python -m mypy
python -m pytest
```

Use the formatter script rather than `ruff format` directly: it applies the project's
one-parameter-per-line function signature layout after Ruff. Write comments and documentation
in UK English. Warnings are errors in the test suite; assert expected warnings explicitly.

For a bug fix, include a small regression test that fails without the fix. For statistical
changes, state the assumptions, cite the method, and compare with hand calculations or an
independent implementation. Do not regenerate golden outputs merely to make a failing test pass.
Explain intended differences from reference implementations and preserve their attribution.

Keep changes focused. Describe the problem, resulting behaviour and checks run in the PR.
Update the README for API changes and the Unreleased section of [CHANGELOG.md](CHANGELOG.md).
Use synthetic or public aggregate data in examples and reports; do not include patient data,
credentials or private paths. Be respectful and make feedback specific to the work.

## Contributions and releases

Submit only material you have the right to contribute. Sign off every commit (`git commit -s`)
to certify the [Developer Certificate of Origin](https://developercertificate.org/); commits made
in the GitHub web interface require sign-off. Contributions are supplied under the
project's [MIT licence](LICENSE); retain any applicable third-party notices. No copyright
assignment is required. The 0.x API can change between minor releases; document breaking changes.
Maintainers follow [RELEASING.md](RELEASING.md) when preparing a release.

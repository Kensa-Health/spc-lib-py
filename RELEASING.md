# Releasing

## Version and validation

1. Update the matching versions in `pyproject.toml` and `spc/__init__.py`. Use patch versions for
   compatible fixes and minor versions for breaking 0.x API changes. Document migrations.
2. Move the relevant Unreleased changelog entries into a version section with the release date
   in dd-mm-yyyy format. Never reuse an already published version.
3. In a clean checkout, use Python 3.11 or later to run all local checks:

   ```bash
   python scripts/check_local.py
   ```

   The runner creates fresh environments and includes formatting, lint, types, tests, build,
   twine, archive-content checks, an isolated core-only wheel smoke test and source-archive
   tests. Keep its `build/local-checks/<run>/report.json` and the checked archives under that
   run's interpreter-specific `dist/` directory. Use `--python` repeatedly for installed
   interpreter versions; missing requested interpreters fail the run.

4. Review recorded full local results for the reviewed commit across Python 3.11–3.14 on
   Linux and Python 3.12 on macOS/Windows, matching the former hosted matrix. Run all groups
   (omit `--checks`) in each environment and retain the reports; no automatic CI run supplies
   this evidence. Verify the recorded commit, clean starting/ending Git status, interpreter,
   OS/architecture and all group outcomes. Reports record provenance but do not freeze source
   contents, so use a clean checkout and avoid edits throughout validation. A Mac-only run
   verifies macOS only; the local runner has so far been verified on macOS arm64/Python 3.12.
   Confirm both archives contain LICENSE and THIRD_PARTY_NOTICES.md. Review the
   [local validation limits](README.md#local-validation-limits), including host-system effects,
   unlocked dependencies and limited wheel smoke coverage, before approving the release.
5. Review the reference fixture provenance and any deliberate numerical differences. Keep the
   NHS/ONS data licences separate from the code licence when distributing test data.

## Publication

Publication is a maintainer action. Releases are distributed from GitHub; the package is not
published on PyPI.

1. Create an annotated `vX.Y.Z` tag on the reviewed commit and a GitHub release with the
   changelog and the checked archives (the sdist and wheel from the validation run).
2. Attach or store the validation reports for that commit with the release record.
3. In a new virtual environment with no GitHub credentials, install the tagged version:

   ```bash
   python -m pip install "spc-lib-py[render] @ git+https://github.com/Kensa-Health/spc-lib-py@vX.Y.Z"
   ```

   Run the README quickstart and check that the installed version matches the tag.
4. Update the install tag in the README to the new version.

Never move or reuse a published tag. If a release is wrong, publish a new patch version.
No workflow in this repository validates or publishes packages.

"""Fetches, converts, and pins real NHS/ONS datasets from the R package
NHSRdatasets.

Run manually, occasionally, never from CI or test setup:
    python scripts/fetch_nhsrdatasets.py

Requires `pyreadr` (pip install pyreadr) -- a dev/fetch-time-only
dependency, not a runtime dependency of spc itself (see pyproject.toml's
`dev` extra).

Pin note (deviation from the source spec, found by actually running this):
`ons_uk_population_2023` does not exist at the `v0.3.3` release tag despite
the spec listing it alongside ae_attendances/ons_mortality as available
there -- confirmed via the GitHub contents API, it was only added on `main`
after that release. Pinned to the exact commit that added it instead of
`v0.3.3`, keeping the spec's own "pin explicitly, don't track a moving
branch" principle intact rather than pinning to the literal branch name.
"""

from __future__ import annotations

import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd
import pyreadr

REPO = "nhs-r-community/NHSRdatasets"
TAG_PINNED_DATASETS = {
    "ae_attendances": "v0.3.3",
    "ons_mortality": "v0.3.3",
    # Not present at v0.3.3 (verified against the GitHub contents API at
    # that tag) -- only added on `main` later. Pinned to that exact commit,
    # not the branch name, so this stays reproducible.
    "ons_uk_population_2023": "43b4c43337124ed40971f71ca7defc87e923fbe0",
}

OUT_DIR = Path(__file__).parent.parent / "tests" / "golden" / "nhsrdatasets"


def fetch_and_convert(
        name: str,
        ref: str
) -> pd.DataFrame:
    url = f"https://raw.githubusercontent.com/{REPO}/{ref}/data/{name}.rda"
    rda_path = OUT_DIR / f"_raw_{name}.rda"
    urllib.request.urlretrieve(url, rda_path)  # noqa: S310 -- fixed https URL, not user input
    result = pyreadr.read_r(str(rda_path))  # OrderedDict: {object_name: DataFrame}
    df = result[name]
    rda_path.unlink()  # don't keep the raw .rda around, only the converted snapshot
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provenance_lines = [
        "# NHSRdatasets snapshot\n",
        f"Retrieved: {date.today().isoformat()}\n",
        "Licence: NHSRdatasets package is CC0-1.0; ae_attendances, ons_mortality, "
        "ons_uk_population_2023 additionally sourced from NHS England / ONS under "
        "Open Government Licence v3.0.\n",
        "\nContains public sector information licensed under the "
        "[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).\n",
        "The source package uses [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/).\n",
        "\nFor the derived R reference outputs, see [parity documentation](../README.md).\n",
        "\n## Per-dataset source ref\n",
    ]
    for name, ref in TAG_PINNED_DATASETS.items():
        df = fetch_and_convert(name, ref)
        df.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        print(f"{name}: {len(df)} rows, {len(df.columns)} cols -> {OUT_DIR / f'{name}.parquet'}")
        provenance_lines.append(f"- `{name}`: https://github.com/{REPO}/tree/{ref}/data/{name}.rda\n")

    (OUT_DIR / "PROVENANCE.md").write_text("".join(provenance_lines))
    print(f"Wrote PROVENANCE.md to {OUT_DIR}")


if __name__ == "__main__":
    main()

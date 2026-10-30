"""Check release archives for licences, typing metadata and runnable test sources.

Run after `python -m build`: python scripts/check_dist.py dist
"""

from __future__ import annotations

import argparse
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LICENCES = ("LICENSE", "THIRD_PARTY_NOTICES.md")
SOURCE_SUFFIXES = {".py", ".R", ".csv", ".parquet", ".md", ".json", ".ipynb"}
PRIVATE_PARTS = {".memory", ".claude", ".agents", ".git", ".venv", "__pycache__", "docs", "reviews"}


def require(
        condition: bool,
        message: str
) -> None:
    """Fail a release check with an actionable message, including under python -O."""
    if not condition:
        raise ValueError(message)


def check_dist(
        directory: Path
) -> None:
    """Inspect exactly one wheel/sdist pair against the source tree and project metadata."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    version = config["project"]["version"]
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    require(len(wheels) == len(sdists) == 1, "Expected exactly one wheel and one sdist; use a fresh output directory")
    prefix = f"spc_lib_py-{version}"
    require(wheels[0].name == f"{prefix}-py3-none-any.whl", "Wheel filename does not match the project version")
    require(sdists[0].name == f"{prefix}.tar.gz", "Source archive filename does not match the project version")

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        info = f"{prefix}.dist-info"
        metadata = Parser().parsestr(wheel.read(f"{info}/METADATA").decode())
        require(metadata["Name"] == "spc-lib-py", "Wrong wheel distribution name")
        require(metadata["Version"] == version, "Wrong wheel version")
        require(metadata["License-Expression"] == "MIT", "Missing MIT SPDX metadata")
        require(set(metadata.get_all("License-File", [])) == set(LICENCES), "Missing licence-file metadata")
        for licence in LICENCES:
            path = f"{info}/licenses/{licence}"
            require(path in names, f"Wheel is missing {licence}")
            require(wheel.read(path) == (ROOT / licence).read_bytes(), f"Wheel has stale {licence}")
        for source in (ROOT / "spc").rglob("*"):
            if source.is_file() and (source.suffix == ".py" or source.name == "py.typed"):
                require(source.relative_to(ROOT).as_posix() in names, f"Wheel is missing {source.name}")
        require(all(Path(n).parts[0] in {"spc", info} for n in names), "Unexpected files in wheel")

    with tarfile.open(sdists[0]) as sdist:
        members = sdist.getmembers()
        require(all(m.name == prefix or m.name.startswith(prefix + "/") for m in members), "Invalid sdist root")
        require(all(m.isfile() or m.isdir() for m in members), "Source archive contains links or special files")
        names = {m.name.removeprefix(prefix + "/") for m in members if m.isfile()}
        require(
            not any(PRIVATE_PARTS.intersection(Path(n).parts) or ".." in Path(n).parts for n in names),
            "Source archive contains private or unsafe paths",
        )
        for licence in LICENCES:
            member = sdist.extractfile(f"{prefix}/{licence}")
            require(member is not None, f"Source archive is missing {licence}")
            if member is not None:
                require(member.read() == (ROOT / licence).read_bytes(), f"Source archive has stale {licence}")
        required = {
            "pyproject.toml",
            "MANIFEST.in",
            "README.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "CHANGELOG.md",
            "RELEASING.md",
            "spc/py.typed",
        }
        for folder in ("spc", "tests", "scripts", "data", "notebooks"):
            required.update(
                p.relative_to(ROOT).as_posix()
                for p in (ROOT / folder).rglob("*")
                if p.is_file()
                and p.suffix in SOURCE_SUFFIXES
                and not {"__pycache__", "demo_output"}.intersection(p.parts)
            )
        missing = sorted(required - names)
        require(not missing, f"Source archive is missing: {', '.join(missing)}")
    print(f"Checked {wheels[0].name} and {sdists[0].name}: licences, package modules and test sources present")


def main() -> None:
    """Read the archive directory and validate its release artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    check_dist(parser.parse_args().directory)


if __name__ == "__main__":
    main()

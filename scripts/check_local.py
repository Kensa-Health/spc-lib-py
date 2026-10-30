"""Run project validation locally in fresh virtual environments.

    python scripts/check_local.py
    python scripts/check_local.py --python python3.11 --python python3.12
    python scripts/check_local.py --checks quality tests

Requires Python 3.11+, venv/ensurepip and network access to install dependencies.
Checks the current working tree on this OS; it does not emulate other operating systems.
Logs, archives and report.json are retained under the ignored build/local-checks/ directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

# Coverage gaps versus the former hosted CI matrix:
# - Only the host OS/architecture and requested, already-installed Python versions run.
# - Fresh venvs still share the host OS/tools; this is not a clean hosted runner image.
# - Runs are manual: there is no automatic push/PR check or required-status enforcement.
# - The working tree can change during a run; Git status is recorded, not a frozen snapshot.
# - Dependencies/build tools resolve current compatible releases, without a lock or CVE scan.
# - The wheel gets a core smoke test; full tests run in the checkout and from the sdist.
# - --checks may omit groups, and the timeout limits each command, not the whole run.
# See README.md's "Local validation limits" and RELEASING.md for required release coverage.
ROOT = Path(__file__).resolve().parent.parent
CHECKS = ("tests", "quality", "packages")
PROBE = """
import json, platform, sys
print(json.dumps({
    "executable": sys.executable,
    "version": list(sys.version_info[:3]),
    "os": platform.system(),
    "architecture": platform.machine(),
}))
"""
WHEEL_SMOKE = """
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
import pandas as pd
import spc

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

require(Path(spc.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),
        "spc was not imported from the isolated wheel environment")
df = pd.DataFrame({"period": pd.date_range("2024-01-01", periods=20),
                   "value": [10, 12, 11, 9, 10] * 4})
result = spc.chart(df, kind="xmr", options=spc.BaseChartOptions(value_field="value", date_field="period"))
require(len(result.points) == 20, "Incorrect point count")
require(spc.ChartResult.model_validate_json(result.model_dump_json()) == result, "JSON roundtrip failed")
require(spc.__version__ == importlib.metadata.version("spc-lib-py"), "Version mismatch")
for name in ("plotly", "scipy"):
    require(name not in sys.modules and importlib.util.find_spec(name) is None,
            name + " must not be required in a core-only environment")
print("Installed core-only wheel: import, calculation, JSON roundtrip and version checks passed")
"""


def positive_seconds(
        value: str
) -> float:
    """Accept a finite, positive subprocess timeout."""
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and greater than zero")
    return seconds


def git_state() -> dict[str, str | None]:
    """Describe the working tree, or leave Git fields empty for an unpacked sdist."""
    state: dict[str, str | None] = {"commit": None, "status": None}
    for key, args in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--porcelain"])):
        try:
            result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            state[key] = result.stdout.strip()
    return state


def extract_source(
        archive: Path,
        destination: Path
) -> Path:
    """Extract regular files/directories only, rejecting paths outside the destination."""
    destination.mkdir()
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            target = destination / member.name
            if "\\" in member.name or not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = source.extractfile(member)
                if stream is None:
                    raise ValueError(f"Unreadable archive member: {member.name}")
                with stream, target.open("wb") as output:
                    shutil.copyfileobj(stream, output)
            else:
                raise ValueError(f"Unsupported archive member: {member.name}")
    roots = list(destination.iterdir())
    if len(roots) != 1 or not (roots[0] / "pyproject.toml").is_file():
        raise ValueError("Expected one source tree containing pyproject.toml")
    return roots[0]


class Runner:
    """Execute commands without a shell, recording failures and retaining their logs."""

    def __init__(
            self,
            directory: Path,
            timeout: float
    ) -> None:
        self.directory = directory
        self.timeout = timeout
        self.steps: list[dict[str, object]] = []
        self.environment = os.environ.copy()
        # Local overrides must not disable assertions or silently narrow the suite.
        for name in list(self.environment):
            if name in {"PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE", "VIRTUAL_ENV"} or name.startswith("PYTEST_"):
                self.environment.pop(name)
        self.environment["PYTHONUNBUFFERED"] = "1"

    def run(
            self,
            name: str,
            command: Sequence[str | Path],
            *,
            cwd: Path = ROOT
    ) -> Path:
        """Run a bounded command; a failed command fails its check group."""
        log = self.directory / f"{len(self.steps) + 1:02d}-{name}.log"
        args = [str(arg) for arg in command]
        step: dict[str, object] = {"name": name, "command": args, "cwd": str(cwd), "log": str(log)}
        self.steps.append(step)
        print(f"  {name} ... (log: {log.name})", flush=True)
        start = time.monotonic()
        try:
            with log.open("w", encoding="utf-8") as output:
                result = subprocess.run(
                    args,
                    cwd=cwd,
                    env=self.environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout,
                    check=False,
                )
            step["returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"{name} exited with code {result.returncode}; see {log}")
            step["status"] = "passed"
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            step["status"] = "failed"
            step["error"] = str(exc)
            raise RuntimeError(f"{name} failed: {exc}") from exc
        finally:
            step["seconds"] = round(time.monotonic() - start, 2)
        return log

    def environment_python(
            self,
            interpreter: str,
            destination: Path,
            name: str
    ) -> Path:
        """Create a fresh environment using the requested interpreter."""
        self.run(f"{name}-venv", [interpreter, "-I", "-m", "venv", destination])
        return destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def install(
            self,
            python: Path,
            name: str,
            requirements: Sequence[str | Path],
            *,
            cwd: Path = ROOT
    ) -> None:
        """Install in the selected environment, ignoring user pip configuration."""
        self.run(
            name,
            [python, "-I", "-m", "pip", "--isolated", "--disable-pip-version-check", "install", *requirements],
            cwd=cwd,
        )

    def check(
            self,
            name: str,
            interpreter: str,
            workspace: Path
    ) -> None:
        """Run one CI job's checks with its own dependencies."""
        python = self.environment_python(interpreter, workspace / name, name)
        if name in ("tests", "quality"):
            self.install(python, f"{name}-install", ["-e", ".[render,test]"])
            if name == "tests":
                self.run("pytest", [python, "-m", "pytest"])
            elif name == "quality":
                self.run("format", [python, "scripts/format_code.py", "--check"])
                self.run("ruff", [python, "-m", "ruff", "check", "."])
                self.run("mypy", [python, "-m", "mypy"])
        elif name == "packages":
            self.packages(interpreter, python, workspace)
        else:
            raise ValueError(f"Unknown check group: {name}")

    def packages(
            self,
            interpreter: str,
            python: Path,
            workspace: Path
    ) -> None:
        """Build/check archives and validate wheel and sdist in separate environments."""
        self.install(python, "build-tools", ["build", "twine"])
        dist = self.directory / "dist"
        self.run("build", [python, "-m", "build", "--outdir", dist])
        archives = sorted(dist.glob("*.whl")) + sorted(dist.glob("*.tar.gz"))
        self.run("archive-content", [python, "scripts/check_dist.py", dist])
        self.run("twine", [python, "-m", "twine", "check", *archives])
        wheel = next(dist.glob("*.whl"))
        sdist = next(dist.glob("*.tar.gz"))
        wheel_python = self.environment_python(interpreter, workspace / "wheel", "wheel")
        self.install(wheel_python, "wheel-install", [wheel], cwd=workspace)
        self.run("wheel-smoke", [wheel_python, "-I", "-c", WHEEL_SMOKE], cwd=workspace)
        source = extract_source(sdist, workspace / "source")
        sdist_python = self.environment_python(interpreter, workspace / "sdist", "sdist")
        self.install(sdist_python, "sdist-install", [".[render,test]"], cwd=source)
        self.run("sdist-pytest", [sdist_python, "-m", "pytest"], cwd=source)


def main(
        argv: Sequence[str] | None = None
) -> int:
    """Run requested groups for each interpreter and return nonzero on any failure."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", action="append", help="Python executable/path; repeat for a version matrix")
    parser.add_argument("--checks", nargs="+", choices=CHECKS, default=list(CHECKS))
    parser.add_argument("--timeout", type=positive_seconds, default=1200, help="seconds per command (default: 1200)")
    args = parser.parse_args(argv)
    checks = list(dict.fromkeys(args.checks))
    interpreters = args.python or [sys.executable]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "build" / "local-checks" / f"{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True)
    runs: list[dict[str, object]] = []
    report: dict[str, object] = {
        "started_utc": datetime.now(UTC).isoformat(),
        "source": str(ROOT),
        "source_mode": "current working tree, including uncommitted changes",
        "git_start": git_state(),
        "checks": checks,
        "runs": runs,
    }
    failed = False
    completed = False
    print(f"Local checks on {platform.system()} / {platform.machine()} only. Results: {output}", flush=True)
    try:
        for index, requested in enumerate(interpreters, start=1):
            directory = output / str(index)
            directory.mkdir()
            runner = Runner(directory, args.timeout)
            outcomes: dict[str, str] = {}
            errors: dict[str, str] = {}
            run: dict[str, object] = {
                "requested_python": requested,
                "checks": outcomes,
                "errors": errors,
                "steps": runner.steps,
            }
            runs.append(run)
            try:
                executable = shutil.which(requested)
                if executable is None:
                    raise ValueError(f"Requested Python not found: {requested}")
                executable = os.path.abspath(executable)
                probe = runner.run("python-info", [executable, "-I", "-c", PROBE])
                info = json.loads(probe.read_text(encoding="utf-8"))
                run["interpreter"] = info
                if tuple(info["version"]) < (3, 11):
                    raise ValueError("Each requested interpreter must be Python 3.11 or newer")
                print(
                    f"Python {'.'.join(map(str, info['version']))} ({info['os']} / {info['architecture']})", flush=True
                )
                with tempfile.TemporaryDirectory(prefix="spc-local-check-") as temporary:
                    for check in checks:
                        try:
                            runner.check(check, executable, Path(temporary))
                            outcomes[check] = "passed"
                        except (RuntimeError, OSError, ValueError, tarfile.TarError) as exc:
                            outcomes[check] = "failed"
                            errors[check] = str(exc)
                            failed = True
                            print(f"FAIL {check}: {exc}", file=sys.stderr, flush=True)
            except (RuntimeError, OSError, ValueError) as exc:
                run["error"] = str(exc)
                outcomes.update({name: "not run" for name in checks if name not in outcomes})
                failed = True
                print(f"FAIL {requested}: {exc}", file=sys.stderr, flush=True)
        completed = True
    except KeyboardInterrupt:
        failed = True
        report["interrupted"] = True
        print("Interrupted; partial results retained.", file=sys.stderr)
    finally:
        failed = failed or not completed
        report["git_end"] = git_state()
        report["finished_utc"] = datetime.now(UTC).isoformat()
        report["status"] = "failed" if failed else "passed"
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for run in runs:
        print(f"{run['requested_python']}: {run['checks']}")
    print(f"{'FAILED' if failed else 'PASSED'} requested local checks. Report: {output / 'report.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Local verification must report failures honestly and isolate archive extraction."""

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_local.py"
_spec = importlib.util.spec_from_file_location("check_local", SCRIPT)
assert _spec is not None and _spec.loader is not None
check_local = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_local)


def test_failed_command_preserves_output_and_exit_code(
        tmp_path
):
    runner = check_local.Runner(tmp_path, 10)
    with pytest.raises(RuntimeError, match="exited with code 7"):
        runner.run("failure", [sys.executable, "-c", "print('failure evidence'); raise SystemExit(7)"], cwd=tmp_path)
    step = runner.steps[0]
    assert step["status"] == "failed"
    assert step["returncode"] == 7
    assert "failure evidence" in Path(step["log"]).read_text()


def test_hung_command_is_a_failure(
        tmp_path
):
    runner = check_local.Runner(tmp_path, 0.1)
    with pytest.raises(RuntimeError, match="timed out"):
        runner.run("timeout", [sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path)
    assert runner.steps[0]["status"] == "failed"


def test_local_overrides_cannot_disable_assertions_or_select_fewer_tests(
        tmp_path,
        monkeypatch
):
    monkeypatch.setenv("PYTHONOPTIMIZE", "2")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k only_one_test")
    runner = check_local.Runner(tmp_path, 10)
    log = runner.run(
        "overrides",
        [sys.executable, "-c", "import os, sys; print(sys.flags.optimize); print(os.environ.get('PYTEST_ADDOPTS'))"],
        cwd=tmp_path,
    )
    assert log.read_text().splitlines() == ["0", "None"]


def test_missing_interpreter_does_not_skip_silently_or_prevent_other_checks(
        tmp_path,
        monkeypatch
):
    monkeypatch.setattr(check_local, "ROOT", tmp_path)
    called = []
    monkeypatch.setattr(check_local.Runner, "check", lambda self, name, interpreter, workspace: called.append(name))
    result = check_local.main(
        ["--python", "spc-python-that-does-not-exist", "--python", sys.executable, "--checks", "tests"]
    )
    report = json.loads(next((tmp_path / "build").rglob("report.json")).read_text())
    assert result == 1
    assert report["status"] == "failed"
    assert report["runs"][0]["checks"] == {"tests": "not run"}
    assert "Requested Python not found" in report["runs"][0]["error"]
    assert report["runs"][1]["checks"] == {"tests": "passed"}
    assert called == ["tests"]


def test_failed_group_does_not_prevent_other_groups_and_report_stays_failed(
        tmp_path,
        monkeypatch
):
    monkeypatch.setattr(check_local, "ROOT", tmp_path)
    called = []

    def fail_tests(
            self,
            name,
            interpreter,
            workspace
    ):
        called.append(name)
        if name == "tests":
            raise RuntimeError("injected test failure")

    monkeypatch.setattr(check_local.Runner, "check", fail_tests)
    assert check_local.main(["--checks", "tests", "quality"]) == 1
    report = json.loads(next((tmp_path / "build").rglob("report.json")).read_text())
    assert report["status"] == "failed"
    assert report["runs"][0]["checks"] == {"tests": "failed", "quality": "passed"}
    assert report["runs"][0]["errors"]["tests"] == "injected test failure"
    assert called == ["tests", "quality"]


def test_unexpected_error_cannot_leave_a_passing_report(
        tmp_path,
        monkeypatch
):
    monkeypatch.setattr(check_local, "ROOT", tmp_path)

    def crash(
            self,
            name,
            interpreter,
            workspace
    ):
        raise LookupError("unexpected bug")

    monkeypatch.setattr(check_local.Runner, "check", crash)
    with pytest.raises(LookupError, match="unexpected bug"):
        check_local.main(["--checks", "tests"])
    report = json.loads(next((tmp_path / "build").rglob("report.json")).read_text())
    assert report["status"] == "failed"


@pytest.mark.parametrize("name", ["../escaped", "/escaped", "..\\escaped"])
def test_source_extraction_rejects_escaping_paths(
        tmp_path,
        name
):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.addfile(tarfile.TarInfo(name), io.BytesIO(b""))
    with pytest.raises(ValueError, match="Unsafe archive path"):
        check_local.extract_source(archive, tmp_path / "unpacked")
    assert not (tmp_path / "escaped").exists()


def test_source_extraction_rejects_links(
        tmp_path
):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("project/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../escaped"
        output.addfile(member)
    with pytest.raises(ValueError, match="Unsupported archive member"):
        check_local.extract_source(archive, tmp_path / "unpacked")


def test_source_extraction_returns_the_packaged_project(
        tmp_path
):
    archive = tmp_path / "source.tar.gz"
    content = b'[project]\nname = "example"\n'
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("project/pyproject.toml")
        member.size = len(content)
        output.addfile(member, io.BytesIO(content))
    project = check_local.extract_source(archive, tmp_path / "unpacked")
    assert (project / "pyproject.toml").read_bytes() == content


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_timeouts_fail_before_starting_checks(
        value
):
    with pytest.raises(SystemExit) as error:
        check_local.main(["--timeout", value])
    assert error.value.code == 2

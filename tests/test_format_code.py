"""scripts/format_code.py: the one-parameter-per-line signature layout."""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "format_code.py"
_spec = importlib.util.spec_from_file_location("format_code", SCRIPT)
assert _spec is not None and _spec.loader is not None
format_code = importlib.util.module_from_spec(_spec)
sys.modules["format_code"] = format_code  # dataclass-free, but keeps tracebacks readable
_spec.loader.exec_module(format_code)


def test_a_packed_signature_becomes_one_parameter_per_line():
    source = "def f(a: int, b: str = 'x',\n      *, c: float | None = None) -> bool:\n    return True\n"
    assert format_code.layout_signatures(source) == (
        "def f(\n"
        "        a: int,\n"
        "        b: str = 'x',\n"
        "        *,\n"
        "        c: float | None = None\n"
        ") -> bool:\n"
        "    return True\n"
    )


def test_no_parameters_and_self_only_stay_on_one_line():
    source = (
        "def now() -> int:\n    return 1\n\n\n"
        "class A:\n    def reset(self) -> None:\n        pass\n\n"
        "    @classmethod\n    def make(cls):\n        pass\n"
    )
    assert format_code.layout_signatures(source) == source


def test_methods_indent_relative_to_their_def():
    source = "class A:\n    def go(self, x, y=1):\n        pass\n"
    assert format_code.layout_signatures(source) == (
        "class A:\n    def go(\n            self,\n            x,\n            y=1\n    ):\n        pass\n"
    )


def test_markers_varargs_async_and_unannotated_defaults():
    source = "async def f(a, /, b, *args: int, c=2, **kwargs: str):\n    pass\n"
    assert format_code.layout_signatures(source) == (
        "async def f(\n        a,\n        /,\n        b,\n        *args: int,\n        c=2,\n        **kwargs: str\n"
        "):\n    pass\n"
    )


def test_a_hash_inside_a_string_default_is_not_a_comment():
    source = 'def f(colour: str = "#a6a6a6", other: int = 1):\n    pass\n'
    assert format_code.layout_signatures(source) == (
        'def f(\n        colour: str = "#a6a6a6",\n        other: int = 1\n):\n    pass\n'
    )


def test_a_signature_with_a_comment_is_left_alone(
        capsys
):
    source = "def f(\n    a,  # the first\n    b,\n):\n    pass\n"
    assert format_code.layout_signatures(source, "demo.py") == source
    assert "comment inside its signature" in capsys.readouterr().out


def test_nested_functions_and_idempotence():
    source = "def outer(a, b):\n    def inner(x, y=2):\n        return x\n    return inner\n"
    once = format_code.layout_signatures(source)
    assert "def inner(\n            x,\n            y=2\n    ):" in once
    assert format_code.layout_signatures(once) == once
    assert ast.dump(ast.parse(once)) == ast.dump(ast.parse(source))


def _ruff_available():
    return subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _ruff_available(), reason="ruff is not installed")
def test_repository_code_is_already_formatted():
    root = SCRIPT.parent.parent
    paths = [str(root / name) for name in ("spc", "tests", "scripts", "data")]
    assert format_code.main(["--check", *paths]) == 0


@pytest.mark.skipif(not _ruff_available(), reason="ruff is not installed")
def test_reindenting_a_docstring_does_not_trip_the_syntax_tree_check():
    source = 'def f(a):\n  """Summary.\n\n      Indented oddly.\n  """\n  return a\n'
    formatted = format_code.format_source(source, "demo.py")
    assert formatted.startswith("def f(\n        a\n):\n")
    assert "Indented oddly." in formatted

"""Formats the code: `ruff format`, then a one-parameter-per-line signature layout.

`ruff format` (Black style) would put every signature back on a 4-space hanging
indent with a trailing comma, so it cannot be run on its own. This runs it first,
then rewrites each signature that has parameters as one parameter per line, indented
eight spaces, with the closing parenthesis on its own line under the `def` and no
trailing comma. A function with no parameters, or a method whose only parameter is
`self` or `cls`, stays on one line.

    python scripts/format_code.py            # format spc, tests, scripts and data in place
    python scripts/format_code.py --check    # exit 1 if any file would change
    python scripts/format_code.py spc/cusum.py
"""

import argparse
import ast
import inspect
import io
import re
import subprocess
import sys
import tokenize
from collections.abc import Sequence
from pathlib import Path

DEFAULT_PATHS = ["spc", "tests", "scripts", "data"]
SKIP_PARTS = {"demo_output", "__pycache__"}
PARAMETER_INDENT = " " * 8


def python_files(
        paths: Sequence[str]
) -> list[Path]:
    """Every .py file under the given paths, skipping generated output."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        candidates = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        found.extend(c for c in candidates if not SKIP_PARTS & set(c.parts))
    return found


def ruff_format(
        source: str,
        filename: str
) -> str:
    """The source formatted by `ruff format`, using the project's configuration."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--stdin-filename", filename, "-"],
        input=source,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _segment(
        source: str,
        node: ast.AST
) -> str:
    """The exact source text of a node (an annotation or a default)."""
    text = ast.get_source_segment(source, node)
    if text is None:
        raise ValueError(f"no source text for {ast.dump(node)}")
    return text


def _parameter(
        source: str,
        arg: ast.arg,
        default: ast.expr | None
) -> str:
    """One parameter as written in a signature: `name: annotation = default`."""
    text = arg.arg
    if arg.annotation is not None:
        text += ": " + _segment(source, arg.annotation)
    if default is not None:
        text += (" = " if arg.annotation is not None else "=") + _segment(source, default)
    return text


def _parameters(
        source: str,
        args: ast.arguments
) -> list[str]:
    """Every parameter in declaration order, including the `/` and `*` markers."""
    positional = args.posonlyargs + args.args
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    parts: list[str] = []
    for index, (arg, default) in enumerate(zip(positional, defaults, strict=True)):
        parts.append(_parameter(source, arg, default))
        if args.posonlyargs and index == len(args.posonlyargs) - 1:
            parts.append("/")
    if args.vararg is not None:
        parts.append("*" + _parameter(source, args.vararg, None))
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        parts.append(_parameter(source, arg, default))
    if args.kwarg is not None:
        parts.append("**" + _parameter(source, args.kwarg, None))
    return parts


def _needs_layout(
        node: ast.FunctionDef | ast.AsyncFunctionDef
) -> bool:
    """True when the function has parameters other than a lone `self` or `cls`."""
    args = node.args
    positional = args.posonlyargs + args.args
    count = len(positional) + len(args.kwonlyargs) + bool(args.vararg) + bool(args.kwarg)
    if count == 0:
        return False
    only_self = count == 1 and bool(positional) and positional[0].arg in ("self", "cls")
    return not only_self


def _signature_span(
        text: str,
        open_paren: int
) -> tuple[int, bool]:
    """The offset just past the signature's closing colon, and whether a comment
    sits inside the signature (which the rewrite would lose).
    """
    remainder = text[open_paren:]
    line_starts = [0]
    for line in io.StringIO(remainder).readlines():
        line_starts.append(line_starts[-1] + len(line))
    depth = 0
    has_comment = False
    for token in tokenize.generate_tokens(io.StringIO(remainder).readline):
        if token.type == tokenize.COMMENT:
            has_comment = True
        elif token.type == tokenize.OP:
            if token.string in ("(", "[", "{"):
                depth += 1
            elif token.string in (")", "]", "}"):
                depth -= 1
            elif token.string == ":" and depth == 0:
                row, col = token.end
                return open_paren + line_starts[row - 1] + col, has_comment
    raise ValueError("no closing colon found for a function signature")


def layout_signatures(
        source: str,
        filename: str = "<source>"
) -> str:
    """The source with every signature that has parameters in the one-per-line layout."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))

    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) or not _needs_layout(node):
            continue
        def_line = lines[node.lineno - 1]
        indent = def_line[: len(def_line) - len(def_line.lstrip())]
        def_start = line_starts[node.lineno - 1] + len(def_line.encode()[: node.col_offset].decode())
        match = re.compile(rf"def\s+{re.escape(node.name)}\s*\(").search(source, def_start)
        if match is None:
            raise ValueError(f"{filename}:{node.lineno}: cannot locate the signature of {node.name}")
        open_paren = match.end() - 1
        end, has_comment = _signature_span(source, open_paren)
        if has_comment:
            print(f"{filename}:{node.lineno}: {node.name} has a comment inside its signature; left alone")
            continue
        inner = indent + PARAMETER_INDENT
        body = ",\n".join(inner + part for part in _parameters(source, node.args))
        returns = f" -> {_segment(source, node.returns)}" if node.returns is not None else ""
        edits.append((open_paren, end, f"(\n{body}\n{indent}){returns}:"))

    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def _comparable_tree(
        source: str
) -> str:
    """The syntax tree as text, with docstring indentation normalised: formatting may
    legitimately re-indent a docstring, which changes its string value but not the program.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                first.value.value = inspect.cleandoc(first.value.value)
    return ast.dump(tree)


def format_source(
        source: str,
        filename: str
) -> str:
    """`ruff format`, then the signature layout, checked not to change the program."""
    formatted = layout_signatures(ruff_format(source, filename), filename)
    if _comparable_tree(formatted) != _comparable_tree(source):
        raise ValueError(f"{filename}: formatting changed the syntax tree; refusing to write it")
    return formatted


def main(
        argv: Sequence[str] | None = None
) -> int:
    """Formats (or with --check, verifies) the given paths; returns the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("paths", nargs="*", default=DEFAULT_PATHS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any file would change; write nothing",
    )
    options = parser.parse_args(argv)

    changed: list[Path] = []
    for path in python_files(options.paths):
        before = path.read_text()
        after = format_source(before, str(path))
        if after != before:
            changed.append(path)
            if not options.check:
                path.write_text(after)
    verb = "would reformat" if options.check else "reformatted"
    for path in changed:
        print(f"{verb} {path}")
    if not changed:
        print("all files already formatted")
    return 1 if (changed and options.check) else 0


if __name__ == "__main__":
    sys.exit(main())

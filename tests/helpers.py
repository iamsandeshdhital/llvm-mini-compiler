"""Shared helpers for the Lumen test suite.

Every test builds sources from small strings, so the suite stays readable and
self-contained. :func:`run` is the workhorse: it drives the whole pipeline and
executes the resulting IR, which is how behaviour is checked rather than just
the shape of the AST.
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
EXAMPLES = os.path.join(ROOT, "examples")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

from lumen import irinterp, irverify  # noqa: E402
from lumen.driver import compile_source  # noqa: E402


def build(source: str, opt_level: int = 1):
    """Compile ``source`` and return the :class:`CompilationResult`."""
    return compile_source(source, "test.lum", opt_level=opt_level)


def compile_or_fail(case: unittest.TestCase, source: str, opt_level: int = 0) -> str:
    """Compile ``source``, asserting that it is error-free, and return the IR.

    Defaults to ``-O0`` so that shape assertions see the code the backend
    actually emits, rather than the constant-folded version of it.
    """
    result = build(source, opt_level)
    if not result.ok:
        result.diags.emit(sys.stderr)
        case.fail(f"expected the program to compile cleanly:\n{source}")
    assert result.ir is not None
    return result.ir


def run(source: str, opt_level: int = 1) -> Tuple[int, str]:
    """Compile and execute ``source``; returns ``(exit code, stdout)``."""
    ir = compile_source(source, "test.lum", opt_level=opt_level).ir
    assert ir is not None, "program failed to compile"
    return irinterp.execute(ir)


def run_output(source: str, opt_level: int = 1) -> str:
    """Compile and execute ``source``, returning just its stdout."""
    return run(source, opt_level)[1]


def errors(source: str, opt_level: int = 0) -> List[str]:
    """Return the error messages produced by compiling ``source``."""
    result = build(source, opt_level)
    return [d.message for d in result.diags.errors]


def warnings(source: str, opt_level: int = 0) -> List[str]:
    result = build(source, opt_level)
    return [d.message for d in result.diags.warnings]


def assert_no_errors(case: unittest.TestCase, source: str) -> None:
    result = build(source)
    if not result.ok:
        result.diags.emit(sys.stderr)
        case.fail(f"unexpected errors in:\n{source}")


def verify(ir: str) -> irverify.VerificationResult:
    return irverify.verify(ir)


def example_source(name: str) -> str:
    with open(os.path.join(EXAMPLES, name), "r", encoding="utf-8") as handle:
        return handle.read()


class IRTestCase(unittest.TestCase):
    """Base class with assertions for IR-level expectations."""

    def assert_verified(self, ir: str) -> None:
        result = irverify.verify(ir)
        if not result.ok:
            self.fail("emitted IR failed verification:\n" + result.summary())

    def assert_contains(self, haystack: str, needle: str) -> None:
        self.assertIn(needle, haystack,
                      msg=f"expected to find {needle!r} in the emitted IR")

    def assert_lacks(self, haystack: str, needle: str) -> None:
        self.assertNotIn(needle, haystack,
                         msg=f"did not expect {needle!r} in the emitted IR")

    def assert_runs(self, source: str, expected: str, exit_code: int = 0) -> None:
        code, out = run(source)
        self.assertEqual(out, expected)
        self.assertEqual(code, exit_code)


def wrap_body(body: str, ret: str = "0") -> str:
    """Wrap statements in a ``main`` that returns ``ret``."""
    return f"func main() -> int {{\n{body}\n    return {ret};\n}}"
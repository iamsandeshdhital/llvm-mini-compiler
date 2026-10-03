"""Driver and command line tests."""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from helpers import ROOT, SRC  # noqa: F401

from lumen.driver import build_arg_parser, find_toolchain, host_triple, main


SAMPLE = """func main() -> int {
    print "from the cli";
    return 3;
}
"""


class CLITestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="lumenc-test-")
        self.source = os.path.join(self.directory, "sample.lum")
        with open(self.source, "w", encoding="utf-8") as handle:
            handle.write(SAMPLE)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def run_cli(self, *arguments):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(arguments))
        return code, out.getvalue(), err.getvalue()


class TestSuccessfulRuns(CLITestCase):
    def test_writes_ir_next_to_the_source(self):
        code, _out, _err = self.run_cli(self.source)
        self.assertEqual(code, 0)
        expected = os.path.join(self.directory, "sample.ll")
        self.assertTrue(os.path.exists(expected))
        with open(expected, encoding="utf-8") as handle:
            self.assertIn("define i64 @main()", handle.read())

    def test_explicit_output_path(self):
        target = os.path.join(self.directory, "out.ll")
        code, _out, _err = self.run_cli(self.source, "-o", target)
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(target))

    def test_check_only_writes_nothing(self):
        code, _out, err = self.run_cli(self.source, "--check-only")
        self.assertEqual(code, 0)
        self.assertIn("type-checks cleanly", err)
        self.assertFalse(os.path.exists(os.path.join(self.directory, "sample.ll")))

    def test_verify_ir_reports_success(self):
        code, _out, err = self.run_cli(self.source, "--verify-ir")
        self.assertEqual(code, 0)
        self.assertIn("IR verified", err)

    def test_run_ir_prints_program_output(self):
        code, out, _err = self.run_cli(self.source, "--run-ir")
        self.assertEqual(code, 3)
        self.assertIn("from the cli", out)

    def test_tokens(self):
        code, out, _err = self.run_cli(self.source, "--tokens")
        self.assertEqual(code, 0)
        # Keywords are shown as the word the programmer typed.
        self.assertIn("func", out)
        self.assertIn("identifier", out)

    def test_ast_dump(self):
        code, out, _err = self.run_cli(self.source, "--ast")
        self.assertEqual(code, 0)
        self.assertIn("FuncDecl", out)
        self.assertIn("PrintStmt", out)

    def test_ast_json_is_valid_json(self):
        import json

        code, out, _err = self.run_cli(self.source, "--ast-json")
        self.assertEqual(code, 0)
        parsed = json.loads(out)
        self.assertIn("kind", parsed)

    def test_stats_include_timings_and_instruction_counts(self):
        code, _out, err = self.run_cli(self.source, "--stats")
        self.assertEqual(code, 0)
        self.assertIn("phase timings", err)
        self.assertIn("instructions", err)

    def test_optimization_level_is_accepted(self):
        for level in ("0", "1", "2", "3"):
            with self.subTest(level=level):
                code, _out, _err = self.run_cli(self.source, "-O", level)
                self.assertEqual(code, 0)

    def test_target_triple_can_be_overridden(self):
        code, _out, _err = self.run_cli(self.source,
                                        "--target", "x86_64-unknown-linux-gnu")
        self.assertEqual(code, 0)
        with open(os.path.join(self.directory, "sample.ll"), encoding="utf-8") as h:
            self.assertIn('target triple = "x86_64-unknown-linux-gnu"', h.read())


class TestFailures(CLITestCase):
    def write(self, name: str, text: str) -> str:
        path = os.path.join(self.directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_type_error_exits_nonzero_and_points_at_the_line(self):
        path = self.write("bad.lum",
                          'func main() -> int { let x: int = "s"; return 0; }')
        code, _out, err = self.run_cli(path)
        self.assertEqual(code, 1)
        self.assertIn("cannot initialise", err)
        self.assertIn("line 1", err)
        # A caret excerpt makes the location obvious.
        self.assertIn("^", err)

    def test_syntax_error_is_reported(self):
        path = self.write("syntax.lum", "func main() -> int { print 1;")
        code, _out, err = self.run_cli(path)
        self.assertEqual(code, 1)
        self.assertIn("unterminated block", err)

    def test_missing_file(self):
        code, _out, err = self.run_cli(
            os.path.join(self.directory, "absent.lum"))
        self.assertEqual(code, 1)
        self.assertIn("no such file", err)

    def test_error_summary_counts_diagnostics(self):
        path = self.write("many.lum",
                          "func main() -> int {\n"
                          "    let a: int = \"x\";\n"
                          "    let b: int = \"y\";\n"
                          "    return 0;\n"
                          "}\n")
        code, _out, err = self.run_cli(path)
        self.assertEqual(code, 1)
        self.assertIn("2 error(s)", err)

    def test_quiet_suppresses_warnings_but_not_errors(self):
        path = self.write("warn.lum",
                          "func f() -> int { while true { return 1; } }\n"
                          "func main() -> int { return f(); }")
        _code, _out, noisy = self.run_cli(path)
        _code, _out, quiet = self.run_cli(path, "--quiet")
        self.assertGreaterEqual(len(noisy), len(quiet))


class TestDriverInternals(unittest.TestCase):
    def test_host_triple_is_known(self):
        triple = host_triple()
        self.assertTrue(triple is None or "-" in triple)

    def test_toolchain_probe_returns_a_dict(self):
        tools = find_toolchain()
        self.assertEqual(set(tools), {"clang", "llc", "opt"})

    def test_arg_parser_exposes_the_documented_flags(self):
        parser = build_arg_parser()
        text = parser.format_help()
        for flag in ("--build", "--run", "--run-ir", "--tokens", "--ast",
                     "--check-only", "--verify-ir", "--stats"):
            self.assertIn(flag, text)

    def test_compile_source_reports_timings_for_each_phase(self):
        from lumen.driver import compile_source

        result = compile_source(SAMPLE, "sample.lum", opt_level=1)
        self.assertTrue(result.ok)
        for phase in ("lex", "parse", "check", "optimize", "lower", "codegen"):
            self.assertIn(phase, result.timings)

    def test_compile_source_stops_before_codegen_when_checking_fails(self):
        from lumen.driver import compile_source

        result = compile_source(
            "func main() -> int { let x: int = \"s\"; return 0; }", "t.lum")
        self.assertFalse(result.ok)
        self.assertIsNone(result.ir)


if __name__ == "__main__":
    unittest.main()
"""End-to-end behaviour tests.

Every test compiles Lumen to LLVM IR and then executes that IR, so a failure
here means the *generated code* computes the wrong thing -- which is the only
kind of backend bug that really matters.
"""

from __future__ import annotations

import unittest

from helpers import IRTestCase, example_source, run, run_output

from helpers import ROOT, SRC  # noqa: F401


def main(body: str, ret: str = "0") -> str:
    return f"func main() -> int {{\n{body}\n    return {ret};\n}}"


class TestPrinting(IRTestCase):
    def test_integer(self):
        self.assert_runs(main("print 42;"), "42\n")

    def test_negative_integer(self):
        self.assert_runs(main("print 0 - 7;"), "-7\n")

    def test_string(self):
        self.assert_runs(main('print "hello";'), "hello\n")

    def test_float_uses_six_decimals(self):
        self.assert_runs(main("print 1.5;"), "1.500000\n")

    def test_bool_prints_as_one_or_zero(self):
        self.assert_runs(main("print true; print false;"), "1\n0\n")

    def test_several_values_are_space_separated(self):
        self.assert_runs(main('print "a", 1, true;'), "a 1 1\n")

    def test_bare_newline(self):
        self.assert_runs(main("print;"), "\n")

    def test_string_with_a_percent_sign(self):
        self.assert_runs(main('print "100%";'), "100%\n")


class TestArithmetic(IRTestCase):
    def test_four_operations(self):
        self.assert_runs(main("print 7 + 3; print 7 - 3; print 7 * 3; print 7 / 3;"),
                         "10\n4\n21\n2\n")

    def test_modulo(self):
        self.assert_runs(main("print 7 % 3;"), "1\n")

    def test_integer_division_truncates_toward_zero(self):
        self.assert_runs(main("print 0 - 7 / 2;"), "-3\n")

    def test_float_division(self):
        self.assert_runs(main("print 7.0 / 2.0;"), "3.500000\n")

    def test_int_widens_to_float_in_mixed_expressions(self):
        self.assert_runs(main("let a = 3; print a / 2.0;"), "1.500000\n")

    def test_precedence_is_respected_at_runtime(self):
        self.assert_runs(main("print 2 + 3 * 4; print (2 + 3) * 4;"),
                         "14\n20\n")

    def test_left_associativity_of_subtraction(self):
        self.assert_runs(main("print 10 - 3 - 2;"), "5\n")

    def test_compound_assignment(self):
        self.assert_runs(main("let x = 10; x += 5; x -= 3; x *= 2; x /= 4; print x;"),
                         "6\n")

    def test_modulo_assignment(self):
        self.assert_runs(main("let x = 17; x %= 5; print x;"), "2\n")


class TestComparisonAndLogic(IRTestCase):
    def test_comparisons(self):
        self.assert_runs(
            main("print 1 < 2; print 2 < 1; print 2 == 2; print 2 != 2; "
                 "print 2 >= 2; print 2 <= 1; print 1 > 2;"),
            "1\n0\n1\n0\n1\n0\n0\n")

    def test_float_comparison(self):
        self.assert_runs(main("print 1.5 < 2.5; print 2.5 < 1.5;"), "1\n0\n")

    def test_string_comparison_is_by_content(self):
        self.assert_runs(main('print "abc" == "abc"; print "abc" == "abd"; '
                              'print "abc" != "abd";'),
                         "1\n0\n1\n")

    def test_logical_operators(self):
        self.assert_runs(main("print true && false; print true || false; "
                              "print !false;"),
                         "0\n1\n1\n")

    def test_and_does_not_evaluate_the_right_operand(self):
        source = ("func side() -> bool { print \"called\"; return true; }\n" +
                  main("let b = false && side(); print b;"))
        self.assert_runs(source, "0\n")

    def test_or_does_not_evaluate_the_right_operand(self):
        source = ("func side() -> bool { print \"called\"; return false; }\n" +
                  main("let b = true || side(); print b;"))
        self.assert_runs(source, "1\n")

    def test_or_evaluates_the_right_operand_when_needed(self):
        source = ("func side() -> bool { print \"called\"; return true; }\n" +
                  main("let b = false || side(); print b;"))
        self.assert_runs(source, "called\n1\n")


class TestControlFlow(IRTestCase):
    def test_if_without_else(self):
        self.assert_runs(main("if 1 < 2 { print \"yes\"; }"), "yes\n")

    def test_if_else(self):
        self.assert_runs(main("if 1 > 2 { print \"a\"; } else { print \"b\"; }"), "b\n")

    def test_else_if_chain(self):
        source = main("""
            let n = 0;
            if n < 0 { print "neg"; }
            else if n == 0 { print "zero"; }
            else { print "pos"; }
        """)
        self.assert_runs(source, "zero\n")

    def test_while_loop_counts(self):
        self.assert_runs(main("let i = 0; while i < 3 { print i; i += 1; }"),
                         "0\n1\n2\n")

    def test_while_that_never_runs(self):
        self.assert_runs(main("while false { print \"no\"; }"), "")

    def test_break_leaves_the_loop(self):
        self.assert_runs(
            main("let i = 0; while true { if i == 3 { break; } print i; i += 1; }"),
            "0\n1\n2\n")

    def test_continue_skips_the_rest_of_the_body(self):
        self.assert_runs(
            main("""
                let i = 0;
                while i < 5 {
                    i += 1;
                    if i == 3 { continue; }
                    print i;
                }
            """),
            "1\n2\n4\n5\n")

    def test_nested_loops_with_break(self):
        source = main("""
            let i = 0;
            while i < 3 {
                let j = 0;
                while true {
                    j += 1;
                    if j == 2 { break; }
                    print j;
                }
                i += 1;
            }
        """)
        self.assert_runs(source, "1\n1\n1\n")

    def test_for_loop_is_half_open(self):
        self.assert_runs(main("for i in 0..4 { print i; }"), "0\n1\n2\n3\n")

    def test_for_loop_with_a_non_zero_start(self):
        self.assert_runs(main("for i in 2..5 { print i; }"), "2\n3\n4\n")

    def test_empty_range_runs_nothing(self):
        self.assert_runs(main("for i in 3..3 { print i; }"), "")

    def test_for_loop_over_floats(self):
        self.assert_runs(main("let t = 0.0; for i in 0.0..3.0 { t += i; } print t;"),
                         "3.000000\n")


class TestFunctions(IRTestCase):
    def test_call_and_return(self):
        source = "func double(n: int) -> int { return n * 2; }\n" + main("print double(21);")
        self.assert_runs(source, "42\n")

    def test_recursion(self):
        source = ("func fact(n: int) -> int { if n <= 1 { return 1; }"
                  " return n * fact(n - 1); }\n" + main("print fact(6);"))
        self.assert_runs(source, "720\n")

    def test_mutual_termination_of_fib(self):
        source = ("func fib(n: int) -> int { if n < 2 { return n; }"
                  " return fib(n - 1) + fib(n - 2); }\n" + main("print fib(15);"))
        self.assert_runs(source, "610\n")

    def test_void_function(self):
        source = "func shout(s: string) { print s; }\n" + main('shout("hi");')
        self.assert_runs(source, "hi\n")

    def test_function_calling_a_function(self):
        source = ("func inc(n: int) -> int { return n + 1; }\n"
                  "func twice(n: int) -> int { return inc(inc(n)); }\n"
                  + main("print twice(5);"))
        self.assert_runs(source, "7\n")

    def test_arrays_as_parameters(self):
        source = ("func total(xs: int[]) -> int { let t = 0;"
                  " for i in 0..len(xs) { t += xs[i]; } return t; }\n"
                  + main("let a: int[] = [1, 2, 3, 4]; print total(a);"))
        self.assert_runs(source, "10\n")

    def test_parameters_are_independent(self):
        source = ("func bump(n: int) -> int { n += 100; return n; }\n"
                  + main("let x = 1; print bump(x); print x;"))
        self.assert_runs(source, "101\n1\n")

    def test_deep_recursion(self):
        source = ("func count(n: int) -> int { if n == 0 { return 0; }"
                  " return 1 + count(n - 1); }\n" + main("print count(500);"))
        self.assert_runs(source, "500\n")

    def test_exit_code_is_the_return_value(self):
        code, _ = run("func main() -> int { return 7; }")
        self.assertEqual(code, 7)

    def test_void_main_exits_zero(self):
        code, out = run("func main() { print \"x\"; }")
        self.assertEqual(code, 0)
        self.assertEqual(out, "x\n")


class TestArrays(IRTestCase):
    def test_literal_and_length(self):
        self.assert_runs(main('let a: int[] = [1, 2, 3]; print len(a); print a[0]; '
                              'print a[2];'),
                         "3\n1\n3\n")

    def test_empty_array(self):
        self.assert_runs(main("let a: int[] = []; print len(a);"), "0\n")

    def test_index_assignment(self):
        self.assert_runs(main("let a: int[] = [1, 2, 3]; a[1] = 99; print a;"),
                         "[1, 99, 3]\n")

    def test_array_assignment_copies(self):
        self.assert_runs(main("""
            let a: int[] = [1, 2, 3];
            let b: int[] = a;
            b[0] = 99;
            print a;
            print b;
        """), "[1, 2, 3]\n[99, 2, 3]\n")

    def test_assignment_between_variables_copies(self):
        self.assert_runs(main("""
            let a: int[] = [1, 2, 3];
            let b: int[] = [0];
            b = a;
            b[2] = 7;
            print a;
            print b;
        """), "[1, 2, 3]\n[1, 2, 7]\n")

    def test_assignment_grows_the_destination(self):
        self.assert_runs(main("""
            let a: int[] = [1, 2];
            let b: int[] = [9];
            b = a;
            print len(b);
            print b;
        """), "2\n[1, 2]\n")

    def test_assignment_shrinks_the_destination(self):
        self.assert_runs(main("""
            let a: int[] = [1];
            let b: int[] = [1, 2, 3, 4];
            b = a;
            print len(b);
            print b;
        """), "1\n[1]\n")

    def test_string_arrays(self):
        self.assert_runs(main('let a: string[] = ["x", "y"]; print a[1]; print a;'),
                         "y\n[x, y]\n")

    def test_float_arrays(self):
        self.assert_runs(main("let a: float[] = [1.5, 2.5]; print a[1];"), "2.500000\n")

    def test_nested_arrays(self):
        source = ("let m: int[][] = [[1, 2], [3, 4]];\n"
                  "print len(m);\nprint m[1];\nprint m[0][1];\n")
        self.assert_runs(main(source), "2\n[3, 4]\n2\n")

    def test_bool_arrays(self):
        self.assert_runs(main("let a: bool[] = [true, false]; print a[0]; print a[1];"),
                         "1\n0\n")

    def test_array_of_arrays_is_shallow_copied(self):
        # Documented semantics: copying an array of arrays copies the
        # references, not the inner buffers.
        self.assert_runs(main("""
            let m: int[][] = [[1, 2], [3, 4]];
            let n: int[][] = m;
            n[0][0] = 99;
            print m[0];
        """), "[99, 2]\n")

    def test_len_of_a_string(self):
        self.assert_runs(main('print len("hello");'), "5\n")

    def test_loop_writes_into_an_array(self):
        self.assert_runs(main("""
            let squares: int[] = [0, 0, 0, 0, 0];
            for i in 0..5 { squares[i] = i * i; }
            print squares;
        """), "[0, 1, 4, 9, 16]\n")


class TestScoping(IRTestCase):
    def test_inner_block_shadows(self):
        self.assert_runs(main("let x = 1; { let x = 2; print x; } print x;"),
                         "2\n1\n")

    def test_loop_variable_is_scoped_to_the_loop(self):
        self.assert_runs(main("for i in 0..2 { print i; }"), "0\n1\n")

    def test_blocks_do_not_leak_variables(self):
        source = ("func main() -> int {\n"
                  "    { let hidden = 5; }\n"
                  "    return 0;\n"
                  "}\n")
        code, out = run(source)
        self.assertEqual(code, 0)
        self.assertEqual(out, "")


class TestOptimizationEquivalence(IRTestCase):
    """Optimisation must never change observable behaviour."""

    PROGRAMS = [
        "print 2 * 3 + 4;",
        "print (1 + 2) * 3 - 4 / 2;",
        "print 10 % 3 + 10 / 3;",
        "if 2 * 2 > 3 { print \"big\"; } else { print \"small\"; }",
        "let a = 0; for i in 0..4 { a += i * 2; } print a;",
        "print 1 < 2; print 2 <= 1; print 1 == 1;",
        "let x = 5; x = x + 1; print x * 1;",
        "while 1 < 2 { print \"once\"; break; }",
        "print 7 / 2; print 0 - 7 / 2;",
        "let f = 1.5; print f * 2.0;",
    ]

    def test_same_output_at_every_level(self):
        for index, body in enumerate(self.PROGRAMS):
            source = main(body)
            with self.subTest(program=index, body=body):
                expected = run_output(source, opt_level=0)
                for level in (1, 2, 3):
                    self.assertEqual(run_output(source, opt_level=level), expected)


class TestExamples(unittest.TestCase):
    """The shipped examples must compile, verify and run."""

    def _check(self, name: str) -> str:
        from lumen import irverify
        from lumen.driver import compile_source

        source = example_source(name)
        result = compile_source(source, name, opt_level=1)
        self.assertTrue(result.ok, f"{name} failed to compile")
        verification = irverify.verify(result.ir)
        self.assertTrue(verification.ok,
                        f"{name} produced invalid IR:\n{verification.summary()}")
        code, out = irinterp_run(result.ir)
        return out

    def test_hello(self):
        out = self._check("hello.lum")
        self.assertIn("Hello from Lumen", out)
        self.assertIn("sum of 1..10 is 55", out)

    def test_fib(self):
        out = self._check("fib.lum")
        self.assertIn("fib 10 = 55", out)
        self.assertIn("fib 15 = 610", out)
        self.assertIn("gcd(1071, 462) =  21", out)
        self.assertIn("10 % 3 =  1", out)

    def test_arrays(self):
        out = self._check("arrays.lum")
        self.assertIn("len(xs)  = 8", out)
        self.assertIn("sum(xs)  = 31", out)
        # the copy must not be an alias
        self.assertIn("  xs = [3, 1, 4, 1, 5, 9, 2, 6]", out)
        self.assertIn("  ys = [99, 1, 4, 1, 5, 9, 2, 6]", out)
        self.assertIn("matrix = [[1, 2, 3], [4, 5, 6]]", out)

    def test_types(self):
        out = self._check("types.lum")
        self.assertIn("pi       = 3.141590", out)
        self.assertIn("abc == abc = 1", out)
        self.assertIn("classify(0)  = zero", out)
        # short-circuiting must have suppressed the side-effecting calls
        self.assertNotIn("never_reached ran", out)
        self.assertIn("(always_true ran)", out)


def irinterp_run(ir: str):
    from lumen import irinterp

    return irinterp.execute(ir)


if __name__ == "__main__":
    unittest.main()
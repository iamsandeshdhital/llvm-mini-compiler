"""Semantic analysis tests: name resolution, types and control flow."""

from __future__ import annotations

import unittest

from helpers import errors, warnings  # noqa: F401

from helpers import ROOT, SRC  # noqa: F401


def main(body: str) -> str:
    return f"func main() -> int {{\n{body}\n    return 0;\n}}"


def with_fn(body: str, extra: str = "") -> str:
    return main(body) + "\n" + extra


class TestScoping(unittest.TestCase):
    def test_undeclared_identifier(self):
        self.assertIn("undeclared identifier 'x'", errors(main("print x;"))[0])

    def test_declaration_then_use(self):
        self.assertEqual(errors(main('let x = 1; print x;')), [])

    def test_redeclaration_in_same_scope(self):
        messages = errors(main("let x = 1; let x = 2;"))
        self.assertTrue(any("already declared" in m for m in messages))

    def test_shadowing_in_nested_scope_is_allowed(self):
        self.assertEqual(errors(main("let x = 1; { let x = 2; print x; }")), [])

    def test_inner_scope_not_visible_outside(self):
        messages = errors(main("{ let inner = 1; } print inner;"))
        self.assertTrue(any("undeclared identifier 'inner'" in m for m in messages))

    def test_function_name_used_as_a_value(self):
        messages = errors(with_fn("print helper;", "func helper() -> int { return 1; }"))
        self.assertTrue(any("is a function" in m for m in messages))

    def test_parameters_are_visible_in_the_body(self):
        self.assertEqual(
            errors(with_fn("print helper(1);",
                           "func helper(x: int) -> int { return x; }")),
            [])


class TestTypes(unittest.TestCase):
    def test_int_widens_to_float(self):
        self.assertEqual(errors(main("let x: float = 1;")), [])

    def test_float_does_not_narrow_to_int_implicitly(self):
        messages = errors(main("let x: int = 1.5;"))
        self.assertTrue(any("cannot initialise" in m for m in messages))

    def test_bool_is_not_a_number(self):
        messages = errors(main("let x: int = true;"))
        self.assertTrue(messages)

    def test_arithmetic_on_strings_is_rejected(self):
        messages = errors(main('let x: string = "a" + "b";'))
        self.assertTrue(any("does not concatenate" in m for m in messages))

    def test_modulo_requires_integers(self):
        messages = errors(main("let a: float = 1.5; let b: float = 2.0;"
                               " print a % b;"))
        self.assertTrue(any("'%' is not defined for float" in m for m in messages))

    def test_condition_must_be_bool(self):
        messages = errors(main("if 1 { }"))
        self.assertTrue(any("condition must be bool" in m for m in messages))

    def test_comparison_yields_bool(self):
        self.assertEqual(errors(main("let b: bool = 1 < 2;")), [])

    def test_logical_operators_require_bool(self):
        messages = errors(main("if 1 && 2 { }"))
        self.assertTrue(any("expects bool" in m for m in messages))

    def test_not_requires_bool(self):
        messages = errors(main("let b: bool = not 1;"))
        self.assertTrue(any("'not' expects a bool" in m for m in messages))

    def test_comparing_incompatible_types(self):
        messages = errors(main('if 1 == "a" { }'))
        self.assertTrue(any("cannot compare" in m for m in messages))

    def test_string_equality_is_allowed(self):
        self.assertEqual(errors(main('let b: bool = "a" == "a";')), [])

    def test_return_type_mismatch(self):
        messages = errors(
            "func f() -> int { return \"x\"; }\n" + main(""))
        self.assertTrue(any("cannot return" in m for m in messages))

    def test_returning_nothing_from_non_void(self):
        messages = errors("func f() -> int { return; }\n" + main(""))
        self.assertTrue(any("expected a value" in m for m in messages))

    def test_array_index_must_be_int(self):
        messages = errors(main("let a: int[] = [1]; print a[true];"))
        self.assertTrue(any("index must be int" in m for m in messages))

    def test_indexing_a_non_array(self):
        messages = errors(main("let x = 1; print x[0];"))
        self.assertTrue(any("cannot index" in m for m in messages))

    def test_len_requires_array_or_string(self):
        self.assertEqual(errors(main('print len("abc");')), [])
        self.assertEqual(errors(main("let a: int[] = [1]; print len(a);")), [])
        messages = errors(main("let x = 1; print len(x);"))
        self.assertTrue(any("needs an array or string" in m for m in messages))

    def test_array_elements_must_agree(self):
        messages = errors(main("let a: int[] = [1, \"two\"];"))
        self.assertTrue(any("share one type" in m for m in messages))

    def test_heterogeneous_array_elements_widen(self):
        self.assertEqual(errors(main("let a: float[] = [1, 2.0];")), [])


class TestAssignment(unittest.TestCase):
    def test_assign_to_const(self):
        messages = errors(main("const x = 1; x = 2;"))
        self.assertTrue(any("declared const" in m for m in messages))

    def test_mutable_by_default(self):
        self.assertEqual(errors(main("let x = 1; x = 2;")), [])

    def test_parameters_are_assignable(self):
        self.assertEqual(
            errors("func f(p: int) -> int { p = p + 1; return p; }\n" + main("")),
            [])

    def test_invalid_assignment_target(self):
        messages = errors(main("1 = 2;"))
        self.assertTrue(any("invalid assignment target" in m for m in messages))

    def test_array_element_is_assignable_even_when_binding_is_const(self):
        self.assertEqual(
            errors(main("const a: int[] = [1]; a[0] = 5;")), [])

    def test_assignment_type_mismatch(self):
        messages = errors(main("let x = 1; x = \"s\";"))
        self.assertTrue(any("cannot assign" in m for m in messages))


class TestCalls(unittest.TestCase):
    def test_unknown_function(self):
        messages = errors(main("print nope(1);"))
        self.assertTrue(any("undeclared function 'nope'" in m for m in messages))

    def test_wrong_arity(self):
        messages = errors(with_fn("print f(1, 2);", "func f(a: int) -> int { return a; }"))
        self.assertTrue(any("takes 1 argument(s) but 2" in m for m in messages))

    def test_wrong_argument_type(self):
        messages = errors(with_fn("print f(\"s\");",
                                  "func f(a: int) -> int { return a; }"))
        self.assertTrue(any("expected int but found string" in m for m in messages))

    def test_print_as_a_function_is_rejected(self):
        messages = errors(main("print len(1);"))
        self.assertTrue(any("needs an array or string" in m for m in messages))


class TestControlFlow(unittest.TestCase):
    def test_break_outside_loop(self):
        messages = errors(main("break;"))
        self.assertTrue(any("'break' outside of a loop" in m for m in messages))

    def test_continue_outside_loop(self):
        messages = errors(main("continue;"))
        self.assertTrue(any("'continue' outside of a loop" in m for m in messages))

    def test_break_inside_loop_is_fine(self):
        self.assertEqual(errors(main("while true { break; }")), [])

    def test_break_inside_nested_loop_is_fine(self):
        self.assertEqual(errors(main("while true { while true { break; } }")), [])

    def test_missing_return_on_all_paths(self):
        messages = errors("func f(x: int) -> int { if x > 0 { return 1; } }\n"
                          + main(""))
        self.assertTrue(any("no return statement on every path" in m for m in messages))

    def test_return_on_both_branches_is_accepted(self):
        self.assertEqual(
            errors("func f(x: int) -> int { if x > 0 { return 1; } else { return 0; } }\n"
                   + main("")),
            [])

    def test_return_inside_a_loop_satisfies_the_check(self):
        self.assertEqual(
            errors("func f(n: int) -> int { while true { return n; } }\n" + main("")),
            [])

    def test_main_needs_no_return(self):
        self.assertEqual(errors("func main() { print 1; }"), [])

    def test_main_must_return_int(self):
        messages = errors("func main() -> bool { return true; }")
        self.assertTrue(any("'main' must return int" in m for m in messages))

    def test_program_needs_main(self):
        messages = errors("func f() -> int { return 1; }")
        self.assertTrue(any("no entry point" in m for m in messages))


class TestDeclarations(unittest.TestCase):
    def test_duplicate_function(self):
        messages = errors("func f() -> int { return 1; }\n"
                          "func f() -> int { return 2; }\n" + main(""))
        self.assertTrue(any("duplicate definition" in m for m in messages))

    def test_duplicate_parameter(self):
        messages = errors("func f(a: int, a: int) -> int { return a; }\n" + main(""))
        self.assertTrue(any("duplicate parameter" in m for m in messages))

    def test_void_parameter_is_rejected(self):
        messages = errors("func f(a: void) -> int { return 1; }\n" + main(""))
        self.assertTrue(any("cannot have type void" in m for m in messages))

    def test_unknown_type_is_a_parse_error(self):
        # Type names are keywords, so an unknown one never reaches the checker.
        from lumen.diagnostics import CompileError
        from lumen.parser import parse_source

        with self.assertRaises(CompileError):
            parse_source("func f(a: widget) -> int { return 1; }", "test.lum")

    def test_array_type_annotation(self):
        self.assertEqual(
            errors(main("let a: int[] = [1, 2]; print len(a);")), [])

    def test_nested_array_type_annotation(self):
        self.assertEqual(
            errors(main("let m: int[][] = [[1]]; print len(m);")), [])


class TestAnnotations(unittest.TestCase):
    def test_expression_nodes_are_typed(self):
        from helpers import build

        result = build(main("let x = 1 + 2;"))
        self.assertTrue(result.ok)
        let = result.program.functions[0].body.stmts[0]
        self.assertEqual(let.init.ty.render(), "int")

    def test_call_nodes_record_their_callee(self):
        from helpers import build

        result = build(main("f(1);") + "\nfunc f(a: int) -> int { return a; }")
        self.assertTrue(result.ok)
        expr = result.program.functions[0].body.stmts[0].expr
        self.assertTrue(hasattr(expr, "callee_symbol"))
        self.assertEqual(expr.param_types[0].render(), "int")


if __name__ == "__main__":
    unittest.main()
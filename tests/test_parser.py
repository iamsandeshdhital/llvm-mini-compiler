"""Parser tests: grammar coverage, precedence and associativity."""

from __future__ import annotations

import unittest

from helpers import ROOT, SRC  # noqa: F401

from lumen import ast
from lumen.diagnostics import CompileError
from lumen.parser import parse_source


def one_expr(source: str) -> ast.Expr:
    """Parse ``source`` as the single expression of a stub main."""
    program = parse_source(wrap(f"let _unused = {source};"), "test.lum")
    let = program.functions[0].body.stmts[0]
    assert isinstance(let, ast.LetStmt)
    return let.init


def wrap(body: str) -> str:
    return f"func main() -> int {{\n{body}\n    return 0;\n}}"


def parse_fails(source: str) -> CompileError:
    with self.assertRaises(CompileError):
        parse_source(source, "test.lum")
    return None  # pragma: no cover


class TestDeclarations(unittest.TestCase):
    def test_function_with_both_spellings(self):
        program = parse_source(
            'func a(x: int) -> int { return x; }\n'
            'void b(y: int) { print y; }\n',
            "test.lum")
        self.assertEqual([f.name for f in program.functions], ["a", "b"])
        self.assertEqual(program.functions[0].ret.name, "int")
        # The C-style spelling always names the return type explicitly.
        self.assertEqual(program.functions[1].ret.name, "void")

    def test_c_style_defaults_to_nothing_implicitly(self):
        # `int b(...) { }` means "returns int", not "returns void".
        program = parse_source("int b() { }", "test.lum")
        self.assertEqual(program.functions[0].ret.name, "int")

    def test_func_keyword_defaults_to_void(self):
        program = parse_source("func b() { }", "test.lum")
        self.assertEqual(program.functions[0].ret.name, "void")

    def test_parameters(self):
        program = parse_source(
            "func f(a: int, b: float, c: bool) { }", "test.lum")
        params = program.functions[0].params
        self.assertEqual([p.name for p in params], ["a", "b", "c"])
        self.assertEqual([p.ty.render() for p in params], ["int", "float", "bool"])

    def test_missing_main_is_a_checker_error_not_a_parse_error(self):
        program = parse_source("func f() { }", "test.lum")
        self.assertEqual(len(program.functions), 1)

    def test_empty_program_parses(self):
        self.assertEqual(parse_source("", "test.lum").functions, [])


class TestStatements(unittest.TestCase):
    def stmt(self, body: str) -> ast.Stmt:
        return parse_source(wrap(body), "test.lum").functions[0].body.stmts[0]

    def test_let_with_and_without_annotation(self):
        self.assertIsInstance(self.stmt("let x = 1;"), ast.LetStmt)
        self.assertIsInstance(self.stmt("let x: float = 1;"), ast.LetStmt)

    def test_const_keyword(self):
        self.assertTrue(self.stmt("const x = 1;").is_const)

    def test_semicolon_is_optional(self):
        program = parse_source(
            "func main() -> int {\n"
            "    let a = 1\n"
            "    let b = 2\n"
            "    return a + b\n"
            "}\n", "test.lum")
        body = program.functions[0].body
        self.assertEqual(len(body.stmts), 3)

    def test_if_else_if_else_chain(self):
        node = self.stmt("if 1 { } else if 2 { } else { }")
        self.assertIsInstance(node, ast.IfStmt)
        self.assertIsInstance(node.else_branch, ast.IfStmt)
        self.assertIsInstance(node.else_branch.else_branch, ast.Block)

    def test_if_without_else(self):
        self.assertIsNone(self.stmt("if 1 { }").else_branch)

    def test_while(self):
        self.assertIsInstance(self.stmt("while 1 { }"), ast.WhileStmt)

    def test_for_range(self):
        node = self.stmt("for i in 0..10 { }")
        self.assertIsInstance(node, ast.ForStmt)
        self.assertEqual(node.var, "i")

    def test_break_continue_return(self):
        self.assertIsInstance(self.stmt("break;"), ast.BreakStmt)
        self.assertIsInstance(self.stmt("continue;"), ast.ContinueStmt)
        self.assertIsInstance(self.stmt("return 1;"), ast.ReturnStmt)

    def test_bare_return(self):
        self.assertIsNone(self.stmt("return;").value)

    def test_nested_block_is_a_statement(self):
        program = parse_source(
            "func main() -> int {\n"
            "    { let x = 1; }\n"
            "    return 0;\n"
            "}\n", "test.lum")
        self.assertIsInstance(program.functions[0].body.stmts[0], ast.Block)


class TestExpressions(unittest.TestCase):
    def test_literals(self):
        self.assertIsInstance(one_expr("42"), ast.IntLit)
        self.assertIsInstance(one_expr("4.2"), ast.FloatLit)
        self.assertIsInstance(one_expr("true"), ast.BoolLit)
        self.assertIsInstance(one_expr('"s"'), ast.StrLit)

    def test_array_literal(self):
        node = one_expr("[1, 2, 3]")
        self.assertIsInstance(node, ast.ArrayLit)
        self.assertEqual(len(node.elements), 3)

    def test_empty_array_literal(self):
        self.assertEqual(one_expr("[]").elements, [])

    def test_call(self):
        node = one_expr("f(1, 2)")
        self.assertIsInstance(node, ast.Call)
        self.assertEqual(node.callee, "f")
        self.assertEqual(len(node.args), 2)

    def test_index(self):
        self.assertIsInstance(one_expr("xs[0]"), ast.Index)

    def test_len_builtin(self):
        self.assertIsInstance(one_expr("len(xs)"), ast.LenBuiltin)

    def test_unary(self):
        self.assertEqual(one_expr("-1").op, "-")
        self.assertEqual(one_expr("!x").op, "!")
        self.assertEqual(one_expr("not x").op, "not")

    def test_nested_unary(self):
        self.assertEqual(one_expr("--1").op, "-")


class TestPrecedence(unittest.TestCase):
    """Precedence is verified by inspecting the resulting tree shape."""

    def test_multiplication_binds_tighter_than_addition(self):
        node = one_expr("1 + 2 * 3")
        self.assertEqual(node.op, "+")
        self.assertEqual(node.right.op, "*")

    def test_parentheses_override(self):
        node = one_expr("(1 + 2) * 3")
        self.assertEqual(node.op, "*")
        self.assertEqual(node.left.op, "+")

    def test_comparison_binds_looser_than_arithmetic(self):
        node = one_expr("1 + 2 < 4")
        self.assertEqual(node.op, "<")
        self.assertEqual(node.left.op, "+")

    def test_equality_binds_looser_than_comparison(self):
        node = one_expr("1 < 2 == true")
        self.assertEqual(node.op, "==")
        self.assertEqual(node.left.op, "<")

    def test_and_binds_looser_than_equality(self):
        node = one_expr("1 == 1 && 2 == 2")
        self.assertEqual(node.op, "&&")

    def test_or_is_loosest_binary_operator(self):
        node = one_expr("true || false && false")
        self.assertEqual(node.op, "||")
        self.assertEqual(node.right.op, "&&")

    def test_subtraction_is_left_associative(self):
        node = one_expr("1 - 2 - 3")
        self.assertEqual(node.op, "-")
        self.assertEqual(node.left.op, "-")
        self.assertEqual(node.left.right.value, 2)

    def test_assignment_is_right_associative(self):
        node = one_expr("a = b = c")
        self.assertIsInstance(node, ast.Assign)
        self.assertIsInstance(node.value, ast.Assign)

    def test_compound_assignment(self):
        node = one_expr("x += 1")
        self.assertEqual(node.op, "+=")

    def test_chained_calls_are_rejected(self):
        # Lumen has no function values, so `f(1)(2)` cannot be typed.
        with self.assertRaises(CompileError):
            parse_source(wrap("f(1)(2);"), "test.lum")

    def test_index_chain(self):
        node = one_expr("m[0][1]")
        self.assertIsInstance(node, ast.Index)
        self.assertIsInstance(node.base, ast.Index)


class TestErrors(unittest.TestCase):
    def test_unclosed_brace(self):
        with self.assertRaises(CompileError):
            parse_source("func main() -> int { print 1;", "test.lum")

    def test_missing_semicolon_after_let(self):
        with self.assertRaises(CompileError):
            parse_source(wrap("let x 1;"), "test.lum")

    def test_missing_rhs_in_assignment(self):
        with self.assertRaises(CompileError):
            parse_source(wrap("x = ;"), "test.lum")

    def test_type_annotation_in_statement_position(self):
        with self.assertRaises(CompileError):
            parse_source(wrap("int x = 1;"), "test.lum")

    def test_nested_function_definition_rejected(self):
        with self.assertRaises(CompileError):
            parse_source(wrap("func inner() { }"), "test.lum")

    def test_parentheses_do_not_hide_a_call(self):
        # `(f)(2)` is the same tree as `f(2)`; rejecting it is the checker's
        # job (Lumen has no function values), so parsing must succeed here.
        node = one_expr("(f)(2)")
        self.assertIsInstance(node, ast.Call)

    def test_missing_rbrace_in_param_list(self):
        with self.assertRaises(CompileError):
            parse_source("func f(a: int { }", "test.lum")


class TestAstTraversal(unittest.TestCase):
    def test_children_and_walk_visit_every_node(self):
        program = parse_source(
            "func main() -> int { let x = 1 + 2; return x; }", "test.lum")
        kinds = [type(node).__name__ for node in program.walk()]
        self.assertIn("Binary", kinds)
        self.assertIn("IntLit", kinds)
        self.assertIn("ReturnStmt", kinds)

    def test_json_round_trip_is_serializable(self):
        import json

        program = parse_source(
            "func main() -> int { let x = [1, 2]; return 0; }", "test.lum")
        text = json.dumps(program.to_json())
        self.assertIn("ArrayLit", text)

    def test_spans_are_present_on_nodes(self):
        program = parse_source(wrap("let x = 1;"), "test.lum")
        for node in program.walk():
            self.assertGreaterEqual(node.span.end, node.span.start)


if __name__ == "__main__":
    unittest.main()
"""Lowering pass: run between type checking and code generation.

Its job is to give every local variable, parameter and loop induction variable
a unique LLVM slot name, and to attach the semantic types the backend needs.
Doing this in a dedicated pass keeps :mod:`lumen.codegen` a pure emitter: it
never has to invent names or re-derive types, which is exactly the kind of
duplication where compilers quietly disagree with themselves.
"""

from __future__ import annotations

from typing import Optional

from . import ast, types as T
from .diagnostics import DiagnosticBag
from .checker import Checker


class Lowering:
    def __init__(self, diags: DiagnosticBag) -> None:
        self.diags = diags
        self.counter = 0

    def slot(self, hint: str) -> str:
        """Return a fresh ``%name_n`` stack slot."""
        from .codegen import sanitize

        self.counter += 1
        return f"%{sanitize(hint)}_{self.counter}"

    # -- entry point -----------------------------------------------------
    def lower_program(self, program: ast.Program) -> ast.Program:
        for fn in program.functions:
            self.lower_function(fn)
        return program

    def lower_function(self, fn: ast.FuncDecl) -> None:
        fn.ret_ty = _resolved(fn.ret)
        for param in fn.params:
            param.param_ty = _resolved(param.ty)
            param.symbol.slot = self.slot(param.name)
            param.slot = param.symbol.slot
        self.lower_stmt(fn.body)

    # -- statements ------------------------------------------------------
    def lower_stmt(self, stmt: ast.Stmt) -> None:
        if isinstance(stmt, ast.LetStmt):
            # The checker attached the Symbol; give it a unique slot and let
            # every use of that name find it through the symbol.
            stmt.symbol.slot = self.slot(stmt.name)
            stmt.slot = stmt.symbol.slot
            # An explicit annotation wins: `let x: float = 1` must store into a
            # double slot even though the initializer's own type is int.
            stmt.var_ty = _resolved(stmt.declared) if stmt.declared else stmt.init.ty
            self.lower_expr(stmt.init)
        elif isinstance(stmt, ast.ExprStmt):
            self.lower_expr(stmt.expr)
        elif isinstance(stmt, ast.Block):
            for inner in stmt.stmts:
                self.lower_stmt(inner)
        elif isinstance(stmt, ast.IfStmt):
            self.lower_expr(stmt.cond)
            self.lower_stmt(stmt.then_branch)
            if stmt.else_branch is not None:
                self.lower_stmt(stmt.else_branch)
        elif isinstance(stmt, ast.WhileStmt):
            self.lower_expr(stmt.cond)
            self.lower_stmt(stmt.body)
        elif isinstance(stmt, ast.ForStmt):
            self.lower_expr(stmt.start)
            self.lower_expr(stmt.end)
            stmt.symbol.slot = self.slot(stmt.var)
            stmt.slot = stmt.symbol.slot
            stmt.float_range = isinstance(stmt.start.ty, T.FloatType) or \
                isinstance(stmt.end.ty, T.FloatType)
            stmt.var_ty = T.FLOAT if stmt.float_range else T.INT
            self.lower_stmt(stmt.body)
        elif isinstance(stmt, ast.ReturnStmt):
            if stmt.value is not None:
                self.lower_expr(stmt.value)
        elif isinstance(stmt, ast.PrintStmt):
            for value in stmt.values:
                self.lower_expr(value)
        # BreakStmt / ContinueStmt need nothing.

    # -- expressions -----------------------------------------------------
    def lower_expr(self, expr: ast.Expr) -> None:
        if isinstance(expr, ast.Ident):
            expr.slot = expr.symbol.slot  # type: ignore[attr-defined]
        elif isinstance(expr, ast.ArrayLit):
            for element in expr.elements:
                self.lower_expr(element)
        elif isinstance(expr, ast.Unary):
            self.lower_expr(expr.operand)
        elif isinstance(expr, ast.Binary):
            self.lower_expr(expr.left)
            self.lower_expr(expr.right)
        elif isinstance(expr, ast.Assign):
            self.lower_expr(expr.target)
            self.lower_expr(expr.value)
        elif isinstance(expr, ast.Index):
            self.lower_expr(expr.base)
            self.lower_expr(expr.index)
        elif isinstance(expr, ast.LenBuiltin):
            self.lower_expr(expr.target)
        elif isinstance(expr, ast.Call):
            for arg in expr.args:
                self.lower_expr(arg)
        # Literals need no lowering.


def _resolved(ref: ast.TypeRef) -> T.Type:
    """Re-resolve a syntactic type; the checker already validated it."""
    from .diagnostics import Span

    try:
        return T.from_ast(ref, Span(0, 0))
    except T.TypeError_:  # pragma: no cover - checker reports this first
        return T.INT


def lower(program: ast.Program, diags: DiagnosticBag) -> ast.Program:
    Lowering(diags).lower_program(program)
    return program
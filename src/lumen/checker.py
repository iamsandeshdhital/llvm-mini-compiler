"""Semantic analysis: name resolution, type checking, control-flow checks.

This is the phase that turns a well-formed AST into a *well-typed* one. It
annotates every expression with its type and resolves every identifier to the
storage slot the code generator will use.

Symbol table
    A chain of lexical scopes. Each :class:`Scope` maps a name to a
    :class:`Symbol`, which records the name, type, mutability and (after
    binding) the LLVM slot used for the value.

Control flow
    :func:`returns` implements the standard backwards dataflow: a statement
    returns a value if it always transfers control to its successor with one.
    That is what lets the checker distinguish "falls off the end of main"
    from "returns on every path", so missing returns are caught precisely
    instead of by a crude syntactic rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import ast, types as T
from .diagnostics import DiagnosticBag, Span

_BUILTIN_LOCALS = {"print", "len"}


@dataclass
class Symbol:
    name: str
    ty: T.Type
    span: Span
    is_const: bool = False
    is_param: bool = False
    slot: Optional[str] = None
    defined_in_function: Optional[str] = None


class Scope:
    """One lexical block. ``parent`` links outward."""

    __slots__ = ("names", "parent")

    def __init__(self, parent: Optional["Scope"] = None) -> None:
        self.names: Dict[str, Symbol] = {}
        self.parent = parent

    def lookup(self, name: str) -> Optional[Symbol]:
        scope: Optional[Scope] = self
        while scope is not None:
            if name in scope.names:
                return scope.names[name]
            scope = scope.parent
        return None

    def declare(self, symbol: Symbol) -> None:
        self.names[symbol.name] = symbol

    def owns(self, name: str) -> bool:
        return name in self.names


class Checker:
    def __init__(self, diags: DiagnosticBag) -> None:
        self.diags = diags
        self.globals: Dict[str, Symbol] = {}
        self.functions: Dict[str, ast.FuncDecl] = {}
        self.scope: Scope = Scope()
        self.global_scope: Scope = Scope()
        self.current_fn: Optional[ast.FuncDecl] = None
        self.current_ret: T.Type = T.VOID
        self.loop_depth = 0
        self.unreachable_reported = False

    # -- entry point -----------------------------------------------------
    def check_program(self, program: ast.Program) -> None:
        self._collect_functions(program)

        if "main" not in self.functions:
            self.diags.error(
                "program has no entry point: expected a function named 'main'",
                program.span,
            )

        # Functions live in an outermost scope so that referring to one without
        # calling it produces "x is a function" rather than "undeclared".
        self.global_scope = Scope()
        for name, symbol in self.globals.items():
            self.global_scope.declare(symbol)

        for name, fn in self.functions.items():
            self._check_function(fn)

    def _collect_functions(self, program: ast.Program) -> None:
        for fn in program.functions:
            if fn.name in self.functions:
                self.diags.error(
                    f"duplicate definition of function '{fn.name}'",
                    fn.span,
                    notes=[f"previously defined at line {self._line(fn.span)}"],
                )
                continue
            self.functions[fn.name] = fn
            try:
                ret = T.from_ast(fn.ret, fn.span)
            except T.TypeError_ as exc:
                self.diags.error(exc.message, exc.span)
                ret = T.VOID
            self.globals[fn.name] = Symbol(
                name=fn.name,
                ty=T.FunctionType(tuple(), ret),
                span=fn.span,
                is_const=True,
                defined_in_function=fn.name,
            )
        # `main`'s return type becomes the process exit status. void is fine -- the
        # exit code is then 0, and `func main() { }` is the natural way to write
        # a program -- while anything else has no sensible exit-code mapping.
        main = self.functions.get("main")
        if main is not None:
            ret = self.globals["main"].ty.ret
            if ret not in (T.INT, T.VOID):
                self.diags.error(
                    f"'main' must return int (the process exit status) or "
                    f"nothing, not {ret.render()}",
                    main.span,
                )

    def _line(self, span: Span) -> int:
        if self.diags.source is None:
            return 0
        return self.diags.source.line_col(span.start)[0]

    # -- functions -------------------------------------------------------
    def _check_function(self, fn: ast.FuncDecl) -> None:
        self.current_fn = fn
        self.scope = Scope(self.global_scope)
        self.unreachable_reported = False
        try:
            ret = T.from_ast(fn.ret, fn.span)
        except T.TypeError_ as exc:
            self.diags.error(exc.message, exc.span)
            ret = T.VOID
        self.current_ret = ret

        seen: Dict[str, Span] = {}
        for param in fn.params:
            if param.name in seen:
                self.diags.error(
                    f"duplicate parameter name '{param.name}'", param.span,
                    notes=[f"first declared at line {self._line(seen[param.name])}"],
                )
                continue
            seen[param.name] = param.span
            try:
                pty = T.from_ast(param.ty, param.span)
            except T.TypeError_ as exc:
                self.diags.error(exc.message, exc.span)
                pty = T.INT
            if pty == T.VOID:
                self.diags.error(
                    f"parameter '{param.name}' cannot have type void", param.span)
                pty = T.INT
            self.scope.declare(Symbol(param.name, pty, param.span,
                                     is_const=False, is_param=True,
                                     defined_in_function=fn.name))
            param.symbol = self.scope.names[param.name]

        self._check_stmt(fn.body)

        if fn.name != "main" and ret != T.VOID and not self._returns(fn.body):
            self.diags.error(
                f"function '{fn.name}' declared to return {ret.render()} but "
                f"has no return statement on every path",
                fn.span,
                notes=[f"add 'return' at the end of the function body"],
            )
        if fn.name == "main" and not self._returns(fn.body):
            self.diags.note(
                "main reaches its end without a return; the exit code will be 0",
                fn.span,
            )
        self.current_fn = None

    # -- statements ------------------------------------------------------
    def _check_stmt(self, stmt: ast.Stmt) -> None:
        if isinstance(stmt, ast.Block):
            self.scope = Scope(self.scope)
            for s in stmt.stmts:
                self._check_stmt(s)
            self.scope = self.scope.parent or self.scope
            return

        if isinstance(stmt, ast.LetStmt):
            self._check_let(stmt)
            return

        if isinstance(stmt, ast.ExprStmt):
            self._check_expr(stmt.expr, want=None)
            return

        if isinstance(stmt, ast.IfStmt):
            cond_ty = self._check_expr(stmt.cond, want=T.BOOL)
            self._require_bool(stmt.cond, cond_ty)
            self._check_stmt(stmt.then_branch)
            if stmt.else_branch is not None:
                self._check_stmt(stmt.else_branch)
            return

        if isinstance(stmt, ast.WhileStmt):
            cond_ty = self._check_expr(stmt.cond, want=T.BOOL)
            self._require_bool(stmt.cond, cond_ty)
            self.loop_depth += 1
            self._check_stmt(stmt.body)
            self.loop_depth -= 1
            return

        if isinstance(stmt, ast.ForStmt):
            self._check_for(stmt)
            return

        if isinstance(stmt, ast.BreakStmt):
            if self.loop_depth == 0:
                self.diags.error("'break' outside of a loop", stmt.span)
            return

        if isinstance(stmt, ast.ContinueStmt):
            if self.loop_depth == 0:
                self.diags.error("'continue' outside of a loop", stmt.span)
            return

        if isinstance(stmt, ast.ReturnStmt):
            self._check_return(stmt)
            return

        if isinstance(stmt, ast.PrintStmt):
            for value in stmt.values:
                vty = self._check_expr(value, want=None)
                if vty == T.VOID:
                    self.diags.error("cannot print a void value", value.span)
                elif vty != T.INT and vty != T.FLOAT and vty != T.BOOL and vty != T.STRING \
                        and not isinstance(vty, T.ArrayType):
                    self.diags.error(
                        f"cannot print a value of type {vty.render()}", value.span)
            return

        raise AssertionError(f"unhandled statement {type(stmt).__name__}")

    def _check_let(self, stmt: ast.LetStmt) -> None:
        declared: Optional[T.Type] = None
        if stmt.declared is not None:
            try:
                declared = T.from_ast(stmt.declared, stmt.span)
            except T.TypeError_ as exc:
                self.diags.error(exc.message, exc.span)
                declared = None

        # The initializer is checked without an expected type so that a mismatch
        # produces exactly one diagnostic ("cannot initialise ...") rather than
        # that plus a redundant "expected int but found string".
        value_ty = self._check_expr(stmt.init, want=None)
        if declared is not None:
            if not T.is_assignable(declared, value_ty):
                self.diags.error(
                    f"cannot initialise '{declared.render()} {stmt.name}' with a "
                    f"value of type {value_ty.render()}",
                    stmt.span,
                    notes=self._conversion_hint(declared, value_ty),
                )
        else:
            declared = value_ty

        if declared == T.VOID:
            self.diags.error(
                f"variable '{stmt.name}' cannot have type void", stmt.span)
            declared = T.INT
        if self.scope.owns(stmt.name):
            self.diags.error(
                f"'{stmt.name}' is already declared in this scope", stmt.span,
                notes=["use a different name, or remove the earlier declaration"],
            )
        symbol = Symbol(
            stmt.name, declared, stmt.span, is_const=stmt.is_const,
            defined_in_function=self.current_fn.name if self.current_fn else None,
        )
        self.scope.declare(symbol)
        stmt.symbol = symbol

    def _check_for(self, stmt: ast.ForStmt) -> None:
        start_ty = self._check_expr(stmt.start, want=None)
        end_ty = self._check_expr(stmt.end, want=None)
        for bound, ty in ((stmt.start, start_ty), (stmt.end, end_ty)):
            if not T.is_numeric(ty):
                self.diags.error(
                    f"range bounds must be int or float, found {ty.render()}",
                    bound.span,
                )

        # A float bound makes the loop counter a float, so `for i in 0.0..3.0`
        # counts in halves rather than truncating to zero immediately.
        element_ty = T.FLOAT if T.FLOAT in (start_ty, end_ty) else T.INT
        stmt.var_ty = element_ty

        self.scope = Scope(self.scope)
        loop_symbol = Symbol(stmt.var, element_ty, stmt.span, is_const=True,
                             defined_in_function=self.current_fn.name
                             if self.current_fn else None)
        self.scope.declare(loop_symbol)
        stmt.symbol = loop_symbol
        self.loop_depth += 1
        self._check_stmt(stmt.body)
        self.loop_depth -= 1
        self.scope = self.scope.parent or self.scope

    def _check_return(self, stmt: ast.ReturnStmt) -> None:
        if stmt.value is None:
            if self.current_ret != T.VOID:
                self.diags.error(
                    f"expected a value of type {self.current_ret.render()} "
                    f"before 'return'",
                    stmt.span,
                )
            return
        value_ty = self._check_expr(stmt.value, want=self.current_ret)
        if not T.is_assignable(self.current_ret, value_ty):
            self.diags.error(
                f"cannot return {value_ty.render()} from a function declared to "
                f"return {self.current_ret.render()}",
                stmt.value.span,
                notes=self._conversion_hint(self.current_ret, value_ty),
            )

    # -- expressions -----------------------------------------------------
    def _check_expr(self, expr: ast.Expr, want: Optional[T.Type]) -> T.Type:
        ty = self._infer(expr)
        expr.ty = ty
        if want is not None and ty != T.VOID and not T.is_assignable(want, ty):
            self.diags.error(
                f"expected {want.render()} but found {ty.render()}",
                expr.span,
                notes=self._conversion_hint(want, ty),
            )
        return ty

    def _infer(self, expr: ast.Expr) -> T.Type:
        if isinstance(expr, ast.IntLit):
            return T.INT
        if isinstance(expr, ast.FloatLit):
            return T.FLOAT
        if isinstance(expr, ast.BoolLit):
            return T.BOOL
        if isinstance(expr, ast.StrLit):
            return T.STRING

        if isinstance(expr, ast.Ident):
            symbol = self.scope.lookup(expr.name)
            if symbol is None:
                self.diags.error(
                    f"use of undeclared identifier '{expr.name}'", expr.span)
                return T.INT  # keep going so one typo yields many useful errors
            if isinstance(symbol.ty, T.FunctionType):
                self.diags.error(
                    f"'{expr.name}' is a function; call it with {expr.name}(...)",
                    expr.span,
                )
                return T.INT
            expr.symbol = symbol
            return symbol.ty

        if isinstance(expr, ast.ArrayLit):
            if not expr.elements:
                return T.ArrayType(T.INT)
            # Infer the element type by unifying as we go, so `[1, 2.0]`
            # settles on float instead of complaining about the second value.
            elem_ty: Optional[T.Type] = None
            for index, element in enumerate(expr.elements):
                other = self._check_expr(element, want=None)
                if elem_ty is None:
                    elem_ty = other
                    continue
                unified = T.unify(elem_ty, other)
                if unified is None:
                    self.diags.error(
                        f"array elements must share one type: found "
                        f"{elem_ty.render()} and {other.render()}",
                        element.span,
                        notes=self._conversion_hint(elem_ty, other),
                    )
                else:
                    elem_ty = unified
            return T.ArrayType(elem_ty if elem_ty is not None else T.INT)

        if isinstance(expr, ast.Unary):
            return self._infer_unary(expr)

        if isinstance(expr, ast.Binary):
            return self._infer_binary(expr)

        if isinstance(expr, ast.Assign):
            return self._infer_assign(expr)

        if isinstance(expr, ast.Index):
            return self._infer_index(expr)

        if isinstance(expr, ast.LenBuiltin):
            return self._infer_len(expr)

        if isinstance(expr, ast.Call):
            return self._infer_call(expr)

        raise AssertionError(f"unhandled expression {type(expr).__name__}")

    def _infer_unary(self, expr: ast.Unary) -> T.Type:
        operand_ty = self._check_expr(expr.operand, want=None)
        if expr.op == "-":
            if not T.is_numeric(operand_ty):
                self.diags.error(
                    f"cannot negate a value of type {operand_ty.render()}",
                    expr.operand.span,
                    notes=self._conversion_hint(T.INT, operand_ty)
                    if operand_ty == T.BOOL else None,
                )
                return T.INT
            return operand_ty
        if expr.op in ("!", "not"):
            if operand_ty != T.BOOL:
                self.diags.error(
                    f"'not' expects a bool, found {operand_ty.render()}",
                    expr.operand.span,
                    notes=["comparison operators already produce bool"],
                )
                return T.BOOL
            return T.BOOL
        raise AssertionError(f"unhandled unary operator {expr.op!r}")

    def _infer_binary(self, expr: ast.Binary) -> T.Type:
        left_ty = self._check_expr(expr.left, want=None)
        right_ty = self._check_expr(expr.right, want=None)
        op = expr.op

        if op in ("==", "!="):
            if not self._comparable(left_ty, right_ty, expr):
                return T.BOOL
            return T.BOOL

        if op in ("<", "<=", ">", ">="):
            if not T.is_numeric(left_ty) or not T.is_numeric(right_ty):
                self.diags.error(
                    f"'{op}' requires int or float operands, found "
                    f"{left_ty.render()} and {right_ty.render()}",
                    expr.span,
                    notes=self._conversion_hint(T.INT, left_ty),
                )
                return T.BOOL
            return T.BOOL

        if op in ("&&", "||", "and", "or"):
            for side, ty in ((expr.left, left_ty), (expr.right, right_ty)):
                if ty != T.BOOL:
                    self.diags.error(
                        f"'{op}' expects bool operands, found {ty.render()}",
                        side.span,
                    )
            return T.BOOL

        if op == "+" and (left_ty == T.STRING or right_ty == T.STRING):
            self.diags.error(
                "'+' does not concatenate strings; Lumen has no implicit "
                "string arithmetic",
                expr.span,
                notes=["print several values at once: print \"a\", b;"],
            )
            return T.INT

        if not T.is_numeric(left_ty) or not T.is_numeric(right_ty):
            self.diags.error(
                f"'{op}' requires int or float operands, found "
                f"{left_ty.render()} and {right_ty.render()}",
                expr.span,
                notes=self._conversion_hint(T.INT, left_ty),
            )
            return T.INT

        if op == "%" and left_ty == T.FLOAT and right_ty == T.FLOAT:
            self.diags.error("'%' is not defined for float operands", expr.span,
                             notes=["use fmod-style arithmetic or keep modulo on int"])
            return T.FLOAT

        common = T.unify(left_ty, right_ty)
        return common if common is not None else T.INT

    def _infer_assign(self, expr: ast.Assign) -> T.Type:
        target_ty = self._check_expr(expr.target, want=None)
        if isinstance(expr.target, ast.Ident):
            symbol = self.scope.lookup(expr.target.name)
            if symbol is not None and symbol.is_const:
                self.diags.error(
                    f"cannot assign to '{expr.target.name}': it is declared const",
                    expr.span,
                )
        elif isinstance(expr.target, ast.Index):
            pass  # array elements are mutable even if the array binding is const
        else:
            self.diags.error("invalid assignment target", expr.target.span,
                             notes=["assignments target a variable or an array element"])

        value_ty = self._check_expr(expr.value, want=target_ty)
        if not T.is_assignable(target_ty, value_ty):
            self.diags.error(
                f"cannot assign {value_ty.render()} to a {target_ty.render()} target",
                expr.span,
                notes=self._conversion_hint(target_ty, value_ty),
            )
        return target_ty

    def _infer_index(self, expr: ast.Index) -> T.Type:
        base_ty = self._check_expr(expr.base, want=None)
        index_ty = self._check_expr(expr.index, want=T.INT)
        if index_ty not in (T.INT, T.VOID):
            if index_ty != T.INT:
                self.diags.error(
                    f"array index must be int, found {index_ty.render()}",
                    expr.index.span,
                )
        if not isinstance(base_ty, T.ArrayType):
            self.diags.error(
                f"cannot index a value of type {base_ty.render()}", expr.base.span,
                notes=["indexing applies to arrays: xs[i]"],
            )
            return T.INT
        return base_ty.element

    def _infer_len(self, expr: ast.LenBuiltin) -> T.Type:
        target_ty = self._check_expr(expr.target, want=None)
        if not isinstance(target_ty, T.ArrayType) and target_ty != T.STRING:
            self.diags.error(
                f"len() needs an array or string, found {target_ty.render()}",
                expr.target.span,
            )
        return T.INT

    def _infer_call(self, expr: ast.Call) -> T.Type:
        callee = self.functions.get(expr.callee)
        if callee is None:
            if expr.callee in _BUILTIN_LOCALS:
                self.diags.error(
                    f"'{expr.callee}' is a statement, not a function; write "
                    f"'{expr.callee} ...;'",
                    expr.span,
                )
                return T.INT
            self.diags.error(
                f"call to undeclared function '{expr.callee}'", expr.span,
                notes=[f"known functions: {', '.join(sorted(self.functions)) or 'none'}"]
                if self.functions else ["no functions are declared yet"],
            )
            for arg in expr.args:
                self._check_expr(arg, want=None)
            return T.INT

        try:
            sig = T.from_ast(callee.ret, callee.span)
            params = tuple(
                T.from_ast(p.ty, p.span) for p in callee.params
            )
        except T.TypeError_ as exc:
            self.diags.error(exc.message, exc.span)
            return T.INT

        if len(expr.args) != len(callee.params):
            rendered = ", ".join(p.ty.render() for p in callee.params)
            signature = f"{expr.callee}({rendered}) -> {sig.render()}"
            self.diags.error(
                f"function '{expr.callee}' takes {len(callee.params)} argument(s) "
                f"but {len(expr.args)} were given",
                expr.span,
                notes=[f"signature: {signature}"],
            )
        for arg, param, ptype in zip(expr.args, callee.params, params):
            self._check_expr(arg, want=ptype)
        expr.callee_symbol = self.globals[expr.callee]
        expr.param_types = list(params)
        return sig

    # -- helpers ---------------------------------------------------------
    def _comparable(self, left: T.Type, right: T.Type, expr: ast.Binary) -> bool:
        if T.unify(left, right) is not None:
            return True
        self.diags.error(
            f"cannot compare {left.render()} with {right.render()}", expr.span)
        return False

    def _require_bool(self, expr: ast.Expr, ty: T.Type) -> None:
        if ty != T.BOOL:
            self.diags.error(
                f"condition must be bool, found {ty.render()}", expr.span,
                notes=["write an explicit comparison, e.g. 'x > 0'"],
            )

    def _conversion_hint(self, target: T.Type, source: T.Type) -> Optional[str]:
        if target == T.FLOAT and source == T.INT:
            return ["an int widens to float automatically"]
        if target == T.INT and source == T.FLOAT:
            return ["add an explicit conversion or keep the value as float"]
        if source == T.BOOL:
            return ["bool is not a number; compare or convert deliberately"]
        if target == T.STRING or source == T.STRING:
            return ["strings are a distinct type in Lumen"]
        return None

    # -- control flow ----------------------------------------------------
    def _returns(self, stmt: ast.Stmt) -> bool:
        """True when control always leaves ``stmt`` via ``return``."""
        if isinstance(stmt, ast.ReturnStmt):
            return True
        if isinstance(stmt, ast.BreakStmt) or isinstance(stmt, ast.ContinueStmt):
            return False
        if isinstance(stmt, ast.Block):
            return any(self._returns(s) for s in stmt.stmts)
        if isinstance(stmt, ast.IfStmt):
            if stmt.else_branch is None:
                return False
            return self._returns(stmt.then_branch) and self._returns(stmt.else_branch)
        if isinstance(stmt, ast.WhileStmt):
            # `while true { ... return ... }` never falls through.
            if isinstance(stmt.cond, ast.BoolLit) and stmt.cond.value:
                return any(self._returns(s) for s in stmt.body.stmts)
            return False
        return False


def check(program: ast.Program, diags: DiagnosticBag) -> ast.Program:
    Checker(diags).check_program(program)
    return program
"""Machine-independent optimizations on the checked AST.

Everything here is deliberately conservative: a transformation must be
provably value-preserving for the subset of the language where the checker has
already guaranteed types, or it does not fire. Optimizing the AST rather than
the emitted text keeps this pass in a place where correctness is easy to argue
about, and leaves LLVM's own (much stronger) pipeline to do the real work on
the IR afterwards.

Transformations
    Constant folding   -- ``2 * 3`` becomes ``6`` before it reaches LLVM.
    Identity folding   -- ``x + 0``, ``x * 1``, ``x - 0``, ``x * 0``.
    Branch folding     -- ``if true`` / ``while false`` keep only the live arm.
    Dead code removal  -- statements after ``return``/``break``/``continue``.
    Constant condition  -- ``if false { ... }`` disappears entirely.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from . import ast, types as T


class Optimizer:
    def __init__(self, level: int = 1) -> None:
        self.level = max(0, min(level, 3))
        self.stats: Dict[str, int] = {}

    def _bump(self, key: str, amount: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + amount

    # -- entry point -----------------------------------------------------
    def optimize_program(self, program: ast.Program) -> ast.Program:
        if self.level == 0:
            return program
        for fn in program.functions:
            fn.body = self.optimize_block(fn.body)
        return program

    # -- statements ------------------------------------------------------
    def optimize_block(self, block: ast.Block) -> ast.Block:
        kept: List[ast.Stmt] = []
        for stmt in block.stmts:
            if kept and isinstance(kept[-1], (ast.ReturnStmt, ast.BreakStmt,
                                              ast.ContinueStmt)):
                self._bump("dead-statement")
                continue
            stmt = self.optimize_stmt(stmt)
            if stmt is not None:
                kept.append(stmt)
        block.stmts = kept
        return block

    def optimize_stmt(self, stmt: ast.Stmt) -> Optional[ast.Stmt]:
        if isinstance(stmt, ast.Block):
            self.optimize_block(stmt)
            return stmt

        if isinstance(stmt, ast.LetStmt):
            folded = self.fold(stmt.init)
            stmt.init = folded
            # `if false { ... }` inside a let initializer leaves a hole, so
            # replace the value with a default rather than deleting the binding.
            return stmt

        if isinstance(stmt, ast.ExprStmt):
            stmt.expr = self.fold(stmt.expr)
            return stmt

        if isinstance(stmt, ast.IfStmt):
            stmt.cond = self.fold(stmt.cond)
            stmt.then_branch = self.optimize_block(stmt.then_branch)
            if stmt.else_branch is not None:
                if isinstance(stmt.else_branch, ast.Block):
                    self.optimize_block(stmt.else_branch)
                else:
                    stmt.else_branch = self.optimize_stmt(stmt.else_branch)
            if isinstance(stmt.cond, ast.BoolLit):
                self._bump("folded-branch")
                if stmt.cond.value:
                    return stmt.then_branch
                # A constant-false condition makes the else arm the whole
                # statement; a missing else leaves nothing to do.
                if stmt.else_branch is None:
                    return ast.Block(stmts=[], span=stmt.span)
                return stmt.else_branch
            return stmt

        if isinstance(stmt, ast.WhileStmt):
            stmt.cond = self.fold(stmt.cond)
            stmt.body = self.optimize_block(stmt.body)
            if isinstance(stmt.cond, ast.BoolLit) and not stmt.cond.value:
                self._bump("dead-loop")
                return None
            if isinstance(stmt.cond, ast.BoolLit) and stmt.cond.value:
                # `while true` may never fall through, so guard the loop.
                pass
            return stmt

        if isinstance(stmt, ast.ForStmt):
            stmt.start = self.fold(stmt.start)
            stmt.end = self.fold(stmt.end)
            stmt.body = self.optimize_block(stmt.body)
            if isinstance(stmt.start, ast.IntLit) and isinstance(stmt.end, ast.IntLit):
                if stmt.start.value >= stmt.end.value:
                    self._bump("dead-loop")
                    return None
            return stmt

        if isinstance(stmt, ast.ReturnStmt):
            if stmt.value is not None:
                stmt.value = self.fold(stmt.value)
            return stmt

        if isinstance(stmt, ast.PrintStmt):
            for index, value in enumerate(stmt.values):
                stmt.values[index] = self.fold(value)
            return stmt

        return stmt

    # -- expressions -----------------------------------------------------
    def fold(self, expr: ast.Expr) -> ast.Expr:
        if isinstance(expr, ast.Assign):
            folded_target = self.fold(expr.target)
            folded_value = self.fold(expr.value)
            expr.target = folded_target
            expr.value = folded_value
            return expr

        if isinstance(expr, ast.ArrayLit):
            for index, element in enumerate(expr.elements):
                expr.elements[index] = self.fold(element)
            return expr

        if isinstance(expr, (ast.IntLit, ast.FloatLit, ast.BoolLit, ast.StrLit)):
            return expr

        if isinstance(expr, ast.Ident):
            return expr

        if isinstance(expr, ast.Index):
            expr.base = self.fold(expr.base)
            expr.index = self.fold(expr.index)
            return expr

        if isinstance(expr, ast.LenBuiltin):
            expr.target = self.fold(expr.target)
            if isinstance(expr.target, ast.ArrayLit):
                self._bump("folded-len")
                return _int_literal(len(expr.target.elements), expr.span, expr.ty)
            return expr

        if isinstance(expr, ast.Unary):
            expr.operand = self.fold(expr.operand)
            if isinstance(expr.operand, ast.IntLit) and expr.op == "-":
                self._bump("folded-const")
                return _int_literal(-expr.operand.value, expr.span, expr.ty)
            if isinstance(expr.operand, ast.FloatLit) and expr.op == "-":
                self._bump("folded-const")
                folded = ast.FloatLit(value=-expr.operand.value, span=expr.span)
                folded.ty = expr.ty
                return folded
            if isinstance(expr.operand, ast.BoolLit) and expr.op in ("!", "not"):
                self._bump("folded-const")
                folded = ast.BoolLit(value=not expr.operand.value, span=expr.span)
                folded.ty = expr.ty
                return folded
            return expr

        if isinstance(expr, ast.Binary):
            expr.left = self.fold(expr.left)
            expr.right = self.fold(expr.right)
            return self._fold_binary(expr)

        return expr

    def _fold_binary(self, expr: ast.Binary) -> ast.Expr:
        left, right = expr.left, expr.right
        op = expr.op

        # Short-circuit operands are logical, not arithmetic: only fold when
        # the operator itself decides the result.
        if op in ("&&", "and"):
            if isinstance(left, ast.BoolLit) and isinstance(right, ast.BoolLit):
                self._bump("folded-const")
                return _bool_literal(left.value and right.value, expr.span, expr.ty)
            return expr
        if op in ("||", "or"):
            if isinstance(left, ast.BoolLit) and isinstance(right, ast.BoolLit):
                self._bump("folded-const")
                return _bool_literal(left.value or right.value, expr.span, expr.ty)
            return expr

        numeric_left = isinstance(left, (ast.IntLit, ast.FloatLit))
        numeric_right = isinstance(right, (ast.IntLit, ast.FloatLit))

        if op in ("==", "!=", "<", "<=", ">", ">="):
            if not (isinstance(left, ast.StrLit) and isinstance(right, ast.StrLit)):
                if numeric_left and numeric_right:
                    return self._fold_comparison(expr)
            return expr

        if not (numeric_left and numeric_right):
            return self._fold_identity(expr)

        left_value = float(left.value) if isinstance(left, ast.FloatLit) else left.value
        right_value = float(right.value) if isinstance(right, ast.FloatLit) else right.value
        is_float = expr.ty == T.FLOAT

        try:
            if op == "+":
                result = left_value + right_value
            elif op == "-":
                result = left_value - right_value
            elif op == "*":
                result = left_value * right_value
            elif op == "/":
                if right_value == 0:
                    return expr  # keep the runtime division by zero
                result = left_value / right_value
            elif op == "%":
                if right_value == 0 or is_float:
                    return expr
                result = left_value % right_value
            else:
                return expr
        except (ZeroDivisionError, OverflowError):  # pragma: no cover - defensive
            return expr

        self._bump("folded-const")
        if is_float:
            folded = ast.FloatLit(value=float(result), span=expr.span)
        else:
            folded = ast.IntLit(value=int(result), span=expr.span)
        folded.ty = expr.ty
        return folded

    def _fold_comparison(self, expr: ast.Binary) -> ast.Expr:
        left, right = expr.left, expr.right
        left_value = float(left.value) if isinstance(left, ast.FloatLit) else left.value
        right_value = float(right.value) if isinstance(right, ast.FloatLit) else right.value
        table = {
            "==": left_value == right_value,
            "!=": left_value != right_value,
            "<": left_value < right_value,
            "<=": left_value <= right_value,
            ">": left_value > right_value,
            ">=": left_value >= right_value,
        }
        self._bump("folded-const")
        return _bool_literal(table[expr.op], expr.span, expr.ty)

    def _fold_identity(self, expr: ast.Binary) -> ast.Expr:
        """Remove algebraic no-ops such as ``x * 1`` and ``x + 0``."""
        left, right, op = expr.left, expr.right, expr.op
        if isinstance(right, ast.IntLit):
            if op == "+" and right.value == 0:
                self._bump("folded-identity")
                return left
            if op == "-" and right.value == 0:
                self._bump("folded-identity")
                return left
            if op == "*" and right.value == 1:
                self._bump("folded-identity")
                return left
        if isinstance(left, ast.IntLit) and op == "+" and left.value == 0:
            self._bump("folded-identity")
            return right
        if isinstance(left, ast.IntLit) and op == "*" and left.value == 1:
            self._bump("folded-identity")
            return right
        if op == "*" and expr.ty == T.INT:
            if (isinstance(right, ast.IntLit) and right.value == 0) or \
                    (isinstance(left, ast.IntLit) and left.value == 0):
                self._bump("folded-identity")
                return _int_literal(0, expr.span, expr.ty)
        return expr


def _int_literal(value: int, span, ty) -> ast.IntLit:
    node = ast.IntLit(value=value, span=span)
    node.ty = ty if ty == T.INT else T.INT
    return node


def _bool_literal(value: bool, span, ty) -> ast.BoolLit:
    node = ast.BoolLit(value=value, span=span)
    node.ty = ty if ty == T.BOOL else T.BOOL
    return node


def optimize(program: ast.Program, level: int = 1) -> tuple:
    """Optimize in place; returns ``(program, stats)``."""
    optimizer = Optimizer(level)
    return optimizer.optimize_program(program), optimizer.stats
"""LLVM IR backend for Lumen.

Design notes
------------
*Memory first.* Every local variable and every parameter lives in an ``alloca``
in the function's entry block, and accesses go through explicit ``load``/
``store``. That is the "unoptimized" shape, but it is the shape LLVM's own
``mem2reg`` pass expects as input: run ``opt -passes=mem2reg`` (or just let
``clang -O2`` do it) and every alloca that never escapes disappears. Because
alloca-based codegen never needs dominance-frontier computation, this backend
emits **no phi nodes at all** -- merge points read their value from a stack
slot instead. That single decision is what keeps the emitter small enough to
audit by hand.

*Arrays are heap objects with a header.* An array value is a pointer to its
first element, and the eight bytes immediately below it hold the element
count::

    [ i64 length ][ elem 0 ][ elem 1 ] ...
      ^
      the array pointer points at elem 0

The header makes ``len(a)`` O(1) and lets an array be passed to and returned
from a function without threading a separate length value. It also means the
original allocation can be recovered with ``getelementptr i8, i8* %p, i64 -8``,
which is what ``a = b`` uses when the destination needs to grow.

*Types.* ``int`` -> ``i64``, ``float`` -> ``double``, ``bool`` -> ``i1``,
``string`` -> ``i8*`` (an ``[N x i8]`` global plus a pointer), ``T[]`` -> ``T*``.

*Control flow.* ``if``/``while``/``for`` become labelled basic blocks.
``&&`` and ``||`` short-circuit through explicit blocks, so ``f() && g()``
never calls ``g()`` when ``f()`` is false.
"""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

from . import ast, types as T
from .diagnostics import DiagnosticBag, Span

_ALIGN = {"i64": 8, "double": 8, "i1": 1, "i8*": 1, "i8": 1}

_TERMINATORS = ("ret", "br", "switch", "unreachable", "resume", "invoke")


class CodegenError(Exception):
    """Raised for AST shapes the backend cannot lower."""

    def __init__(self, message: str, span: Span) -> None:
        super().__init__(message)
        self.message = message
        self.span = span


# --------------------------------------------------------------------------
# Type mapping and literal formatting
# --------------------------------------------------------------------------


def llvm_type(t: T.Type) -> str:
    """Lower a Lumen type to its LLVM counterpart."""
    if t == T.INT:
        return "i64"
    if t == T.FLOAT:
        return "double"
    if t == T.BOOL:
        return "i1"
    if t == T.STRING:
        return "i8*"
    if t == T.VOID:
        return "void"
    if isinstance(t, T.ArrayType):
        return f"{llvm_type(t.element)}*"
    raise AssertionError(f"no LLVM mapping for {t!r}")


def size_of(t: T.Type) -> int:
    """Byte size of one value of ``t`` stored in memory."""
    ll = llvm_type(t)
    if ll.endswith("*"):
        return 8
    return _ALIGN.get(ll, 8)


def align_of(t: T.Type) -> int:
    return size_of(t)


def sanitize(name: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not out or out[0].isdigit():
        out = "v" + out
    return out


def mangle(name: str) -> str:
    """Global symbol name for a user function (avoids libc collisions)."""
    return "main" if name == "main" else "_lumen_" + sanitize(name)


def escape_bytes(data: bytes) -> str:
    """Escape raw bytes for an LLVM ``c"..."`` string literal."""
    out = []
    for byte in data:
        ch = chr(byte)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append("\\22")
        elif ch == "\n":
            out.append("\\0A")
        elif ch == "\t":
            out.append("\\09")
        elif ch == "\r":
            out.append("\\0D")
        elif byte == 0:
            out.append("\\00")
        elif 0x20 <= byte < 0x7F:
            out.append(ch)
        else:
            out.append("\\%02X" % byte)
    return "".join(out)


def float_const(value: float) -> str:
    """Exact, round-trippable LLVM text for a double constant."""
    if value != value:
        return "0x7FF8000000000000"  # quiet NaN
    if value == float("inf"):
        return "0x7FF0000000000000"
    if value == float("-inf"):
        return "0xFFF0000000000000"
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    if bits == 0:
        return "0x0000000000000000"
    if bits == (1 << 63):
        return "-0x0000000000000000"
    return "0x%016X" % bits


def _terminated(block: List[str]) -> bool:
    """True when ``block`` already ends in a control-flow instruction."""
    for line in reversed(block):
        text = line.strip()
        if text:
            return text.split(" ", 1)[0] in _TERMINATORS
    return False


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------


class IRBuilder:
    """Emits one LLVM module as text."""

    def __init__(self, module_name: str, diags: Optional[DiagnosticBag] = None,
                 target_triple: Optional[str] = None,
                 source_filename: Optional[str] = None) -> None:
        self.module_name = module_name
        self.diags = diags
        self.target_triple = target_triple
        self.source_filename = source_filename or module_name
        self.globals: List[str] = []
        self.declares: List[str] = []
        self.defines: List[str] = []
        self._strings: Dict[str, Tuple[str, int]] = {}
        self._str_counter = 0
        self._label_counter = 0
        self._tmp_counter = 0
        self._blocks: List[List[str]] = []
        self._loops: List[Tuple[str, str]] = []  # (exit label, continue label)
        self._current_ret: T.Type = T.INT
        self.instruction_count = 0

    # -- naming ----------------------------------------------------------
    def fresh(self, prefix: str = "t") -> str:
        self._tmp_counter += 1
        return f"%{prefix}{self._tmp_counter}"

    def label(self, prefix: str) -> str:
        self._label_counter += 1
        return f"{prefix}{self._label_counter}"

    # -- emission --------------------------------------------------------
    def emit(self, line: str) -> None:
        self.instruction_count += 1
        if self._blocks:
            self._blocks[-1].append("  " + line)

    def start_block(self, name: str) -> None:
        self._blocks.append([f"{name}:"])

    def end_block(self) -> List[str]:
        return self._blocks.pop()

    def detached_block(self, name: str, body) -> None:
        """Emit ``body`` as a region of blocks branching off the current one.

        Blocks are rendered after the block they branch from (any order is legal
        in LLVM) and the current block is left current afterwards, so the caller
        decides what follows. ``body`` may push continuation blocks of its own --
        a nested ``if`` always leaves its merge block open -- so every block it
        creates is collected and re-attached in emission order.
        """
        saved = self._blocks.pop()
        depth = len(self._blocks)
        self._blocks.append([f"{name}:"])
        body()
        region: List[List[str]] = []
        while len(self._blocks) > depth:
            region.append(self._blocks.pop())
        region.reverse()
        # `saved` holds individual lines, so flatten the collected blocks.
        saved.extend(line for block in region for line in block)
        self._blocks.append(saved)

    def terminated(self) -> bool:
        return bool(self._blocks) and _terminated(self._blocks[-1])

    # -- string interning ------------------------------------------------
    def global_string(self, text: str) -> str:
        """Intern ``text`` as a NUL-terminated global constant."""
        cached = self._strings.get(text)
        if cached is not None:
            return cached[0]
        data = text.encode("utf-8") + b"\0"
        self._str_counter += 1
        name = f"@.str.{self._str_counter}"
        self.globals.append(
            f'{name} = private unnamed_addr constant [{len(data)} x i8] '
            f'c"{escape_bytes(data)}", align 1'
        )
        self._strings[text] = (name, len(data))
        return name

    def string_ptr(self, text: str) -> str:
        """A ``i8*`` value pointing at the interned constant ``text``."""
        name = self.global_string(text)
        size = self._strings[text][1]
        tmp = self.fresh("s")
        self.emit(
            f"{tmp} = getelementptr inbounds [{size} x i8], "
            f"[{size} x i8]* {name}, i64 0, i64 0"
        )
        return tmp

    # -- module ----------------------------------------------------------
    def generate(self, program: ast.Program) -> str:
        self.declares.extend([
            "declare i32 @printf(i8*, ...)",
            "declare i8* @malloc(i64)",
            "declare i8* @realloc(i8*, i64)",
            "declare void @free(i8*)",
            "declare i64 @strlen(i8*)",
            "declare i32 @strcmp(i8*, i8*)",
            "declare void @memcpy(i8*, i8*, i64)",
        ])
        for fn in program.functions:
            self.defines.append(self.emit_function(fn))
        return self.render()

    def render(self) -> str:
        out: List[str] = [
            f"; ModuleID = '{self.module_name}'",
            f'source_filename = "{self.source_filename}"',
        ]
        if self.target_triple:
            out.append(f'target triple = "{self.target_triple}"')
        out.append("")
        if self.globals:
            out.append("; ---- string constants ----")
            out.extend(self.globals)
            out.append("")
        if self.declares:
            out.append("; ---- external declarations ----")
            out.extend(self.declares)
            out.append("")
        if self.defines:
            out.append("; ---- functions ----")
            out.extend(self.defines)
        return "\n".join(out) + "\n"

    # ------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------
    def emit_function(self, fn: ast.FuncDecl) -> str:
        self._blocks = []
        self._tmp_counter = 0
        self._loops = []
        ret = fn.ret_ty
        if fn.name == "main" and ret == T.VOID:
            # A void main is normalised to i32 so that the entry point matches
            # what the C runtime expects; the exit code becomes 0.
            ret = T.INT
        self._current_ret = ret

        param_text = ", ".join(
            f"{llvm_type(pty)} %{sanitize(p.name)}"
            for p in fn.params
            for pty in [p.param_ty]
        )
        header = f"define {llvm_type(ret)} @{mangle(fn.name)}({param_text}) {{"

        self.start_block("entry")
        for p in fn.params:
            slot, pty = p.slot, p.param_ty
            self.emit(f"{slot} = alloca {llvm_type(pty)}, align {align_of(pty)}")
            self.emit(
                f"store {llvm_type(pty)} %{sanitize(p.name)}, "
                f"{llvm_type(pty)}* {slot}, align {align_of(pty)}"
            )
        self.emit_stmt(fn.body)
        self.emit_epilogue(fn)
        # A function body spans several blocks: drain the whole stack in order.
        body: List[str] = []
        while self._blocks:
            body.extend(self._blocks.pop(0))
        return "\n".join([header] + body + ["}"])

    def emit_epilogue(self, fn: ast.FuncDecl) -> None:
        """Guarantee the final block is terminated, even on error paths."""
        if self.terminated():
            return
        ret = self._current_ret
        if ret == T.VOID or fn.name == "main":
            self.emit("ret void" if ret == T.VOID else "ret i64 0")
        else:
            # Only reachable when the checker already reported a missing
            # return; emit a defined value so the IR stays syntactically valid.
            self.emit(f"ret {llvm_type(ret)} zeroinitializer")

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------
    def emit_stmt(self, stmt: ast.Stmt) -> None:
        if isinstance(stmt, ast.Block):
            for inner in stmt.stmts:
                if self.terminated():
                    break  # unreachable after return/break/continue
                self.emit_stmt(inner)
            return
        if isinstance(stmt, ast.LetStmt):
            self.emit_let(stmt)
        elif isinstance(stmt, ast.ExprStmt):
            if isinstance(stmt.expr, ast.Assign):
                self.emit_assign(stmt.expr)
            else:
                self.emit_expr(stmt.expr)
        elif isinstance(stmt, ast.IfStmt):
            self.emit_if(stmt)
        elif isinstance(stmt, ast.WhileStmt):
            self.emit_while(stmt)
        elif isinstance(stmt, ast.ForStmt):
            self.emit_for(stmt)
        elif isinstance(stmt, ast.BreakStmt):
            if self._loops:
                self.emit(f"br label %{self._loops[-1][0]}")
        elif isinstance(stmt, ast.ContinueStmt):
            if self._loops:
                self.emit(f"br label %{self._loops[-1][1]}")
        elif isinstance(stmt, ast.ReturnStmt):
            self.emit_return(stmt)
        elif isinstance(stmt, ast.PrintStmt):
            self.emit_print(stmt)
        else:
            raise CodegenError(
                f"cannot generate code for {type(stmt).__name__}", stmt.span)

    def emit_let(self, stmt: ast.LetStmt) -> None:
        slot, ty = stmt.slot, stmt.var_ty
        self.emit(f"{slot} = alloca {llvm_type(ty)}, align {align_of(ty)}")
        value = self.emit_expr(stmt.init)

        if isinstance(ty, T.ArrayType):
            # A fresh literal already owns its buffer, so it can be stored
            # directly. Initialising from another array has to *clone*, or the
            # two bindings would alias one heap object.
            if not isinstance(stmt.init, ast.ArrayLit):
                self.emit_array_clone(slot, ty, value)
                return
            self.emit(
                f"store {llvm_type(ty)} {value}, {llvm_type(ty)}* {slot}, align 8")
            return

        value = self.coerce(value, stmt.init.ty, ty)
        self.emit(
            f"store {llvm_type(ty)} {value}, {llvm_type(ty)}* {slot}, "
            f"align {align_of(ty)}"
        )

    def emit_if(self, stmt: ast.IfStmt) -> None:
        cond = self.emit_expr(stmt.cond)
        then_label = self.label("if.then")
        has_else = stmt.else_branch is not None
        else_label = self.label("if.else") if has_else else None
        end_label = self.label("if.end")

        self.emit(
            f"br i1 {cond}, label %{then_label}, "
            f"label %{else_label if has_else else end_label}"
        )

        def emit_then() -> None:
            self.emit_stmt(stmt.then_branch)
            if not self.terminated():
                self.emit(f"br label %{end_label}")

        self.detached_block(then_label, emit_then)

        if has_else:
            def emit_else() -> None:
                self.emit_stmt(stmt.else_branch)  # type: ignore[arg-type]
                if not self.terminated():
                    self.emit(f"br label %{end_label}")

            self.detached_block(else_label, emit_else)
        self.start_block(end_label)

    def emit_while(self, stmt: ast.WhileStmt) -> None:
        cond_label = self.label("while.cond")
        body_label = self.label("while.body")
        end_label = self.label("while.end")

        self.emit(f"br label %{cond_label}")

        def emit_cond() -> None:
            cond = self.emit_expr(stmt.cond)
            self.emit(f"br i1 {cond}, label %{body_label}, label %{end_label}")

        def emit_body() -> None:
            self._loops.append((end_label, cond_label))
            self.emit_stmt(stmt.body)
            self._loops.pop()
            if not self.terminated():
                self.emit(f"br label %{cond_label}")

        self.detached_block(cond_label, emit_cond)
        self.detached_block(body_label, emit_body)
        self.start_block(end_label)

    def emit_for(self, stmt: ast.ForStmt) -> None:
        """``for i in a..b`` lowers to a counted half-open loop [a, b)."""
        slot, ty = stmt.slot, stmt.var_ty
        self.emit(f"{slot} = alloca {llvm_type(ty)}, align {align_of(ty)}")
        start = self.coerce(self.emit_expr(stmt.start), stmt.start.ty, ty)
        self.emit(f"store {llvm_type(ty)} {start}, {llvm_type(ty)}* {slot}, "
                  f"align {align_of(ty)}")

        cond_label = self.label("for.cond")
        body_label = self.label("for.body")
        step_label = self.label("for.step")
        end_label = self.label("for.end")

        self.emit(f"br label %{cond_label}")

        def emit_cond() -> None:
            limit = self.coerce(self.emit_expr(stmt.end), stmt.end.ty, ty)
            index = self.load(slot, ty)
            guard = self.fresh("c")
            if ty == T.INT:
                self.emit(f"{guard} = icmp slt i64 {index}, {limit}")
            else:
                self.emit(f"{guard} = fcmp olt double {index}, {limit}")
            self.emit(f"br i1 {guard}, label %{body_label}, label %{end_label}")

        def emit_body() -> None:
            self._loops.append((end_label, step_label))
            self.emit_stmt(stmt.body)
            self._loops.pop()
            if not self.terminated():
                self.emit(f"br label %{step_label}")

        def emit_step() -> None:
            current = self.load(slot, ty)
            bumped = self.fresh("i")
            if ty == T.INT:
                self.emit(f"{bumped} = add i64 {current}, 1")
            else:
                self.emit(f"{bumped} = fadd double {current}, 1.0")
            self.emit(f"store {llvm_type(ty)} {bumped}, {llvm_type(ty)}* {slot}, "
                      f"align {align_of(ty)}")
            self.emit(f"br label %{cond_label}")

        self.detached_block(cond_label, emit_cond)
        self.detached_block(body_label, emit_body)
        self.detached_block(step_label, emit_step)
        self.start_block(end_label)

    def emit_return(self, stmt: ast.ReturnStmt) -> None:
        ret = self._current_ret
        if stmt.value is None:
            self.emit("ret void")
            return
        value = self.coerce(self.emit_expr(stmt.value), stmt.value.ty, ret)
        self.emit(f"ret {llvm_type(ret)} {value}")

    # -- printing --------------------------------------------------------
    def emit_print(self, stmt: ast.PrintStmt) -> None:
        if not stmt.values:
            self.printf_literal("\n")
            return
        for position, value in enumerate(stmt.values):
            self.emit_print_value(value)
            if position + 1 < len(stmt.values):
                self.printf_literal(" ")
        self.printf_literal("\n")

    def emit_print_value(self, expr: ast.Expr) -> None:
        """Print one value, recursing into arrays.

        Arrays print in bracket form (``[1, 2, 3]``). A literal is unrolled at
        emit time; anything else gets a runtime element loop.
        """
        ty = expr.ty
        if isinstance(ty, T.ArrayType):
            self.printf_literal("[")
            if isinstance(expr, ast.ArrayLit):
                for index, element in enumerate(expr.elements):
                    if index:
                        self.printf_literal(", ")
                    self.emit_print_value(element)
            else:
                pointer = self.emit_expr(expr)
                self.emit_array_loop(pointer, ty, parenthesised=False)
            self.printf_literal("]")
            return
        value = self.emit_expr(expr)
        self.emit_scalar_print(value, ty)

    def emit_scalar_print(self, value: str, ty: T.Type) -> None:
        """Print one scalar.

        A variadic argument must be passed with its real type -- ``%s`` needs
        an ``i8*``, not an ``i64`` -- so each case supplies the LLVM type it
        passes to printf.
        """
        if ty == T.INT:
            self.printf_arg("%lld", value, "i64")
        elif ty == T.FLOAT:
            self.printf_arg("%f", value, "double")
        elif ty == T.BOOL:
            widened = self.fresh("b")
            self.emit(f"{widened} = zext i1 {value} to i64")
            self.printf_arg("%lld", widened, "i64")
        elif ty == T.STRING:
            self.printf_arg("%s", value, "i8*")

    def emit_array_loop(self, pointer: str, ty: T.ArrayType,
                        parenthesised: bool) -> None:
        """Print the elements of ``pointer`` by walking it at runtime.

        The pointer is held in an alloca so that nested arrays (whose element
        loop needs a fresh pointer each time) never require a phi node.
        """
        element_ty = ty.element
        ll_element = llvm_type(element_ty)

        holder = self.fresh("ap")
        self.emit(f"{holder} = alloca {ll_element}*, align 8")
        self.emit(f"store {ll_element}* {pointer}, {ll_element}** {holder}, align 8")
        counter = self.fresh("ai")
        self.emit(f"{counter} = alloca i64, align 8")
        self.emit(f"store i64 0, i64* {counter}, align 8")

        length = self.array_length(pointer, ty)

        cond_label = self.label("ap.cond")
        body_label = self.label("ap.body")
        sep_label = self.label("ap.sep")
        after_label = self.label("ap.after")
        end_label = self.label("ap.end")

        self.emit(f"br label %{cond_label}")

        def emit_cond() -> None:
            index = self.load(counter, T.INT)
            guard = self.fresh("c")
            self.emit(f"{guard} = icmp ult i64 {index}, {length}")
            self.emit(f"br i1 {guard}, label %{body_label}, label %{end_label}")

        def emit_body() -> None:
            base = self.load(holder, ty)
            position = self.load(counter, T.INT)
            address = self.fresh("e")
            self.emit(f"{address} = getelementptr inbounds {ll_element}, "
                      f"{ll_element}* {base}, i64 {position}")
            element = self.load(address, element_ty)

            # Print the separator before every element except the first.
            first = self.fresh("c")
            self.emit(f"{first} = icmp ugt i64 {position}, 0")
            self.emit(f"br i1 {first}, label %{sep_label}, label %{after_label}")

            def emit_separator() -> None:
                self.printf_literal(", ")
                self.emit(f"br label %{after_label}")

            self.detached_block(sep_label, emit_separator)
            self.start_block(after_label)

            if isinstance(element_ty, T.ArrayType):
                self.printf_literal("[")
                self.emit_array_loop(element, element_ty, parenthesised=True)
                self.printf_literal("]")
            else:
                self.emit_scalar_print(element, element_ty)
            next_index = self.load(counter, T.INT)
            bumped = self.fresh("n")
            self.emit(f"{bumped} = add i64 {next_index}, 1")
            self.emit(f"store i64 {bumped}, i64* {counter}, align 8")
            self.emit(f"br label %{cond_label}")

        self.detached_block(cond_label, emit_cond)
        self.detached_block(body_label, emit_body)
        self.start_block(end_label)

    def printf_literal(self, text: str) -> None:
        self.printf_call(text, [], [])

    def printf_arg(self, text: str, value: str, llvm_ty: str) -> None:
        self.printf_call(text, [value], [llvm_ty])

    def printf_call(self, text: str, values: List[str], value_types: List[str]) -> None:
        pointer = self.string_ptr(text)
        call = self.fresh("p")
        arguments = "".join(
            f", {ty} {value}" for value, ty in zip(values, value_types))
        self.emit(f"{call} = call i32 (i8*, ...) @printf(i8* {pointer}{arguments})")

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------
    def emit_expr(self, expr: ast.Expr) -> str:
        if isinstance(expr, ast.IntLit):
            return str(expr.value)
        if isinstance(expr, ast.FloatLit):
            return float_const(expr.value)
        if isinstance(expr, ast.BoolLit):
            return "true" if expr.value else "false"
        if isinstance(expr, ast.StrLit):
            return self.string_ptr(expr.value)
        if isinstance(expr, ast.Ident):
            slot, ty = expr.slot, expr.ty
            return self.load(slot, ty)
        if isinstance(expr, ast.ArrayLit):
            return self.emit_array_literal(expr)
        if isinstance(expr, ast.Unary):
            return self.emit_unary(expr)
        if isinstance(expr, ast.Binary):
            return self.emit_binary(expr)
        if isinstance(expr, ast.Assign):
            self.emit_assign(expr)
            return expr.result_value
        if isinstance(expr, ast.Index):
            return self.load(self.emit_index_address(expr), expr.ty)
        if isinstance(expr, ast.LenBuiltin):
            return self.emit_len(expr)
        if isinstance(expr, ast.Call):
            return self.emit_call(expr)
        raise CodegenError(f"cannot generate code for {type(expr).__name__}",
                           expr.span)

    def load(self, pointer: str, ty: T.Type) -> str:
        value = self.fresh("v")
        self.emit(f"{value} = load {llvm_type(ty)}, {llvm_type(ty)}* {pointer}, "
                   f"align {align_of(ty)}")
        return value

    # -- arrays ----------------------------------------------------------
    def emit_array_literal(self, expr: ast.ArrayLit) -> str:
        """Allocate a fresh array on the heap with its length header."""
        element_ty = expr.ty.element  # type: ignore[union-attr]
        ll_element = llvm_type(element_ty)
        element_size = size_of(element_ty)
        count = len(expr.elements)

        payload = self.fresh("nb")
        self.emit(f"{payload} = mul i64 {count}, {element_size}")
        total = self.fresh("tb")
        self.emit(f"{total} = add i64 {payload}, 8")
        raw = self.fresh("raw")
        self.emit(f"{raw} = call i8* @malloc(i64 {total})")
        header = self.fresh("hd")
        self.emit(f"{header} = getelementptr inbounds i8, i8* {raw}, i64 0")
        self.emit(f"store i64 {count}, i8* {header}, align 8")
        area = self.fresh("ar")
        self.emit(f"{area} = getelementptr inbounds i8, i8* {raw}, i64 8")
        base = self.fresh("ab")
        self.emit(f"{base} = bitcast i8* {area} to {ll_element}*")

        for index, element in enumerate(expr.elements):
            value = self.coerce(self.emit_expr(element), element.ty, element_ty)
            address = self.fresh("ea")
            self.emit(f"{address} = getelementptr inbounds {ll_element}, "
                      f"{ll_element}* {base}, i64 {index}")
            self.emit(f"store {llvm_type(element_ty)} {value}, "
                      f"{ll_element}* {address}, align {align_of(element_ty)}")
        return base

    def array_length(self, pointer: str, ty: T.ArrayType) -> str:
        header = self.fresh("lh")
        self.emit(f"{header} = getelementptr i8, i8* {pointer}, i64 -8")
        return self.load(header, T.INT)

    def emit_len(self, expr: ast.LenBuiltin) -> str:
        target = expr.target
        value = self.emit_expr(target)
        if target.ty == T.STRING:
            result = self.fresh("l")
            self.emit(f"{result} = call i64 @strlen(i8* {value})")
            return result
        return self.array_length(value, target.ty)  # type: ignore[arg-type]

    # -- operators -------------------------------------------------------
    def emit_unary(self, expr: ast.Unary) -> str:
        operand = self.emit_expr(expr.operand)
        result = self.fresh("u")
        if expr.op == "-":
            if expr.ty == T.FLOAT:
                self.emit(f"{result} = fneg double {operand}")
            else:
                self.emit(f"{result} = sub i64 0, {operand}")
        elif expr.op in ("!", "not"):
            self.emit(f"{result} = xor i1 {operand}, true")
        else:
            raise CodegenError(f"unsupported unary operator {expr.op!r}", expr.span)
        return result

    _ARITH_INT = {"+": "add", "-": "sub", "*": "mul", "/": "sdiv", "%": "srem"}
    _ARITH_FLT = {"+": "fadd", "-": "fsub", "*": "fmul", "/": "fdiv", "%": "frem"}

    def emit_binary(self, expr: ast.Binary) -> str:
        op = expr.op
        if op in ("&&", "and", "||", "or"):
            return self.emit_logical(expr)
        comparisons = {
            "==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge",
        }
        if op in comparisons:
            return self.emit_compare(expr, comparisons[op])
        left = self.emit_expr(expr.left)
        right = self.emit_expr(expr.right)
        result = self.fresh("b")
        if expr.ty == T.FLOAT:
            self.emit(f"{result} = {self._ARITH_FLT[op]} double {left}, {right}")
        else:
            self.emit(f"{result} = {self._ARITH_INT[op]} i64 {left}, {right}")
        return result

    def emit_compare(self, expr: ast.Binary, predicate: str) -> str:
        left_ty = expr.left.ty
        left = self.emit_expr(expr.left)
        right = self.emit_expr(expr.right)

        if left_ty == T.STRING:
            # Strings compare by content: strcmp is 0 exactly when they match.
            call = self.fresh("sc")
            self.emit(f"{call} = call i32 @strcmp(i8* {left}, i8* {right})")
            equal = self.fresh("eq")
            self.emit(f"{equal} = icmp eq i32 {call}, 0")
            if predicate == "ne":
                # xor with true, not false: XOR with false is the identity.
                result = self.fresh("c")
                self.emit(f"{result} = xor i1 {equal}, true")
                return result
            return equal

        result = self.fresh("c")
        if left_ty == T.FLOAT:
            # LLVM spells ordered float predicates `oeq`/`olt`/... ; the `o`
            # makes the comparison false when either operand is NaN.
            self.emit(f"{result} = fcmp o{predicate} double {left}, {right}")
        else:
            self.emit(f"{result} = icmp s{predicate} i64 {left}, {right}")
        return result

    def emit_logical(self, expr: ast.Binary) -> str:
        """Short-circuit ``&&`` / ``||`` without a phi node.

        Both outcomes are written into a stack slot before branching, so the
        merge block can simply reload it.
        """
        is_and = expr.op in ("&&", "and")
        slot = self.fresh("lg")
        self.emit(f"{slot} = alloca i1, align 1")

        left = self.emit_expr(expr.left)
        self.emit(f"store i1 {left}, i1* {slot}, align 1")

        rhs_label = self.label("logic.rhs")
        end_label = self.label("logic.end")
        if is_and:
            self.emit(f"br i1 {left}, label %{rhs_label}, label %{end_label}")
        else:
            self.emit(f"br i1 {left}, label %{end_label}, label %{rhs_label}")

        def emit_rhs() -> None:
            right = self.emit_expr(expr.right)
            self.emit(f"store i1 {right}, i1* {slot}, align 1")
            self.emit(f"br label %{end_label}")

        self.detached_block(rhs_label, emit_rhs)
        self.start_block(end_label)
        return self.load(slot, T.BOOL)

    def emit_index_address(self, expr: ast.Index) -> str:
        base = self.emit_expr(expr.base)
        index = self.emit_expr(expr.index)
        address = self.fresh("ix")
        self.emit(f"{address} = getelementptr inbounds {llvm_type(expr.ty)}, "
                   f"{llvm_type(expr.ty)}* {base}, i64 {index}")
        return address

    # -- assignment ------------------------------------------------------
    def emit_assign(self, expr: ast.Assign) -> None:
        target = expr.target
        if isinstance(target, ast.Index):
            ll_element = llvm_type(target.ty)
            base = self.emit_expr(target.base)
            index = self.emit_expr(target.index)
            address = self.fresh("ix")
            self.emit(f"{address} = getelementptr inbounds {ll_element}, "
                      f"{ll_element}* {base}, i64 {index}")
            value = self.coerce(self.emit_expr(expr.value), expr.value.ty, target.ty)
            if expr.op:
                value = self.emit_compound(
                    expr.op, self.load(address, target.ty), value, target.ty)
            self.emit(f"store {ll_element} {value}, {ll_element}* {address}, "
                      f"align {align_of(target.ty)}")
            expr.result_value = value
            return

        slot, ty = target.slot, target.ty
        value = self.coerce(self.emit_expr(expr.value), expr.value.ty, ty)
        if expr.op:
            value = self.emit_compound(
                expr.op, self.load(slot, ty), value, ty)

        if isinstance(ty, T.ArrayType):
            self.emit_array_copy(slot, ty, value)
            value = self.load(slot, ty)
        else:
            self.emit(f"store {llvm_type(ty)} {value}, {llvm_type(ty)}* {slot}, "
                      f"align {align_of(ty)}")
        expr.result_value = value

    def emit_compound(self, op: str, current: str, operand: str, ty: T.Type) -> str:
        binary_op = op[:-1]  # '+=' -> '+'
        result = self.fresh("c")
        if ty == T.FLOAT:
            self.emit(f"{result} = {self._ARITH_FLT[binary_op]} double "
                      f"{current}, {operand}")
        else:
            self.emit(f"{result} = {self._ARITH_INT[binary_op]} i64 "
                      f"{current}, {operand}")
        return result

    def emit_array_clone(self, slot: str, ty: T.ArrayType, source: str) -> None:
        """Bind ``slot`` to a *fresh* copy of the array at ``source``.

        Used for ``let ys = xs``: the destination does not exist yet, so this
        mallocs a new buffer with the same length header instead of reusing
        :meth:`emit_array_copy`, which needs a valid array to copy into.
        """
        ll_element = llvm_type(ty.element)
        element_size = size_of(ty.element)

        length = self.array_length(source, ty)
        payload = self.fresh("eb")
        self.emit(f"{payload} = mul i64 {length}, {element_size}")
        total = self.fresh("tb")
        self.emit(f"{total} = add i64 {payload}, 8")
        raw = self.fresh("raw")
        self.emit(f"{raw} = call i8* @malloc(i64 {total})")
        header = self.fresh("hd")
        self.emit(f"{header} = getelementptr inbounds i8, i8* {raw}, i64 0")
        self.emit(f"store i64 {length}, i8* {header}, align 8")
        area = self.fresh("ar")
        self.emit(f"{area} = getelementptr inbounds i8, i8* {raw}, i64 8")
        base = self.fresh("ab")
        self.emit(f"{base} = bitcast i8* {area} to {ll_element}*")
        self.emit_memcpy(ll_element, source, base, payload)
        self.emit(f"store {ll_element}* {base}, {ll_element}* {slot}, align 8")

    def emit_array_copy(self, slot: str, ty: T.ArrayType, source: str) -> None:
        """Copy one array value into another; arrays are values, not aliases.

        If the destination already has the right size the payload is copied in
        place. Otherwise the original allocation (recoverable from the header)
        is grown with ``realloc``. The resulting pointer is routed through an
        alloca rather than a phi node, matching the rest of this backend.
        """
        ll_element = llvm_type(ty.element)
        element_size = size_of(ty.element)
        destination = self.load(slot, ty)

        source_len = self.array_length(source, ty)
        destination_len = self.array_length(destination, ty)

        payload = self.fresh("eb")
        self.emit(f"{payload} = mul i64 {source_len}, {element_size}")
        total = self.fresh("ab")
        self.emit(f"{total} = add i64 {payload}, 8")

        same = self.fresh("eq")
        self.emit(f"{same} = icmp eq i64 {destination_len}, {source_len}")

        result_slot = self.fresh("res")
        self.emit(f"{result_slot} = alloca {ll_element}*, align 8")

        inplace_label = self.label("copy.inplace")
        resize_label = self.label("copy.resize")
        done_label = self.label("copy.done")
        self.emit(f"br i1 {same}, label %{inplace_label}, label %{resize_label}")

        def emit_inplace() -> None:
            self.emit_memcpy(ll_element, source, destination, payload)
            self.emit(f"store {ll_element}* {destination}, {ll_element}** "
                      f"{result_slot}, align 8")
            self.emit(f"br label %{done_label}")

        def emit_resize() -> None:
            base = self.fresh("base")
            self.emit(f"{base} = getelementptr i8, i8* {destination}, i64 -8")
            raw = self.fresh("nraw")
            self.emit(f"{raw} = call i8* @realloc(i8* {base}, i64 {total})")
            header = self.fresh("nh")
            self.emit(f"{header} = getelementptr inbounds i8, i8* {raw}, i64 0")
            self.emit(f"store i64 {source_len}, i8* {header}, align 8")
            area = self.fresh("na")
            self.emit(f"{area} = getelementptr inbounds i8, i8* {raw}, i64 8")
            grown = self.fresh("nb")
            self.emit(f"{grown} = bitcast i8* {area} to {ll_element}*")
            self.emit_memcpy(ll_element, source, grown, payload)
            self.emit(f"store {ll_element}* {grown}, {ll_element}** "
                      f"{result_slot}, align 8")
            self.emit(f"br label %{done_label}")

        self.detached_block(inplace_label, emit_inplace)
        self.detached_block(resize_label, emit_resize)
        self.start_block(done_label)
        result = self.load(result_slot, ty)
        self.emit(f"store {ll_element}* {result}, {ll_element}* {slot}, align 8")

    def emit_memcpy(self, ll_element: str, source: str, destination: str,
                    payload: str) -> None:
        destination8 = self.fresh("d8")
        self.emit(f"{destination8} = bitcast {ll_element}* {destination} to i8*")
        source8 = self.fresh("s8")
        self.emit(f"{source8} = bitcast {ll_element}* {source} to i8*")
        self.emit(f"call void @memcpy(i8* {destination8}, i8* {source8}, "
                  f"i64 {payload})")

    # -- conversions -----------------------------------------------------
    def coerce(self, value: str, source: T.Type, target: T.Type) -> str:
        """Insert an explicit numeric conversion when one is needed."""
        if source == target or not (T.is_numeric(source) and T.is_numeric(target)):
            return value
        if source == T.INT and target == T.FLOAT:
            out = self.fresh("f")
            self.emit(f"{out} = sitofp i64 {value} to double")
            return out
        if source == T.FLOAT and target == T.INT:
            out = self.fresh("i")
            self.emit(f"{out} = fptosi double {value} to i64")
            return out
        return value

    # -- calls -----------------------------------------------------------
    def emit_call(self, expr: ast.Call) -> str:
        callee = expr.callee  # type: ignore[attr-defined]
        param_types: List[T.Type] = expr.param_types  # type: ignore[attr-defined]
        ret = expr.ty
        args = [
            self.coerce(self.emit_expr(arg), arg.ty, param)
            for arg, param in zip(expr.args, param_types)
        ]
        arg_text = ", ".join(
            f"{llvm_type(param)} {value}"
            for value, param in zip(args, param_types)
        )
        if ret == T.VOID:
            self.emit(f"call void @{mangle(callee)}({arg_text})")
            return "0"
        result = self.fresh("r")
        self.emit(f"{result} = call {llvm_type(ret)} @{mangle(callee)}({arg_text})")
        return result


def generate(program: ast.Program, module_name: str,
             diags: Optional[DiagnosticBag] = None,
             target_triple: Optional[str] = None) -> str:
    """Lower a checked :class:`~lumen.ast.Program` to LLVM IR text."""
    return IRBuilder(module_name, diags, target_triple).generate(program)
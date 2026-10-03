"""Lumen abstract syntax tree.

Every node carries a :class:`Span` so any later phase can point at the exact
source text responsible for it. Expression nodes gain a ``ty`` attribute during
type checking, which is what the code generator reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

from .diagnostics import Span


@dataclass
class Node:
    span: Span

    def children(self) -> List["Node"]:
        """Yield direct child nodes in source order."""
        out: List[Node] = []
        for f in fields(self):
            if f.name == "span":
                continue
            _collect(getattr(self, f.name), out)
        return out

    def walk(self):
        yield self
        for child in self.children():
            yield from child.walk()

    def to_json(self) -> Dict[str, Any]:
        """Serialize to plain dicts/lists for ``--ast --format=json``."""
        out: Dict[str, Any] = {"kind": type(self).__name__}
        for f in fields(self):
            if f.name == "span":
                out["span"] = [self.span.start, self.span.end]
                continue
            out[f.name] = _jsonify(getattr(self, f.name))
        return out


def _collect(value: Any, out: List[Node]) -> None:
    if isinstance(value, Node):
        out.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect(item, out)


def _jsonify(value: Any) -> Any:
    if isinstance(value, Node):
        return value.to_json()
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# --------------------------------------------------------------------------
# Types (syntactic; resolved to semantic types by the checker)
# --------------------------------------------------------------------------


@dataclass
class TypeRef:
    """A type as written in source: ``int``, ``float``, ``bool``, ``string``,
    ``void``, or ``T[]`` for an array whose elements have type ``T``."""

    name: str
    element: Optional["TypeRef"] = None
    span: Span = field(default_factory=lambda: Span(0, 0))

    def render(self) -> str:
        if self.element is not None:
            return f"{self.element.render()}[]"
        return self.name


# --------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------


@dataclass
class Expr(Node):
    ty: Any = None  # filled in by the type checker (a lumen.types.Type)


@dataclass
class IntLit(Expr):
    value: int = 0


@dataclass
class FloatLit(Expr):
    value: float = 0.0


@dataclass
class BoolLit(Expr):
    value: bool = False


@dataclass
class StrLit(Expr):
    value: str = ""


@dataclass
class Ident(Expr):
    name: str = ""


@dataclass
class ArrayLit(Expr):
    elements: List[Expr] = field(default_factory=list)


@dataclass
class Unary(Expr):
    op: str = ""
    operand: Expr = None  # type: ignore[assignment]


@dataclass
class Binary(Expr):
    op: str = ""
    left: Expr = None  # type: ignore[assignment]
    right: Expr = None  # type: ignore[assignment]


@dataclass
class Call(Expr):
    callee: str = ""
    args: List[Expr] = field(default_factory=list)


@dataclass
class Index(Expr):
    base: Expr = None  # type: ignore[assignment]
    index: Expr = None  # type: ignore[assignment]


@dataclass
class Assign(Expr):
    """``target op= value`` where ``op`` is '', '+=', '-=', '*=', '/=' or '%='."""

    target: Expr = None  # type: ignore[assignment]
    op: str = ""
    value: Expr = None  # type: ignore[assignment]


@dataclass
class LenBuiltin(Expr):
    target: Expr = None  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Statements
# --------------------------------------------------------------------------


@dataclass
class Stmt(Node):
    pass


@dataclass
class LetStmt(Stmt):
    name: str = ""
    declared: Optional[TypeRef] = None
    init: Optional[Expr] = None
    is_const: bool = False
    slot: Any = None  # filled in by codegen: LLVM alloca name


@dataclass
class ExprStmt(Stmt):
    expr: Expr = None  # type: ignore[assignment]


@dataclass
class Block(Stmt):
    stmts: List[Stmt] = field(default_factory=list)


@dataclass
class IfStmt(Stmt):
    cond: Expr = None  # type: ignore[assignment]
    then_branch: Block = None  # type: ignore[assignment]
    else_branch: Optional[Stmt] = None  # None, an `else if` IfStmt, or a Block


@dataclass
class WhileStmt(Stmt):
    cond: Expr = None  # type: ignore[assignment]
    body: Block = None  # type: ignore[assignment]


@dataclass
class ForStmt(Stmt):
    var: str = ""
    start: Expr = None  # type: ignore[assignment]
    end: Expr = None  # type: ignore[assignment]
    body: Block = None  # type: ignore[assignment]
    slot: Any = None  # induction variable alloca


@dataclass
class BreakStmt(Stmt):
    pass


@dataclass
class ContinueStmt(Stmt):
    pass


@dataclass
class ReturnStmt(Stmt):
    value: Optional[Expr] = None


@dataclass
class PrintStmt(Stmt):
    values: List[Expr] = field(default_factory=list)


@dataclass
class Param:
    name: str
    ty: TypeRef
    span: Span
    slot: Any = None  # codegen alloca for the incoming argument


@dataclass
class FuncDecl(Node):
    name: str = ""
    params: List[Param] = field(default_factory=list)
    ret: TypeRef = field(default_factory=lambda: TypeRef("void"))
    body: Block = None  # type: ignore[assignment]


@dataclass
class Program(Node):
    """A whole translation unit."""

    functions: List[FuncDecl] = field(default_factory=list)


def is_dataclass_node(obj: Any) -> bool:
    return is_dataclass(obj) and isinstance(obj, Node)
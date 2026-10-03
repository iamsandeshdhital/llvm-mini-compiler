"""The Lumen type system.

Five base types (``int``, ``float``, ``bool``, ``string``, ``void``) plus
one compound type, ``T[]``. Arrays are *reference* types: the value of an
array-typed expression is a pointer to its first element, which is exactly how
they are modelled in LLVM.

Numeric subtyping is a small lattice: ``int <: float``, and ``bool``/``string``
are unrelated to numbers. Nothing converts implicitly between unrelated types,
so ``if x { }`` with an integer ``x`` is a hard error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .ast import TypeRef
from .diagnostics import Span


class Type:
    name = "?"

    def render(self) -> str:
        return self.name

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.render()

    __repr__ = __str__


class VoidType(Type):
    name = "void"


class BoolType(Type):
    name = "bool"


class IntType(Type):
    name = "int"


class FloatType(Type):
    name = "float"


class StringType(Type):
    name = "string"


@dataclass(frozen=True)
class ArrayType(Type):
    element: Type

    name = "array"

    def render(self) -> str:
        return f"{self.element.render()}[]"


@dataclass(frozen=True)
class FunctionType(Type):
    params: tuple
    ret: Type
    name = "function"

    def render(self) -> str:
        args = ", ".join(p.render() for p in self.params)
        return f"fn({args}) -> {self.ret.render()}"


VOID = VoidType()
BOOL = BoolType()
INT = IntType()
FLOAT = FloatType()
STRING = StringType()

NUMERIC = (INT, FLOAT)
_BY_NAME = {
    "int": INT,
    "float": FLOAT,
    "bool": BOOL,
    "string": STRING,
    "void": VOID,
}


class TypeError_(Exception):
    """Internal signal; user-facing messages are produced by the checker."""

    def __init__(self, message: str, span: Span):
        super().__init__(message)
        self.message = message
        self.span = span


def base_type(name: str, span: Span) -> Type:
    if name not in _BY_NAME:
        raise TypeError_(f"unknown type '{name}'", span)
    return _BY_NAME[name]


def from_ast(ref: TypeRef, span: Span) -> Type:
    """Resolve a syntactic :class:`TypeRef` into a semantic :class:`Type`."""
    if ref.element is not None:
        return ArrayType(from_ast(ref.element, span))
    return base_type(ref.name, span)


def is_assignable(target: Type, source: Type) -> bool:
    """Can a value of type ``source`` be stored into ``target``?"""
    if isinstance(target, ArrayType):
        return isinstance(source, ArrayType) and source.element == target.element
    if target == source:
        return True
    # int widens to float; nothing else converts implicitly.
    if target == FLOAT and source == INT:
        return True
    return False


def unify(left: Type, right: Type) -> Optional[Type]:
    """Least common supertype of two operand types, or ``None`` if unrelated."""
    if is_assignable(left, right):
        return left
    if is_assignable(right, left):
        return right
    return None


def is_numeric(t: Type) -> bool:
    return isinstance(t, (IntType, FloatType))


def is_printable(t: Type) -> bool:
    from .ast import Node  # noqa: F401  (documentation of intent)

    return isinstance(t, (IntType, FloatType, BoolType, StringType, ArrayType))


def element_of(t: Type) -> Optional[Type]:
    return t.element if isinstance(t, ArrayType) else None
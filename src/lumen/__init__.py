"""Lumen: a small statically typed language that compiles to LLVM IR.

Public entry points:

``lumenc`` / ``python -m lumen``   command line driver
``compile_source``                 programmatic one-shot compilation
``Lexer`` / ``Parser`` / ``Checker`` / ``IRBuilder``   individual phases
"""

from .diagnostics import CompileError, DiagnosticBag, Severity, Span
from .driver import VERSION, compile_source, main
from .lexer import Lexer, tokenize
from .parser import Parser, parse, parse_source

__all__ = [
    "VERSION",
    "CompileError",
    "DiagnosticBag",
    "Severity",
    "Span",
    "Lexer",
    "Parser",
    "compile_source",
    "main",
    "parse",
    "parse_source",
    "tokenize",
]

__version__ = VERSION
"""Token definitions for the Lumen language."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict

from .diagnostics import Span


class Tok(Enum):
    """Token kinds. ``IDENT``/literals carry a value; the rest are fixed."""

    EOF = auto()

    IDENT = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()

    # Keywords
    KW_LET = auto()
    KW_CONST = auto()
    KW_IF = auto()
    KW_ELSE = auto()
    KW_WHILE = auto()
    KW_FOR = auto()
    KW_IN = auto()
    KW_BREAK = auto()
    KW_CONTINUE = auto()
    KW_RETURN = auto()
    KW_FUNC = auto()
    KW_PRINT = auto()
    KW_TRUE = auto()
    KW_FALSE = auto()
    KW_AND = auto()
    KW_OR = auto()
    KW_NOT = auto()
    KW_LEN = auto()
    # Type keywords
    KW_INT = auto()
    KW_FLOAT = auto()
    KW_BOOL = auto()
    KW_STRING = auto()
    KW_VOID = auto()

    # Operators / punctuation
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    ASSIGN = auto()
    PLUS_EQ = auto()
    MINUS_EQ = auto()
    STAR_EQ = auto()
    SLASH_EQ = auto()
    PERCENT_EQ = auto()
    EQ_EQ = auto()
    BANG_EQ = auto()
    LT = auto()
    LTE = auto()
    GT = auto()
    GTE = auto()
    BANG = auto()
    AMP_AMP = auto()
    PIPE_PIPE = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    COLON = auto()
    SEMI = auto()
    DOT = auto()
    ARROW = auto()


KEYWORDS: Dict[str, Tok] = {
    "let": Tok.KW_LET,
    "const": Tok.KW_CONST,
    "if": Tok.KW_IF,
    "else": Tok.KW_ELSE,
    "while": Tok.KW_WHILE,
    "for": Tok.KW_FOR,
    "in": Tok.KW_IN,
    "break": Tok.KW_BREAK,
    "continue": Tok.KW_CONTINUE,
    "return": Tok.KW_RETURN,
    "func": Tok.KW_FUNC,
    "fn": Tok.KW_FUNC,
    "print": Tok.KW_PRINT,
    "true": Tok.KW_TRUE,
    "false": Tok.KW_FALSE,
    "and": Tok.KW_AND,
    "or": Tok.KW_OR,
    "not": Tok.KW_NOT,
    "len": Tok.KW_LEN,
    "int": Tok.KW_INT,
    "float": Tok.KW_FLOAT,
    "bool": Tok.KW_BOOL,
    "string": Tok.KW_STRING,
    "void": Tok.KW_VOID,
}

#: Human-readable names, used by ``--tokens`` and by parse-error messages.
TOKEN_NAMES: Dict[Tok, str] = {t: t.name for t in Tok}
TOKEN_NAMES.update(
    {
        Tok.EOF: "end of file",
        Tok.IDENT: "identifier",
        Tok.INT: "integer literal",
        Tok.FLOAT: "float literal",
        Tok.STRING: "string literal",
        Tok.ASSIGN: "=",
        Tok.EQ_EQ: "==",
        Tok.BANG_EQ: "!=",
        Tok.LTE: "<=",
        Tok.GTE: ">=",
        Tok.AMP_AMP: "&&",
        Tok.PIPE_PIPE: "||",
        Tok.ARROW: "->",
        Tok.PLUS_EQ: "+=",
        Tok.MINUS_EQ: "-=",
        Tok.STAR_EQ: "*=",
        Tok.SLASH_EQ: "/=",
        Tok.PERCENT_EQ: "%=",
    }
)


# Keywords read better in diagnostics as the word the user typed
# ("expected 'if'" rather than "expected KW_IF").
for _word, _kind in KEYWORDS.items():
    TOKEN_NAMES[_kind] = _word
del _word, _kind


def token_name(tok: Tok) -> str:
    return TOKEN_NAMES.get(tok, tok.name)


@dataclass(frozen=True)
class Token:
    kind: Tok
    span: Span
    value: Any = None  # str for IDENT/STRING, int for INT, float for FLOAT

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.value is None:
            return f"Token({token_name(self.kind)})"
        return f"Token({token_name(self.kind)}, {self.value!r})"
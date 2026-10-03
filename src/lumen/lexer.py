"""Hand-written lexer (scanner) for Lumen.

Scanning rules:

* Whitespace is insignificant except that a newline token is emitted so that
  optional-semicolon statement termination can be reported precisely.
* Comments: ``// line``, ``# line`` and ``/* block */`` (non-nesting).
* Numbers: decimal, ``0x`` hex, ``0b`` binary, ``0o`` octal, underscores as
  digit separators, and floats via a ``.`` fraction and/or exponent.
* Strings: double-quoted with ``\\n \\t \\r \\0 \\" \\\\ \\xNN`` escapes.

The lexer raises :class:`LumenSyntaxError` for malformed input rather than
returning error tokens, which keeps the parser free of error recovery.
"""

from __future__ import annotations

from typing import List, Optional

from .diagnostics import CompileError, Span
from .tokens import KEYWORDS, Tok, Token

_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "0": "\0",
    "\\": "\\",
    '"': '"',
    "'": "'",
    "e": "\x1b",
}

# Longest-first so that '<=' wins over '<' and '&&' over nothing else.
_OPERATORS = [
    ("==", Tok.EQ_EQ),
    ("!=", Tok.BANG_EQ),
    ("<=", Tok.LTE),
    (">=", Tok.GTE),
    ("&&", Tok.AMP_AMP),
    ("||", Tok.PIPE_PIPE),
    ("+=", Tok.PLUS_EQ),
    ("-=", Tok.MINUS_EQ),
    ("*=", Tok.STAR_EQ),
    ("/=", Tok.SLASH_EQ),
    ("%=", Tok.PERCENT_EQ),
    ("->", Tok.ARROW),
    ("+", Tok.PLUS),
    ("-", Tok.MINUS),
    ("*", Tok.STAR),
    ("/", Tok.SLASH),
    ("%", Tok.PERCENT),
    ("=", Tok.ASSIGN),
    ("<", Tok.LT),
    (">", Tok.GT),
    ("!", Tok.BANG),
    ("(", Tok.LPAREN),
    (")", Tok.RPAREN),
    ("{", Tok.LBRACE),
    ("}", Tok.RBRACE),
    ("[", Tok.LBRACKET),
    ("]", Tok.RBRACKET),
    (",", Tok.COMMA),
    (":", Tok.COLON),
    (";", Tok.SEMI),
    (".", Tok.DOT),
]


class Lexer:
    def __init__(self, text: str, filename: str = "<input>") -> None:
        self.text = text
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.col = 1

    # -- helpers ---------------------------------------------------------
    def _fail(self, message: str, start: Optional[int] = None) -> "CompileError":
        s = self.pos if start is None else start
        span = Span(s, max(s, min(self.pos + 1, len(self.text))))
        return CompileError(f"{self.filename}:{self.line}:{self.col}: {message}")

    def _advance(self, n: int = 1) -> str:
        chunk = self.text[self.pos: self.pos + n]
        for ch in chunk:
            if ch == "\n":
                self.line += 1
                self.col = 1
            else:
                self.col += 1
        self.pos += n
        return chunk

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        return self.text[idx] if idx < len(self.text) else ""

    def _match(self, ch: str) -> bool:
        if self._peek() == ch:
            self._advance()
            return True
        return False

    # -- main loop -------------------------------------------------------
    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while True:
            self._skip_trivia()
            if self.pos >= len(self.text):
                tokens.append(Token(Tok.EOF, Span(self.pos, self.pos)))
                return tokens
            start = self.pos
            ch = self._peek()

            if ch.isdigit():
                tokens.append(self._lex_number())
            elif ch == '"' or ch == "'":
                tokens.append(self._lex_string())
            elif ch.isalpha() or ch == "_":
                tokens.append(self._lex_identifier())
            else:
                tok = None
                for text, kind in _OPERATORS:
                    if self.text.startswith(text, self.pos):
                        self._advance(len(text))
                        tok = Token(kind, Span(start, self.pos))
                        break
                if tok is None:
                    raise self._fail(f"unexpected character {ch!r}")
                tokens.append(tok)

    def _skip_trivia(self) -> None:
        while self.pos < len(self.text):
            ch = self._peek()
            if ch in " \t\r\n\f\v":
                # Newlines are insignificant: statement terminators are
                # optional, so the parser works off keywords and braces.
                self._advance()
            elif ch == "\\" and self._peek(1) == "\n":  # explicit line continuation
                self._advance(2)
            elif ch == "/" and self._peek(1) == "/":
                while self.pos < len(self.text) and self._peek() != "\n":
                    self._advance()
            elif ch == "#":
                while self.pos < len(self.text) and self._peek() != "\n":
                    self._advance()
            elif ch == "/" and self._peek(1) == "*":
                start = self.pos
                self._advance(2)
                depth = 1
                while self.pos < len(self.text) and depth:
                    if self._peek() == "*" and self._peek(1) == "/":
                        self._advance(2)
                        depth -= 1
                    elif self._peek() == "/" and self._peek(1) == "*":
                        self._advance(2)
                        depth += 1  # tolerate nesting
                    else:
                        self._advance()
                if depth:
                    raise self._fail("unterminated block comment", start)
            else:
                return

    def _lex_identifier(self) -> Token:
        start = self.pos
        while self.pos < len(self.text) and (
            self._peek().isalnum() or self._peek() == "_"
        ):
            self._advance()
        word = self.text[start:self.pos]
        kind = KEYWORDS.get(word, Tok.IDENT)
        return Token(kind, Span(start, self.pos), word)

    def _lex_number(self) -> Token:
        start = self.pos
        if self._peek() == "0" and self._peek(1) in "xX":
            self._advance(2)
            digits = self._consume_digits("0123456789abcdefABCDEF")
            if not digits:
                raise self._fail("malformed hexadecimal literal", start)
            return Token(Tok.INT, Span(start, self.pos), int(digits, 16))

        if self._peek() == "0" and self._peek(1) in "bB":
            self._advance(2)
            digits = self._consume_digits("01")
            if not digits:
                raise self._fail("malformed binary literal", start)
            return Token(Tok.INT, Span(start, self.pos), int(digits, 2))

        if self._peek() == "0" and self._peek(1) in "oO":
            self._advance(2)
            digits = self._consume_digits("01234567")
            if not digits:
                raise self._fail("malformed octal literal", start)
            return Token(Tok.INT, Span(start, self.pos), int(digits, 8))

        int_part = self._consume_digits("0123456789")
        is_float = False
        frac_part = ""
        # A '.' is a fraction only when followed by a digit, so `1.max` and
        # member access on an integer literal still parse.
        if self._peek() == "." and self._peek(1).isdigit():
            is_float = True
            self._advance()
            frac_part = self._consume_digits("0123456789")

        exp_part = ""
        if self._peek() in "eE":
            save = self.pos
            self._advance()
            if self._peek() in "+-":
                self._advance()
            digits = self._consume_digits("0123456789")
            if digits:
                is_float = True
                exp_part = "e" + self.text[save + 1:self.pos]
            else:
                self.pos = save  # not an exponent after all

        if is_float:
            raw = f"{int_part.replace('_', '')}.{frac_part.replace('_', '')}{exp_part}"
            return Token(Tok.FLOAT, Span(start, self.pos), float(raw))
        return Token(Tok.INT, Span(start, self.pos), int(int_part.replace("_", "")))

    def _consume_digits(self, allowed: str) -> str:
        start = self.pos
        while self.pos < len(self.text) and (self._peek() in allowed or self._peek() == "_"):
            self._advance()
        return self.text[start:self.pos]

    def _lex_string(self) -> Token:
        start = self.pos
        quote = self._advance()
        out = []
        while True:
            if self.pos >= len(self.text) or self._peek() == "\n":
                raise self._fail("unterminated string literal", start)
            ch = self._advance()
            if ch == quote:
                return Token(Tok.STRING, Span(start, self.pos), "".join(out))
            if ch != "\\":
                out.append(ch)
                continue
            esc = self._advance()
            if esc in _ESCAPES:
                out.append(_ESCAPES[esc])
            elif esc == "x":
                hex_digits = self._advance(2)
                try:
                    out.append(chr(int(hex_digits, 16)))
                except ValueError:
                    raise self._fail("malformed \\x escape", start) from None
            elif esc == "u":
                hex_digits = ""
                if self._match("{"):
                    while self.pos < len(self.text) and self._peek() != "}":
                        hex_digits += self._advance()
                    self._match("}")
                else:
                    hex_digits = self._advance(4)
                try:
                    out.append(chr(int(hex_digits, 16)))
                except ValueError:
                    raise self._fail("malformed \\u escape", start) from None
            else:
                raise self._fail(f"unknown escape sequence '\\{esc}'", start)


def tokenize(text: str, filename: str = "<input>") -> List[Token]:
    return Lexer(text, filename).tokenize()
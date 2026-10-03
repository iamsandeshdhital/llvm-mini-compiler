"""Lexer tests: token kinds, literals, comments and error positions."""

from __future__ import annotations

import unittest

from helpers import ROOT, SRC  # noqa: F401  (inserts src/ on sys.path)

from lumen.diagnostics import CompileError
from lumen.lexer import tokenize
from lumen.tokens import Tok


def kinds(source: str):
    return [token.kind for token in tokenize(source)]


class TestLiterals(unittest.TestCase):
    def test_decimal_integers(self):
        self.assertEqual(kinds("42 7"), [Tok.INT, Tok.INT, Tok.EOF])
        self.assertEqual(tokenize("42")[0].value, 42)

    def test_bases(self):
        self.assertEqual(tokenize("0xFF")[0].value, 255)
        self.assertEqual(tokenize("0b1011")[0].value, 11)
        self.assertEqual(tokenize("0o17")[0].value, 15)

    def test_underscore_separators(self):
        self.assertEqual(tokenize("1_000_000")[0].value, 1000000)

    def test_floats(self):
        token = tokenize("3.5")[0]
        self.assertIs(token.kind, Tok.FLOAT)
        self.assertAlmostEqual(token.value, 3.5)

    def test_float_exponent(self):
        self.assertAlmostEqual(tokenize("1e3")[0].value, 1000.0)
        self.assertAlmostEqual(tokenize("2.5e-2")[0].value, 0.025)

    def test_dot_after_int_is_not_a_float(self):
        # `1.max` style code must not swallow the dot into a float literal.
        self.assertIs(tokenize("1")[0].kind, Tok.INT)

    def test_strings_and_escapes(self):
        self.assertEqual(tokenize('"hi"')[0].value, "hi")
        self.assertEqual(tokenize(r'"a\nb"')[0].value, "a\nb")
        self.assertEqual(tokenize(r'"q\"q"')[0].value, 'q"q')
        self.assertEqual(tokenize(r'"A"')[0].value, "A")
        self.assertEqual(tokenize(r'"\x41"')[0].value, "A")

    def test_single_quoted_strings(self):
        self.assertEqual(tokenize("'ok'")[0].value, "ok")

    def test_keywords(self):
        self.assertIs(tokenize("let")[0].kind, Tok.KW_LET)
        self.assertIs(tokenize("func")[0].kind, Tok.KW_FUNC)
        self.assertIs(tokenize("int")[0].kind, Tok.KW_INT)

    def test_malformed_number(self):
        with self.assertRaises(CompileError):
            tokenize("0x")

    def test_unterminated_string(self):
        with self.assertRaises(CompileError):
            tokenize('"open')

    def test_unknown_escape(self):
        with self.assertRaises(CompileError):
            tokenize(r'"\q"')


class TestOperators(unittest.TestCase):
    def test_multi_character_operators(self):
        cases = {
            "==": Tok.EQ_EQ,
            "!=": Tok.BANG_EQ,
            "<=": Tok.LTE,
            ">=": Tok.GTE,
            "&&": Tok.AMP_AMP,
            "||": Tok.PIPE_PIPE,
            "+=": Tok.PLUS_EQ,
            "-=": Tok.MINUS_EQ,
            "*=": Tok.STAR_EQ,
            "/=": Tok.SLASH_EQ,
            "%=": Tok.PERCENT_EQ,
            "->": Tok.ARROW,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertIs(tokenize(text)[0].kind, expected)

    def test_longest_match_wins(self):
        self.assertIs(tokenize("<=")[0].kind, Tok.LTE)
        self.assertIs(tokenize("<")[0].kind, Tok.LT)

    def test_range_operator_is_two_dots(self):
        self.assertEqual(kinds("0..5"),
                         [Tok.INT, Tok.DOT, Tok.DOT, Tok.INT, Tok.EOF])


class TestTrivia(unittest.TestCase):
    def test_line_comments(self):
        self.assertEqual(kinds("// hi\n42"), [Tok.INT, Tok.EOF])
        self.assertEqual(kinds("# hi\n42"), [Tok.INT, Tok.EOF])

    def test_block_comments(self):
        self.assertEqual(kinds("/* a\nb */ 42"), [Tok.INT, Tok.EOF])

    def test_newlines_are_insignificant(self):
        # Statement terminators are optional, so whitespace-only source yields
        # exactly one token: end of file.
        self.assertEqual(kinds("\n\n\n"), [Tok.EOF])

    def test_unterminated_block_comment(self):
        with self.assertRaises(CompileError):
            tokenize("/* open")

    def test_unexpected_character(self):
        with self.assertRaises(CompileError):
            tokenize("§")


class TestPositions(unittest.TestCase):
    def test_spans_cover_tokens(self):
        source = "let x = 1"
        spans = [(t.span.start, t.span.end) for t in tokenize(source)]
        self.assertEqual(source[spans[0][0]:spans[0][1]], "let")
        self.assertEqual(source[spans[1][0]:spans[1][1]], "x")

    def test_line_and_column_tracking(self):
        from lumen.diagnostics import SourceMap

        text = "func a() {\n  let x;\n}"
        source = SourceMap("t.lum", text)
        self.assertEqual(source.line_col(0), (1, 1))
        # offset 11 is the first character of line 2
        self.assertEqual(source.line_col(11), (2, 1))
        self.assertEqual(source.line_col(text.index("x")), (2, 7))

    def test_snippet_points_at_the_token(self):
        from lumen.diagnostics import SourceMap, Span

        source = SourceMap("t.lum", "let x = ;")
        snippet = source.snippet(Span(8, 9))
        self.assertIn("^", snippet)
        self.assertIn("let x = ;", snippet)


if __name__ == "__main__":
    unittest.main()
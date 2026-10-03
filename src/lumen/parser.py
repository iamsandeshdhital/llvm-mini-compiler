"""Recursive-descent parser with precedence climbing for expressions.

Top level
    program    := funcDecl*

funcDecl    := 'func' IDENT '(' params? ')' ('->' type)? block
             | type IDENT '(' params? ')' block          (C-style spelling)

stmt         := letStmt | ifStmt | whileStmt | forStmt | returnStmt
              | breakStmt | continueStmt | printStmt | block | exprStmt

exprStmt     := expr semi?
semi         := ';' | <nothing>

Statement terminators are optional, which is why a grammar driven by
statement keywords is convenient here; ``semi()`` simply eats a ``;`` when one
is present. Assignment is the only right-associative operator and is parsed in
``parse_assignment`` after the binary precedence chain, so ``a = b = c`` works
while ``a + b = c`` is rejected.
"""

from __future__ import annotations

from typing import List, Optional

from .ast import (
    ArrayLit,
    Assign,
    Binary,
    Block,
    BoolLit,
    BreakStmt,
    Call,
    ContinueStmt,
    Expr,
    ExprStmt,
    FloatLit,
    ForStmt,
    FuncDecl,
    Ident,
    IfStmt,
    Index,
    IntLit,
    LenBuiltin,
    LetStmt,
    Param,
    PrintStmt,
    Program,
    ReturnStmt,
    StrLit,
    Stmt,
    TypeRef,
    Unary,
    WhileStmt,
)
from .diagnostics import CompileError, Span
from .lexer import tokenize
from .tokens import Tok, Token, token_name

# Binary operator precedence, loosest first.
_PRECEDENCE = {
    Tok.PIPE_PIPE: 1,
    Tok.KW_OR: 1,
    Tok.AMP_AMP: 2,
    Tok.KW_AND: 2,
    Tok.EQ_EQ: 3,
    Tok.BANG_EQ: 3,
    Tok.LT: 4,
    Tok.LTE: 4,
    Tok.GT: 4,
    Tok.GTE: 4,
    Tok.PLUS: 5,
    Tok.MINUS: 5,
    Tok.STAR: 6,
    Tok.SLASH: 6,
    Tok.PERCENT: 6,
}

_SPELLING = {
    Tok.PIPE_PIPE: "||",
    Tok.KW_OR: "or",
    Tok.AMP_AMP: "&&",
    Tok.KW_AND: "and",
    Tok.EQ_EQ: "==",
    Tok.BANG_EQ: "!=",
    Tok.LT: "<",
    Tok.LTE: "<=",
    Tok.GT: ">",
    Tok.GTE: ">=",
    Tok.PLUS: "+",
    Tok.MINUS: "-",
    Tok.STAR: "*",
    Tok.SLASH: "/",
    Tok.PERCENT: "%",
}

_ASSIGN_OPS = {
    Tok.ASSIGN: "",
    Tok.PLUS_EQ: "+=",
    Tok.MINUS_EQ: "-=",
    Tok.STAR_EQ: "*=",
    Tok.SLASH_EQ: "/=",
    Tok.PERCENT_EQ: "%=",
}

_TYPE_TOKENS = {
    Tok.KW_INT: "int",
    Tok.KW_FLOAT: "float",
    Tok.KW_BOOL: "bool",
    Tok.KW_STRING: "string",
    Tok.KW_VOID: "void",
}


class Parser:
    def __init__(self, tokens: List[Token], filename: str = "<input>") -> None:
        self.tokens = tokens
        self.filename = filename
        self.index = 0

    # -- token plumbing --------------------------------------------------
    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def peek(self, offset: int = 0) -> Token:
        idx = min(self.index + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def at(self, kind: Tok) -> bool:
        return self.current.kind is kind

    def at_type_start(self) -> bool:
        return self.current.kind in _TYPE_TOKENS

    def advance(self) -> Token:
        tok = self.tokens[self.index]
        if tok.kind is not Tok.EOF:
            self.index += 1
        return tok

    def accept(self, kind: Tok) -> Optional[Token]:
        if self.at(kind):
            return self.advance()
        return None

    def expect(self, kind: Tok, context: str = "") -> Token:
        if not self.at(kind):
            where = f" {context}" if context else ""
            raise self.error(f"expected '{token_name(kind)}'{where}, found {token_name(self.current.kind)}")
        return self.advance()

    def error(self, message: str) -> CompileError:
        tok = self.current
        return CompileError(f"{self.filename}: {message} (at token {token_name(tok.kind)})")

    def semi(self) -> None:
        """Consume an optional statement terminator."""
        self.accept(Tok.SEMI)

    # -- program ---------------------------------------------------------
    def parse_program(self) -> Program:
        start = self.current.span
        functions: List[FuncDecl] = []
        while not self.at(Tok.EOF):
            functions.append(self.parse_func_decl())
        return Program(span=start.merge(self.current.span), functions=functions)

    def parse_func_decl(self) -> FuncDecl:
        start = self.current.span
        if self.accept(Tok.KW_FUNC):
            name_tok = self.expect(Tok.IDENT, "after 'func'")
            params = self.parse_params()
            ret = TypeRef("void", span=name_tok.span)
            if self.accept(Tok.ARROW):
                ret = self.parse_type()
            body = self.parse_block()
            span = start.merge(body.span)
        else:
            # C-style: `int main(args) { ... }`
            ret = self.parse_type()
            name_tok = self.expect(Tok.IDENT, "after return type")
            params = self.parse_params()
            body = self.parse_block()
            span = start.merge(body.span)
        return FuncDecl(name=name_tok.value, params=params, ret=ret, body=body, span=span)

    def parse_params(self) -> List[Param]:
        self.expect(Tok.LPAREN, "to begin the parameter list")
        params: List[Param] = []
        if not self.at(Tok.RPAREN):
            while True:
                name_tok = self.expect(Tok.IDENT, "as a parameter name")
                self.expect(Tok.COLON, "after the parameter name")
                ty = self.parse_type()
                params.append(Param(name_tok.value, ty, name_tok.span))
                if not self.accept(Tok.COMMA):
                    break
        self.expect(Tok.RPAREN, "to close the parameter list")
        return params

    def parse_type(self) -> TypeRef:
        tok = self.current
        if tok.kind not in _TYPE_TOKENS:
            raise self.error(f"expected a type, found {token_name(tok.kind)}")
        self.advance()
        name = _TYPE_TOKENS[tok.kind]
        # `int[][]` binds left, so peel suffixes right-to-left.
        inner: Optional[TypeRef] = TypeRef(name, span=tok.span)
        suffix: List[Span] = []
        while self.accept(Tok.LBRACKET):
            self.expect(Tok.RBRACKET, "to close an array suffix")
            suffix.append(tok.span)
        for _ in suffix:
            inner = TypeRef("array", element=inner, span=inner.span)
        return inner

    # -- statements ------------------------------------------------------
    def parse_block(self) -> Block:
        open_tok = self.expect(Tok.LBRACE, "to begin a block")
        stmts: List[Stmt] = []
        while not self.at(Tok.RBRACE):
            if self.at(Tok.EOF):
                raise self.error("unterminated block: expected '}'")
            stmts.append(self.parse_stmt())
        close_tok = self.expect(Tok.RBRACE, "to close a block")
        return Block(stmts=stmts, span=open_tok.span.merge(close_tok.span))

    def parse_stmt(self) -> Stmt:
        tok = self.current
        if tok.kind is Tok.LBRACE:
            return self.parse_block()
        if tok.kind is Tok.KW_LET or tok.kind is Tok.KW_CONST:
            return self.parse_let()
        if tok.kind is Tok.KW_IF:
            return self.parse_if()
        if tok.kind is Tok.KW_WHILE:
            return self.parse_while()
        if tok.kind is Tok.KW_FOR:
            return self.parse_for()
        if tok.kind is Tok.KW_RETURN:
            return self.parse_return()
        if tok.kind is Tok.KW_BREAK:
            self.advance()
            self.semi()
            return BreakStmt(span=tok.span)
        if tok.kind is Tok.KW_CONTINUE:
            self.advance()
            self.semi()
            return ContinueStmt(span=tok.span)
        if tok.kind is Tok.KW_PRINT:
            return self.parse_print()
        if tok.kind in _TYPE_TOKENS:
            raise self.error(
                f"a type annotation may only appear on a function, not on "
                f"'{token_name(tok.kind)}' here"
            )
        expr = self.parse_expr()
        self.semi()
        return ExprStmt(expr=expr, span=expr.span)

    def parse_let(self) -> LetStmt:
        kw = self.advance()  # let / const
        name_tok = self.expect(Tok.IDENT, "after 'let'")
        declared: Optional[TypeRef] = None
        if self.accept(Tok.COLON):
            declared = self.parse_type()
        self.expect(Tok.ASSIGN, "after the variable name ('=' with an initializer)")
        init = self.parse_expr()
        self.semi()
        return LetStmt(
            name=name_tok.value,
            declared=declared,
            init=init,
            is_const=kw.kind is Tok.KW_CONST,
            span=kw.span.merge(init.span),
        )

    def parse_if(self) -> IfStmt:
        kw = self.advance()
        cond = self.parse_expr()
        then_branch = self.parse_block()
        else_branch: Optional[Stmt] = None
        span = kw.span.merge(then_branch.span)
        if self.accept(Tok.KW_ELSE):
            if self.at(Tok.KW_IF):
                else_branch = self.parse_if()
            else:
                else_branch = self.parse_block()
            span = span.merge(else_branch.span)
        return IfStmt(cond=cond, then_branch=then_branch, else_branch=else_branch, span=span)

    def parse_while(self) -> WhileStmt:
        kw = self.advance()
        cond = self.parse_expr()
        body = self.parse_block()
        return WhileStmt(cond=cond, body=body, span=kw.span.merge(body.span))

    def parse_for(self) -> ForStmt:
        kw = self.advance()
        name_tok = self.expect(Tok.IDENT, "after 'for'")
        self.expect(Tok.KW_IN, "after the loop variable ('in')")
        start = self.parse_expr()
        self.expect(Tok.DOT, "in a range ('..')")
        self.expect(Tok.DOT, "in a range ('..')")
        end = self.parse_expr()
        body = self.parse_block()
        return ForStmt(
            var=name_tok.value, start=start, end=end, body=body,
            span=kw.span.merge(body.span),
        )

    def parse_return(self) -> ReturnStmt:
        kw = self.advance()
        value: Optional[Expr] = None
        if not self.at(Tok.SEMI) and not self.at(Tok.RBRACE):
            value = self.parse_expr()
        self.semi()
        return ReturnStmt(value=value, span=kw.span.merge(value.span if value else kw.span))

    def parse_print(self) -> PrintStmt:
        kw = self.advance()
        values: List[Expr] = []
        if not self.at(Tok.SEMI) and not self.at(Tok.RBRACE):
            values.append(self.parse_expr())
            while self.accept(Tok.COMMA):
                values.append(self.parse_expr())
        self.semi()
        span = kw.span.merge(values[-1].span if values else kw.span)
        return PrintStmt(values=values, span=span)

    # -- expressions -----------------------------------------------------
    def parse_expr(self) -> Expr:
        return self.parse_assignment()

    def parse_assignment(self) -> Expr:
        lhs = self.parse_binary(0)
        tok = self.current
        if tok.kind in _ASSIGN_OPS:
            self.advance()
            rhs = self.parse_assignment()  # right associative
            return Assign(target=lhs, op=_ASSIGN_OPS[tok.kind], value=rhs, span=lhs.span.merge(rhs.span))
        return lhs

    def parse_binary(self, min_prec: int) -> Expr:
        lhs = self.parse_unary()
        while True:
            kind = self.current.kind
            prec = _PRECEDENCE.get(kind)
            if prec is None or prec < min_prec:
                return lhs
            self.advance()
            # All Lumen binary operators are left-associative.
            rhs = self.parse_binary(prec + 1)
            lhs = Binary(
                op=_SPELLING[kind], left=lhs, right=rhs,
                span=lhs.span.merge(rhs.span),
            )

    def parse_unary(self) -> Expr:
        tok = self.current
        if tok.kind in (Tok.MINUS, Tok.BANG, Tok.KW_NOT):
            self.advance()
            operand = self.parse_unary()
            op = "!" if tok.kind is Tok.BANG else ("not" if tok.kind is Tok.KW_NOT else "-")
            return Unary(op=op, operand=operand, span=tok.span.merge(operand.span))
        return self.parse_postfix()

    def parse_postfix(self) -> Expr:
        expr = self.parse_primary()
        while True:
            if self.at(Tok.LPAREN):
                if not isinstance(expr, Ident):
                    raise self.error("only named functions can be called")
                self.advance()  # '('
                args = self.parse_args()
                close = self.expect(Tok.RPAREN, "to close an argument list")
                expr = Call(callee=expr.name, args=args,
                            span=expr.span.merge(close.span))
            elif self.at(Tok.LBRACKET):
                self.advance()
                index = self.parse_expr()
                close = self.expect(Tok.RBRACKET, "to close an index")
                expr = Index(base=expr, index=index, span=expr.span.merge(close.span))
            else:
                return expr

    def parse_args(self) -> List[Expr]:
        args: List[Expr] = []
        if not self.at(Tok.RPAREN):
            while True:
                args.append(self.parse_expr())
                if not self.accept(Tok.COMMA):
                    break
        return args

    def parse_primary(self) -> Expr:
        tok = self.current
        kind = tok.kind

        if kind is Tok.INT:
            self.advance()
            return IntLit(value=tok.value, span=tok.span)
        if kind is Tok.FLOAT:
            self.advance()
            return FloatLit(value=tok.value, span=tok.span)
        if kind is Tok.STRING:
            self.advance()
            return StrLit(value=tok.value, span=tok.span)
        if kind in (Tok.KW_TRUE, Tok.KW_FALSE):
            self.advance()
            return BoolLit(value=kind is Tok.KW_TRUE, span=tok.span)
        if kind is Tok.IDENT:
            self.advance()
            # A following '(' is consumed by the postfix loop, which turns the
            # identifier into a Call node.
            return Ident(name=tok.value, span=tok.span)
        if kind is Tok.KW_LEN:
            self.advance()
            if not self.at(Tok.LPAREN):
                raise self.error("'len' must be applied to a value: len(x)")
            self.advance()
            inner = self.parse_expr()
            close = self.expect(Tok.RPAREN, "to close 'len('")
            return LenBuiltin(target=inner, span=tok.span.merge(close.span))
        if kind is Tok.LPAREN:
            self.advance()
            inner = self.parse_expr()
            self.expect(Tok.RPAREN, "to close '('")
            return inner
        if kind is Tok.LBRACKET:
            self.advance()
            elems: List[Expr] = []
            if not self.at(Tok.RBRACKET):
                while True:
                    elems.append(self.parse_expr())
                    if not self.accept(Tok.COMMA):
                        break
            self.expect(Tok.RBRACKET, "to close an array literal")
            return ArrayLit(elements=elems, span=tok.span.merge(self.tokens[self.index - 1].span))
        if kind is Tok.KW_FUNC:
            raise self.error("nested functions are not supported")
        if kind in _TYPE_TOKENS:
            raise self.error(f"'{_TYPE_TOKENS[kind]}' is a type, not a value")
        raise self.error(f"unexpected {token_name(kind)} in expression")


def parse(tokens: List[Token], filename: str = "<input>") -> Program:
    return Parser(tokens, filename).parse_program()


def parse_source(text: str, filename: str = "<input>") -> Program:
    return parse(tokenize(text, filename), filename)
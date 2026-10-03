"""Compilation driver: the phase pipeline and the command line interface.

Pipeline::

    source text
      -> lexer.Lexer            tokens
      -> parser.Parser          AST
      -> checker.Checker        typed AST      (errors are fatal here)
      -> optimize.Optimizer     typed AST      (optional, -O0..-O3)
      -> lower.Lowering         slot-annotated AST
      -> codegen.IRBuilder      LLVM IR text
      -> irverify.verify        checked IR
      -> clang/llc              native binary (only if LLVM is installed)

The last stage is optional by design: emitting correct IR is the backend's job,
turning IR into an executable is LLVM's. ``lumenc --build`` shells out to
whichever of ``clang`` or ``llc`` it can find, and reports clearly when neither
is installed rather than silently doing nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import ast as ast_mod
from . import codegen, irverify, lower, optimize
from .checker import Checker
from .diagnostics import CompileError, DiagnosticBag, Severity, SourceMap
from .lexer import Lexer
from .parser import Parser
from .tokens import token_name

VERSION = "0.1.0"

_TRIPLES = {
    ("win32", "AMD64"): "x86_64-pc-windows-msvc",
    ("win32", "x86_64"): "x86_64-pc-windows-msvc",
    ("win32", "ARM64"): "aarch64-pc-windows-msvc",
    ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("linux", "aarch64"): "aarch64-unknown-linux-gnu",
    ("darwin", "x86_64"): "x86_64-apple-darwin",
    ("darwin", "arm64"): "arm64-apple-darwin",
}


def host_triple() -> Optional[str]:
    import platform

    return _TRIPLES.get((sys.platform, platform.machine()))


def executable_suffix() -> str:
    return ".exe" if sys.platform == "win32" else ""


# --------------------------------------------------------------------------


@dataclass
class CompilationResult:
    source: SourceMap
    tokens: List = field(default_factory=list)
    program: Optional[ast_mod.Program] = None
    ir: Optional[str] = None
    diags: DiagnosticBag = field(default_factory=DiagnosticBag)
    timings: Dict[str, float] = field(default_factory=dict)
    optimization_stats: Dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.diags.has_errors()

    @property
    def warning_count(self) -> int:
        return self.diags.warning_count


def compile_source(text: str, filename: str = "<input>", opt_level: int = 1,
                   target: Optional[str] = None,
                   keep_tokens: bool = False) -> CompilationResult:
    """Run lexing, parsing, checking, lowering and IR generation over ``text``."""
    source = SourceMap(filename, text)
    diags = DiagnosticBag(source)
    result = CompilationResult(source=source, diags=diags)

    def phase(name, function):
        start = time.perf_counter()
        try:
            return function()
        finally:
            result.timings[name] = (time.perf_counter() - start) * 1000.0

    tokens = phase("lex", lambda: Lexer(text, filename).tokenize())
    if keep_tokens:
        result.tokens = tokens

    program = phase("parse", lambda: Parser(tokens, filename).parse_program())
    result.program = program

    phase("check", lambda: Checker(diags).check_program(program))
    if diags.has_errors():
        return result

    program, stats = phase("optimize", lambda: optimize.optimize(program, opt_level))
    result.program = program
    result.optimization_stats = stats

    phase("lower", lambda: lower.lower(program, diags))
    if diags.has_errors():
        return result

    result.ir = phase(
        "codegen",
        lambda: codegen.generate(program, os.path.basename(filename), diags,
                                 target_triple=target),
    )
    return result


# --------------------------------------------------------------------------
# Optional native toolchain
# --------------------------------------------------------------------------


def find_toolchain() -> Dict[str, Optional[str]]:
    return {
        "clang": shutil.which("clang") or shutil.which("clang-cl"),
        "llc": shutil.which("llc"),
        "opt": shutil.which("opt"),
    }


def _run(command: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(command), capture_output=True, text=True)


def build_native(ir_path: str, binary_path: str,
                 toolchain: Optional[Dict[str, Optional[str]]] = None,
                 opt_level: int = 1) -> str:
    """Assemble, optimize and link ``ir_path`` into ``binary_path``.

    Raises :class:`CompileError` with an actionable message when no LLVM tool is
    installed -- emitting a helpful error beats silently producing nothing.
    """
    tools = toolchain or find_toolchain()
    clang, llc = tools.get("clang"), tools.get("llc")

    if clang:
        flag = "-O2" if opt_level >= 2 else ("-O1" if opt_level == 1 else "-O0")
        completed = _run([clang, flag, ir_path, "-o", binary_path, "-lm"])
        if completed.returncode == 0:
            return clang
        detail = (completed.stderr or completed.stdout).strip()
        raise CompileError(f"clang failed to build {binary_path}:\n{detail}")

    if llc:
        obj = os.path.splitext(binary_path)[0] + ".o"
        completed = _run([llc, "-O2", "-filetype=obj", ir_path, "-o", obj])
        if completed.returncode == 0:
            linker = shutil.which("link") if sys.platform == "win32" else "cc"
            linked = _run([linker, obj, "-o", binary_path])
            if linked.returncode == 0:
                return llc
            detail = (linked.stderr or linked.stdout).strip()
            raise CompileError(f"linking {binary_path} failed:\n{detail}")
        detail = (completed.stderr or completed.stdout).strip()
        raise CompileError(f"llc failed to compile {ir_path}:\n{detail}")

    raise CompileError(
        "no LLVM toolchain on PATH, so the IR was emitted but not linked.\n"
        "Install LLVM (clang or llc) and re-run with --build, or finish the job "
        "yourself:\n"
        f"    llc -O2 -filetype=obj {ir_path} -o out.o\n"
        f"    cc out.o -o program{'' if sys.platform != 'win32' else '.exe'}")


# --------------------------------------------------------------------------
# Pretty printing for --tokens / --ast
# --------------------------------------------------------------------------


def format_tokens(tokens: Sequence) -> str:
    lines = []
    for token in tokens:
        value = "" if token.value is None else f"  {token.value!r}"
        lines.append(f"{token.span.start:>5}  {token_name(token.kind):<18}{value}")
    return "\n".join(lines)


def format_ast(node: ast_mod.Node, indent: int = 0) -> str:
    """Compact indented AST dump, one node per line."""
    pad = "  " * indent
    head = f"{pad}{type(node).__name__}"
    extras = []
    for name in ("name", "op", "callee", "value", "is_const"):
        if hasattr(node, name):
            extras.append(f"{name}={getattr(node, name)!r}")
    if extras:
        head += "  " + " ".join(extras)
    ty = getattr(node, "ty", None)
    render = getattr(ty, "render", None)
    if callable(render):
        head += f"  :{render()}"
    lines = [head]
    for child in node.children():
        lines.extend(format_ast(child, indent + 1).splitlines())
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lumenc",
        description="The Lumen compiler: source -> LLVM IR -> native binary.",
        epilog="Without --build the output is LLVM IR, written to <input>.ll",
    )
    parser.add_argument("input", help="Lumen source file to compile")
    parser.add_argument("-o", "--output", help="output path (IR, or the binary with --build)")
    parser.add_argument("-O", "--opt-level", default="1", choices=["0", "1", "2", "3"],
                        help="optimization level (default: 1)")
    parser.add_argument("--build", action="store_true",
                        help="invoke clang/llc to produce a native executable")
    parser.add_argument("--run", action="store_true",
                        help="execute the program (natively, else via --run-ir)")
    parser.add_argument("--run-ir", action="store_true",
                        help="execute the IR with the built-in interpreter "
                             "(needs no LLVM toolchain)")
    parser.add_argument("--tokens", action="store_true", help="print the token stream")
    parser.add_argument("--ast", action="store_true", help="print the AST")
    parser.add_argument("--ast-json", action="store_true", help="print the AST as JSON")
    parser.add_argument("--check-only", action="store_true",
                        help="run lexing, parsing and checking only")
    parser.add_argument("--verify-ir", action="store_true",
                        help="structurally verify the emitted IR")
    parser.add_argument("--target", default=None, help="override the LLVM target triple")
    parser.add_argument("--stats", action="store_true",
                        help="print phase timings and IR statistics")
    parser.add_argument("--quiet", "-q", action="store_true", help="suppress warnings")
    parser.add_argument("--version", action="version", version=f"lumenc {VERSION}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return _run_cli(args)
    except CompileError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(f"error: no such file: {args.input}", file=sys.stderr)
        return 1


def _emit_diagnostics(diags: DiagnosticBag, source: SourceMap, quiet: bool) -> None:
    for diag in diags.items:
        if quiet and diag.severity is Severity.WARNING:
            continue
        print(diag.render(source), file=sys.stderr)
    if diags.has_errors():
        print(f"{diags.error_count} error(s), {diags.warning_count} warning(s)",
              file=sys.stderr)


def _resolve_outputs(args: argparse.Namespace) -> tuple:
    """Decide where the IR and (optionally) the binary should be written."""
    stem = os.path.splitext(args.input)[0]
    if args.run_ir:
        return (args.output if args.output and args.output.endswith(".ll")
                else stem + ".ll"), None
    if args.build or args.run:
        if args.output and not args.output.endswith(".ll"):
            binary = args.output
        else:
            binary = stem + executable_suffix()
        ir_path = args.output if (args.output and args.output.endswith(".ll")) \
            else stem + ".ll"
        return ir_path, binary
    return args.output or (stem + ".ll"), None


def _run_cli(args: argparse.Namespace) -> int:
    with open(args.input, "r", encoding="utf-8") as handle:
        text = handle.read()

    opt_level = int(args.opt_level)
    result = compile_source(
        text,
        filename=args.input,
        opt_level=opt_level,
        target=args.target or host_triple(),
        keep_tokens=args.tokens,
    )

    if args.tokens:
        print(format_tokens(result.tokens))
    if args.ast and result.program is not None:
        print(format_ast(result.program))
    if args.ast_json and result.program is not None:
        print(json.dumps(result.program.to_json(), indent=2))

    _emit_diagnostics(result.diags, result.source, args.quiet)
    if not result.ok:
        return 1

    if args.check_only:
        print(f"ok: {args.input} type-checks cleanly", file=sys.stderr)
        return 0

    assert result.ir is not None
    if args.verify_ir:
        verification = irverify.verify(result.ir)
        print(verification.summary(), file=sys.stderr)
        if not verification.ok:
            return 1

    ir_path, binary_path = _resolve_outputs(args)
    with open(ir_path, "w", encoding="utf-8") as handle:
        handle.write(result.ir)

    if binary_path is not None:
        try:
            tool = build_native(ir_path, binary_path, opt_level=opt_level)
        except CompileError as exc:
            if args.run and not args.build:
                # --run without --build: fall back to the IR interpreter
                # rather than failing, so the compiler is usable with no LLVM.
                print(f"note: {exc.message.splitlines()[0]}", file=sys.stderr)
                print("note: running the IR with the built-in interpreter",
                      file=sys.stderr)
                return _run_interpreter(result.ir)
            print(f"error: {exc.message}", file=sys.stderr)
            return 1
        if not args.quiet:
            print(f"built {binary_path} with {tool}", file=sys.stderr)

    if args.stats:
        _print_stats(result)

    if args.run_ir:
        return _run_interpreter(result.ir)

    if args.run and binary_path is not None:
        return subprocess.run([os.path.abspath(binary_path)]).returncode
    return 0


def _run_interpreter(ir: str) -> int:
    """Execute the emitted IR with :mod:`lumen.irinterp`.

    Output is streamed straight to stdout so the program behaves exactly like a
    natively built one.
    """
    from . import irinterp

    try:
        program = irinterp.Program(ir, stdout=sys.stdout)
        return program.run()
    except irinterp.IRError as exc:
        print(f"\nerror: IR execution failed: {exc}", file=sys.stderr)
        return 1


def _print_stats(result: CompilationResult) -> None:
    print("\n--- phase timings (ms) ---", file=sys.stderr)
    for name, elapsed in result.timings.items():
        print(f"  {name:<10} {elapsed:7.3f}", file=sys.stderr)
    if result.ir is not None:
        verification = irverify.verify(result.ir)
        instructions = sum(
            1 for line in result.ir.splitlines()
            if line.startswith("  ") and not line.strip().startswith(";"))
        print("\n--- IR statistics ---", file=sys.stderr)
        print(f"  lines        {len(result.ir.splitlines())}", file=sys.stderr)
        print(f"  instructions {instructions}", file=sys.stderr)
        print(f"  blocks       {verification.blocks}", file=sys.stderr)
        print(f"  functions    {verification.functions}", file=sys.stderr)
        print(f"  phi nodes    {result.ir.count(' phi ')}", file=sys.stderr)
    if result.optimization_stats:
        print("\n--- AST optimizations ---", file=sys.stderr)
        for name, count in sorted(result.optimization_stats.items()):
            print(f"  {name:<18} {count}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
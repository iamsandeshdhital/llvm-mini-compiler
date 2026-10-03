# Lumen

**A compiler from a small statically typed language to LLVM IR, written from
scratch in pure Python — no dependencies, no code generation frameworks.**

Lumen is a teaching/research language designed to make the whole compilation
pipeline legible: you can read one file per phase, and every decision the
compiler makes is written down in a comment.

```
   hello.lum  ──▶  lexer  ──▶  parser  ──▶  checker  ──▶  optimizer  ──▶  lowering
                                                                              │
                                                                              ▼
                                            hello.ll  ◀──  IR verifier  ◀──  LLVM codegen
                                                                              │
                              ┌───────────────────────────────────────────────┴──────────────┐
                              ▼                                                              ▼
                     clang / llc (if installed)                            built-in IR interpreter
                        native .exe / binary                            (works with zero toolchain)
```

---

## Why this exists

Most "compiler" projects either wrap an existing backend (Clang, LLVM's ORC
JIT, `llvmlite`) or are a parser plus a toy interpreter. This one goes all the
way to **textual LLVM IR** with nothing but the Python standard library, and
then does the thing that usually gets skipped: it **executes** the IR it emits
to prove the generated code computes the right answer.

Concretely, that means:

- **Lexer → AST → typed AST → LLVM IR**, hand-written at every step.
- **Real semantic analysis**: lexical scoping, a numeric subtyping lattice,
  control-flow analysis that catches missing returns on *any* path, and
  diagnostics with source excerpts and carets.
- **A real backend**: alloca-based frames, explicit `load`/`store`, labelled
  basic blocks, short-circuiting `&&`/`||`, heap arrays with an O(1) length
  header, and value (copy) semantics for arrays.
- **A built-in LLVM IR interpreter**, so the compiler is runnable and testable
  with **no LLVM installation at all** — which is exactly the situation if you
  want to try the project on a machine with nothing installed.

## Quick start

Python 3.9+ is the only requirement.

```bash
git clone https://github.com/iamsandeshdhital/llvm-mini-compiler.git
cd llvm-mini-compiler

# Run a program immediately -- no LLVM needed, thanks to the IR interpreter
python lumenc.py examples/hello.lum --run-ir

# Compile to LLVM IR and inspect it
python lumenc.py examples/fib.lum -o fib.ll --stats

# See the token stream and the AST
python lumenc.py examples/arrays.lum --tokens
python lumenc.py examples/arrays.lum --ast

# Type-check only (useful in CI)
python lumenc.py examples/types.lum --check-only
```

Expected output:

```
$ python lumenc.py examples/hello.lum --run-ir
Hello from Lumen
sum of 1..10 is 55
```

If you would rather use the module form, set `PYTHONPATH=src` once and call
`python -m lumen ...`. Installing the package (`pip install -e .`) gives you a
`lumenc` command that works from anywhere.

### Building a native binary (optional)

If `clang` or `llc` is on your `PATH`:

```bash
python lumenc.py examples/hello.lum --build --run
```

Without a toolchain, `--run` falls back to the IR interpreter and says so.
`clang`/`llc` are *never* required to use this compiler.

## The language in one screen

```
// comments start with //, # or /* ... */

func main() -> int {
    // types: int (i64), float (double), bool, string, T[], void
    let n: int = 7;
    let xs: int[] = [3, 1, 4];
    let name: string = "lumen";

    print name, "has", len(xs), "numbers";   // values are space separated

    for i in 0..len(xs) {                     // half-open range [0, len)
        print "xs[", i, "] =", xs[i];
    }

    while n > 0 {
        n -= 1;
    }

    if n == 0 && len(name) > 0 {
        print "done";
    } else {
        print "something else";
    }

    return 0;
}

func greet(who: string) {      // no -> type means void
    print "hello,", who;
}
```

Full grammar and semantics: [`docs/LANGUAGE.md`](docs/LANGUAGE.md).

## Diagnostics

The compiler reports problems with a source excerpt rather than a line number
alone, and keeps going so one run shows several errors:

```
$ python -m lumen broken.lum
error: cannot initialise 'int count' with a value of type string (line 3, col 5)
  note: strings are a distinct type in Lumen
3 |     let count: int = "three";
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
error: 'counted' is not defined; did you mean 'count'? (line 5, col 11)
5 |     if counted > 1 {
  |           ^^^^^^^
2 error(s), 0 warning(s)
```

Some checks worth calling out, all implemented in `checker.py`:

| Check | Example |
| --- | --- |
| Missing return on *any* path | `func f(x) -> int { if x > 0 { return 1; } }` |
| Missing `main` | a program with no `main` |
| `break` outside a loop | `break;` at top level |
| bool is not a number | `let x: int = true;` |
| float does not narrow to int | `let x: int = 1.5;` |
| no string concatenation | `"a" + "b"` |
| assignment to `const` | `const k = 1; k = 2;` |
| wrong arity / argument type | `f(1, 2)` when `f` takes one `int` |

## Repository layout

```
src/lumen/
  diagnostics.py   source map, spans, caret excerpts, diagnostic bag
  tokens.py        token kinds and spellings
  lexer.py         hand-written scanner (comments, bases, escapes)
  ast.py           node definitions, spans, JSON serialization
  parser.py        recursive descent + precedence climbing
  types.py         type lattice (int <: float, T[] reference types)
  checker.py       scopes, type checking, control-flow analysis
  optimize.py      constant folding, dead branches, unreachable code
  lower.py         assigns LLVM stack slots to every local
  codegen.py       the LLVM IR backend
  irverify.py      structural verification of the emitted IR
  irinterp.py      executes the emitted IR (no LLVM required)
  driver.py        phase pipeline and CLI
examples/          programs that exercise every language feature
tests/             265 tests: lexer, parser, checker, IR, execution, CLI
docs/LANGUAGE.md   grammar and semantics
docs/ARCHITECTURE.md design notes and the lowering rules
```

## How the backend works

Three decisions shape everything in `codegen.py`:

**1. Memory-first codegen.** Every local and parameter is an `alloca` in the
entry block, accessed with explicit `load`/`store`:

```llvm
define i64 @sum_to(i64 %n) {
entry:
  %n_1 = alloca i64, align 8
  store i64 %n, i64* %n_1, align 8
  %total_2 = alloca i64, align 8
  store i64 0, i64* %total_2, align 8
  br label %while.cond1
while.cond1:
  %v1 = load i64, i64* %n_1, align 8
  ...
```

This is the shape LLVM's own `mem2reg` pass consumes, so `--build` at `-O2`
promotes every non-escaping local to a register. The payoff is that the emitter
never needs dominance-frontier computation — and therefore **never emits a phi
node**. Merge points read their value from a stack slot instead. There is a
test asserting `ir.count(" phi ") == 0`.

**2. Arrays are heap objects with a header.**

```
[ i64 length ][ elem 0 ][ elem 1 ] ...
  ^
  the array pointer points at elem 0
```

`len(a)` is a load from eight bytes *below* the pointer
(`getelementptr i8, i8* %p, i64 -8`), which is O(1) and lets arrays cross
function boundaries without a separate length parameter. Because arrays are
*values*, `a = b` copies the payload and grows the destination with `realloc`
when the sizes differ.

**3. `print` is variadic, so arguments keep their real types.** `%s` needs an
`i8*`, `%lld` needs an `i64`; passing a pointer as `i64` happens to work on
x86-64 and is still wrong IR.

## Testing

```bash
make test          # or: python -m unittest discover -s tests -v
```

265 tests, no dependencies, ~4 seconds. They are layered so a failure tells you
*where* it is:

| File | What it pins down |
| --- | --- |
| `test_lexer.py` | token kinds, literals, comments, error positions |
| `test_parser.py` | grammar coverage, operator precedence, associativity |
| `test_checker.py` | scoping, the type lattice, control-flow analysis |
| `test_codegen.py` | the mapping from language constructs to IR instructions |
| `test_execution.py` | **runs the emitted IR** and checks the output |
| `test_cli.py` | exit codes, flags, diagnostics |

Two techniques are worth highlighting:

- **`test_execution.py` executes the IR.** A test asserts
  `print 2 * 3 + 4;` produces `10\n`, so a backend regression is a failing
  assertion rather than a silently wrong binary.
- **Optimization equivalence.** The same ten programs are run at `-O0`, `-O1`,
  `-O2` and `-O3` and every level must produce byte-identical output.

That approach has already paid for itself. Bugs the execution tests caught that
IR inspection alone would not have:

| Bug | Symptom |
| --- | --- |
| `xor i1 %eq, false` for string `!=` | XOR with `false` is the identity, so `"a" != "b"` was false |
| Continuation block emitted before its predecessors | `else if` chains produced malformed IR |
| `printf` varargs always passed as `i64` | pointers passed with the wrong type |
| `fcmp lt` instead of `fcmp olt` | not a valid LLVM predicate |
| Optimizer kept *both* arms of a constant `if` | `if 2 * 2 > 3` printed both branches |
| `let ys = xs` aliased instead of copying | two bindings shared one heap buffer |

## Verifying IR without LLVM

`llvm-as` would be the normal way to check generated IR, but it needs a ~1 GB
install. `irverify.py` re-implements the checks that matter for a text emitter:

1. every basic block ends with exactly one terminator, and nothing follows it;
2. every `br`/`switch` target is a label that exists;
3. every `%name` use is preceded by a definition (no uninitialized reads);
4. no `%name` is defined twice;
5. lines are well formed (balanced parentheses, no stray tabs).

```
$ python -m lumen examples/arrays.lum --verify-ir
IR verified: 2 function(s), 12 block(s), 74 instruction(s)
```

`irinterp.go` executes the subset of IR this backend emits, including the C
runtime functions (`printf`, `malloc`, `realloc`, `strlen`, `strcmp`,
`memcpy`). `--run-ir` runs any program with it. It also gives the test suite a
way to check behaviour without a toolchain.

## Optimization

`-O0` disables the optimizer; `-O1` and above run these AST-level transforms:

- constant folding of arithmetic, comparisons and negation
- algebraic identities (`x + 0`, `x * 1`, `x - 0`, `x * 0` on integers)
- constant conditions (`if true`, `while false`, empty ranges)
- unreachable statement removal after `return`/`break`/`continue`
- `len` of an array literal

Keeping these on the AST rather than on the IR text means each transform is
small enough to audit, and it leaves LLVM's own (much stronger) pipeline to do
the real work once `--build` hands the IR over.

## Command line

| Flag | Effect |
| --- | --- |
| `-o PATH` | output path (IR, or the binary with `--build`) |
| `-O0..-O3` | optimization level (default `1`) |
| `--build` | invoke `clang`/`llc` to produce a native executable |
| `--run` | execute natively, falling back to the IR interpreter |
| `--run-ir` | always execute with the built-in interpreter |
| `--check-only` | lex, parse, type-check; emit no IR |
| `--verify-ir` | structurally verify the emitted IR |
| `--tokens` | print the token stream |
| `--ast`, `--ast-json` | print the AST |
| `--stats` | phase timings and IR statistics |
| `--target TRIPLE` | override the LLVM target triple |
| `-q, --quiet` | suppress warnings |

## Limitations

Being explicit about what is *not* here:

- No user-defined types, generics, closures, function values or modules.
- Nested arrays are copied **shallowly** (the outer buffer is copied, inner
  arrays are shared) — `examples/arrays.lum` demonstrates and documents this.
- Arrays are never freed; programs are short-lived, and adding GC to a teaching
  compiler would bury the interesting parts.
- The optimizer is AST-level and modest by design.
- The IR interpreter implements the subset this backend emits, not all of LLVM.
- The backend emits only `entry`-anchored allocas and no `switch`, `phi`, or
  vector types.

## License

MIT — see [LICENSE](LICENSE).
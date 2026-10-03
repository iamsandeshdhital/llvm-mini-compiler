# Architecture

How the compiler is put together, and *why* each piece makes the decision it
does. Read this if you want to extend the compiler; read the code comments if
you just want to understand it.

## 1. The pipeline

```
lexer.py     text            -> List[Token]
parser.py    tokens          -> Program (AST)
checker.py   AST             -> typed AST + diagnostics
optimize.py  typed AST       -> folded AST          [-O1 and above]
lower.py     typed AST       -> slot-annotated AST
codegen.py   slot AST        -> LLVM IR text
irverify.py  IR text         -> structural report
irinterp.py  IR text         -> program output      [--run-ir]
driver.py    CLI             -> runs the above
```

Each phase is a separate module with one public entry point, and each is usable
on its own:

```python
from lumen.driver import compile_source
result = compile_source("func main() -> int { return 0; }", "t.lum")
print(result.ir)
```

`compile_source` records per-phase timings in `result.timings`, which is what
`--stats` prints.

### Why the phases are separated this way

- **`checker.py` owns every decision about meaning.** Type rules, scoping,
  `const`-ness and the "does this function return on every path" question all
  live in one file, so the answer to "is this program well typed?" is auditable
  without reading the backend.
- **`lower.py` exists so `codegen.py` never invents names.** The checker creates
  a `Symbol` per binding; lowering gives that symbol a unique LLVM slot name
  (`%total_2`) and attaches it to both the declaration and every use. Without
  this split, name mangling would be duplicated in the emitter and the two
  copies would eventually disagree.
- **`optimize.py` runs before lowering.** Constant folding on the AST is far
  easier to verify than folding on IR text, and the optimizer only ever sees
  nodes the checker has already typed.

## 2. Type checking

`types.py` is deliberately tiny. There are five base types and one compound
type:

```
int <: float          (numeric widening)
bool, string          (unrelated to numbers and to each other)
T[]                   (reference to T)
```

`is_assignable(target, source)` is the only place that answers "may I store
this?", and it is used by `let`, assignment, argument passing and return. The
consequence worth noticing: because `bool` and `string` are not numbers,
`if x { }` with an integer `x` is a hard error rather than a silent truthiness
rule.

### Control-flow analysis

`Checker._returns` is a small backwards analysis — a statement returns a value
if every path out of it goes through `return`:

```python
if isinstance(stmt, ast.IfStmt):
    if stmt.else_branch is None:
        return False                      # the implicit else falls through
    return self._returns(stmt.then_branch) and self._returns(stmt.else_branch)
if isinstance(stmt, ast.WhileStmt):
    if isinstance(stmt.cond, ast.BoolLit) and stmt.cond.value:
        return any(...)                   # `while true` never falls out
    return False
```

This is why `func f(x) -> int { if x > 0 { return 1; } }` is rejected while
`func f(x) -> int { if x > 0 { return 1; } else { return 0; } }` is accepted —
a much better diagnostic than "expected a return at the end of the function".

## 3. Diagnostics

`diagnostics.py` keeps a `SourceMap` per compilation unit and turns a byte span
into a line/column plus a caret excerpt:

```
error: cannot initialise 'int count' with a value of type string (line 3, col 5)
  note: strings are a distinct type in Lumen
3 |     let count: int = "three";
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Every node carries a `Span`, so every error can point at the exact expression
responsible. The `DiagnosticBag` accumulates rather than throwing, which is why
one run reports every problem in the file instead of only the first.

## 4. The backend

### 4.1 Memory-first, and therefore phi-free

Every local and parameter becomes an `alloca` in the function's `entry` block:

```llvm
define i64 @_lumen_sum_to(i64 %n) {
entry:
  %n_1 = alloca i64, align 8
  store i64 %n, i64* %n_1, align 8
  %total_2 = alloca i64, align 8
  store i64 0, i64* %total_2, align 8
  %i_3 = alloca i64, align 8
  store i64 1, i64* %i_3, align 8
  br label %while.cond1
while.cond1:
  %v1 = load i64, i64* %i_3, align 8
  %v2 = load i64, i64* %n_1, align 8
  %c3 = icmp sle i64 %v1, %v2
  br i1 %c3, label %while.body2, label %while.end3
while.body2:
  ...
  br label %while.cond1
while.end3:
  %v9 = load i64, i64* %total_2, align 8
  ret i64 %v9
}
```

This is the canonical shape that LLVM's `mem2reg` pass is designed to consume,
so `--build` at `-O2` promotes every non-escaping local straight into a
register. The strategic benefit is that alloca-based codegen never needs a
dominance frontier, which is the only reason a real SSA backend needs `phi`
nodes at all. Two places needed a value at a merge point — the result of `&&`
and the result of an array copy — and both route the value through a temporary
`alloca` instead:

```llvm
%lg3 = alloca i1, align 1
store i1 %left, i1* %lg3, align 1
br i1 %left, label %logic.rhs4, label %logic.end5
logic.rhs4:
  %right = ...
  store i1 %right, i1* %lg3, align 1
  br label %logic.end5
logic.end5:
  %result = load i1, i1* %lg3, align 1
```

`test_codegen.py` asserts `ir.count(" phi ") == 0`, so the property cannot rot
silently.

### 4.2 Arrays are heap objects with a header

```
        0        8                16
        +--------+-------------------+
        | length | element 0 | ...  |      malloc(8 + n*sizeof(T))
        +--------+-------------------+
              ^
              the array pointer
```

- **O(1) `len`.** `getelementptr i8, i8* %p, i64 -8` reaches the header, so
  arrays can be passed to functions and returned without a length parameter.
- **Recoverable allocation.** Because the header sits immediately below the
  elements, the original `malloc` pointer is `getelementptr i8, i8* %p, i64 -8`.
  That is what lets `a = b` grow the destination with `realloc` instead of
  leaking a new buffer.
- **Value semantics.** Three cases, and all three are real code in `codegen.py`:

  | Situation | Lowering |
  | --- | --- |
  | `let a: int[] = [1, 2]` | `malloc` + header + element stores (`emit_array_literal`) |
  | `let b: int[] = a` | `malloc` + `memcpy` into a **fresh** buffer (`emit_array_clone`) |
  | `a = b`, same length | `memcpy` in place (`emit_array_copy`, fast path) |
  | `a = b`, different length | `realloc` then `memcpy` (`emit_array_copy`, grow path) |

  A length check (`icmp eq`) chooses between the last two, and the resulting
  pointer is delivered through an `alloca` so no phi is needed.

### 4.3 Block construction

`IRBuilder.detached_block` is the only place block ordering is decided. It
emits a region of blocks branching off the current one, collects every block the
region created (a nested `if` always leaves its merge block open), and
re-attaches them in order:

```python
def detached_block(self, name, body):
    saved = self._blocks.pop()
    depth = len(self._blocks)
    self._blocks.append([f"{name}:"])
    body()
    region = []
    while len(self._blocks) > depth:
        region.append(self._blocks.pop())
    region.reverse()
    saved.extend(line for block in region for line in block)
    self._blocks.append(saved)
```

This is subtle enough to deserve the note: the first version of this function
restored `saved` with `append`, which put it *after* the region when a nested
construct had pushed a continuation block. The result was `else if` chains
emitting their blocks before `entry`, which the verifier caught. Every
`if`/`while`/`for`/logical/array-copy construct now goes through this one
function rather than manipulating the block list itself.

### 4.4 Lowering rules

| Lumen | LLVM |
| --- | --- |
| `let x = e` | `%x = alloca T` then `store` |
| `x` | `%v = load T, T* %x` |
| `a + b` (int) | `add i64` |
| `a / b` (int) | `sdiv i64` (truncates toward zero) |
| `a + b` (float) | `fadd double` |
| `a < b` | `icmp slt i64` / `fcmp olt double` |
| `!a` | `xor i1 %a, true` |
| `a && b` | two blocks; the result is stored to a slot |
| `a[0] != "s"` | `strcmp` + `icmp eq` + `xor ... true` |
| `len(xs)` | load the header via `getelementptr ... i64 -8` |
| `len("s")` | `call i64 @strlen` |
| `if c {} else {}` | `br i1` to `if.then`/`if.else`, joined at `if.end` |
| `while c {}` | `while.cond` / `while.body` / `while.end` |
| `for i in a..b` | induction slot + `icmp slt` in `for.cond`, bump in `for.step` |
| `print 1` | `call i32 (i8*, ...) @printf(i8* %fmt, i64 1)` |
| `print 1.5` | `... @printf(i8* %fmt, double ...)` |
| `print "s"` | `... @printf(i8* %fmt, i8* %str)` — the pointer keeps its type |
| `int` → `float` | `sitofp` |
| `float` → `int` (explicit) | `fptosi` |
| any call to `f` | `call ... @_lumen_f` (`main` keeps its name) |

User functions are prefixed `_lumen_` so they cannot collide with libc symbols,
and string literals are interned into `private unnamed_addr constant [N x i8]`
globals with a `\00` terminator.

Float constants are emitted as exact 64-bit hex (`0x400921F9F01B866E` for
`pi`), never as decimal, so no precision is lost between the AST and the
backend.

## 5. Optimization

`optimize.py` works on the checked AST and is conservative by construction: a
transform fires only when it is provably value-preserving.

| Transform | Example |
| --- | --- |
| Constant folding | `2 * 3 + 4` → `10` |
| Identity folding | `x + 0`, `x - 0`, `x * 1`, `x * 0` (integers only) |
| Constant condition | `if true` keeps only the then-arm |
| Empty range | `for i in 5..5` disappears |
| Unreachable code | statements after `return`/`break`/`continue` |
| Folded `len` | `len([1,2,3])` → `3` |

`-O0` skips it entirely. The test suite runs ten programs at every level and
requires byte-identical output, which is how an inverted branch-folding bug was
caught.

## 6. Verification and execution without LLVM

### `irverify.py`

A structural verifier for the emitted text: one terminator per block and
nothing after it, every branch target defined, every `%name` used after it is
defined, no name defined twice, balanced parentheses and no tabs. Forward
branch references are legal in LLVM and are allowed.

The definition-before-use check is textual rather than a dominance analysis. It
is *sound* for this backend because the emitter only ever produces a value's
definition ahead of its uses, and it is still worth keeping as a regression net
if that ever stops holding.

### `irinterp.py`

Executes the subset of IR the backend emits, in a flat address space where
pointers are integers and `alloca`/`malloc` hand out 8-byte aligned addresses.
`getelementptr` is therefore plain arithmetic, which makes the array header
behave exactly as it does in LLVM. It implements the binary and float
arithmetic, comparisons (`seq`/`slt`/`olt`/...), conversions, `getelementptr`,
`bitcast`, `br`, `ret`, and the C runtime subset `printf`, `malloc`, `realloc`,
`free`, `strlen`, `strcmp`, `memcpy`.

Three things this buys:

1. `--run-ir` runs any Lumen program with no toolchain installed.
2. `test_execution.py` can assert that generated code produces specific output,
   which is what turns "the IR looks right" into "the IR is right".
3. Interpreter bugs found during development were real IR bugs. Executing
   `xor i1 %eq, false` showed `!=` returning false, because XOR with `false` is
   the identity rather than negation.

## 7. Testing strategy

| Layer | Question it answers |
| --- | --- |
| `test_lexer.py` | does the scanner produce the right tokens and positions? |
| `test_parser.py` | does precedence/associativity build the intended tree? |
| `test_checker.py` | is every ill-typed program rejected with a good message? |
| `test_codegen.py` | does each construct lower to the documented IR? |
| `test_execution.py` | does the emitted IR compute the right answers? |
| `test_cli.py` | do the flags, exit codes and diagnostics behave? |

Precedence is tested by *inspecting the tree shape* rather than by parsing a
string, so a change to the precedence table cannot silently pass. Optimization
is tested by *differential execution* across `-O0..-O3`. Backend behaviour is
tested by *running the IR*.

## 8. Ideas for extending it

Roughly in order of how much they would teach:

1. **Structs and struct literals** — exercises field offsets, aggregate
   lowering, and `insertvalue`/`extractvalue` or pointer-to-member schemes.
2. **A register allocator** — emit virtual registers, build an interference
   graph, colour it, then rewrite the allocas away. The existing alloca output
   is a convenient starting point and would make mem2reg's job unnecessary.
3. **SSA construction with real phi nodes** — dominance frontiers, then the
   renaming pass. The current alloca design would become the "before" state.
4. **Closure conversion** — environment pointers, or a flat closure record.
5. **A `--emit-asm` mode** — target a specific instruction set directly.
6. **Constant folding on the IR** instead of the AST, with a real benefit
   analysis to decide what is safe to move or delete.
7. **A garbage collector**, or at least `free` on unreachable arrays, which
   needs liveness analysis to be worth doing.

## 9. Things deliberately not done

- **No bitwise operators.** They would widen the IR surface without teaching
  anything about the pipeline.
- **No exception handling or closures.** Each drags in representation questions
  (landing pads, environment chains) that would bury the ten IR patterns that
  make this project readable.
- **No garbage collector.** Programs are short-lived; the interesting part is
  how values are represented, not how they are reclaimed.
- **No incremental or parallel compilation.** The whole program is a few
  milliseconds; there is nothing to parallelise.
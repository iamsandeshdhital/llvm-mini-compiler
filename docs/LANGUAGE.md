# The Lumen language

A small, statically typed, imperative language. The grammar below is
authoritative: `parser.py` is a direct transcription of it.

## 1. Lexical structure

| Element | Rule |
| --- | --- |
| Comments | `// to end of line`, `# to end of line`, `/* ... */` (nesting tolerated) |
| Whitespace | Insignificant. Newlines never terminate a statement. |
| Identifiers | `[A-Za-z_][A-Za-z0-9_]*` |
| Integers | `123`, `0xFF`, `0b1011`, `0o17`; `_` may separate digits |
| Floats | `1.5`, `.5` is *not* allowed, `2e10`, `2.5e-3` |
| Strings | `"..."` or `'...'`, escapes `\n \t \r \0 \\ \" \' \e \xNN \uNNNN` |
| Operators | `+ - * / %` `== != < <= > >=` `&& \|\|` `!` `= += -= *= /= %=` `->` `..` |

A `.` immediately followed by a digit continues a numeric literal, so `1.5` is
a float while `1.max` and `a.b` are unaffected.

### Keywords

`let` `const` `if` `else` `while` `for` `in` `break` `continue` `return`
`func` `print` `true` `false` `and` `or` `not` `len` `int` `float` `bool`
`string` `void`

`and`/`or`/`not` are synonyms for `&&`/`||`/`!`.

## 2. Grammar

```ebnf
program      ::= function*

function     ::= "func" IDENT "(" params? ")" ( "->" type )? block
               | type IDENT "(" params? ")" block          (* C-style spelling *)

params       ::= param ( "," param )*
param        ::= IDENT ":" type

type         ::= "int" | "float" | "bool" | "string" | "void"   type suffix*
type suffix  ::= "[" "]"

block        ::= "{" statement* "}"

statement    ::= letStmt | ifStmt | whileStmt | forStmt | returnStmt
               | breakStmt | continueStmt | printStmt | block | exprStmt
               ;

letStmt      ::= ( "let" | "const" ) IDENT ( ":" type )? "=" expr semi?
ifStmt       ::= "if" expr block ( "else" ( ifStmt | block ) )?
whileStmt    ::= "while" expr block
forStmt      ::= "for" IDENT "in" expr ".." expr block
returnStmt   ::= "return" expr? semi?
breakStmt    ::= "break" semi?
continueStmt ::= "continue" semi?
printStmt    ::= "print" ( expr ( "," expr )* )? semi?
exprStmt     ::= expr semi?

semi?        ::= ";" | ε                            (* optional *)

(* expressions, loosest precedence first *)
expr         ::= assignment
assignment   ::= binary ( ( "=" | "+=" | "-=" | "*=" | "/=" | "%=" )
                           assignment )?          (* right associative *)
binary       ::= unary ( binop unary )*           (* left associative *)
unary        ::= ( "-" | "!" | "not" ) unary | postfix
postfix      ::= primary ( "(" args? ")" | "[" expr "]" )*
args         ::= expr ( "," expr )*
primary      ::= INT | FLOAT | STRING | "true" | "false"
               | IDENT | "len" "(" expr ")"
               | "(" expr ")" | arrayLit
arrayLit     ::= "[" ( expr ( "," expr )* )? "]"

binop        ::= "*" | "/" | "%"                   (7)
               | "+" | "-"                          (6)
               | "<" | "<=" | ">" | ">="            (5)
               | "==" | "!="                        (4)
               | "&&" | "and"                       (3)
               | "||" | "or"                        (2)
```

`semi?` is what makes the newline-terminated style work. The one place it can
bite: a statement that ends in an expression followed by a line starting with
`-` or `(` continues onto the next line. Write `;` there.

## 3. Types

| Type | LLVM | Notes |
| --- | --- | --- |
| `int` | `i64` | 64-bit signed; `/` truncates toward zero, `%` follows the sign |
| `float` | `double` | IEEE-754 binary64 |
| `bool` | `i1` | `true`/`false`, stored as 0/1 |
| `string` | `i8*` | NUL-terminated, immutable, compared by content |
| `void` | `void` | only as a return type |
| `T[]` | `T*` | reference to a heap array; see §6 |

### Conversions

`int` widens to `float` implicitly and anywhere. Nothing else converts:

```
let a: float = 1;      // ok, 1 becomes 1.0
let b: int = 1.5;      // error: no implicit narrowing
let c: int = true;     // error: bool is not a number
```

### Typing rules

- Arithmetic (`+ - * / %`) requires `int` or `float`; `%` is integer-only.
- `< <= > >=` require two numeric operands and produce `bool`.
- `== !=` require operands to unify (numerics, strings, or arrays) and produce
  `bool`. Strings compare by content.
- `&& ||` require `bool` and produce `bool`. **They short-circuit.**
- `!`/`not` requires `bool`.
- `if`/`while` conditions must be `bool`.
- An array literal's element type is the unification of its elements, so
  `[1, 2.5]` is `float[]`.
- `len` accepts an array or a string and returns `int`.

## 4. Declarations and statements

```
let x = 1;              // type inferred
let y: float = 1;       // annotated; stores into a double slot
const k = 10;           // immutable after initialisation
```

Parameters are mutable (`p = p + 1;` is legal) and are spilled to the stack on
entry.

A block introduces a scope; declarations in it do not leak. Shadowing is
allowed:

```
let x = 1;
{
    let x = 2;   // shadows
    print x;     // 2
}
print x;         // 1
```

`for i in a..b` binds `i` in a scope containing only the loop body. The range is
**half-open**: `for i in 0..3` runs with `i` equal to 0, 1, 2. If either bound
is a float the counter is a float, so `for i in 0.0..3.0` steps by 1.0.

`main` must return `int` (its value becomes the process exit code) or nothing
(exit code 0). The backend normalises a void `main` to `i32` so the entry point
matches what a C runtime expects.

## 5. Values and assignment

Assignment is an expression, right-associative, so `a = b = 0` works:

```
let a = 0;
a = a + 1;
a += 2;   // also -= *= /= %=
```

Valid targets are variables and array elements. Assigning to a `const`, or to
anything that is not assignable, is a compile error. Note that a `const` array
still has mutable *elements*: `const a: int[] = [1]; a[0] = 5;` is legal.

## 6. Arrays

An array value is a pointer to its first element, with the element count stored
in the eight bytes below it (see [ARCHITECTURE.md](ARCHITECTURE.md#2-arrays-are-heap-objects-with-a-header)).

```
let xs: int[] = [3, 1, 4];
print len(xs);    // 3, O(1)
print xs[0];      // 3
xs[0] = 99;
```

**Arrays are values, not references.** Binding or assigning one copies the
payload, so the two bindings are independent:

```
let a: int[] = [1, 2];
let b: int[] = a;
b[0] = 99;
print a;   // [1, 2]
print b;   // [99, 2]
```

If the destination is a different size, its buffer is grown to fit. Arrays of
arrays are supported and copied **shallowly** — the outer buffer is copied, the
inner arrays are shared:

```
let m: int[][] = [[1, 2], [3, 4]];
let n: int[][] = m;
n[0][0] = 99;
print m[0];   // [99, 2]  (inner array is shared)
```

Printing an array shows it in bracket form: `print [1, 2, 3];` prints
`[1, 2, 3]`.

## 7. `print`

`print` is a statement, not a function. Arguments are space-separated and a
newline ends the line:

```
print "a", 1, true;
// a 1 1
```

| Value | Rendered as |
| --- | --- |
| `int` | the number, e.g. `-42` |
| `float` | six decimal places, e.g. `1.500000` |
| `bool` | `1` or `0` |
| `string` | the characters |
| `T[]` | `[e0, e1, e2]`, recursing into nested arrays |

There is no string concatenation in Lumen; `print "fib", n, "=", fib(n);` is
how output is composed.

## 8. A complete program

```
// examples/hello.lum
func main() -> int {
    print "Hello from Lumen";
    print "sum of 1..10 is", sum_to(10);
    return 0;
}

func sum_to(n: int) -> int {
    let total: int = 0;
    let i: int = 1;
    while i <= n {
        total += i;
        i += 1;
    }
    return total;
}
```

## 9. Deliberate omissions

Lumen has no user-defined types, generics, closures, function values, modules,
imports, exceptions, or garbage collector. Each of those would add machinery
that obscures the pipeline this project exists to show. The language is small
enough to read in one sitting and complete enough to write real programs in —
sieve, matrix multiply, sorting, interpreters — all of which lower to the same
ten IR patterns documented in [ARCHITECTURE.md](ARCHITECTURE.md).
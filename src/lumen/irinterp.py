"""A small LLVM IR interpreter for the subset this compiler emits.

Why this exists
---------------
Emitting IR is only half of a backend; the other half is knowing the emitted
instructions mean what you intended. Normally ``llvm-as`` plus ``lli`` settle
that, but they are a ~1 GB install. This module executes the IR directly,
which buys three things:

* ``lumenc --run-ir`` runs a Lumen program with **no LLVM toolchain at all**,
  so the compiler is usable and demonstrable on a bare Python install;
* the test suite executes real programs and compares their output, turning
  "the IR looks plausible" into "the IR computes the right answer";
* any disagreement between the AST semantics and the generated machine code
  shows up as a failing test rather than as silent corruption.

Model
-----
* One flat address space. ``alloca`` and ``malloc`` hand out 8-byte aligned
  addresses and pointers are plain integers, so ``getelementptr`` is plain
  arithmetic and the ``[length][elements...]`` array header behaves exactly as
  it does in LLVM.
* Values are Python ints (for i1, i64 and pointers), floats, or ``None`` for
  void.
* Globals are materialized into memory at load time, so a ``getelementptr`` on
  a string constant behaves the same as one on a heap object.
* Control flow follows ``br``/``ret`` between real basic blocks.

Supported opcodes: alloca, load, store, getelementptr, bitcast, call, add,
sub, mul, sdiv, srem, udiv, urem, fadd, fsub, fmul, fdiv, frem, fneg, icmp,
fcmp, zext, sext, trunc, sitofp, fptosi, fptoui, uitofp, br, ret -- plus the C
runtime functions printf, malloc, realloc, free, strlen, strcmp and memcpy.
"""

from __future__ import annotations

import io
import math
import re
import struct
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: Interpreting a call means recursing in Python, so Lumen's call depth is
#: bounded by the interpreter's own stack. Raise the limit to match
#: :attr:`Interpreter.max_depth` instead of hitting Python's default of 1000.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 40000))

# --------------------------------------------------------------------------
# Parsed representation
# --------------------------------------------------------------------------


@dataclass
class Instr:
    opcode: str
    text: str
    result: Optional[str] = None


@dataclass
class Block:
    label: str
    instructions: List[Instr] = field(default_factory=list)


@dataclass
class Function:
    name: str
    ret_type: str
    params: List[Tuple[str, str]] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    is_declaration: bool = False

    def block_map(self) -> Dict[str, Block]:
        return {block.label: block for block in self.blocks}


@dataclass
class Module:
    functions: Dict[str, Function] = field(default_factory=dict)
    globals: Dict[str, bytes] = field(default_factory=dict)
    triple: Optional[str] = None


class IRParseError(Exception):
    pass


class IRError(Exception):
    """A fault while interpreting the IR (bad index, null pointer, ...)."""


_GLOBAL_CONST = re.compile(
    r'^(@[\w.$-]+)\s*=\s*(?:private\s+)?(?:unnamed_addr\s+)?'
    r'constant\s+\[\d+ x i8\]\s+c"(.*)",\s*align\s*\d+$'
)


def parse_module(ir: str) -> Module:
    """Parse the textual IR produced by :mod:`lumen.codegen`."""
    module = Module()
    lines = ir.splitlines()
    index = 0
    total = len(lines)

    while index < total:
        line = lines[index].strip()
        index += 1
        if not line or line.startswith(";"):
            continue

        if line.startswith("target triple"):
            module.triple = line.split("=", 1)[1].strip().strip('"')
            continue
        if line.startswith(("source_filename", "ModuleID")):
            continue

        if match := _GLOBAL_CONST.match(line):
            module.globals[match.group(1)] = _unescape(match.group(2))
            continue

        if line.startswith("declare "):
            header = line[len("declare"):].strip()
            name_part = header.partition("(")[0]
            _, _, name = name_part.rpartition("@")
            ret_type = name_part.rsplit("@", 1)[0].strip()
            module.functions[name.strip()] = Function(
                name.strip(), ret_type, is_declaration=True)
            continue

        if line.startswith("define "):
            function = _parse_signature(line)
            current: Optional[Block] = None
            while index < total:
                inner = lines[index].strip()
                index += 1
                if inner == "}":
                    break
                if not inner or inner.startswith(";"):
                    continue
                if inner.endswith(":") and "=" not in inner:
                    current = Block(inner[:-1])
                    function.blocks.append(current)
                    continue
                if current is None:
                    raise IRParseError(
                        f"instruction outside a block in @{function.name}: {inner!r}")
                current.instructions.append(_parse_instruction(inner))
            module.functions[function.name] = function
            continue

        raise IRParseError(f"cannot parse module line: {line!r}")

    return module


def _parse_signature(line: str) -> Function:
    """Parse ``define <ret> @name(<params>) {``.

    The name has to be split off *before* the parameter list, because the
    return type contains no parentheses but the argument list does.
    """
    header = line[len("define"):].strip().rstrip("{").strip()
    name_part, _, arg_part = header.partition("(")
    _, _, name = name_part.rpartition("@")
    ret_type = name_part.rsplit("@", 1)[0].strip()
    arg_text = arg_part.rsplit(")", 1)[0]

    params: List[Tuple[str, str]] = []
    for piece in _split_top(arg_text, ","):
        piece = piece.strip()
        if not piece:
            continue
        ty, _, value = piece.rpartition(" ")
        params.append((value.strip(), ty.strip()))
    return Function(name.strip(), ret_type, params)


def _parse_instruction(line: str) -> Instr:
    result: Optional[str] = None
    if "=" in line.split(" ", 1)[0] or line.startswith("%"):
        result, _, rest = line.partition("=")
        result = result.strip()
        rest = rest.strip()
    else:
        rest = line
    opcode, _, tail = rest.partition(" ")
    return Instr(opcode=opcode.strip(), text=tail.strip(), result=result)


def _unescape(body: str) -> bytes:
    """Decode an LLVM ``c"..."`` body back into raw bytes."""
    out = bytearray()
    index = 0
    hexdigits = set("0123456789ABCDEFabcdef")
    while index < len(body):
        char = body[index]
        if char != "\\":
            out.extend(char.encode("utf-8"))
            index += 1
            continue
        index += 1
        if index >= len(body):
            break
        nxt = body[index]
        if nxt in hexdigits and index + 1 < len(body) and body[index + 1] in hexdigits:
            out.append(int(body[index:index + 2], 16))
            index += 2
        elif nxt == "n":
            out.append(0x0A)
            index += 1
        elif nxt == "t":
            out.append(0x09)
            index += 1
        else:
            out.append(ord(nxt))
            index += 1
    return bytes(out)


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


class Memory:
    """Flat byte-addressed memory with 8-byte allocation granularity."""

    def __init__(self) -> None:
        self.cells: Dict[int, object] = {}
        self.next_address = 1024

    def allocate(self, size: int) -> int:
        address = self.next_address
        self.next_address += max(8, ((size + 7) // 8) * 8)
        return address

    def read(self, address: int) -> object:
        address = int(address)
        if address not in self.cells:
            raise IRError(f"read from uninitialized address 0x{address:x}")
        return self.cells[address]

    def write(self, address: int, value: object) -> None:
        self.cells[int(address)] = value

    def read_bytes(self, address: int, count: int) -> bytes:
        out = bytearray()
        for offset in range(count):
            value = self.cells.get(int(address) + offset)
            out.append(int(value) & 0xFF if isinstance(value, (int, bool)) else 0)
        return bytes(out)

    def write_bytes(self, address: int, data: bytes) -> None:
        for offset, byte in enumerate(data):
            self.cells[int(address) + offset] = byte

    def read_cstring(self, address: int) -> bytes:
        out = bytearray()
        cursor = int(address)
        while True:
            if cursor in self.cells:
                value = self.cells[cursor]
                if isinstance(value, str):
                    out.extend(value.encode("utf-8"))
                    return bytes(out)
                if int(value) == 0:
                    return bytes(out)
                out.append(int(value) & 0xFF)
                cursor += 1
                continue
            return bytes(out)


# --------------------------------------------------------------------------
# Opcode tables
# --------------------------------------------------------------------------

#: LLVM spells integer comparisons ``icmp <pred>`` with signed/unsigned
#: prefixes, so ``==`` becomes ``seq`` and ``<=`` becomes ``sle``.
_ICMP = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "slt": lambda a, b: a < b,
    "sle": lambda a, b: a <= b,
    "sgt": lambda a, b: a > b,
    "sge": lambda a, b: a >= b,
    "ult": lambda a, b: a < b,
    "ule": lambda a, b: a <= b,
    "ugt": lambda a, b: a > b,
    "uge": lambda a, b: a >= b,
    # Unsigned spellings are interpreted as signed: every integer the compiler
    # emits is i64, and the interpreter models both the same way.
    "seq": lambda a, b: a == b,
    "sne": lambda a, b: a != b,
}

_FCMP = {
    "oeq": lambda a, b: a == b,
    "one": lambda a, b: a != b,
    "olt": lambda a, b: a < b,
    "ole": lambda a, b: a <= b,
    "ogt": lambda a, b: a > b,
    "oge": lambda a, b: a >= b,
    "ord": lambda a, b: not (math.isnan(a) or math.isnan(b)),
    "uno": lambda a, b: math.isnan(a) or math.isnan(b),
}

_ARITH_INT = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "sdiv": "int_div",
    "srem": "int_rem",
    "udiv": "int_div",
    "urem": "int_rem",
    # `xor i1 %a, true` is how the backend spells logical negation.
    "xor": lambda a, b: bool(a) != bool(b),
    "and": lambda a, b: bool(a) and bool(b),
    "or": lambda a, b: bool(a) or bool(b),
}

_ARITH_FLOAT = {
    "fadd": lambda a, b: a + b,
    "fsub": lambda a, b: a - b,
    "fmul": lambda a, b: a * b,
    "fdiv": "float_div",
    "frem": "float_rem",
}

_SUPPORTED = set(_ARITH_INT) | set(_ARITH_FLOAT) | {
    "alloca", "load", "store", "getelementptr", "bitcast", "call", "icmp",
    "fcmp", "zext", "sext", "trunc", "sitofp", "fptosi", "fptoui", "uitofp",
    "br", "ret", "fneg", "unreachable", "switch",
}


# --------------------------------------------------------------------------
# Interpreter
# --------------------------------------------------------------------------


class Interpreter:
    def __init__(self, module: Module, memory: Memory,
                 stdout: Optional[io.StringIO] = None) -> None:
        self.module = module
        self.memory = memory
        self.stdout = stdout
        self.call_depth = 0
        self.max_depth = 5000
        self.steps = 0
        self.max_steps = 20_000_000
        self._blocks: Dict[str, Dict[str, Block]] = {}
        #: Address of each global, filled in by :class:`Program`.
        self.globals: Dict[str, int] = {}

    # -- execution -------------------------------------------------------
    def run_main(self) -> int:
        if "main" not in self.module.functions:
            raise IRError("module has no @main")
        result = self.call("main", [])
        return 0 if result is None else int(result)

    def call(self, name: str, arguments: List[object]) -> object:
        function = self.module.functions.get(name)
        if function is None or function.is_declaration:
            return self._call_runtime(name, arguments)
        if self.call_depth > self.max_depth:
            raise IRError("maximum call depth exceeded (runaway recursion?)")

        values: Dict[str, object] = {}
        for (param, _ty), argument in zip(function.params, arguments):
            values[param] = argument

        blocks = self._blocks.get(name)
        if blocks is None:
            blocks = function.block_map()
            self._blocks[name] = blocks
        if not function.blocks:
            return None

        label = "entry" if "entry" in blocks else function.blocks[0].label
        self.call_depth += 1
        try:
            while True:
                block = blocks.get(label)
                if block is None:
                    raise IRError(f"branch to unknown block %{label}")
                jump: Optional[str] = None
                for instruction in block.instructions:
                    self.steps += 1
                    if self.steps > self.max_steps:
                        raise IRError(
                            "step limit exceeded -- the program probably loops "
                            "forever")
                    outcome = self._execute(instruction, values)
                    if outcome is not None:
                        kind, payload = outcome
                        if kind == "br":
                            jump = payload
                            break
                        return payload  # ret
                if jump is None:
                    return None  # implicit ret void
                label = jump
        finally:
            self.call_depth -= 1

    def _execute(self, instruction: Instr, values: Dict[str, object]):
        opcode = instruction.opcode
        if opcode not in _SUPPORTED:
            raise IRError(f"unsupported opcode {opcode!r} ({instruction.text})")
        if opcode in _ARITH_INT:
            return self._arithmetic(instruction, values, _ARITH_INT[opcode])
        if opcode in _ARITH_FLOAT:
            return self._arithmetic(instruction, values, _ARITH_FLOAT[opcode])
        if opcode == "icmp":
            return self._compare(instruction, values, _ICMP)
        if opcode == "fcmp":
            return self._compare(instruction, values, _FCMP)
        if opcode in ("zext", "sext", "trunc", "sitofp", "fptosi", "fptoui",
                      "uitofp", "bitcast", "fptrunc", "fpext"):
            return self._convert(instruction, values)
        if opcode == "fneg":
            operand = self._value(_operand(instruction.text), values)
            values[instruction.result] = -float(operand)
            return None
        handler = getattr(self, f"_op_{opcode}")
        return handler(instruction, values)

    # -- memory ----------------------------------------------------------
    def _op_alloca(self, instruction: Instr, values: Dict[str, object]):
        match = re.search(r",\s*i64\s+(\d+)", instruction.text)
        count = int(match.group(1)) if match else 1
        values[instruction.result] = self.memory.allocate(count * 8)
        return None

    def _op_load(self, instruction: Instr, values: Dict[str, object]):
        parts = _split_top(instruction.text, ",", 1)
        value_type = parts[0].strip()
        pointer = self._value(_operand(parts[1]), values)
        value = self.memory.read(pointer)
        if value_type in ("double", "float"):
            value = float(value)
        elif value_type == "i1":
            value = bool(value)
        else:
            value = int(value)
        values[instruction.result] = value
        return None

    def _op_store(self, instruction: Instr, values: Dict[str, object]):
        parts = _split_top(instruction.text, ",", 1)
        value_type, operand = _split_type_and_operand(parts[0])
        value = self._value(operand, values)
        if value_type == "i1":
            value = bool(value)
        elif value_type in ("double", "float"):
            value = float(value)
        pointer = self._value(_operand(parts[1]), values)
        self.memory.write(pointer, value)
        return None

    def _op_getelementptr(self, instruction: Instr, values: Dict[str, object]):
        text = re.sub(r"^inbounds\s+", "", instruction.text)
        parts = _split_top(text, ",")
        element_type = parts[0].strip()
        base = self._value(_operand(parts[1]), values)
        indices = [self._value(_operand(piece), values) for piece in parts[2:]]
        if len(indices) == 1:
            address = base + indices[0] * _type_size(element_type)
        else:
            array_size = _type_size(element_type)
            address = base + indices[0] * array_size + \
                indices[1] * _type_size(parts[1].strip())
        values[instruction.result] = address
        return None

    def _op_bitcast(self, instruction: Instr, values: Dict[str, object]):
        source, _target = _split_conversion(instruction.text)
        values[instruction.result] = self._value(_operand(source), values)
        return None

    # -- operators -------------------------------------------------------
    def _arithmetic(self, instruction: Instr, values: Dict[str, object], op) -> None:
        parts = _split_top(instruction.text, ",")
        left = self._value(_operand(parts[0]), values)
        right = self._value(_operand(parts[1]), values)
        if op == "int_div":
            if right == 0:
                raise IRError("integer division by zero")
            magnitude = abs(left) // abs(right)
            result = magnitude if (left < 0) == (right < 0) else -magnitude
        elif op == "int_rem":
            if right == 0:
                raise IRError("integer remainder by zero")
            magnitude = abs(left) % abs(right)
            result = magnitude if left >= 0 else -magnitude
        elif op == "float_div":
            if right == 0:
                raise IRError("float division by zero")
            result = left / right
        elif op == "float_rem":
            result = math.fmod(left, right) if right else float("nan")
        else:
            result = op(left, right)
        values[instruction.result] = result

    def _compare(self, instruction: Instr, values: Dict[str, object],
                 table: Dict[str, object]) -> None:
        predicate, _, rest = instruction.text.partition(" ")
        parts = _split_top(rest.strip(), ",")
        left = self._value(_operand(parts[0]), values)
        right = self._value(_operand(parts[1]), values)
        values[instruction.result] = table[predicate](left, right)

    def _convert(self, instruction: Instr, values: Dict[str, object]) -> None:
        source, target = _split_conversion(instruction.text)
        operand = self._value(_operand(source), values)
        if target.startswith(("double", "float")):
            result = float(operand)
        elif instruction.opcode in ("fptosi", "fptoui"):
            result = abs(int(operand)) if instruction.opcode == "fptoui" \
                else int(operand)
        elif instruction.opcode == "uitofp":
            result = float(operand)
        elif target == "i1":
            result = bool(operand)
        else:
            result = int(operand)
        values[instruction.result] = result

    # -- control flow ----------------------------------------------------
    def _op_br(self, instruction: Instr, values: Dict[str, object]):
        if instruction.text.startswith("i1"):
            parts = _split_top(instruction.text, ",")
            condition = bool(self._value(_operand(parts[0]), values))
            true_label = _label(parts[1])
            false_label = _label(parts[2])
            return ("br", true_label if condition else false_label)
        return ("br", _label(instruction.text))

    def _op_ret(self, instruction: Instr, values: Dict[str, object]):
        text = instruction.text.strip()
        if not text or text == "void":
            return ("ret", None)
        return ("ret", self._value(_operand(text), values))

    def _op_unreachable(self, instruction: Instr, values: Dict[str, object]):
        raise IRError("executed an `unreachable` instruction")

    def _op_switch(self, instruction: Instr, values: Dict[str, object]):
        parts = _split_top(instruction.text, "[", 1)
        operand = self._value(_operand(parts[0]), values)
        for case in _split_top(parts[1], " ")[1:]:
            value, _, label = case.partition(" ")
            if int(value) == operand:
                return ("br", _label(label))
        return ("br", _label(parts[1].split("]")[-1] if "]" in parts[1] else ""))

    # -- calls -----------------------------------------------------------
    def _op_call(self, instruction: Instr, values: Dict[str, object]):
        text = instruction.text
        name = text.rsplit("@", 1)[-1].split("(")[0].strip()
        arguments = [
            self._value(_operand(piece), values)
            for piece in _split_top(_arguments_of(text), ",")
            if piece.strip()
        ]
        result: object
        function = self.module.functions.get(name)
        if function is not None and not function.is_declaration:
            result = self.call(name, arguments)
        else:
            result = self._call_runtime(name, arguments)
        if instruction.result:
            values[instruction.result] = 0 if result is None else result
        return None

    def _call_runtime(self, name: str, arguments: List[object]) -> object:
        handler = getattr(self, f"_rt_{name}", None)
        if handler is None:
            raise IRError(f"call to unknown external function @{name}")
        return handler(*arguments)

    # -- C runtime -------------------------------------------------------
    def _rt_printf(self, format_pointer: int, *arguments: object) -> int:
        spec = self.memory.read_cstring(format_pointer).decode("utf-8", "replace")
        text, consumed = format_c(spec, list(arguments), self.memory)
        if self.stdout is not None:
            self.stdout.write(text)
        return consumed

    def _rt_malloc(self, size: int) -> int:
        return self.memory.allocate(int(size))

    def _rt_realloc(self, address: int, size: int) -> int:
        address, size = int(address), int(size)
        new_address = self.memory.allocate(size)
        # Copy the payload, including the 8-byte length header.
        for offset in range(0, max(8, size), 8):
            value = self.memory.cells.get(address + offset)
            if value is None:
                break
            self.memory.cells[new_address + offset] = value
        return new_address

    def _rt_free(self, address: int) -> None:
        return None

    def _rt_strlen(self, address: int) -> int:
        return len(self.memory.read_cstring(address))

    def _rt_strcmp(self, left: int, right: int) -> int:
        first = self.memory.read_cstring(left)
        second = self.memory.read_cstring(right)
        return (first > second) - (first < second)

    def _rt_memcpy(self, destination: int, source: int, count: int) -> None:
        destination, source, count = int(destination), int(source), int(count)
        for offset in range(count):
            value = self.memory.cells.get(source + offset)
            if value is not None:
                self.memory.cells[destination + offset] = value
        return None

    # -- operand resolution ----------------------------------------------
    def _value(self, text: str, values: Dict[str, object]) -> object:
        text = text.strip()
        if not text:
            raise IRError("missing operand")
        if text.startswith("%"):
            if text not in values:
                raise IRError(f"use of undefined SSA value {text}")
            return values[text]
        if text.startswith("@"):
            address = self.globals.get(text)
            if address is None:
                raise IRError(f"reference to unknown global {text}")
            return address
        if text == "true":
            return True
        if text == "false":
            return False
        if text in ("null", "none"):
            return 0
        if text.startswith("0x") or text.startswith("-0x"):
            return _llvm_hex_number(text)
        try:
            return int(text)
        except ValueError:
            return float(text)


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------


def _split_top(text: str, separator: str = ",", limit: int = -1) -> List[str]:
    """Split on ``separator``, ignoring anything inside brackets or parens.

    With ``limit`` set, the final element keeps the unsplit remainder (so
    ``split("store i64 %v, i64* %p, align 8", ",", 1)`` still sees the
    trailing ``align`` clause).
    """
    out: List[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == separator and depth == 0:
            out.append(text[start:index])
            start = index + 1
            if 0 <= limit <= len(out):
                out.append(text[start:])
                return out
    out.append(text[start:])
    return out


#: Trailing instruction attributes that must be stripped before reading an
#: operand -- otherwise the ``8`` in ``align 8`` looks like an integer constant.
_TAIL_ATTR_RE = re.compile(
    r",\s*(?:align|align2|volatile|nontemporal|noundef|zeroinitializer)"
    r"(?:\s+[\w*]+)*\s*$"
)

_SYMBOL_RE = re.compile(r"[@%][\w.$-]+")

_CONSTANT_RE = re.compile(
    r"^(?:true|false|null|none|-?0x[0-9A-Fa-f]+|"
    r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+)$"
)


def _operand(text: str) -> str:
    """Pull the operand out of ``<type> <operand>[, align N]``.

    Trailing attributes are stripped first, then a symbol reference wins over a
    numeric constant. Doing it in that order matters: ``i64* %n_1, align 8``
    must resolve to ``%n_1``, not to the ``8``.
    """
    text = _TAIL_ATTR_RE.sub("", text.strip()).strip()
    if not text:
        return text
    symbol = _SYMBOL_RE.search(text)
    if symbol:
        return symbol.group(0)
    if _CONSTANT_RE.match(text.split(" ", 1)[-1].strip()):
        return text.split(" ", 1)[-1].strip()
    return text.split(" ", 1)[-1].strip() if " " in text else text


def _split_conversion(text: str) -> Tuple[str, str]:
    """Split ``<srctype> <operand> to <dsttype>`` into its two halves.

    The ``to`` clause has to come off first, or the operand extractor sees
    ``true to i64`` in ``zext i1 true to i64``.
    """
    text = text.strip()
    if " to " in text:
        source, _, target = text.rpartition(" to ")
        return source.strip(), target.strip()
    return text, ""


def _split_type_and_operand(text: str) -> Tuple[str, str]:
    text = text.strip()
    if " " not in text:
        return "", text
    head, _, rest = text.partition(" ")
    if head.startswith(("i", "double", "float", "ptr", "[")):
        return head, rest.strip()
    return "", text


def _label(text: str) -> str:
    """Extract a block label from ``%foo``, ``label %foo`` or ``foo:``."""
    text = text.strip()
    if text.startswith("label"):
        text = text[len("label"):]
    return text.strip().rstrip(":").strip().lstrip("%")


def _arguments_of(text: str) -> str:
    """Extract the argument list from ``<ret> <sig> @name(<args>)``.

    The callee's own signature contains parentheses (``i32 (i8*, ...)``), so the
    argument list is found by matching brackets from the ``(`` that follows the
    callee name rather than by taking the last ``(`` in the line.
    """
    at = text.rfind("@")
    start = text.find("(", at if at != -1 else 0)
    if start == -1:
        return ""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:index].strip()
    return ""


def _llvm_hex_number(text: str) -> float:
    """Decode LLVM's hexadecimal floating-point literal into a double.

    ``0x400921F9F01B866E`` is the raw IEEE-754 bit pattern, *not* a C99 hex
    float, so ``float.fromhex`` would misread it as a huge integer.
    """
    negative = text.startswith("-")
    digits = text.lstrip("-+")[2:]
    if len(digits) == 16:
        value = struct.unpack(">d", struct.pack(">Q", int(digits, 16)))[0]
    else:
        value = float.fromhex(digits)
    return -value if negative else value


def _type_size(ty: str) -> int:
    ty = ty.strip()
    if ty.endswith("*") or ty == "ptr":
        return 8
    if ty in ("double", "float", "i64", "i32"):
        return 8
    if ty in ("i1", "i8"):
        return 1
    if match := re.match(r"\[(\d+) x", ty):
        return int(match.group(1))
    return 8


def format_c(spec: str, arguments: List[object],
             memory: Memory) -> Tuple[str, int]:
    """Minimal printf: %d, %i, %lld, %f, %g, %s, %c, %% and newlines."""
    out: List[str] = []
    index = 0
    consumed = 0
    while index < len(spec):
        char = spec[index]
        if char != "%":
            out.append(char)
            index += 1
            continue
        if spec[index + 1:index + 2] == "%":
            out.append("%")
            index += 2
            continue
        cursor = index + 1
        while cursor < len(spec) and spec[cursor] in "lhzj":
            cursor += 1
        if cursor >= len(spec):
            break
        conversion = spec[cursor]
        index = cursor + 1

        def take() -> object:
            nonlocal consumed
            value = arguments[consumed] if consumed < len(arguments) else 0
            consumed += 1
            return value

        if conversion in "di":
            out.append(str(int(take())))
        elif conversion == "f":
            out.append(f"{float(take()):.6f}")
        elif conversion in "ge":
            out.append(str(float(take())))
        elif conversion == "s":
            out.append(memory.read_cstring(int(take())).decode("utf-8", "replace"))
        elif conversion == "c":
            out.append(chr(int(take())))
        else:
            out.append(f"%{conversion}")
    return "".join(out), consumed


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


class Program:
    """A loaded module plus its memory, ready to execute."""

    def __init__(self, ir: str, stdout: Optional[io.StringIO] = None) -> None:
        self.ir = ir
        self.module = parse_module(ir)
        self.memory = Memory()
        self.stdout = stdout if stdout is not None else io.StringIO()
        self.interpreter = Interpreter(self.module, self.memory, self.stdout)
        # Globals live at low addresses so their pointers stay printable.
        for name, data in self.module.globals.items():
            address = self.memory.allocate(len(data))
            self.memory.write_bytes(address, data)
            self.interpreter.globals[name] = address

    def run(self) -> int:
        return self.interpreter.run_main()

    @property
    def output(self) -> str:
        return self.stdout.getvalue()


def execute(ir: str) -> Tuple[int, str]:
    """Run ``@main`` from ``ir``; returns ``(exit code, captured stdout)``."""
    program = Program(ir)
    code = program.run()
    return code, program.output
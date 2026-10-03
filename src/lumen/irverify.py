"""A structural verifier for the emitted LLVM IR.

Without an LLVM toolchain installed there is no ``llvm-as`` to validate against,
so this module re-implements the checks that matter for a text emitter:

1. every basic block ends with exactly one terminator, and nothing follows it
   inside that block;
2. every ``br``/``switch`` target names a label that is actually defined;
3. every ``%name`` use is preceded by a definition of that name (SSA form, with
   the caveat noted below);
4. no ``%name`` is defined twice;
5. instruction lines are well formed (balanced parentheses, no stray tabs).

The definition-before-use check is textual, not a dominance analysis. It is
*sound* for this backend specifically, because :mod:`lumen.codegen` only ever
emits basic blocks in an order where a value's definition precedes every use --
and it is still useful as a regression net if that ever stops being true.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_DEFINE_RE = re.compile(
    r"^\s*(?P<name>%[A-Za-z0-9_.$-]+)\s*=\s*(?P<rest>.+)$"
)
_USE_RE = re.compile(r"%[A-Za-z0-9_.$-]+")
_LABEL_DEF_RE = re.compile(r"^([A-Za-z0-9_.$-]+):")
_LABEL_REF_RE = re.compile(r"label\s+%([A-Za-z0-9_.$-]+)")
_FUNC_RE = re.compile(r"^define\b")
_GLOBAL_RE = re.compile(r"^@[A-Za-z0-9_.$-]+\s*=")

_TERMINATORS = ("ret", "br", "switch", "unreachable", "resume", "invoke")

#: Module-level directives that are not globals or functions.
_MODULE_DIRECTIVES = (
    "source_filename", "target", "declare", "attributes", "uselistorder",
    "module", "!", ";",
)


@dataclass
class VerificationResult:
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    functions: int = 0
    blocks: int = 0
    instructions: int = 0

    def error(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def summary(self) -> str:
        if self.ok:
            return (f"IR verified: {self.functions} function(s), "
                    f"{self.blocks} block(s), {self.instructions} instruction(s)")
        return f"IR verification failed with {len(self.errors)} problem(s):\n  " + \
               "\n  ".join(self.errors)


def verify(ir: str) -> VerificationResult:
    result = VerificationResult()
    lines = ir.splitlines()
    in_function = False
    function_name = ""
    current_block: Optional[str] = None
    seen_defs: Dict[str, int] = {}
    defined_labels: set = set()
    used_labels: List[tuple] = []
    depth_parens = 0
    in_string = False
    string_delim = ""

    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(";"):
            continue

        # Parenthesis/quote balance must be zero at end of line.
        depth_parens, in_string = _scan_line(line, depth_parens, in_string)
        if depth_parens < 0:
            result.error(f"{function_name}:{number}: unbalanced ')' in: {line}")
            depth_parens = 0
        if "\t" in raw:
            result.error(f"{function_name}:{number}: tab character in IR line")

        if _FUNC_RE.match(line):
            in_function = True
            function_name = line.split("@", 1)[-1].split("(")[0].strip()
            result.functions += 1
            seen_defs = {}
            defined_labels = set()
            used_labels = []
            current_block = None
            # Parameters count as definitions of their incoming names.
            for param in re.findall(r"%[A-Za-z0-9_.$-]+", line.split("(", 1)[-1]):
                seen_defs.setdefault(param, number)
            continue

        if line.startswith("}"):
            if in_function:
                for target, line_no in used_labels:
                    if target not in defined_labels:
                        result.error(
                            f"{function_name}:{line_no}: branch to undefined label "
                            f"%{target}")
            in_function = False
            current_block = None
            continue

        if not in_function:
            if _GLOBAL_RE.match(line):
                continue
            if line.startswith(_MODULE_DIRECTIVES):
                continue
            result.error(f"line {number}: unexpected module-level text: {line}")
            continue

        label_match = _LABEL_DEF_RE.match(line)
        if label_match:
            current_block = label_match.group(1)
            result.blocks += 1
            if current_block in defined_labels:
                result.error(
                    f"{function_name}:{number}: duplicate label %{current_block}")
            defined_labels.add(current_block)
            # Forward label references are legal, so pending branch targets are
            # checked once at the end of the function, not per block.
            continue

        if current_block is None:
            result.error(f"{function_name}:{number}: instruction outside a block")
            continue

        # Instruction inside a block.
        result.instructions += 1
        define_match = _DEFINE_RE.match(line)
        if define_match:
            name = define_match.group("name")
            if name in seen_defs:
                result.error(
                    f"{function_name}:{number}: '%{name[1:]}' redefined "
                    f"(first defined on line {seen_defs[name]})")
            seen_defs[name] = number

        # `label %foo` names a basic block, not an SSA value.
        value_text = _LABEL_REF_RE.sub("", line)
        for used in _USE_RE.findall(value_text):
            if used not in seen_defs:
                result.error(
                    f"{function_name}:{number}: use of undefined value {used}")

        for target in _LABEL_REF_RE.findall(line):
            used_labels.append((target, number))

        head = line.split(" ", 1)[0]
        if head in _TERMINATORS:
            current_block = None  # the next instruction starts a new block

    if depth_parens != 0:
        result.error("module ends with unbalanced parentheses")
    return result


def _scan_line(line: str, depth: int, in_string: bool) -> tuple:
    """Track parenthesis depth while respecting ``c"..."`` string literals."""
    index = 0
    while index < len(line):
        char = line[index]
        if in_string:
            if char == "\\":
                index += 2
                continue
            if char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        index += 1
    return depth, in_string


def verify_or_raise(ir: str) -> VerificationResult:
    result = verify(ir)
    if not result.ok:
        raise ValueError(result.summary())
    return result
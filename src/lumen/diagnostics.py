"""Source management and diagnostics for the Lumen compiler.

A :class:`SourceMap` owns the original text of one compilation unit and can
render a caret-underlined excerpt for any span, which is what makes compiler
errors actionable instead of merely correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional, TextIO


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"


@dataclass(frozen=True)
class Span:
    """A half-open region ``[start, end)`` of a single source file."""

    start: int
    end: int

    def merge(self, other: "Span") -> "Span":
        lo = min(self.start, other.start)
        hi = max(self.end, other.end)
        return Span(lo, hi)

    def __len__(self) -> int:
        return max(0, self.end - self.start)


@dataclass(frozen=True)
class Location:
    file: str
    line: int
    col: int


class SourceMap:
    """Tracks one file's text and maps byte offsets to human positions."""

    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.text = text
        # Offsets of the first character of every line (line 1 starts at 0).
        self._line_starts: List[int] = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                self._line_starts.append(i + 1)

    @classmethod
    def from_path(cls, path: str) -> "SourceMap":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(path, fh.read())

    def line_col(self, offset: int) -> tuple[int, int]:
        """Return 1-based ``(line, column)`` for a byte offset."""
        offset = max(0, min(offset, len(self.text)))
        lo, hi = 0, len(self._line_starts) - 1
        while lo < hi:  # bisect for the last line start <= offset
            mid = (lo + hi + 1) // 2
            if self._line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1, offset - self._line_starts[lo] + 1

    def line_text(self, line: int) -> str:
        if line < 1 or line > len(self._line_starts):
            return ""
        start = self._line_starts[line - 1]
        end = self.text.find("\n", start)
        if end == -1:
            end = len(self.text)
        return self.text[start:end].rstrip("\r")

    def snippet(self, span: Span) -> str:
        """Render ``span`` with a caret line, e.g. ``1 | let x = ;`` / ``    ^``."""
        line, col = self.line_col(span.start)
        text = self.line_text(line)
        caret_len = max(1, span.end - span.start)
        caret_len = min(caret_len, max(1, len(text) - col + 1))
        gutter = str(line)
        pad = " " * len(gutter)
        caret = " " * (col - 1) + "^" * caret_len
        return f"{gutter} | {text}\n{pad} | {caret}"


@dataclass
class Diagnostic:
    severity: Severity
    message: str
    span: Optional[Span] = None
    location: Optional[Location] = None
    notes: List[str] = field(default_factory=list)

    def render(self, source: Optional[SourceMap] = None) -> str:
        head = f"{self.severity.value}: {self.message}"
        if self.location is not None:
            head += f" (line {self.location.line}, col {self.location.col})"
        lines = [head]
        for note in self.notes:
            lines.append(f"  note: {note}")
        if source is not None and self.span is not None:
            lines.append(source.snippet(self.span))
        return "\n".join(lines)


class CompileError(Exception):
    """Raised to abort compilation after a fatal, already-reported error."""

    def __init__(self, message: str, diagnostics: Optional[Iterable[Diagnostic]] = None):
        super().__init__(message)
        self.message = message
        self.diagnostics = list(diagnostics or ())


class DiagnosticBag:
    """Collects diagnostics so a phase can report many problems in one run."""

    def __init__(self, source: Optional[SourceMap] = None) -> None:
        self.source = source
        self.items: List[Diagnostic] = []

    def _add(self, severity: Severity, message: str, span: Optional[Span],
             notes: Optional[Iterable[str]] = None) -> Diagnostic:
        loc = None
        if span is not None and self.source is not None:
            line, col = self.source.line_col(span.start)
            loc = Location(self.source.name, line, col)
        diag = Diagnostic(severity, message, span, loc, list(notes or ()))
        self.items.append(diag)
        return diag

    def error(self, message: str, span: Optional[Span] = None,
              notes: Optional[Iterable[str]] = None) -> Diagnostic:
        return self._add(Severity.ERROR, message, span, notes)

    def warning(self, message: str, span: Optional[Span] = None,
                notes: Optional[Iterable[str]] = None) -> Diagnostic:
        return self._add(Severity.WARNING, message, span, notes)

    def note(self, message: str, span: Optional[Span] = None) -> Diagnostic:
        return self._add(Severity.NOTE, message, span)

    @property
    def errors(self) -> List[Diagnostic]:
        return [d for d in self.items if d.severity is Severity.ERROR]

    @property
    def warnings(self) -> List[Diagnostic]:
        return [d for d in self.items if d.severity is Severity.WARNING]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def has_errors(self) -> bool:
        return any(d.severity is Severity.ERROR for d in self.items)

    def emit(self, stream: TextIO, limit: int = 50) -> None:
        """Print up to ``limit`` diagnostics, then a summary line."""
        for diag in self.items[:limit]:
            print(diag.render(self.source), file=stream)
        hidden = len(self.items) - limit
        if hidden > 0:
            print(f"... and {hidden} more diagnostic(s) suppressed", file=stream)

    def raise_if_failed(self) -> None:
        if self.has_errors():
            raise CompileError(
                f"{self.error_count} error(s) during compilation",
                self.items,
            )
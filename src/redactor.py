"""Validated, deterministic redaction.

Applies category-specific placeholders to selected findings using character
offsets against the ORIGINAL text. Replacements are applied right-to-left so
that earlier offsets never shift. Overlapping selections are resolved
deterministically to avoid corrupting the output.

The original input is never mutated; a new string is returned.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .schemas import Finding


class RedactionError(ValueError):
    """Raised when a finding's span is invalid for the given text."""


def _validate_span(finding: Finding, text_len: int) -> None:
    if finding.start < 0 or finding.end < 0:
        raise RedactionError("negative offset")
    if finding.start >= finding.end:
        raise RedactionError("start must be before end")
    if finding.end > text_len:
        raise RedactionError("span extends beyond text")


def _validate_evidence(finding: Finding, text: str) -> None:
    if finding.evidence and text[finding.start:finding.end] != finding.evidence:
        raise RedactionError("finding evidence does not match the original text")


def _resolve_overlaps(findings: Sequence[Finding]) -> List[Finding]:
    """Drop findings that overlap an already-accepted finding.

    Processed in order of start offset, then by widest span first, so the
    broadest redaction wins and later overlapping spans are skipped.
    """
    ordered = sorted(findings, key=lambda f: (f.start, -(f.end - f.start)))
    accepted: List[Finding] = []
    for f in ordered:
        if any(f.start < a.end and a.start < f.end for a in accepted):
            continue
        accepted.append(f)
    return accepted


def apply_redactions(original_text: str, selected: Iterable[Finding]) -> str:
    """Return a sanitized copy of ``original_text`` with selected spans redacted.

    Invalid spans raise :class:`RedactionError`. Overlapping selected findings
    are resolved so that output text is never corrupted.
    """
    selected = list(selected)
    if not selected:
        return original_text

    text_len = len(original_text)
    for f in selected:
        _validate_span(f, text_len)
        _validate_evidence(f, original_text)

    resolved = _resolve_overlaps(selected)

    # Apply from right to left so offsets remain valid as we edit.
    resolved.sort(key=lambda f: f.start, reverse=True)

    result = original_text
    for f in resolved:
        placeholder = f.suggested_replacement or "[REDACTED]"
        result = result[: f.start] + placeholder + result[f.end:]
    return result

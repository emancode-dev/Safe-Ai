"""Finding schemas, validation, and normalization for SafeShare AI.

This module defines the canonical ``Finding`` data structure used across the
deterministic detectors and the AI analyzer, plus validation helpers that
sanitize untrusted model output before it is ever used to modify user text.

Nothing here performs I/O or logging of user content.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Allowed vocabularies
# ---------------------------------------------------------------------------

VALID_SEVERITIES: Tuple[str, ...] = ("low", "medium", "high")

VALID_CATEGORIES: Tuple[str, ...] = (
    "email",
    "phone",
    "ip_address",
    "api_key",
    "private_key",
    "cloud_credential",
    "secret_assignment",
    "personal_data",
    "internal_detail",
    "other",
)

# Detection source labels.
METHOD_PATTERN = "pattern"
METHOD_AI = "ai"

# Maximum length of a suggested replacement placeholder. Guards against a
# malicious model returning a huge replacement string.
MAX_REPLACEMENT_LEN = 64

# Maximum length of an evidence excerpt we are willing to store/display.
MAX_EVIDENCE_LEN = 160

# Category -> default redaction placeholder.
CATEGORY_PLACEHOLDERS: Dict[str, str] = {
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "ip_address": "[REDACTED_IP]",
    "api_key": "[REDACTED_SECRET]",
    "private_key": "[REDACTED_SECRET]",
    "cloud_credential": "[REDACTED_SECRET]",
    "secret_assignment": "[REDACTED_SECRET]",
    "personal_data": "[REDACTED_PERSONAL_DATA]",
    "internal_detail": "[REDACTED_INTERNAL]",
    "other": "[REDACTED]",
}


def default_placeholder(category: str) -> str:
    """Return the default redaction placeholder for a category."""
    return CATEGORY_PLACEHOLDERS.get(category, "[REDACTED]")


@dataclass
class Finding:
    """A single potential privacy risk located in the input text.

    Offsets are character indices into the *original* submitted text such that
    ``text[start:end]`` yields the matched span.
    """

    category: str
    severity: str
    start: int
    end: int
    evidence: str
    reason: str
    suggested_replacement: str
    method: str  # METHOD_PATTERN or METHOD_AI
    finding_id: str = ""

    def __post_init__(self) -> None:
        if not self.finding_id:
            self.finding_id = self.compute_id()

    def compute_id(self) -> str:
        """Deterministic, stable ID for a finding within one analysis run."""
        raw = f"{self.method}|{self.category}|{self.start}|{self.end}|{self.evidence}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        return f"{self.method}-{digest}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ValidationError(ValueError):
    """Raised when a finding cannot be validated or normalized."""


def _coerce_int(value: Any) -> Optional[int]:
    """Best-effort integer coercion that never raises."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def validate_finding(raw: Dict[str, Any], original_text: str, *, method: str) -> Finding:
    """Validate and normalize a single raw finding dict against the source text.

    Raises :class:`ValidationError` if the finding cannot be made safe. The
    caller is expected to drop rejected findings rather than guess at a fix.
    """
    if not isinstance(raw, dict):
        raise ValidationError("finding is not an object")

    category = str(raw.get("category", "")).strip().lower()
    if category not in VALID_CATEGORIES:
        raise ValidationError("unsupported category")

    severity = str(raw.get("severity", "")).strip().lower()
    if severity not in VALID_SEVERITIES:
        raise ValidationError("unsupported severity")

    start = _coerce_int(raw.get("start"))
    end = _coerce_int(raw.get("end"))
    if start is None or end is None:
        raise ValidationError("offsets are missing or non-numeric")
    if start < 0 or end < 0:
        raise ValidationError("offsets must be non-negative")
    if start >= end:
        raise ValidationError("start must be strictly less than end")
    if end > len(original_text):
        raise ValidationError("end offset is beyond text length")

    actual_span = original_text[start:end]

    evidence = str(raw.get("evidence", "")).strip()
    # Evidence must be consistent with the actual span. We require the span to
    # be present rather than trusting the model's quoted evidence verbatim.
    if evidence and evidence != actual_span:
        raise ValidationError("evidence does not match the supplied offsets")
    evidence = actual_span[:MAX_EVIDENCE_LEN]

    reason = str(raw.get("reason", "")).strip()
    if not reason:
        reason = "Potential privacy risk."
    reason = reason[:500]

    replacement = str(raw.get("suggested_replacement", "")).strip()
    if (
        not replacement
        or len(replacement) > MAX_REPLACEMENT_LEN
        or not (replacement.startswith("[") and replacement.endswith("]"))
    ):
        replacement = default_placeholder(category)

    return Finding(
        category=category,
        severity=severity,
        start=start,
        end=end,
        evidence=evidence,
        reason=reason,
        suggested_replacement=replacement,
        method=method,
    )


def validate_ai_payload(payload: Dict[str, Any], original_text: str) -> Tuple[List[Finding], str]:
    """Validate a full AI response payload.

    Returns ``(findings, summary)``. Invalid individual findings are dropped;
    the whole payload is rejected only if it is structurally unusable.
    """
    if not isinstance(payload, dict):
        raise ValidationError("AI payload is not a JSON object")

    raw_findings = payload.get("findings", [])
    if not isinstance(raw_findings, list):
        raise ValidationError("'findings' must be a list")

    findings: List[Finding] = []
    for raw in raw_findings:
        try:
            findings.append(validate_finding(raw, original_text, method=METHOD_AI))
        except ValidationError:
            # Silently drop individual bad findings; never crash on bad input.
            continue

    summary = str(payload.get("summary", "")).strip()[:1000]
    return findings, summary


def _overlaps(a: Finding, b: Finding) -> bool:
    return a.start < b.end and b.start < a.end


def deduplicate(findings: Iterable[Finding]) -> List[Finding]:
    """Remove duplicate and redundant overlapping findings.

    Rules:
    - Exact duplicate spans+category collapse to one.
    - When two findings overlap, prefer the pattern-detected one (deterministic
      and trustworthy); otherwise keep the one with the wider span.
    """
    items = list(findings)
    # Stable ordering: by start, then wider span first, then pattern before AI.
    items.sort(key=lambda f: (f.start, -(f.end - f.start), 0 if f.method == METHOD_PATTERN else 1))

    kept: List[Finding] = []
    for cand in items:
        redundant = False
        for existing in kept:
            if cand.category == existing.category and cand.start == existing.start and cand.end == existing.end:
                redundant = True
                break
            if _overlaps(cand, existing):
                # Decide which to keep.
                if existing.method == METHOD_PATTERN and cand.method == METHOD_AI:
                    redundant = True
                    break
                if existing.method == cand.method:
                    # Keep wider existing (already sorted wider-first) -> drop candidate.
                    redundant = True
                    break
                # cand is pattern, existing is AI overlapping: replace existing.
                if cand.method == METHOD_PATTERN and existing.method == METHOD_AI:
                    kept.remove(existing)
                    break
        if not redundant:
            kept.append(cand)

    kept.sort(key=lambda f: (f.start, f.end))
    return kept


def severity_counts(findings: Iterable[Finding]) -> Dict[str, int]:
    counts = {s: 0 for s in VALID_SEVERITIES}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    return counts

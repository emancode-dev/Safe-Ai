"""Tests for schema validation and normalization."""

import pytest

from src.schemas import (
    Finding,
    METHOD_AI,
    METHOD_PATTERN,
    ValidationError,
    deduplicate,
    severity_counts,
    validate_ai_payload,
    validate_finding,
)

TEXT = "Contact jane.doe@example.org about the Riverside clinic incident."


def _raw(**over):
    base = {
        "category": "personal_data",
        "severity": "medium",
        "start": 8,
        "end": 28,
        "evidence": "jane.doe@example.org",
        "reason": "email identifies a person",
        "suggested_replacement": "[REDACTED_PERSONAL_DATA]",
    }
    base.update(over)
    return base


def test_valid_finding():
    f = validate_finding(_raw(), TEXT, method=METHOD_AI)
    assert f.category == "personal_data"
    assert TEXT[f.start:f.end] == f.evidence
    assert f.finding_id.startswith("ai-")


def test_invalid_json_payload_type():
    with pytest.raises(ValidationError):
        validate_ai_payload([], TEXT)


def test_missing_findings_key_defaults_empty():
    findings, summary = validate_ai_payload({"summary": "none"}, TEXT)
    assert findings == []
    assert summary == "none"


def test_findings_not_list_rejected():
    with pytest.raises(ValidationError):
        validate_ai_payload({"findings": {}}, TEXT)


def test_invalid_severity_rejected():
    with pytest.raises(ValidationError):
        validate_finding(_raw(severity="catastrophic"), TEXT, method=METHOD_AI)


def test_unknown_category_rejected():
    with pytest.raises(ValidationError):
        validate_finding(_raw(category="banana"), TEXT, method=METHOD_AI)


def test_evidence_must_match_exact_offset():
    with pytest.raises(ValidationError):
        validate_finding(_raw(evidence="jane"), TEXT, method=METHOD_AI)


def test_negative_offset_rejected():
    with pytest.raises(ValidationError):
        validate_finding(_raw(start=-1), TEXT, method=METHOD_AI)


def test_offset_beyond_text_rejected():
    with pytest.raises(ValidationError):
        validate_finding(_raw(start=0, end=9999), TEXT, method=METHOD_AI)


def test_start_not_before_end_rejected():
    with pytest.raises(ValidationError):
        validate_finding(_raw(start=10, end=10), TEXT, method=METHOD_AI)


def test_evidence_not_in_text_rejected():
    with pytest.raises(ValidationError):
        validate_finding(_raw(evidence="not-present-anywhere"), TEXT, method=METHOD_AI)


def test_bad_findings_dropped_not_crash():
    payload = {
        "findings": [
            _raw(),                      # valid
            _raw(start=-5),              # invalid offset
            {"category": "x"},           # missing offsets
        ],
        "summary": "mixed",
    }
    findings, summary = validate_ai_payload(payload, TEXT)
    assert len(findings) == 1
    assert summary == "mixed"


def test_oversized_replacement_falls_back_to_placeholder():
    f = validate_finding(_raw(suggested_replacement="X" * 500), TEXT, method=METHOD_AI)
    assert f.suggested_replacement == "[REDACTED_PERSONAL_DATA]"


def _f(cat, start, end, method=METHOD_PATTERN):
    return Finding(cat, "low", start, end, "x", "r", "[R]", method)


def test_dedup_exact_duplicates():
    a = _f("email", 0, 5)
    b = _f("email", 0, 5)
    assert len(deduplicate([a, b])) == 1


def test_dedup_prefers_pattern_over_ai_on_overlap():
    pattern = _f("email", 0, 10, METHOD_PATTERN)
    ai = _f("personal_data", 2, 8, METHOD_AI)
    result = deduplicate([ai, pattern])
    assert len(result) == 1
    assert result[0].method == METHOD_PATTERN


def test_severity_counts():
    findings = [_f("email", 0, 1), Finding("email", "high", 2, 3, "x", "r", "[R]", METHOD_AI)]
    counts = severity_counts(findings)
    assert counts["low"] == 1
    assert counts["high"] == 1

"""Tests for the redaction engine."""

import pytest

from src.redactor import apply_redactions, RedactionError
from src.schemas import Finding, METHOD_PATTERN


def _f(cat, start, end, repl, evidence=None):
    return Finding(cat, "medium", start, end, evidence or "", "r", repl, METHOD_PATTERN)


def test_redact_selected_email():
    text = "mail me at a@b.co now"
    f = _f("email", 11, 17, "[REDACTED_EMAIL]", "a@b.co")
    assert apply_redactions(text, [f]) == "mail me at [REDACTED_EMAIL] now"


def test_redact_token_value():
    text = "token=abc123xyz end"
    f = _f("secret_assignment", 6, 15, "[REDACTED_SECRET]", "abc123xyz")
    assert apply_redactions(text, [f]) == "token=[REDACTED_SECRET] end"


def test_unselected_findings_untouched():
    text = "a@b.co and 10.0.0.1"
    email = _f("email", 0, 6, "[REDACTED_EMAIL]", "a@b.co")
    # Only redact the email; IP left as-is.
    assert apply_redactions(text, [email]) == "[REDACTED_EMAIL] and 10.0.0.1"


def test_multiple_redactions_correct_order():
    text = "a@b.co ip 10.0.0.1 x"
    email = _f("email", 0, 6, "[E]", "a@b.co")
    ip = _f("ip_address", 10, 18, "[I]", "10.0.0.1")
    assert apply_redactions(text, [email, ip]) == "[E] ip [I] x"


def test_overlapping_findings_resolved():
    text = "abcdefghij"
    wide = _f("other", 0, 8, "[W]", "abcdefgh")
    narrow = _f("other", 2, 5, "[N]", "cde")
    # Wider span wins; result must not be corrupted.
    out = apply_redactions(text, [wide, narrow])
    assert out == "[W]ij"


def test_invalid_span_rejected():
    text = "short"
    bad = _f("other", 0, 999, "[X]")
    with pytest.raises(RedactionError):
        apply_redactions(text, [bad])


def test_negative_offset_rejected():
    with pytest.raises(RedactionError):
        apply_redactions("abc", [_f("other", -1, 2, "[X]")])


def test_empty_selection_returns_original():
    text = "nothing to do"
    assert apply_redactions(text, []) == text


def test_original_not_mutated():
    text = "a@b.co"
    f = _f("email", 0, 6, "[E]", "a@b.co")
    apply_redactions(text, [f])
    assert text == "a@b.co"


def test_no_offset_corruption_with_many_findings():
    text = "x@y.co A@B.co p@q.co"
    f1 = _f("email", 0, 6, "[1]", "x@y.co")
    f2 = _f("email", 7, 13, "[2]", "A@B.co")
    f3 = _f("email", 14, 20, "[3]", "p@q.co")
    assert apply_redactions(text, [f1, f2, f3]) == "[1] [2] [3]"

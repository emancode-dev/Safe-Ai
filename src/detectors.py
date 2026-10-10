"""Deterministic, pattern-based sensitive-data detection.

This engine works entirely offline and never calls the language model. It is
intentionally conservative: it prefers to miss an ambiguous value over flooding
the user with false positives.

No function here logs or prints the submitted text, and raw matched values are
never embedded in exception messages.
"""

from __future__ import annotations

import re
from typing import List

from .schemas import Finding, METHOD_PATTERN

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Email: pragmatic RFC-ish pattern, good enough for log/config text.
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# IPv4: four octets 0-255. We validate ranges to avoid matching version
# strings like 1.2.3.4567 or arbitrary dotted numbers.
IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
)

# Phone numbers: conservative. Require an international prefix or a clearly
# formatted grouping to reduce false positives against ordinary numbers.
PHONE_RE = re.compile(
    r"(?<![\w.])"
    r"(?:\+\d{1,3}[\s.\-]?)?"            # optional country code
    r"(?:\(\d{2,4}\)[\s.\-]?)?"          # optional area code in parens
    r"\d{3}[\s.\-]\d{3,4}[\s.\-]\d{3,4}" # grouped digits with separators
    r"(?![\w.])"
)

# Common cloud/provider credential patterns.
AWS_ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")
SLACK_TOKEN_RE = re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")
GOOGLE_API_KEY_RE = re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")
STRIPE_KEY_RE = re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[0-9A-Za-z]{10,}\b")

# Private key header markers (PEM style).
PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
)

# Password / secret assignments in config-like text:
#   PASSWORD=... , api_token: "..." , secret_key = '...'
# We require a plausible value (>= 6 chars, not an obvious placeholder word).
SECRET_ASSIGN_RE = re.compile(
    r"(?i)\b([a-z0-9_.\-]*(?:password|passwd|pwd|secret|token|api[_\-]?key|apikey|access[_\-]?key|auth)[a-z0-9_.\-]*)\s*[:=]\s*"
    r"(['\"]?)([^\s'\"]{6,})\2"
)

# Values that look like a placeholder rather than a real secret.
_PLACEHOLDER_VALUES = {
    "changeme", "example", "placeholder", "your_token_here", "yourtoken",
    "xxxxxx", "redacted", "none", "null", "true", "false", "example_value",
    "<your-token>", "secret", "password", "todo",
}


def _add(findings: List[Finding], *, category: str, severity: str, start: int,
         end: int, text: str, reason: str, replacement: str) -> None:
    findings.append(
        Finding(
            category=category,
            severity=severity,
            start=start,
            end=end,
            evidence=text[start:end],
            reason=reason,
            suggested_replacement=replacement,
            method=METHOD_PATTERN,
        )
    )


def detect(text: str) -> List[Finding]:
    """Run all deterministic detectors over ``text`` and return findings.

    Offsets are character indices into ``text``.
    """
    findings: List[Finding] = []
    if not text:
        return findings

    # Private keys (highest severity, check first).
    for m in PRIVATE_KEY_RE.finditer(text):
        _add(findings, category="private_key", severity="high",
             start=m.start(), end=m.end(), text=text,
             reason="PEM private key header detected.",
             replacement="[REDACTED_SECRET]")

    # Cloud / provider credentials.
    for regex, name in (
        (AWS_ACCESS_KEY_RE, "AWS access key ID"),
        (GITHUB_TOKEN_RE, "GitHub token"),
        (SLACK_TOKEN_RE, "Slack token"),
        (GOOGLE_API_KEY_RE, "Google API key"),
        (STRIPE_KEY_RE, "Stripe key"),
    ):
        for m in regex.finditer(text):
            _add(findings, category="cloud_credential", severity="high",
                 start=m.start(), end=m.end(), text=text,
                 reason=f"Matches a known {name} format.",
                 replacement="[REDACTED_SECRET]")

    # Secret assignments (config-style).
    for m in SECRET_ASSIGN_RE.finditer(text):
        value = m.group(3)
        if value.lower() in _PLACEHOLDER_VALUES:
            continue
        # Skip values that are obviously angle-bracket placeholders.
        if value.startswith("<") and value.endswith(">"):
            continue
        v_start = m.start(3)
        v_end = m.end(3)
        _add(findings, category="secret_assignment", severity="high",
             start=v_start, end=v_end, text=text,
             reason="Looks like a secret or credential assigned in configuration-style text.",
             replacement="[REDACTED_SECRET]")

    # Emails.
    for m in EMAIL_RE.finditer(text):
        _add(findings, category="email", severity="medium",
             start=m.start(), end=m.end(), text=text,
             reason="Email address detected.",
             replacement="[REDACTED_EMAIL]")

    # IPv4 addresses.
    for m in IPV4_RE.finditer(text):
        _add(findings, category="ip_address", severity="low",
             start=m.start(), end=m.end(), text=text,
             reason="IPv4 address detected.",
             replacement="[REDACTED_IP]")

    # Phone numbers (conservative).
    for m in PHONE_RE.finditer(text):
        span = text[m.start():m.end()]
        digits = re.sub(r"\D", "", span)
        # Require 7-15 digits to resemble a real phone number.
        if 7 <= len(digits) <= 15:
            _add(findings, category="phone", severity="low",
                 start=m.start(), end=m.end(), text=text,
                 reason="Possible phone number detected.",
                 replacement="[REDACTED_PHONE]")

    return findings

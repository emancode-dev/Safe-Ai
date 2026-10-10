"""Tests for deterministic detection."""

from src import detectors
from src.schemas import METHOD_PATTERN


def _categories(text):
    return {f.category for f in detectors.detect(text)}


def test_email_detection():
    text = "Reach me at dev-example@example.com please."
    findings = [f for f in detectors.detect(text) if f.category == "email"]
    assert len(findings) == 1
    f = findings[0]
    assert text[f.start:f.end] == "dev-example@example.com"
    assert f.method == METHOD_PATTERN


def test_ipv4_detection():
    text = "The staging server is 192.0.2.10 and ok."
    findings = [f for f in detectors.detect(text) if f.category == "ip_address"]
    assert len(findings) == 1
    assert text[findings[0].start:findings[0].end] == "192.0.2.10"


def test_ipv4_rejects_out_of_range():
    # 999 is not a valid octet; should not be detected as an IP.
    assert "ip_address" not in _categories("version 999.1.1.1 build")


def test_phone_conservative_match():
    text = "Call +1 202 555 0143 for support."
    findings = [f for f in detectors.detect(text) if f.category == "phone"]
    assert len(findings) == 1


def test_phone_does_not_flag_plain_number():
    # A bare number without grouping/separators should not be flagged.
    assert "phone" not in _categories("The total was 1234567 units.")


def test_aws_access_key_detection():
    text = "key AKIAIOSFODNN7EXAMPLE here"
    findings = [f for f in detectors.detect(text) if f.category == "cloud_credential"]
    assert len(findings) == 1
    assert text[findings[0].start:findings[0].end] == "AKIAIOSFODNN7EXAMPLE"


def test_github_token_detection():
    text = "token ghp_" + "a" * 36 + " end"
    assert "cloud_credential" in _categories(text)


def test_private_key_header_detection():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIB...\n-----END RSA PRIVATE KEY-----"
    findings = [f for f in detectors.detect(text) if f.category == "private_key"]
    assert len(findings) == 1
    assert findings[0].severity == "high"


def test_secret_assignment_detection():
    text = "db_password=Sup3rSecretValue"
    findings = [f for f in detectors.detect(text) if f.category == "secret_assignment"]
    assert len(findings) == 1
    assert text[findings[0].start:findings[0].end] == "Sup3rSecretValue"


def test_secret_assignment_skips_placeholder():
    text = "api_token=changeme"
    assert "secret_assignment" not in _categories(text)


def test_ordinary_text_not_flagged():
    text = "The quick brown fox jumps over the lazy dog on a sunny afternoon."
    assert detectors.detect(text) == []


def test_empty_text():
    assert detectors.detect("") == []


def test_offsets_are_consistent():
    text = "mail a@b.co and ip 10.0.0.1"
    for f in detectors.detect(text):
        assert text[f.start:f.end] == f.evidence

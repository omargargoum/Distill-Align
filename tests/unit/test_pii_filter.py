"""
Unit tests for the PII/secret scanner and redaction logic.

The filter is security-sensitive: it decides what sensitive data is kept or
redacted before data leaves the user's machine, so detection, exclusion, and
redaction behavior are pinned down here.
"""

import pytest

from distill_align.core.pii_filter import PIIFilter


@pytest.fixture
def pii_filter() -> PIIFilter:
    """A PII filter with default settings."""
    return PIIFilter()


def _finding_types(result) -> list[str]:
    """Return finding types in scan order."""
    return [f.type for f in result.findings]


class TestDetection:
    """Each pattern family should be detected with correct metadata."""

    def test_email(self, pii_filter):
        result = pii_filter.scan_text("Contact alice.smith@corp.example.net please")
        finding = result.findings[0]
        assert finding.type == "email"
        assert finding.category == "pii"
        assert finding.severity == "medium"
        assert finding.value == "alice.smith@corp.example.net"

    def test_us_phone(self, pii_filter):
        result = pii_filter.scan_text("Call (555) 123-4567 now")
        assert _finding_types(result) == ["phone_us"]
        assert result.findings[0].severity == "medium"

    def test_international_phone(self, pii_filter):
        result = pii_filter.scan_text("Reach me at +44 20 7946 0958")
        assert _finding_types(result) == ["phone_intl"]

    def test_ssn(self, pii_filter):
        result = pii_filter.scan_text("Employee SSN is 123-45-6789 on file")
        assert _finding_types(result) == ["ssn"]
        assert result.findings[0].severity == "critical"

    def test_credit_card_pattern_matches(self):
        # The credit_card regex itself matches a 16-digit Visa...
        from distill_align.core.pii_filter import PIIScanner

        assert PIIScanner.CREDIT_CARD.search("4111111111111111") is not None

    def test_plain_card_number_shadowed_by_phone(self, pii_filter):
        # ...but pattern ordering means scan_text classifies a consecutive-digit
        # card as phone_us first (phone_us precedes credit_card in PATTERNS and
        # its 10-digit run overlaps the card span).
        result = pii_filter.scan_text("Card: 4111111111111111 expires soon")
        assert _finding_types(result) == ["phone_us"]

    def test_private_ip(self, pii_filter):
        result = pii_filter.scan_text("DB host 10.0.0.1 is internal")
        assert _finding_types(result) == ["private_ip"]
        assert result.findings[0].severity == "low"

    def test_dob(self, pii_filter):
        result = pii_filter.scan_text("Birth date: 12/31/1990")
        assert _finding_types(result) == ["dob"]
        assert result.findings[0].severity == "medium"

    def test_aws_access_key(self, pii_filter):
        result = pii_filter.scan_text("key AKIAIOSFODNN7EXAMPLE")
        assert _finding_types(result) == ["aws_access_key"]
        assert result.findings[0].severity == "critical"

    def test_aws_secret_key(self, pii_filter):
        result = pii_filter.scan_text("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        assert _finding_types(result) == ["aws_secret_key"]

    def test_github_token(self, pii_filter):
        result = pii_filter.scan_text(f"token ghp_{'a' * 40}")
        assert _finding_types(result) == ["github_token"]

    def test_gitlab_token(self, pii_filter):
        result = pii_filter.scan_text(f"token glpat-{'b' * 24}")
        assert _finding_types(result) == ["gitlab_token"]

    def test_hf_token(self, pii_filter):
        result = pii_filter.scan_text(f"token hf_{'c' * 40}")
        assert _finding_types(result) == ["hf_token"]

    def test_slack_token(self, pii_filter):
        result = pii_filter.scan_text(f"token xoxb-{'d' * 28}")
        assert _finding_types(result) == ["slack_token"]

    def test_stripe_key(self, pii_filter):
        result = pii_filter.scan_text(f"key sk_live_{'e' * 30}")
        assert _finding_types(result) == ["stripe_key"]

    def test_bearer_token(self, pii_filter):
        result = pii_filter.scan_text(f"Authorization: Bearer {'f' * 20}")
        assert _finding_types(result) == ["bearer_token"]
        assert result.findings[0].severity == "high"

    def test_jwt_token(self, pii_filter):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        result = pii_filter.scan_text(f"session {jwt}")
        assert _finding_types(result) == ["jwt_token"]
        assert result.findings[0].severity == "high"

    def test_ssh_private_key(self, pii_filter):
        key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA7eZexampleonlycontent\n-----END RSA PRIVATE KEY-----"
        result = pii_filter.scan_text(key)
        assert _finding_types(result) == ["ssh_private_key"]
        assert result.findings[0].severity == "critical"

    def test_google_api_key(self, pii_filter):
        result = pii_filter.scan_text(f"key AIza{'g' * 35}")
        assert _finding_types(result) == ["google_api_key"]

    def test_connection_string(self, pii_filter):
        result = pii_filter.scan_text("postgres password=hunter2secret1 host=db")
        assert _finding_types(result) == ["connection_string"]
        assert result.findings[0].severity == "high"


class TestFalsePositivesAndOverlaps:
    """Exclusion and overlap handling should prevent noisy or duplicate findings."""

    def test_example_dot_com_email_excluded(self, pii_filter):
        result = pii_filter.scan_text("Contact user@example.com for details")
        assert result.findings == []

    def test_192_168_ip_excluded(self, pii_filter):
        result = pii_filter.scan_text("Server at 192.168.1.100")
        assert result.findings == []

    def test_ssn_shaped_phone_reports_phone_not_ssn(self, pii_filter):
        # 555-123-4567 matches both patterns; the earlier phone_us wins.
        result = pii_filter.scan_text("Call 555-123-4567 today")
        assert _finding_types(result) == ["phone_us"]

    def test_multiple_findings_sorted_by_position(self, pii_filter):
        result = pii_filter.scan_text("Email alice@corp.org on (555) 123-4567")
        types = _finding_types(result)
        assert types == ["email", "phone_us"]
        starts = [f.start for f in result.findings]
        assert starts == sorted(starts)

    def test_empty_text(self, pii_filter):
        result = pii_filter.scan_text("")
        assert result.total_findings == 0
        assert result.redacted_text == ""


class TestCategoryToggles:
    """PII and secret scanning can be enabled/disabled independently."""

    def test_disable_pii(self):
        filtered = PIIFilter(enable_pii=False)
        result = filtered.scan_text(f"email alice@corp.org token ghp_{'a' * 40}")
        assert _finding_types(result) == ["github_token"]

    def test_disable_secrets(self):
        filtered = PIIFilter(enable_secrets=False)
        result = filtered.scan_text(f"email alice@corp.org token ghp_{'a' * 40}")
        assert _finding_types(result) == ["email"]

    def test_disable_redaction(self):
        filtered = PIIFilter(redact=False)
        text = "Email alice@corp.org today"
        result = filtered.scan_text(text)
        assert result.total_findings == 1
        assert result.redacted_text == text


class TestRedaction:
    """Redaction should replace findings without disturbing surrounding text."""

    def test_redact_text(self, pii_filter):
        redacted = pii_filter.redact_text("Email alice@corp.org today")
        assert redacted == "Email [REDACTED: email] today"

    def test_custom_placeholder(self):
        filtered = PIIFilter(redact_placeholder="<{type}>")
        redacted = filtered.redact_text("Email alice@corp.org today")
        assert redacted == "Email <email> today"

    def test_repeated_findings_all_redacted(self, pii_filter):
        redacted = pii_filter.redact_text("a@b.co and c@d.co")
        assert redacted == "[REDACTED: email] and [REDACTED: email]"


class TestResultHelpers:
    """Summary helpers should reflect severity counts."""

    def test_has_critical_and_high(self):
        filtered = PIIFilter()
        result = filtered.scan_text("SSN 123-45-6789 Bearer ffffffffffffffffffff")
        assert result.has_critical
        assert result.has_high
        assert result.critical_count == 1
        assert result.high_count == 1

    def test_summary(self, tmp_path):
        filtered = PIIFilter()
        result = filtered.scan_text("SSN 123-45-6789 on 10.0.0.1")
        assert result.summary == "PII scan: 2 findings (1 critical, 1 low)"

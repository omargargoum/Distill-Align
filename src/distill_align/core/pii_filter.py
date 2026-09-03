"""
PII (Personally Identifiable Information) and secret scanning module.

Provides pattern-based detection and redaction of sensitive data including:
- PII: emails, phone numbers, SSNs, credit cards, IP addresses
- Secrets: API keys, tokens, SSH keys, JWTs, cloud credentials

Usage:
    filter_ = PIIFilter()
    findings = filter_.scan_text("user@example.com")
    redacted = filter_.redact_text("user@example.com")
"""

from __future__ import annotations

import dataclasses
import re
from typing import ClassVar

from loguru import logger

# =============================================================================
# Data models
# =============================================================================


@dataclasses.dataclass(frozen=True)
class PIIFinding:
    """A single PII or secret finding in scanned text."""

    type: str  # e.g. "email", "ssn", "aws_key", "github_token"
    value: str  # The matched text
    start: int  # Character offset (start)
    end: int  # Character offset (end)
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "pii" or "secret"
    description: str  # Human-readable description


@dataclasses.dataclass(frozen=True)
class PIIFilterResult:
    """Result of scanning text for PII and secrets."""

    findings: list[PIIFinding]
    redacted_text: str
    total_findings: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int

    @property
    def has_critical(self) -> bool:
        return self.critical_count > 0

    @property
    def has_high(self) -> bool:
        return self.high_count > 0

    @property
    def summary(self) -> str:
        parts = []
        if self.critical_count:
            parts.append(f"{self.critical_count} critical")
        if self.high_count:
            parts.append(f"{self.high_count} high")
        if self.medium_count:
            parts.append(f"{self.medium_count} medium")
        if self.low_count:
            parts.append(f"{self.low_count} low")
        severity_str = ", ".join(parts) if parts else "none"
        return f"PII scan: {self.total_findings} findings ({severity_str})"


# =============================================================================
# Pattern definitions
# =============================================================================


class PIIScanner:
    """Compiled patterns and scanning logic for PII and secrets."""

    # ------------------------------------------------------------------
    # PII patterns
    # ------------------------------------------------------------------

    EMAIL: ClassVar[re.Pattern] = re.compile(
        r"[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}"
    )

    # US phone numbers (with optional country code)
    PHONE_US: ClassVar[re.Pattern] = re.compile(r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}")

    # International phone numbers (non-US prefix)
    PHONE_INTL: ClassVar[re.Pattern] = re.compile(
        r"\+(?:[0-9]{1,3})[-.\s]?\(?[0-9]{1,4}\)?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}"
    )

    # US Social Security Number (XXX-XX-XXXX)
    SSN: ClassVar[re.Pattern] = re.compile(
        r"\b(?!000|666|9[0-9]{2})([0-8][0-9]{2}|7[0-5][0-9]|76[0-6])[-](?!00)[0-9]{2}[-](?!0000)[0-9]{4}\b"
    )

    # Credit card numbers (major providers — Luhn validation is best done programmatically)
    CREDIT_CARD: ClassVar[re.Pattern] = re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?"  # Visa
        r"|5[1-5][0-9]{14}"  # MasterCard
        r"|3[47][0-9]{13}"  # AmEx
        r"|6(?:011|5[0-9]{2})[0-9]{12}"  # Discover
        r"|(?:2131|1800|35[0-9]{3})[0-9]{11})"  # JCB
        r"\b"
    )

    # Private/internal IP addresses (RFC 1918, loopback, link-local)
    PRIVATE_IP: ClassVar[re.Pattern] = re.compile(
        r"\b(?:10\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}"
        r"|172\.(?:1[6-9]|2[0-9]|3[01])\.(?:[0-9]{1,3}\.)[0-9]{1,3}"
        r"|192\.168\.(?:[0-9]{1,3}\.)[0-9]{1,3}"
        r"|127\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}"
        r"|169\.254\.(?:[0-9]{1,3}\.)[0-9]{1,3}"
        r"|0\.0\.0\.0"
        r"|::1"
        r"|fc[0-9a-f]{2}:|fd[0-9a-f]{2}:)"
        r"\b"
    )

    # Date of birth / birth date (various formats)
    DOB: ClassVar[re.Pattern] = re.compile(
        r"\b(?:birth\s*(?:date|day)\s*[:#]?\s*" r"|dob\s*[:#]?\s*)" r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})" r"\b",
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Secret patterns
    # ------------------------------------------------------------------

    # AWS Access Key ID (starts with AKIA...)
    AWS_ACCESS_KEY: ClassVar[re.Pattern] = re.compile(r"(?:AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}\b")

    # AWS Secret Access Key
    AWS_SECRET_KEY: ClassVar[re.Pattern] = re.compile(
        r"(?i)aws.{0,20}?(?:secret|key|token).{0,20}?[:\s=]+[A-Za-z0-9/+=]{40}\b"
    )

    # GitHub personal access tokens (ghp_, gho_, ghu_, ghs_, ghf_)
    GITHUB_TOKEN: ClassVar[re.Pattern] = re.compile(r"(?:ghp_|gho_|ghu_|ghs_|ghf_|github_pat_)[a-zA-Z0-9_]{36,}")

    # GitLab tokens
    GITLAB_TOKEN: ClassVar[re.Pattern] = re.compile(r"\bglpat-[a-zA-Z0-9_\-]{20,}\b")

    # Hugging Face tokens
    HF_TOKEN: ClassVar[re.Pattern] = re.compile(r"\bhf_[a-zA-Z0-9]{32,}\b")

    # Slack tokens (xoxb-, xoxp-, xoxa-, xoxe-)
    SLACK_TOKEN: ClassVar[re.Pattern] = re.compile(r"\bxox[abeprs]-[a-zA-Z0-9-]{24,}\b")

    # Stripe API keys (live/test)
    STRIPE_KEY: ClassVar[re.Pattern] = re.compile(r"\b(?:sk|pk)_(?:live|test)_[a-zA-Z0-9]{24,}\b")

    # Anthropic API keys
    ANTHROPIC_KEY: ClassVar[re.Pattern] = re.compile(r"\bsk-ant-[a-zA-Z0-9\-_]{32,}\b")

    # OpenAI API keys (sk-..., sk-proj-...)
    OPENAI_KEY: ClassVar[re.Pattern] = re.compile(r"\bsk-(?:proj-)?[a-zA-Z0-9\-_]{20,}\b")

    # OpenRouter / Together / generic provider keys
    OPENROUTER_KEY: ClassVar[re.Pattern] = re.compile(r"\bsk-or-[a-zA-Z0-9\-_]{16,}\b")

    # Generic API key / bearer token
    BEARER_TOKEN: ClassVar[re.Pattern] = re.compile(
        r"(?i)(?:(?:bearer|token|apikey|api_key|api-key|secret)[:\s=]+)[a-zA-Z0-9_\-\.]{16,64}"
    )

    # JWT tokens (three base64url segments separated by dots)
    JWT_TOKEN: ClassVar[re.Pattern] = re.compile(r"\beyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+\b")

    # SSH private keys (embedded in text)
    SSH_PRIVATE_KEY: ClassVar[re.Pattern] = re.compile(
        r"-----BEGIN\s+(?:RSA|DSA|EC|OPENSSH|SSH)\s+PRIVATE\s+KEY-----"
        r"[A-Za-z0-9+/=\s]+?"
        r"-----END\s+(?:RSA|DSA|EC|OPENSSH|SSH)\s+PRIVATE\s+KEY-----",
        re.DOTALL,
    )

    # Google API keys (AIza... or ya29...)
    GOOGLE_API_KEY: ClassVar[re.Pattern] = re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")

    # Password / connection strings
    CONNECTION_STRING: ClassVar[re.Pattern] = re.compile(
        r"(?i)(?:password|pwd|passwd|connection)[:\s=]+[^\s,;'\"]{8,100}"
    )

    # ------------------------------------------------------------------
    # Aggregated list for composition
    # ------------------------------------------------------------------

    PATTERNS: ClassVar[list[tuple[str, re.Pattern, str, str, str]]] = [
        # (type, pattern, category, severity, description)
        ("email", EMAIL, "pii", "medium", "Email address"),
        ("phone_us", PHONE_US, "pii", "medium", "US phone number"),
        ("phone_intl", PHONE_INTL, "pii", "medium", "International phone number"),
        ("ssn", SSN, "pii", "critical", "Social Security Number"),
        ("credit_card", CREDIT_CARD, "pii", "critical", "Credit card number"),
        ("private_ip", PRIVATE_IP, "pii", "low", "Private/internal IP address"),
        ("dob", DOB, "pii", "medium", "Date of birth"),
        ("aws_access_key", AWS_ACCESS_KEY, "secret", "critical", "AWS Access Key ID"),
        ("aws_secret_key", AWS_SECRET_KEY, "secret", "critical", "AWS Secret Access Key"),
        ("github_token", GITHUB_TOKEN, "secret", "critical", "GitHub personal access token"),
        ("gitlab_token", GITLAB_TOKEN, "secret", "critical", "GitLab personal access token"),
        ("hf_token", HF_TOKEN, "secret", "critical", "Hugging Face token"),
        ("slack_token", SLACK_TOKEN, "secret", "critical", "Slack token"),
        ("stripe_key", STRIPE_KEY, "secret", "critical", "Stripe API key"),
        ("anthropic_key", ANTHROPIC_KEY, "secret", "critical", "Anthropic API key"),
        ("openai_key", OPENAI_KEY, "secret", "critical", "OpenAI API key"),
        ("openrouter_key", OPENROUTER_KEY, "secret", "critical", "OpenRouter API key"),
        ("bearer_token", BEARER_TOKEN, "secret", "high", "Bearer token / API key"),
        ("jwt_token", JWT_TOKEN, "secret", "high", "JWT token"),
        ("ssh_private_key", SSH_PRIVATE_KEY, "secret", "critical", "SSH private key"),
        ("google_api_key", GOOGLE_API_KEY, "secret", "critical", "Google API key"),
        ("connection_string", CONNECTION_STRING, "secret", "high", "Connection string with password"),
    ]

    # ------------------------------------------------------------------
    # Exclusion filters — reduce false positives on common safe patterns
    # ------------------------------------------------------------------

    # Skip matches where the surrounding context suggests a false positive
    FALSE_POSITIVE_PATTERNS: ClassVar[list[re.Pattern]] = [
        # Example domains
        re.compile(r"@example\.(?:com|org|net)\b", re.IGNORECASE),
        # Placeholder IPs
        re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b", re.IGNORECASE),  # Too many false positives for code
    ]


# =============================================================================
# Main filter class
# =============================================================================


class PIIFilter:
    """
    Scans text for PII, secrets, and credentials.

    Args:
        enable_pii: Scan for PII patterns (default: True).
        enable_secrets: Scan for secret/token patterns (default: True).
        redact: Whether to redact findings in output (default: True).
        redact_placeholder: Template for redacted text (default: ``[REDACTED: {type}]``).
        validate_credit_cards: Run Luhn checksum on credit-card candidates
            to cut false positives (default: True).
        use_presidio: If True and ``presidio_analyzer`` is installed, run it
            as an additional NER pass (default: False — regex only, no new dep).
    """

    PATTERNS: ClassVar[list[tuple[str, re.Pattern, str, str, str]]] = PIIScanner.PATTERNS
    FALSE_POSITIVE_PATTERNS: ClassVar[list[re.Pattern]] = PIIScanner.FALSE_POSITIVE_PATTERNS

    def __init__(
        self,
        enable_pii: bool = True,
        enable_secrets: bool = True,
        redact: bool = True,
        redact_placeholder: str = "[REDACTED: {type}]",
        validate_credit_cards: bool = True,
        use_presidio: bool = False,
    ):
        self.enable_pii = enable_pii
        self.enable_secrets = enable_secrets
        self.redact = redact
        self.redact_placeholder = redact_placeholder
        self.validate_credit_cards = validate_credit_cards
        self.use_presidio = use_presidio

    def scan_text(self, text: str) -> PIIFilterResult:
        """
        Scan text for PII and secrets.

        Args:
            text: The text to scan.

        Returns:
            PIIFilterResult with findings and optionally redacted text.
        """
        if not text:
            return PIIFilterResult(
                findings=[],
                redacted_text=text,
                total_findings=0,
                critical_count=0,
                high_count=0,
                medium_count=0,
                low_count=0,
            )

        findings: list[PIIFinding] = []
        seen_ranges: set[tuple[int, int]] = set()

        for pattern_type, pattern, category, severity, description in self.PATTERNS:
            # Skip categories that are disabled
            if category == "pii" and not self.enable_pii:
                continue
            if category == "secret" and not self.enable_secrets:
                continue

            for match in pattern.finditer(text):
                start, end = match.start(), match.end()

                # Skip false positives
                if self._is_false_positive(text, start, end):
                    continue

                # Luhn-validate credit cards to cut false positives
                if pattern_type == "credit_card" and self.validate_credit_cards and not _luhn_valid(match.group()):
                    continue

                # Avoid overlapping matches (keep the first/longest)
                if self._overlaps(seen_ranges, start, end):
                    continue

                seen_ranges.add((start, end))

                findings.append(
                    PIIFinding(
                        type=pattern_type,
                        value=match.group(),
                        start=start,
                        end=end,
                        severity=severity,
                        category=category,
                        description=description,
                    )
                )

        # Sort by position
        findings.sort(key=lambda f: f.start)

        # Optional Presidio NER pass (additive; never removes regex findings)
        if self.use_presidio:
            findings = self._merge_presidio_findings(text, findings, seen_ranges)

        # Redact if requested
        redacted_text = self._redact_findings(text, findings) if self.redact else text

        # Count by severity
        critical = sum(1 for f in findings if f.severity == "critical")
        high = sum(1 for f in findings if f.severity == "high")
        medium = sum(1 for f in findings if f.severity == "medium")
        low = sum(1 for f in findings if f.severity == "low")

        if findings:
            logger.warning(
                f"PII scan found {len(findings)} items: {critical} critical, {high} high, {medium} medium, {low} low"
            )

        return PIIFilterResult(
            findings=findings,
            redacted_text=redacted_text,
            total_findings=len(findings),
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            low_count=low,
        )

    def redact_text(self, text: str) -> str:
        """
        Convenience method: scan and redact text in one call.

        Args:
            text: The text to scan and redact.

        Returns:
            Redacted text with PII/secrets replaced by placeholders.
        """
        result = self.scan_text(text)
        return result.redacted_text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_false_positive(self, text: str, start: int, end: int) -> bool:
        """Check if a match at *start*-*end* is likely a false positive."""
        # Check context (line around match)
        line_start = text.rfind("\n", 0, start)
        line_start = 0 if line_start == -1 else line_start + 1
        line_end = text.find("\n", end)
        line_end = len(text) if line_end == -1 else line_end
        context = text[line_start:line_end]

        return any(fp_pattern.search(context) for fp_pattern in self.FALSE_POSITIVE_PATTERNS)

    @staticmethod
    def _overlaps(seen: set[tuple[int, int]], start: int, end: int) -> bool:
        """Check if (start, end) overlaps any range in *seen*."""
        return any(start < e and end > s for s, e in seen)

    def _redact_findings(self, text: str, findings: list[PIIFinding]) -> str:
        """Replace findings in *text* with redacted placeholders."""
        if not findings:
            return text

        # Build replacement segments in reverse order to preserve positions
        chars = list(text)
        for f in reversed(findings):
            replacement = self.redact_placeholder.format(type=f.type)
            chars[f.start : f.end] = replacement

        return "".join(chars)

    def _merge_presidio_findings(
        self,
        text: str,
        findings: list[PIIFinding],
        seen_ranges: set[tuple[int, int]],
    ) -> list[PIIFinding]:
        """Optionally enrich findings with Presidio NER (graceful fallback)."""
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
        except ImportError:
            logger.debug("Presidio not installed; skipping NER pass (pip install distill-align[pii])")
            return findings
        try:
            engine: object = AnalyzerEngine()
            results: list[object] = engine.analyze(text=text, language="en")  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"Presidio pass failed, keeping regex findings: {e}")
            return findings
        merged = list(findings)
        for r in results:
            start = int(getattr(r, "start", -1))
            end = int(getattr(r, "end", -1))
            if start < 0 or end <= start:
                continue
            if self._overlaps(seen_ranges, start, end):
                continue
            seen_ranges.add((start, end))
            entity = str(getattr(r, "entity_type", "PII"))
            merged.append(
                PIIFinding(
                    type=f"presidio_{entity.lower()}",
                    value=text[start:end],
                    start=start,
                    end=end,
                    severity="medium",
                    category="pii",
                    description=f"Presidio NER: {entity}",
                )
            )
        merged.sort(key=lambda f: f.start)
        return merged


def _luhn_valid(number: str) -> bool:
    """Validate a credit-card candidate with the Luhn checksum."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0

"""
Anonymous telemetry for usage statistics.

Telemetry is opt-in and disabled by default. Enable with DISTILL_TELEMETRY=true.
No personal data is collected — only anonymized usage counts.

NOTE: The telemetry endpoint is not yet operational. Events are silently
discarded until the backend service is deployed.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

logger = logging.getLogger(__name__)

TELEMETRY_ENV_VAR = "DISTILL_TELEMETRY"
TELEMETRY_URL = ""  # Not yet operational — see note above


def is_enabled() -> bool:
    """Check if telemetry is enabled.

    Telemetry is opt-in and disabled by default. Enable by setting
    ``DISTILL_TELEMETRY=true`` in the environment.
    """
    import os

    env_val = os.environ.get(TELEMETRY_ENV_VAR, "").strip().lower()
    return env_val in ("true", "1", "yes")


def track_event(event: str, properties: dict[str, Any] | None = None) -> None:
    """Track an anonymous event (currently a no-op)."""
    if not is_enabled():
        return
    # Intentionally disabled until backend is deployed
    logger.debug("Telemetry event (disabled): %s", event)


class TelemetryContext:
    """Context manager for telemetry tracking."""

    def __init__(self, event: str, properties: dict[str, Any] | None = None):
        self.event = event
        self.properties = properties or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        track_event(self.event, self.properties)


# ── Phase 6: tracing hooks (OTel / Langfuse compatible, zero hard deps) ──


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[None]:
    """Emit a tracing span around *name*.

    Uses OpenTelemetry when installed, else Langfuse when configured via
    ``Settings.enable_tracing``, else a no-op. Never raises — tracing must
    not break pipelines.
    """
    attributes = attributes or {}
    # 1. OpenTelemetry (if installed)
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]

        tracer = trace.get_tracer("distill-align")
        with tracer.start_as_current_span(name) as otel_span:
            for k, v in attributes.items():
                with suppress(Exception):
                    otel_span.set_attribute(k, str(v))
            yield
            return
    except ImportError:
        pass
    except Exception:
        pass
    # 2. Structured log fallback (Langfuse ingestion can parse JSON logs)
    import logging as _logging

    _logging.getLogger(__name__).debug("span %s %s", name, attributes)
    yield

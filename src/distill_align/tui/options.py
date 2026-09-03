"""Shared TUI option helpers — single source of truth for dropdowns.

Every Select in the TUI derives its choices from the same modules the CLI
and pipelines use (registry, catalog, ConversationMode, FORMATTER_MAP,
IngestionConfig), so new providers / models / modes / formats / chunkers
appear in the dashboard automatically. All helpers are pure functions —
importable and testable without running the Textual app.
"""

from __future__ import annotations

from typing import get_args


def conversation_mode_choices() -> list[tuple[str, str]]:
    """``(value, label)`` pairs for conversation-mode dropdowns.

    ``"default"`` (legacy Socratic path) first, then every ConversationMode.
    """
    from ..synthesis.conversation_builder import ConversationMode

    return [("default", "default")] + [(m.value, m.value) for m in ConversationMode]


def valid_conversation_modes() -> set[str]:
    """All accepted ``--mode`` values (including ``"default"``)."""
    return {value for value, _ in conversation_mode_choices()}


def export_format_choices() -> list[tuple[str, str]]:
    """``(value, label)`` pairs for export-format dropdowns (sharegpt first)."""
    from ..exporter.pipeline import FORMATTER_MAP

    ordered = ["sharegpt", *[f for f in FORMATTER_MAP if f != "sharegpt"]]
    return [(f, f) for f in ordered]


def valid_export_formats() -> set[str]:
    """All accepted ``--format`` values."""
    from ..exporter.pipeline import FORMATTER_MAP

    return set(FORMATTER_MAP)


def _ordered_chunkers() -> list[str]:
    """Chunker names in ``IngestionConfig`` Literal order (``"auto"`` first)."""
    from ..core.schemas import IngestionConfig

    annotation = IngestionConfig.model_fields["chunker"].annotation
    ordered: list[str] = []
    for a in get_args(annotation):
        if isinstance(a, str) and a not in ordered:
            ordered.append(a)
    return ordered or ["auto", "markdown", "code", "recursive", "semantic", "parent_child", "late"]


def chunker_choices() -> list[tuple[str, str]]:
    """``(value, label)`` pairs for chunker dropdowns (``"auto"`` first)."""
    return [(c, c) for c in _ordered_chunkers()]


def valid_chunkers() -> set[str]:
    """All accepted chunker names, derived from ``IngestionConfig``."""
    return set(_ordered_chunkers())


def provider_choices() -> list[tuple[str, str]]:
    """``(display, value)`` pairs for provider dropdowns (registry-driven)."""
    from ..synthesis.models.registry import list_select_choices

    return list_select_choices()


def default_model_for(provider: str) -> str:
    """Catalog production default model for *provider* (fallback: gpt-5-mini)."""
    from ..synthesis.models.registry import get as get_provider_info

    info = get_provider_info(provider)
    if info and info.default_model:
        return info.default_model
    return "gpt-5-mini"


def validate_mode(mode: str) -> bool:
    """True when *mode* is a known conversation mode."""
    return mode in valid_conversation_modes()


def validate_export_format(fmt: str) -> bool:
    """True when *fmt* is a known export format."""
    return fmt in valid_export_formats()


def validate_chunker(chunker: str) -> bool:
    """True when *chunker* is a known chunker name."""
    return chunker in valid_chunkers()

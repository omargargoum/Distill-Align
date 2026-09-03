"""Shared mode utilities and template definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Mode Constants ──────────────────────────────────────────────────────────

MODE_SIMPLE = "simple"
MODE_EXPERT = "expert"

MODE_LABELS = {
    MODE_SIMPLE: "🌱 Simple",
    MODE_EXPERT: "🧪 Expert",
}


# ── Pipeline Templates ──────────────────────────────────────────────────────


@dataclass
class PipelineTemplate:
    """A named preset that pre-fills pipeline forms with sensible defaults."""

    name: str
    description: str
    icon: str = "📋"
    config: dict[str, Any] = field(default_factory=dict)


# Well-known templates users can pick from
BUILTIN_TEMPLATES: list[PipelineTemplate] = [
    PipelineTemplate(
        name="Simple Q&A from docs",
        description="Turn markdown or text documents into question-answer conversation pairs. Great for documentation, articles, and guides.",
        icon="🗣️",
        config={
            "chunk_size": 1500,
            "overlap": 200,
            "recursive": True,
            "auto_detect": True,
            "provider": "openai",
            "model": "gpt-5-mini",
            "concurrency": 5,
            "rpm": 60,
            "mode": "qa",
            "cache": True,
            "checkpoint": True,
            "format": "sharegpt",
        },
    ),
    PipelineTemplate(
        name="Code instruction dataset",
        description="Generate instruction-following pairs from code repositories. Best for fine-tuning on code understanding and generation.",
        icon="💻",
        config={
            "chunk_size": 1000,
            "overlap": 150,
            "recursive": True,
            "auto_detect": True,
            "provider": "openai",
            "model": "gpt-5-mini",
            "concurrency": 3,
            "rpm": 30,
            "mode": "default",
            "cache": True,
            "checkpoint": True,
            "format": "sharegpt",
        },
    ),
    PipelineTemplate(
        name="Multi-turn reasoning",
        description="Create deep multi-turn conversations that show step-by-step reasoning. Ideal for teaching models to think through problems.",
        icon="📚",
        config={
            "chunk_size": 2000,
            "overlap": 300,
            "recursive": True,
            "auto_detect": True,
            "provider": "openai",
            "model": "gpt-5-mini",
            "concurrency": 3,
            "rpm": 30,
            "mode": "teach",
            "cache": True,
            "checkpoint": True,
            "format": "chatml",
        },
    ),
    PipelineTemplate(
        name="Explain like I'm 5",
        description="Transform complex technical content into simple, accessible explanations with analogies.",
        icon="🎓",
        config={
            "chunk_size": 1200,
            "overlap": 200,
            "recursive": True,
            "auto_detect": True,
            "provider": "openai",
            "model": "gpt-5-mini",
            "concurrency": 5,
            "rpm": 60,
            "mode": "explain",
            "cache": True,
            "checkpoint": True,
            "format": "sharegpt",
        },
    ),
    PipelineTemplate(
        name="Lightning fast (Ollama)",
        description="Use a local Ollama model for fast, free generation. No API keys needed — perfect for prototyping.",
        icon="⚡",
        config={
            "chunk_size": 1000,
            "overlap": 200,
            "recursive": True,
            "auto_detect": True,
            "provider": "ollama",
            "model": "qwen3:30b",
            "concurrency": 4,
            "rpm": 999,
            "mode": "default",
            "cache": True,
            "checkpoint": True,
            "format": "sharegpt",
        },
    ),
    PipelineTemplate(
        name="Production-grade (judged)",
        description="Full pipeline with LLM-as-judge evaluation for quality scoring. Slower but produces higher-quality, scored datasets.",
        icon="🏆",
        config={
            "chunk_size": 1000,
            "overlap": 200,
            "recursive": True,
            "auto_detect": True,
            "provider": "openai",
            "model": "gpt-5-mini",
            "concurrency": 3,
            "rpm": 30,
            "mode": "default",
            "judge": True,
            "cache": True,
            "checkpoint": True,
            "format": "sharegpt",
            "card": True,
        },
    ),
]


def get_template(name: str) -> PipelineTemplate | None:
    """Look up a template by name."""
    for t in BUILTIN_TEMPLATES:
        if t.name == name:
            return t
    return None

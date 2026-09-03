"""Provider registry for LLM backends.

Allows registering both built-in and custom providers, and looking them up
by name across the CLI, TUI, config, and pre-flight systems.

Usage:
    from distill_align.synthesis.models.registry import register, get, list_all

    # Built-ins are registered at import time via register_builtins()
    # Custom providers can be registered from config files:
    register(ProviderInfo(
        name="groq",
        label="Groq",
        api_format="openai",
        env_vars=["GROQ_API_KEY"],
        default_base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
    ))
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProviderInfo:
    """Metadata about an LLM provider in the registry.

    Attributes:
        name: Internal key used in configs and the TUI (e.g. ``"openai"``).
        label: Human-readable display name (e.g. ``"OpenAI"``).
        api_format: The wire-protocol family: ``"openai"``, ``"anthropic"``,
            ``"ollama"``, or ``"vllm"``.  Determines which client class is used.
        env_vars: Environment variable(s) to check for an API key, in priority
            order.  Empty list means no key is required.
        default_base_url: Base URL used when none is explicitly provided.
        default_model: Model name to pre-fill in forms.
        is_builtin: True for providers shipped with the package.
        requires_api_key: Whether this provider needs an API key to operate.
        concurrency_limit: Suggested max concurrency for this provider.
        rpm_limit: Suggested max requests-per-minute for this provider.
    """

    name: str
    label: str
    api_format: str = "openai"
    env_vars: list[str] = field(default_factory=list)
    default_base_url: str = ""
    default_model: str = ""
    is_builtin: bool = True
    requires_api_key: bool = True
    concurrency_limit: int = 10
    rpm_limit: int = 60


# ── Module-level registry ───────────────────────────────────────────────────

_registry: dict[str, ProviderInfo] = {}
_alias_registry: dict[str, str] = {}  # alias → canonical name


def register(info: ProviderInfo, alias: str | None = None) -> None:
    """Register a provider.

    Args:
        info: Provider metadata.
        alias: An optional alias (e.g. ``"openai-compatible"`` → ``"openai"``).
    """
    _registry[info.name] = info
    if alias:
        _alias_registry[alias] = info.name


def get(name: str) -> ProviderInfo | None:
    """Look up a provider by name (or alias)."""
    canonical = _alias_registry.get(name)
    return _registry.get(canonical or name)


def list_all() -> list[ProviderInfo]:
    """Return every registered provider (built-in + custom)."""
    return list(_registry.values())


def list_names() -> list[str]:
    """Return every registered provider name as a sorted list."""
    return sorted(_registry.keys())


def list_select_choices() -> list[tuple[str, str]]:
    """Return ``(display, value)`` pairs suitable for a TUI ``Select`` widget."""
    return [(p.label, p.name) for p in list_all()]


def clear_custom() -> None:
    """Remove all non-built-in providers (e.g. before re-loading config)."""
    to_remove = [name for name, p in _registry.items() if not p.is_builtin]
    for name in to_remove:
        del _registry[name]


# ── Built-in providers ──────────────────────────────────────────────────────


def register_builtins() -> None:
    """Register the built-in providers shipped with Distill-Align.

    Per-provider ``default_model`` values track :mod:`catalog.PROVIDER_DEFAULTS`
    (Sep-2026 production defaults). Safe to call multiple times — idempotent.
    """
    # Avoid double-registration
    if "openai" in _registry:
        return

    from .catalog import PROVIDER_DEFAULTS

    register(
        ProviderInfo(
            name="openai",
            label="OpenAI",
            api_format="openai",
            env_vars=["OPENAI_API_KEY", "DISTILL_LLM_API_KEY"],
            default_base_url="https://api.openai.com/v1",
            default_model=PROVIDER_DEFAULTS["openai"],
            concurrency_limit=10,
            rpm_limit=60,
        ),
        alias="openai-compatible",
    )

    register(
        ProviderInfo(
            name="anthropic",
            label="Anthropic Claude",
            api_format="anthropic",
            env_vars=["ANTHROPIC_API_KEY"],
            default_base_url="https://api.anthropic.com/v1",
            default_model=PROVIDER_DEFAULTS["anthropic"],
            concurrency_limit=10,
            rpm_limit=50,
        ),
        alias="anthropic-compatible",
    )

    register(
        ProviderInfo(
            name="gemini",
            label="Google Gemini",
            api_format="gemini",
            env_vars=["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            default_base_url="https://generativelanguage.googleapis.com/v1beta",
            default_model=PROVIDER_DEFAULTS["gemini"],
            concurrency_limit=15,
            rpm_limit=120,
        ),
    )

    register(
        ProviderInfo(
            name="azure",
            label="Azure OpenAI",
            api_format="openai",
            env_vars=["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
            default_base_url="https://api.openai.azure.com",
            default_model=PROVIDER_DEFAULTS["azure"],
            concurrency_limit=10,
            rpm_limit=60,
        ),
    )

    register(
        ProviderInfo(
            name="ollama",
            label="Ollama (local)",
            api_format="ollama",
            env_vars=[],
            default_base_url="http://localhost:11434",
            default_model=PROVIDER_DEFAULTS["ollama"],
            requires_api_key=False,
            concurrency_limit=8,
            rpm_limit=999,
        ),
    )

    register(
        ProviderInfo(
            name="vllm",
            label="vLLM (local)",
            api_format="vllm",
            env_vars=[],
            default_base_url="http://localhost:8000/v1",
            default_model=PROVIDER_DEFAULTS["vllm"],
            requires_api_key=False,
            concurrency_limit=16,
            rpm_limit=999,
        ),
    )

    register(
        ProviderInfo(
            name="qwen",
            label="Alibaba Qwen (DashScope)",
            api_format="openai",
            env_vars=["DASHSCOPE_API_KEY", "QWEN_API_KEY", "DISTILL_LLM_API_KEY"],
            default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            default_model=PROVIDER_DEFAULTS["qwen"],
            concurrency_limit=10,
            rpm_limit=120,
        ),
    )

    # ── 2026 gateway / hosted open-weight providers (all OpenAI-compatible) ──
    gateways = [
        (
            "openrouter",
            "OpenRouter (gateway)",
            ["OPENROUTER_API_KEY", "DISTILL_LLM_API_KEY"],
            "https://openrouter.ai/api/v1",
            10,
            120,
        ),
        ("litellm", "LiteLLM Proxy", ["LITELLM_API_KEY", "DISTILL_LLM_API_KEY"], "http://localhost:4000/v1", 12, 300),
        (
            "together",
            "Together AI",
            ["TOGETHER_API_KEY", "DISTILL_LLM_API_KEY"],
            "https://api.together.xyz/v1",
            10,
            120,
        ),
        ("groq", "Groq", ["GROQ_API_KEY", "DISTILL_LLM_API_KEY"], "https://api.groq.com/openai/v1", 10, 120),
        ("mistral", "Mistral AI", ["MISTRAL_API_KEY", "DISTILL_LLM_API_KEY"], "https://api.mistral.ai/v1", 10, 120),
        ("deepseek", "DeepSeek", ["DEEPSEEK_API_KEY", "DISTILL_LLM_API_KEY"], "https://api.deepseek.com/v1", 10, 120),
        (
            "cohere",
            "Cohere",
            ["COHERE_API_KEY", "DISTILL_LLM_API_KEY"],
            "https://api.cohere.ai/compatibility/v1",
            10,
            120,
        ),
    ]
    for name, label, env_vars, base_url, conc, rpm in gateways:
        register(
            ProviderInfo(
                name=name,
                label=label,
                api_format="openai",
                env_vars=env_vars,
                default_base_url=base_url,
                default_model=PROVIDER_DEFAULTS[name],
                concurrency_limit=conc,
                rpm_limit=rpm,
            ),
        )


# ── Import-time initialisation ──────────────────────────────────────────────

register_builtins()

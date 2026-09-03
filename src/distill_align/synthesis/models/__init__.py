"""LLM model integrations for various providers.

The :mod:`~.registry` module provides a provider registry that maps names
(e.g. ``"openai"``, ``"anthropic"``) to metadata and determines which client
class to use.  Built-in providers are registered at import time; custom
providers can be added from config files.

The :mod:`~.catalog` module is the single source of truth for model IDs,
pricing, context windows, and lifecycle status (Sep-2026 refresh).
"""

from .base import (
    BaseLLMClient,
    LLMMessage,
    LLMResponse,
    async_retry,
    build_strict_response_format,
    strip_code_fences,
)
from .catalog import (
    DEFAULT_TRAINING_MODEL,
    MODEL_ALIASES,
    MODEL_CATALOG,
    PROVIDER_DEFAULTS,
    current_models,
    pricing_table,
    resolve_alias,
)
from .gateway import GatewayClient
from .registry import (
    ProviderInfo,
    clear_custom,
    get,
    list_all,
    list_names,
    list_select_choices,
    register,
    register_builtins,
)

__all__ = [
    "BaseLLMClient",
    "LLMMessage",
    "LLMResponse",
    "async_retry",
    "build_strict_response_format",
    "strip_code_fences",
    "DEFAULT_TRAINING_MODEL",
    "MODEL_ALIASES",
    "MODEL_CATALOG",
    "PROVIDER_DEFAULTS",
    "current_models",
    "pricing_table",
    "resolve_alias",
    "GatewayClient",
    "ProviderInfo",
    "clear_custom",
    "get",
    "list_all",
    "list_names",
    "list_select_choices",
    "register",
    "register_builtins",
]

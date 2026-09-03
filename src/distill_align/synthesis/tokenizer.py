"""
Token counting and cost estimation for LLM API calls.

Provides accurate token counting using tiktoken and cost estimation
based on model pricing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    import tiktoken  # type: ignore[import-not-found]

# Pricing per 1M tokens (USD). Single source of truth lives in
# synthesis/models/catalog.py (Sep-2026 model IDs, context windows, status).
# Kept here as an alias so existing imports keep working.
from .models import catalog as _catalog
from .models.catalog import pricing_table

# Re-exported for backwards compat (single source lives in catalog.py).
MODEL_ALIASES = _catalog.MODEL_ALIASES

MODEL_PRICING: dict[str, dict[str, float]] = pricing_table()


# Known prefixes for Azure deployments — strip to get the base model
AZURE_DEPLOYMENT_PREFIXES = ("gpt-", "o1", "o3", "o4")


# Model to tiktoken encoding mapping
MODEL_ENCODING_MAP: dict[str, str] = {
    "gpt-5.6-sol": "o200k_base",
    "gpt-5.6-terra": "o200k_base",
    "gpt-5.6-luna": "o200k_base",
    "gpt-5.5": "o200k_base",
    "gpt-5.4": "o200k_base",
    "gpt-5.4-mini": "o200k_base",
    "gpt-5.4-nano": "o200k_base",
    "gpt-5": "o200k_base",
    "gpt-5-mini": "o200k_base",
    "gpt-5-nano": "o200k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4.1": "o200k_base",
    "gpt-4.1-mini": "o200k_base",
    "gpt-4.1-nano": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o3-mini": "o200k_base",
    "o3": "o200k_base",
    "o4-mini": "o200k_base",
    "gemini-2.0-flash": "o200k_base",
    "gemini-2.5-pro": "o200k_base",
    "gemini-3-flash": "o200k_base",
    "gemini-3-pro": "o200k_base",
    "gemini-3.5-flash": "o200k_base",
    "gemini-3.8-flash": "o200k_base",
    "claude-sonnet-4": "cl100k_base",
    "claude-sonnet-5": "cl100k_base",
    "claude-opus-4-8": "cl100k_base",
    "claude-fable-5": "cl100k_base",
    "claude-haiku-4-5": "cl100k_base",
    "claude-3-5-sonnet": "cl100k_base",
}


class TokenCount(BaseModel):
    """Token count result for a single text."""

    text: str
    token_count: int
    model: str = ""
    estimated_cost: float = 0.0


class CostEstimate(BaseModel):
    """Cost estimate for a batch of API calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: str = ""
    num_requests: int = 0
    avg_tokens_per_request: float = 0.0


class Tokenizer:
    """
    Token counter and cost estimator.

    Uses tiktoken for accurate OpenAI token counting, with fallback
    to character-based estimation for other models.
    """

    def __init__(self, model: str = "gpt-5-mini"):
        """
        Initialize the tokenizer.

        Args:
            model: Model name for token counting and cost estimation.
        """
        self.model = model
        self._encoder: tiktoken.Encoding | None = None

        # Cumulative stats
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._num_requests = 0

    def _get_encoder(self):
        """Get or create the tiktoken encoder."""
        if self._encoder is None:
            try:
                import tiktoken  # type: ignore[import-not-found]

                encoding_name = MODEL_ENCODING_MAP.get(self.model)
                if encoding_name:
                    self._encoder = tiktoken.get_encoding(encoding_name)
                else:
                    try:
                        self._encoder = tiktoken.encoding_for_model(self.model)
                    except KeyError:
                        self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                logger.warning("tiktoken not installed, using character-based estimation")
                return None
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        encoder = self._get_encoder()
        if encoder:
            return len(encoder.encode(text))
        # Fallback: ~4 characters per token (English)
        return max(1, len(text) // 4)

    def count_message_tokens(self, messages: list[dict[str, str]]) -> int:
        """
        Count tokens in a list of OpenAI-format messages.

        Accounts for per-message overhead (role, formatting).

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            Total token count including message overhead.
        """
        encoder = self._get_encoder()
        if not encoder:
            # Fallback estimation
            total = 0
            for msg in messages:
                total += self.count_tokens(msg.get("content", ""))
                total += 4  # Per-message overhead
            return total

        # Tokens per message
        tokens_per_message = 3
        if "gpt-3.5" in self.model:
            tokens_per_message = 4

        total_tokens = 0
        for message in messages:
            total_tokens += tokens_per_message
            for _key, value in message.items():
                total_tokens += len(encoder.encode(str(value)))
            total_tokens += 1  # Role marker

        total_tokens += 3  # Reply primer
        return total_tokens

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost in USD for given token counts.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        pricing = self._resolve_pricing(self.model)

        # Pricing is per 1M tokens
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    @staticmethod
    def _resolve_pricing(model: str) -> dict[str, float]:
        """Resolve pricing for a model, handling Azure deployment names.

        Azure deployments often use names like ``gpt-4o-deployment``.
        This strips known prefixes to find the base model pricing, or
        tries increasingly shorter suffixes as a fallback. Also normalises
        common separators (``:``, ``/``, ``@``) used by gateways
        (OpenRouter, LiteLLM, Together) to the bare model id.

        Args:
            model: Model name or Azure deployment name.

        Returns:
            Dict with ``input`` and ``output`` prices per 1M tokens.
        """
        # Direct match first
        if model in MODEL_PRICING:
            return MODEL_PRICING[model]

        # Gateway-style ids: "openai/gpt-5-mini", "anthropic/claude-sonnet-5",
        # "meta-llama/Llama-4-Scout...:free" → try trailing segment + lower variants
        candidates: list[str] = [model]
        for sep in ("/", ":", "@"):
            if sep in model:
                candidates.append(model.split(sep)[-1])
        # Case-insensitive fallback (e.g. "Meta-Llama-..." vs "meta-llama/...")
        lowered = {k.lower(): v for k, v in MODEL_PRICING.items()}
        for cand in candidates:
            if cand in MODEL_PRICING:
                return MODEL_PRICING[cand]
            if cand.lower() in lowered:
                return lowered[cand.lower()]

        # Azure deployment names: strip known prefixes
        for prefix in AZURE_DEPLOYMENT_PREFIXES:
            if model.startswith(prefix):
                # Try exact prefix match first
                if prefix in MODEL_PRICING:
                    return MODEL_PRICING[prefix]
                # Try progressively shorter suffixes
                base = model
                while base:
                    if base in MODEL_PRICING:
                        return MODEL_PRICING[base]
                    # Remove last dash-segment
                    if "-" in base:
                        base = base.rsplit("-", 1)[0]
                    else:
                        break

        # Fallback: generic zero pricing
        logger.debug(f"No pricing found for model '{model}', assuming $0")
        return {"input": 0.0, "output": 0.0}

    def record_usage_from_response(self, response_data: dict[str, Any]) -> float:
        """Record token usage from an LLM API response dict.

        Expects a dict with a ``usage`` key containing ``prompt_tokens``
        (or ``input_tokens``) and ``completion_tokens`` (or
        ``output_tokens``) integers.

        Args:
            response_data: The raw response dict from an LLM call.

        Returns:
            Estimated cost in USD for this request.
        """
        usage = response_data.get("usage", {})
        if not isinstance(usage, dict):
            return 0.0

        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("promptTokenCount", 0) or 0
        output_tokens = (
            usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("candidatesTokenCount", 0) or 0
        )
        # Note: OpenAI/Anthropic already include reasoning + cached tokens in
        # the top-level completion/prompt counts, so details are only a
        # fallback when top-level keys are absent (e.g. partial usage dicts).
        details_in = usage.get("prompt_tokens_details") or {}
        details_out = usage.get("completion_tokens_details") or {}
        if not isinstance(details_in, dict):
            details_in = {}
        if not isinstance(details_out, dict):
            details_out = {}
        if not input_tokens:
            input_tokens = int(details_in.get("cached_tokens", 0) or 0)
        if not output_tokens:
            output_tokens = int(details_out.get("reasoning_tokens", 0) or 0)
        try:
            input_tokens = int(input_tokens)
        except (TypeError, ValueError):
            input_tokens = 0
        try:
            output_tokens = int(output_tokens)
        except (TypeError, ValueError):
            output_tokens = 0

        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._num_requests += 1

        return self.estimate_cost(input_tokens, output_tokens)

    def record_usage(self, input_tokens: int, output_tokens: int = 0) -> None:
        """
        Record token usage for batch statistics.

        Args:
            input_tokens: Input token count.
            output_tokens: Output token count.
        """
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._num_requests += 1

    def get_stats(self) -> CostEstimate:
        """
        Get cumulative usage statistics.

        Returns:
            CostEstimate with totals and cost.
        """
        total_tokens = self._total_input_tokens + self._total_output_tokens
        cost = self.estimate_cost(self._total_input_tokens, self._total_output_tokens)
        avg = total_tokens / self._num_requests if self._num_requests > 0 else 0

        return CostEstimate(
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            model=self.model,
            num_requests=self._num_requests,
            avg_tokens_per_request=avg,
        )

    def get_cost_report_string(self) -> str:
        """Get a human-readable cost report.

        Returns:
            Multi-line string with cost summary.
        """
        stats = self.get_stats()
        lines = [
            "── Cost Report ──────────────────────",
            f"  Model:               {stats.model}",
            f"  Requests:             {stats.num_requests}",
            f"  Input tokens:         {stats.total_input_tokens:,}",
            f"  Output tokens:        {stats.total_output_tokens:,}",
            f"  Total tokens:         {stats.total_tokens:,}",
            f"  Avg tokens/request:   {stats.avg_tokens_per_request:.1f}",
            f"  Estimated cost (USD): ${stats.estimated_cost_usd:.4f}",
            "─────────────────────────────────────",
        ]
        return "\n".join(lines)

    def estimate_batch_cost(
        self,
        texts: list[str],
        estimated_output_tokens: int = 500,
    ) -> CostEstimate:
        """
        Estimate the cost of processing a batch of texts.

        Args:
            texts: List of input texts.
            estimated_output_tokens: Estimated output tokens per request.

        Returns:
            CostEstimate for the batch.
        """
        total_input = sum(self.count_tokens(t) for t in texts)
        total_output = estimated_output_tokens * len(texts)
        cost = self.estimate_cost(total_input, total_output)

        return CostEstimate(
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_input + total_output,
            estimated_cost_usd=cost,
            model=self.model,
            num_requests=len(texts),
            avg_tokens_per_request=(total_input + total_output) / len(texts) if texts else 0,
        )

    def reset_stats(self) -> None:
        """Reset cumulative statistics."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._num_requests = 0


class CostTrackingClient:
    """A transparent proxy around a :class:`BaseLLMClient` that records
    token usage into a :class:`Tokenizer` on every API response.

    All calls are forwarded to the wrapped client, and token usage from
    the response is automatically recorded via :meth:`Tokenizer.record_usage`.

    Usage::

        tokenizer = Tokenizer(model="gpt-4o")
        raw_client = OpenAIClient(...)
        client = CostTrackingClient(raw_client, tokenizer)

        # Use ``client`` everywhere — calls are forwarded transparently
        response = await client.chat(messages=[...])
        text = await client.generate(system_prompt=..., user_prompt=...)
    """

    def __init__(self, client: Any, tokenizer: Tokenizer) -> None:
        object.__setattr__(self, "_wrapped_client", client)
        object.__setattr__(self, "_tokenizer", tokenizer)

    async def chat(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy chat() — records usage from the raw response."""
        response = await self._wrapped_client.chat(*args, **kwargs)
        if response.usage:
            self._tokenizer.record_usage_from_response(response.raw_response)
        return response

    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        """Proxy complete() — records usage from the raw response."""
        response = await self._wrapped_client.complete(*args, **kwargs)
        if response.usage:
            self._tokenizer.record_usage_from_response(response.raw_response)
        return response

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Proxy generate() — routes through our chat() for recording."""
        from .models.base import LLMMessage

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        response = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )
        return response.content  # type: ignore[no-any-return]

    def __getattr__(self, name: str) -> Any:
        """Forward other attribute access to the wrapped client."""
        return getattr(self._wrapped_client, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attributes on the wrapped client."""
        setattr(self._wrapped_client, name, value)

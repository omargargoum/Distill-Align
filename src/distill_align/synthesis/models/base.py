"""
Base LLM client interface.

All LLM provider clients should inherit from this base class.
"""

import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from ...core.exceptions import LLMClientError


class LLMMessage(BaseModel):
    """A message in a conversation."""

    role: str  # system, user, assistant, tool
    content: str
    name: str | None = None


class LLMResponse(BaseModel):
    """Response from an LLM."""

    content: str
    model: str
    usage: dict[str, int] = {}  # prompt_tokens, completion_tokens, total_tokens
    finish_reason: str = "stop"
    raw_response: dict[str, Any] = {}
    refusal: str | None = None  # Safety refusal text (structured-output APIs)


def build_strict_response_format(
    json_schema: dict[str, Any],
    name: str = "response",
    strict: bool = True,
) -> dict[str, Any]:
    """Build an OpenAI-style strict ``response_format`` payload.

    Provider clients translate this into their native equivalent:
    Anthropic ``output_config.format``, Gemini ``responseSchema``,
    Ollama ``format`` (schema), vLLM ``guided_json``.

    Args:
        json_schema: JSON Schema dict for the expected object.
        name: Schema name for the wrapper.
        strict: Request strict enforcement where supported.
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": json_schema, "strict": strict},
    }


def strip_code_fences(text: str) -> str:
    """Remove Markdown code fences (```json ... ```) around JSON payloads.

    Local models (Ollama/vLLM without grammar constraints) often wrap JSON
    in fences despite ``format=json``. Strict APIs never need this, but it
    is a harmless normalisation for the legacy path.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Drop opening fence (``` or ```json)
        lines = lines[1:]
        # Drop closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


async def async_retry(
    fn,
    retry_attempts: int = 3,
    base_delay: float = 0.5,
    retry_on: tuple[type[BaseException], ...] | None = None,
):
    """Run async *fn* with exponential backoff + jitter.

    Args:
        fn: Zero-arg async callable.
        retry_attempts: Total attempts (not retries).
        base_delay: Base seconds; wait = base * 2**attempt + jitter.
        retry_on: Exception types to retry (default: any Exception except
            asyncio.CancelledError).

    Raises:
        The last exception if all attempts fail.
    """
    import asyncio
    import random

    if retry_on is None:
        retry_on = (Exception,)
    last: BaseException | None = None
    for attempt in range(max(1, retry_attempts)):
        try:
            return await fn()
        except asyncio.CancelledError:
            raise
        except retry_on as e:  # type: ignore[misc]
            last = e
            if attempt >= retry_attempts - 1:
                break
            await asyncio.sleep(base_delay * (2**attempt) + random.uniform(0, 0.25))
    assert last is not None
    raise last


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "default",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Initialize the LLM client.

        Args:
            base_url: Base URL for the API.
            api_key: Optional API key for authentication.
            model: Model name/identifier.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries on failure.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of conversation messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            response_format: Optional structured output format, e.g.
                ``{"type": "json_object"}`` or
                ``{"type": "json_schema", "json_schema": {...}}``.
                Only supported by OpenAI-compatible endpoints.
            **kwargs: Additional provider-specific parameters.

        Returns:
            LLMResponse object.

        Raises:
            LLMClientError: If the request fails.
        """
        pass

    async def chat_structured(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.3,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        strict: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Send a chat request with JSON mode enabled and parse the response.

        This is a convenience wrapper around ``chat()`` that:
        1. Sets ``response_format={"type": "json_object"}`` (or a strict
           ``json_schema`` payload when *json_schema* is provided)
        2. Parses the response content as JSON
        3. Returns the parsed dictionary

        Args:
            messages: List of conversation messages.
            temperature: Sampling temperature (default 0.3 for deterministic).
            max_tokens: Maximum tokens to generate.
            json_schema: Optional JSON Schema for strict structured outputs
                (OpenAI strict mode, Anthropic ``output_config``,
                Gemini ``response_schema`` — mapped per client).
            schema_name: Name for the schema wrapper (OpenAI strict mode).
            strict: Request strict schema enforcement where supported.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            LLMClientError: If the response cannot be parsed as JSON.
        """
        if json_schema is not None:
            response_format: dict[str, Any] | None = build_strict_response_format(
                json_schema, name=schema_name, strict=strict
            )
        else:
            response_format = {"type": "json_object"}
        response = await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )
        content = strip_code_fences(response.content)
        try:
            return json.loads(content)  # type: ignore[no-any-return]
        except json.JSONDecodeError as e:
            raise LLMClientError(
                f"Failed to parse structured response as JSON: {e}. Raw content: {response.content[:200]}"
            ) from e

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a text completion request.

        Args:
            prompt: Text prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Returns:
            LLMResponse object.

        Raises:
            LLMClientError: If the request fails.
        """
        pass

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """
        Convenience method to generate a response with system and user prompts.

        Args:
            system_prompt: System prompt.
            user_prompt: User prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            response_format: Optional structured output format.

        Returns:
            Generated text content.
        """
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
        return response.content

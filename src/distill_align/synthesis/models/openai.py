"""
OpenAI-compatible LLM client.

Supports OpenAI API and compatible endpoints (vLLM, Ollama with OpenAI compatibility mode).
"""

from typing import Any

import httpx

from ...core.exceptions import LLMClientError, ModelNotFoundError, RateLimitError
from .base import BaseLLMClient, LLMMessage, LLMResponse


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI-compatible APIs."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        model: str = "gpt-5-mini",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Initialize the OpenAI client.

        Args:
            base_url: Base URL for the OpenAI API.
            api_key: OpenAI API key.
            model: Model name (e.g., "gpt-5.6-terra", "gpt-5-mini", "gpt-5.6-sol").
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
        """
        super().__init__(base_url, api_key, model, timeout, max_retries)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a chat completion request to OpenAI.

        Args:
            messages: List of conversation messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            response_format: Optional structured output format, e.g.
                ``{"type": "json_object"}``. Requires model supporting
                structured outputs (gpt-5-mini, gpt-5.6-terra, etc.).
            **kwargs: Additional parameters (e.g., top_p, frequency_penalty).

        Returns:
            LLMResponse object.

        Raises:
            LLMClientError: If the request fails.
        """
        client = await self._get_client()

        # Build request payload
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            # Newer reasoning models (o1+, gpt-5) only accept
            # max_completion_tokens; send both-tolerant single key.
            if self.model.startswith(("o1", "o3", "o4", "gpt-5")):
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens
        if response_format:
            payload["response_format"] = response_format
            # vLLM/OpenRouter guided_json passthrough: strict json_schema
            # doubles as guided_json for OpenAI-compatible servers.
            try:
                schema = response_format.get("json_schema", {}).get("schema")
            except AttributeError:
                schema = None
            if schema:
                payload["guided_json"] = schema
        payload.update(kwargs)

        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            # Parse response
            choice = data["choices"][0]
            message = choice.get("message", {})
            # Structured-output refusals arrive in a separate field —
            # surface them instead of returning empty content.
            refusal = message.get("refusal")
            content = message.get("content") or ""
            usage = data.get("usage", {})

            return LLMResponse(
                content=content,
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=data,
                refusal=refusal,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError("Rate limit exceeded") from e
            elif e.response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {self.model}") from e
            else:
                raise LLMClientError(f"API error: {e.response.status_code}") from e
        except Exception as e:
            raise LLMClientError(f"Request failed: {e}") from e

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a text completion request to OpenAI.

        Args:
            prompt: Text prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse object.

        Raises:
            LLMClientError: If the request fails.
        """
        client = await self._get_client()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        try:
            response = await client.post("/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            usage = data.get("usage", {})

            return LLMResponse(
                content=choice["text"],
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError("Rate limit exceeded") from e
            elif e.response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {self.model}") from e
            else:
                raise LLMClientError(f"API error: {e.response.status_code} - {e.response.text}") from e
        except Exception as e:
            raise LLMClientError(f"Request failed: {e}") from e

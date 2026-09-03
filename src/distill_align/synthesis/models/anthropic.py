"""
Anthropic Claude LLM client.

Uses the Anthropic Messages API directly via httpx.
"""

from typing import Any

import httpx

from ...core.exceptions import LLMClientError, RateLimitError
from .base import BaseLLMClient, LLMMessage, LLMResponse


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude API.

    Uses the ``/v1/messages`` endpoint with ``x-api-key`` authentication.
    """

    def __init__(
        self,
        base_url: str = "https://api.anthropic.com/v1",
        api_key: str | None = None,
        model: str = "claude-sonnet-5",
        timeout: float = 120.0,
        max_retries: int = 3,
        anthropic_version: str = "2023-06-01",
    ):
        """
        Args:
            base_url: Anthropic API base URL.
            api_key: Anthropic API key.
            model: Claude model name.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
            anthropic_version: Anthropic API version header.
        """
        super().__init__(base_url, api_key, model, timeout, max_retries)
        self.anthropic_version = anthropic_version
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            headers = {
                "x-api-key": self.api_key or "",
                "anthropic-version": self.anthropic_version,
                "Content-Type": "application/json",
            }
            if not self.api_key:
                headers["x-api-key"] = ""  # Will fail with 401
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
        """Send a chat completion request to Anthropic.

        Anthropic uses a different message format::

            {"role": "user", "content": "..."}
            {"role": "assistant", "content": "..."}

        System prompts are passed as a separate ``system`` parameter.
        """
        client = await self._get_client()

        # Separate system message from others
        system_content: str | None = None
        api_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
        }
        if system_content:
            payload["system"] = system_content
        if response_format:
            # GA structured outputs (Feb 2026): output_config.format with
            # JSON Schema. Legacy json_object → prompt-level JSON mode.
            schema = None
            try:
                schema = response_format.get("json_schema", {}).get("schema")
            except AttributeError:
                schema = None
            if schema is not None:
                payload["output_config"] = {"format": {"type": "json_schema", "json_schema": schema}}
            elif response_format.get("type") == "json_object":
                # Best-effort JSON mode: reinforce via system suffix.
                if system_content:
                    payload["system"] = system_content + "\nRespond with valid JSON only."
                else:
                    payload["system"] = "Respond with valid JSON only."
            else:
                payload["response_format"] = response_format
        payload.update(kwargs)

        try:
            response = await client.post("/messages", json=payload)
            response.raise_for_status()
            data = response.json()

            usage = data.get("usage", {})
            content_blocks = data.get("content", [])
            text = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    text += block.get("text", "")

            return LLMResponse(
                content=text,
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                },
                finish_reason=data.get("stop_reason", "stop"),
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError("Rate limit exceeded") from e
            raise LLMClientError(f"Anthropic API error: {e.response.status_code}") from e
        except Exception as e:
            raise LLMClientError(f"Anthropic request failed: {e}") from e

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a text completion (not supported by Anthropic).

        Falls back to chat with a single user message.
        """
        return await self.chat(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

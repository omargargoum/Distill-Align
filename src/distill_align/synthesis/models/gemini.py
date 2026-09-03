"""
Google Gemini LLM client.

Uses the Gemini API directly via httpx.
"""

from typing import Any

import httpx

from ...core.exceptions import LLMClientError, RateLimitError
from .base import BaseLLMClient, LLMMessage, LLMResponse


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini API.

    Uses the ``/v1beta/models/{model}:generateContent`` endpoint with
    API key query parameter authentication.
    """

    def __init__(
        self,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        api_key: str | None = None,
        model: str = "gemini-3.5-flash",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Args:
            base_url: Gemini API base URL.
            api_key: Google API key.
            model: Gemini model name.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
        """
        super().__init__(base_url, api_key, model, timeout, max_retries)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _convert_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert internal messages to Gemini format.

        Gemini uses ``contents`` with ``role`` and ``parts``::

            {"role": "user", "parts": [{"text": "..."}]}
        """
        contents: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                # Gemini uses system_instruction instead of system messages
                continue
            # Gemini requires role "model" for assistant messages
            gemini_role = "model" if msg.role == "assistant" else msg.role
            contents.append(
                {
                    "role": gemini_role,
                    "parts": [{"text": msg.content}],
                }
            )
        return contents

    async def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a chat completion request to Gemini.

        System prompts are extracted and sent as ``system_instruction``.
        """
        client = await self._get_client()

        # Extract system instruction
        system_instruction: str | None = None
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
                break

        contents = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if system_instruction:
            payload["system_instruction"] = {"parts": [{"text": system_instruction}]}
        if response_format:
            # Gemini native: response_mime_type=json + responseSchema for
            # constrained decoding. Accept both json_object and strict
            # json_schema wrappers from build_strict_response_format().
            payload["generationConfig"]["response_mime_type"] = "application/json"
            try:
                schema = response_format.get("json_schema", {}).get("schema")
            except AttributeError:
                schema = None
            if schema:
                payload["generationConfig"]["responseSchema"] = schema
        payload.update(kwargs)

        endpoint = f"/models/{self.model}:generateContent"

        try:
            params = {}
            if self.api_key:
                params["key"] = self.api_key

            response = await client.post(endpoint, json=payload, params=params)
            response.raise_for_status()
            data = response.json()

            # Parse Gemini response
            candidates = data.get("candidates", [])
            text = ""
            usage_data: dict[str, int] = {}
            if candidates:
                candidate = candidates[0]
                content_parts = candidate.get("content", {}).get("parts", [])
                for part in content_parts:
                    text += part.get("text", "")
                usage_data = data.get("usageMetadata", {})

            return LLMResponse(
                content=text,
                model=data.get("modelVersion", self.model),
                usage={
                    "prompt_tokens": usage_data.get("promptTokenCount", 0),
                    "completion_tokens": usage_data.get("candidatesTokenCount", 0),
                    "total_tokens": usage_data.get("totalTokenCount", 0),
                },
                finish_reason=candidates[0].get("finishReason", "stop") if candidates else "stop",
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError("Rate limit exceeded") from e
            raise LLMClientError(f"Gemini API error: {e.response.status_code}") from e
        except Exception as e:
            raise LLMClientError(f"Gemini request failed: {e}") from e

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a text completion (falls back to chat with single user message)."""
        return await self.chat(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

"""Gateway client (Phase 3): OpenAI-compatible routing for 2026 providers.

Covers OpenRouter, LiteLLM proxy, Together, Groq, Mistral, DeepSeek,
Cohere compatibility endpoints — all speak ``/chat/completions`` but need
per-request headers (OpenRouter ``HTTP-Referer``/``X-Title``) and optional
provider-routing extras. Falls back to plain OpenAIClient behaviour for
unknown gateways.
"""

from typing import Any

from .base import LLMMessage, LLMResponse
from .openai import OpenAIClient


class GatewayClient(OpenAIClient):
    """OpenAI-compatible client with gateway headers + routing extras."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "default",
        timeout: float = 120.0,
        max_retries: int = 3,
        app_name: str = "distill-align",
        app_url: str = "https://github.com/omargargoum/Distill-Align",
        provider_routing: dict[str, Any] | None = None,
    ):
        super().__init__(base_url, api_key, model, timeout, max_retries)
        self.app_name = app_name
        self.app_url = app_url
        self.provider_routing = provider_routing or {}

    async def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        **kwargs,
    ) -> LLMResponse:
        client = await self._get_client()
        # Gateway attribution headers (OpenRouter honors these; harmless elsewhere)
        headers = {
            "HTTP-Referer": self.app_url,
            "X-Title": self.app_name,
        }
        if extra_headers:
            headers.update(extra_headers)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            payload["max_completion_tokens"] = max_tokens
        if response_format:
            payload["response_format"] = response_format
            try:
                schema = response_format.get("json_schema", {}).get("schema")
            except AttributeError:
                schema = None
            if schema:
                payload["guided_json"] = schema
        if self.provider_routing:
            payload["provider"] = self.provider_routing
        payload.update(kwargs)

        import httpx

        from ...core.exceptions import LLMClientError, ModelNotFoundError, RateLimitError

        try:
            response = await client.post("/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            message = choice.get("message", {})
            usage = data.get("usage", {})
            return LLMResponse(
                content=message.get("content") or "",
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=data,
                refusal=message.get("refusal"),
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError("Rate limit exceeded") from e
            if e.response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {self.model}") from e
            raise LLMClientError(f"Gateway API error: {e.response.status_code}") from e
        except Exception as e:
            raise LLMClientError(f"Gateway request failed: {e}") from e

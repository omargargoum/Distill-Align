"""
Azure OpenAI LLM client with Entra ID (Azure AD) authentication.

Uses the Azure OpenAI service API with either API key or OAuth2 token
(Entra ID / Azure AD) authentication.
"""

from typing import Any

import httpx

from ...core.exceptions import LLMClientError, RateLimitError
from .base import BaseLLMClient, LLMMessage, LLMResponse


class AzureClient(BaseLLMClient):
    """Client for Azure OpenAI service.

    Supports two authentication modes:
    1. **API key** — Pass ``api_key`` directly.
    2. **Entra ID (Azure AD)** — Pass ``azure_ad_token`` or provide a
       callable ``token_provider`` that returns a valid OAuth2 token.

    The ``base_url`` should be the full Azure OpenAI endpoint, e.g.::

        https://my-resource.openai.azure.com

    The deployment (model) name is set via the ``model`` parameter.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.azure.com",
        api_key: str | None = None,
        model: str = "gpt-5-mini",  # Deployment name in Azure
        timeout: float = 120.0,
        max_retries: int = 3,
        azure_ad_token: str | None = None,
        api_version: str = "2024-10-01-preview",
    ):
        """
        Args:
            base_url: Azure OpenAI endpoint URL (resource name).
            api_key: Azure API key (Resource Management -> Keys and Endpoint).
            model: Deployment name (not the base model name).
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
            azure_ad_token: OAuth2 token for Entra ID auth (alternative to api_key).
            api_version: Azure OpenAI API version.
        """
        super().__init__(base_url, api_key, model, timeout, max_retries)
        self.azure_ad_token = azure_ad_token
        self.api_version = api_version
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            headers: dict[str, str] = {
                "Content-Type": "application/json",
            }
            if self.api_key:
                headers["api-key"] = self.api_key
            elif self.azure_ad_token:
                headers["Authorization"] = f"Bearer {self.azure_ad_token}"

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
        """Send a chat completion request to Azure OpenAI.

        Azure uses the same API schema as OpenAI but with a different
        endpoint format::

            POST /openai/deployments/{deployment}/chat/completions?api-version={version}
        """
        client = await self._get_client()

        payload: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format:
            payload["response_format"] = response_format
        payload.update(kwargs)

        endpoint = f"/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}"

        try:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            usage = data.get("usage", {})

            return LLMResponse(
                content=choice["message"]["content"],
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
            raise LLMClientError(f"Azure OpenAI API error: {e.response.status_code}") from e
        except Exception as e:
            raise LLMClientError(f"Azure OpenAI request failed: {e}") from e

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a text completion request to Azure OpenAI."""
        client = await self._get_client()

        payload: dict[str, Any] = {
            "prompt": prompt,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        endpoint = f"/openai/deployments/{self.model}/completions?api-version={self.api_version}"

        try:
            response = await client.post(endpoint, json=payload)
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
            raise LLMClientError(f"Azure OpenAI API error: {e.response.status_code} - {e.response.text}") from e
        except Exception as e:
            raise LLMClientError(f"Azure OpenAI request failed: {e}") from e

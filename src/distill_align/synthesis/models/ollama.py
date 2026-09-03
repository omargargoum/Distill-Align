"""
Ollama LLM client.

Supports local Ollama server for running models locally.
"""

from typing import Any

import httpx

from ...core.exceptions import LLMClientError, ModelNotFoundError
from .base import BaseLLMClient, LLMMessage, LLMResponse


class OllamaClient(BaseLLMClient):
    """Client for Ollama local LLM server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        timeout: float = 300.0,  # Longer timeout for local models
        max_retries: int = 3,
    ):
        """
        Initialize the Ollama client.

        Args:
            base_url: Base URL for the Ollama server.
            model: Model name (e.g., "llama3.1", "mistral", "codellama").
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
        """
        super().__init__(base_url, api_key=None, model=model, timeout=timeout, max_retries=max_retries)
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

    async def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a chat request to Ollama.

        Args:
            messages: List of conversation messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters.

        Returns:
            LLMResponse object.

        Raises:
            LLMClientError: If the request fails.
        """
        client = await self._get_client()

        # Build request payload
        options: dict[str, object] = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": options,
        }
        if response_format:
            # Ollama ≥0.3: `format` accepts a full JSON Schema for
            # XGrammar-constrained decoding. Accept strict wrappers and
            # legacy {"type": "json_object"} (→ "json").
            schema = None
            try:
                schema = response_format.get("json_schema", {}).get("schema")
            except AttributeError:
                schema = None
            if isinstance(schema, dict):
                payload["format"] = schema
            else:
                payload["format"] = "json"
        payload.update(kwargs)

        try:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            # Parse response
            message = data.get("message", {})
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }

            return LLMResponse(
                content=message.get("content", ""),
                model=data.get("model", self.model),
                usage=usage,
                finish_reason="stop" if data.get("done", False) else "length",
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {self.model}") from e
            else:
                raise LLMClientError(f"Ollama API error: {e.response.status_code}") from e
        except Exception as e:
            raise LLMClientError(f"Ollama request failed: {e}") from e

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a text completion request to Ollama.

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

        options: dict[str, object] = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        payload.update(kwargs)

        try:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }

            return LLMResponse(
                content=data.get("response", ""),
                model=data.get("model", self.model),
                usage=usage,
                finish_reason="stop" if data.get("done", False) else "length",
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {self.model}") from e
            else:
                raise LLMClientError(f"Ollama API error: {e.response.status_code}") from e
        except Exception as e:
            raise LLMClientError(f"Ollama request failed: {e}") from e

    async def list_models(self) -> list[str]:
        """
        List available models on the Ollama server.

        Returns:
            List of model names.
        """
        client = await self._get_client()
        try:
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            raise LLMClientError(f"Failed to list models: {e}") from e

    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model on the Ollama server.

        Args:
            model_name: Name of the model to pull.

        Returns:
            True if successful.
        """
        client = await self._get_client()
        try:
            response = await client.post("/api/pull", json={"name": model_name, "stream": False})
            response.raise_for_status()
            return True
        except Exception as e:
            raise LLMClientError(f"Failed to pull model: {e}") from e

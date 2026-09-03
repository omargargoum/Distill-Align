"""
Tests for HTTP-based LLM clients using pytest-httpx mocking.

Covers:
- Successful chat completion
- Rate limiting (429)
- Model not found (404)
- Server errors (500)
- Network timeouts
- Completion endpoint
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from distill_align.core.exceptions import LLMClientError, ModelNotFoundError, RateLimitError
from distill_align.synthesis.models.base import LLMMessage
from distill_align.synthesis.models.openai import OpenAIClient

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def client() -> OpenAIClient:
    """Create an OpenAI client for testing."""
    return OpenAIClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key-not-real",
        model="gpt-4o-mini",
        timeout=10.0,
    )


def _msg(role: str = "user", content: str = "Hello!") -> LLMMessage:
    """Shortcut for creating an LLMMessage."""
    return LLMMessage(role=role, content=content)


def _mock_chat_response(model: str = "gpt-4o-mini") -> dict[str, Any]:
    """Build a standard chat completion response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }


def _mock_completion_response(model: str = "gpt-4o-mini") -> dict[str, Any]:
    """Build a standard text completion response."""
    return {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": 1677652288,
        "model": model,
        "choices": [
            {
                "text": "Once upon a time...",
                "index": 0,
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 5,
            "total_tokens": 10,
        },
    }


# ------------------------------------------------------------------
# Chat completion tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_success(client: OpenAIClient, httpx_mock: HTTPXMock) -> None:
    """Test a successful chat completion request."""
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        method="POST",
        json=_mock_chat_response(),
        status_code=200,
    )

    response = await client.chat(
        messages=[_msg()],
        temperature=0.5,
    )

    assert response.content == "Hello! How can I help you today?"
    assert response.model == "gpt-4o-mini"
    assert response.finish_reason == "stop"
    assert response.usage["total_tokens"] == 18
    assert response.raw_response is not None


@pytest.mark.asyncio
async def test_chat_rate_limit(client: OpenAIClient, httpx_mock: HTTPXMock) -> None:
    """Test rate limit handling (429)."""
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        method="POST",
        status_code=429,
        text='{"error": {"message": "Rate limit exceeded"}}',
    )

    with pytest.raises(RateLimitError, match="Rate limit exceeded"):
        await client.chat(messages=[_msg()])


@pytest.mark.asyncio
async def test_chat_model_not_found(client: OpenAIClient, httpx_mock: HTTPXMock) -> None:
    """Test model not found (404)."""
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        method="POST",
        status_code=404,
        text='{"error": {"message": "Model not found"}}',
    )

    with pytest.raises(ModelNotFoundError, match="Model not found"):
        await client.chat(messages=[_msg()])


@pytest.mark.asyncio
async def test_chat_server_error(client: OpenAIClient, httpx_mock: HTTPXMock) -> None:
    """Test server error (500)."""
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        method="POST",
        status_code=500,
        text="Internal Server Error",
    )

    with pytest.raises(LLMClientError, match="API error"):
        await client.chat(messages=[_msg()])


@pytest.mark.asyncio
async def test_chat_timeout(client: OpenAIClient, httpx_mock: HTTPXMock) -> None:
    """Test network timeout."""
    httpx_mock.add_exception(
        httpx.TimeoutException("Request timed out"),
        url="https://api.openai.com/v1/chat/completions",
        method="POST",
    )

    with pytest.raises(LLMClientError, match="Request failed"):
        await client.chat(messages=[_msg()])


@pytest.mark.asyncio
async def test_chat_with_max_tokens(client: OpenAIClient, httpx_mock: HTTPXMock) -> None:
    """Test that max_tokens is sent in the payload."""
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        method="POST",
        json=_mock_chat_response(),
        status_code=200,
    )

    await client.chat(messages=[_msg(content="Hi")], max_tokens=100)

    # Verify max_tokens was included
    request = httpx_mock.get_request()
    assert request is not None
    import json as _json

    body = _json.loads(request.content)
    assert body["max_tokens"] == 100


@pytest.mark.asyncio
async def test_chat_custom_kwargs(client: OpenAIClient, httpx_mock: HTTPXMock) -> None:
    """Test that additional kwargs are passed through."""
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        method="POST",
        json=_mock_chat_response(),
        status_code=200,
    )

    await client.chat(
        messages=[_msg(content="Hi")],
        top_p=0.9,
        frequency_penalty=0.2,
    )

    request = httpx_mock.get_request()
    assert request is not None
    import json as _json

    body = _json.loads(request.content)
    assert body["top_p"] == 0.9
    assert body["frequency_penalty"] == 0.2


# ------------------------------------------------------------------
# Completion endpoint tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_success(client: OpenAIClient, httpx_mock: HTTPXMock) -> None:
    """Test a successful text completion request."""
    httpx_mock.add_response(
        url="https://api.openai.com/v1/completions",
        method="POST",
        json=_mock_completion_response(),
        status_code=200,
    )

    response = await client.complete(prompt="Tell me a story", temperature=0.8)

    assert response.content == "Once upon a time..."
    assert response.model == "gpt-4o-mini"
    assert response.finish_reason == "length"
    assert response.usage["total_tokens"] == 10


@pytest.mark.asyncio
async def test_complete_rate_limit(client: OpenAIClient, httpx_mock: HTTPXMock) -> None:
    """Test rate limit on completion endpoint."""
    httpx_mock.add_response(
        url="https://api.openai.com/v1/completions",
        method="POST",
        status_code=429,
    )

    with pytest.raises(RateLimitError):
        await client.complete(prompt="Test")


# ------------------------------------------------------------------
# Client lifecycle tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_reuses_session(client: OpenAIClient, httpx_mock: HTTPXMock) -> None:
    """Test that the HTTP client is reused across calls."""
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        method="POST",
        json=_mock_chat_response(),
        status_code=200,
    )
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        method="POST",
        json=_mock_chat_response(),
        status_code=200,
    )

    # First call creates the client
    await client.chat(messages=[_msg()])
    # Second call should reuse it
    await client.chat(messages=[_msg(content="Hello again")])

    # The internal client should be the same object
    internal_client = await client._get_client()
    assert isinstance(internal_client, httpx.AsyncClient)

    # Close cleanup
    await client.close()


@pytest.mark.asyncio
async def test_client_close(client: OpenAIClient) -> None:
    """Test that close() cleans up resources."""
    # The client is lazily created, so it should be None initially
    assert client._client is None
    await client.close()  # Should be a no-op

    # Force client creation then close
    _ = await client._get_client()
    assert client._client is not None
    await client.close()
    assert client._client is None


@pytest.mark.asyncio
async def test_chat_structured_json_mode(client: OpenAIClient, httpx_mock: HTTPXMock) -> None:
    """Test that chat_structured sends response_format and parses JSON."""
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": '{"result": "success", "score": 95}'}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )

    result = await client.chat_structured(
        messages=[LLMMessage(role="user", content="Test structured output")],
    )

    assert result == {"result": "success", "score": 95}

    # Verify response_format was sent
    request = httpx_mock.get_requests()[0]
    body = json.loads(request.content)
    assert body["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_chat_structured_invalid_json(client: OpenAIClient, httpx_mock: HTTPXMock) -> None:
    """Test that chat_structured raises error on invalid JSON."""
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "Not valid JSON"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )

    from distill_align.core.exceptions import LLMClientError

    with pytest.raises(LLMClientError, match="Failed to parse structured response"):
        await client.chat_structured(
            messages=[LLMMessage(role="user", content="Test")],
        )


@pytest.mark.asyncio
async def test_chat_with_response_format(client: OpenAIClient, httpx_mock: HTTPXMock) -> None:
    """Test that response_format is passed through in the payload."""
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "test"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        },
    )

    schema = {"type": "json_schema", "json_schema": {"name": "test", "schema": {"type": "object"}}}
    await client.chat(
        messages=[LLMMessage(role="user", content="Test")],
        response_format=schema,
    )

    request = httpx_mock.get_requests()[0]
    body = json.loads(request.content)
    assert body["response_format"] == schema


# =============================================================================
# Anthropic Client Tests
# =============================================================================


@pytest.mark.asyncio
async def test_anthropic_chat_success(httpx_mock: HTTPXMock) -> None:
    """Test successful Anthropic chat completion."""
    from distill_align.synthesis.models.anthropic import AnthropicClient

    client = AnthropicClient(api_key="test-key")
    httpx_mock.add_response(
        method="POST",
        url="https://api.anthropic.com/v1/messages",
        json={
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )

    response = await client.chat(
        messages=[LLMMessage(role="user", content="Hi")],
    )

    assert response.content == "Hello from Claude!"
    assert response.usage["prompt_tokens"] == 10
    assert response.usage["completion_tokens"] == 5

    # Verify headers
    request = httpx_mock.get_requests()[0]
    assert request.headers["x-api-key"] == "test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"

    await client.close()


@pytest.mark.asyncio
async def test_anthropic_with_system_prompt(httpx_mock: HTTPXMock) -> None:
    """Test Anthropic with system message extracted."""
    from distill_align.synthesis.models.anthropic import AnthropicClient

    client = AnthropicClient(api_key="test-key")
    httpx_mock.add_response(
        method="POST",
        url="https://api.anthropic.com/v1/messages",
        json={
            "content": [{"type": "text", "text": "Understood."}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        },
    )

    await client.chat(
        messages=[
            LLMMessage(role="system", content="Be concise."),
            LLMMessage(role="user", content="Explain AI"),
        ],
    )

    request = httpx_mock.get_requests()[0]
    body = json.loads(request.content)
    assert body["system"] == "Be concise."
    # System message should NOT be in messages array
    roles = [m["role"] for m in body["messages"]]
    assert "system" not in roles

    await client.close()


@pytest.mark.asyncio
async def test_anthropic_rate_limit(httpx_mock: HTTPXMock) -> None:
    """Test Anthropic rate limit handling."""
    from distill_align.synthesis.models.anthropic import AnthropicClient

    client = AnthropicClient(api_key="test-key")
    httpx_mock.add_response(
        method="POST",
        url="https://api.anthropic.com/v1/messages",
        status_code=429,
        json={"error": {"message": "Rate limited"}},
    )

    with pytest.raises(RateLimitError):
        await client.chat(
            messages=[LLMMessage(role="user", content="Hi")],
        )

    await client.close()


# =============================================================================
# Gemini Client Tests
# =============================================================================


@pytest.mark.asyncio
async def test_gemini_chat_success(httpx_mock: HTTPXMock) -> None:
    """Test successful Gemini chat completion."""
    from distill_align.synthesis.models.gemini import GeminiClient

    client = GeminiClient(api_key="test-key")
    httpx_mock.add_response(
        method="POST",
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key=test-key",
        json={
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello from Gemini!"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-3.5-flash",
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        },
    )

    response = await client.chat(
        messages=[LLMMessage(role="user", content="Hi")],
    )

    assert response.content == "Hello from Gemini!"
    assert response.usage["total_tokens"] == 15

    await client.close()


@pytest.mark.asyncio
async def test_gemini_with_system_instruction(httpx_mock: HTTPXMock) -> None:
    """Test Gemini with system instruction."""
    from distill_align.synthesis.models.gemini import GeminiClient

    client = GeminiClient(api_key="test-key")
    httpx_mock.add_response(
        method="POST",
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key=test-key",
        json={
            "candidates": [
                {
                    "content": {"parts": [{"text": "OK"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-3.5-flash",
            "usageMetadata": {},
        },
    )

    await client.chat(
        messages=[
            LLMMessage(role="system", content="Be concise."),
            LLMMessage(role="user", content="Say hi"),
        ],
    )

    request = httpx_mock.get_requests()[0]
    body = json.loads(request.content)
    assert body["system_instruction"]["parts"][0]["text"] == "Be concise."

    await client.close()


@pytest.mark.asyncio
async def test_gemini_structured_output(httpx_mock: HTTPXMock) -> None:
    """Test Gemini with response_mime_type for JSON."""
    from distill_align.synthesis.models.gemini import GeminiClient

    client = GeminiClient(api_key="test-key")
    httpx_mock.add_response(
        method="POST",
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key=test-key",
        json={
            "candidates": [
                {
                    "content": {"parts": [{"text": '{"ok": true}'}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-3.5-flash",
            "usageMetadata": {},
        },
    )

    await client.chat(
        messages=[LLMMessage(role="user", content="JSON pls")],
        response_format={"type": "json_object"},
    )

    request = httpx_mock.get_requests()[0]
    body = json.loads(request.content)
    assert body["generationConfig"]["response_mime_type"] == "application/json"

    await client.close()


# =============================================================================
# Azure Client Tests
# =============================================================================


@pytest.mark.asyncio
async def test_azure_chat_success(httpx_mock: HTTPXMock) -> None:
    """Test successful Azure OpenAI chat completion."""
    from distill_align.synthesis.models.azure import AzureClient

    client = AzureClient(api_key="test-key", model="gpt-4o-deployment")
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.azure.com/openai/deployments/gpt-4o-deployment/chat/completions?api-version=2024-10-01-preview",
        json={
            "choices": [{"message": {"content": "Hello from Azure!"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )

    response = await client.chat(
        messages=[LLMMessage(role="user", content="Hi")],
    )

    assert response.content == "Hello from Azure!"
    assert response.usage["total_tokens"] == 15

    # Verify API key header
    request = httpx_mock.get_requests()[0]
    assert request.headers["api-key"] == "test-key"

    await client.close()


@pytest.mark.asyncio
async def test_azure_ad_auth(httpx_mock: HTTPXMock) -> None:
    """Test Azure with Entra ID token auth instead of API key."""
    from distill_align.synthesis.models.azure import AzureClient

    client = AzureClient(azure_ad_token="entra-token", model="gpt-4o-deployment")
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.azure.com/openai/deployments/gpt-4o-deployment/chat/completions?api-version=2024-10-01-preview",
        json={
            "choices": [{"message": {"content": "Authenticated"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        },
    )

    await client.chat(
        messages=[LLMMessage(role="user", content="Hi")],
    )

    request = httpx_mock.get_requests()[0]
    assert request.headers["Authorization"] == "Bearer entra-token"
    assert "api-key" not in request.headers

    await client.close()


@pytest.mark.asyncio
async def test_azure_chat_with_response_format(httpx_mock: HTTPXMock) -> None:
    """Test Azure with response_format."""
    from distill_align.synthesis.models.azure import AzureClient

    client = AzureClient(api_key="test-key", model="gpt-4o-deployment")
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.azure.com/openai/deployments/gpt-4o-deployment/chat/completions?api-version=2024-10-01-preview",
        json={
            "choices": [{"message": {"content": "json"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {},
        },
    )

    await client.chat(
        messages=[LLMMessage(role="user", content="JSON")],
        response_format={"type": "json_object"},
    )

    request = httpx_mock.get_requests()[0]
    body = json.loads(request.content)
    assert body["response_format"] == {"type": "json_object"}

    await client.close()

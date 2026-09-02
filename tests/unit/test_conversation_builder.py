"""
Unit tests for the multi-turn conversation builder.

The builder turns raw chunks into training conversations, so prompt assembly,
JSON parsing, fallback behavior, and error handling are pinned down here.
"""

import json

import pytest

from distill_align.core.exceptions import LLMClientError
from distill_align.core.schemas import DataChunk, SourceMetadata
from distill_align.synthesis.conversation_builder import ConversationBuilder, ConversationMode
from distill_align.synthesis.models.base import BaseLLMClient, LLMMessage, LLMResponse


class FakeLLMClient(BaseLLMClient):
    """A scriptable LLM client for conversation builder tests."""

    def __init__(self, response: str = "", error: Exception | None = None):
        super().__init__(base_url="http://fake", model="fake-model")
        self.response = response
        self.error = error
        self.calls: list[tuple[str, str, float]] = []

    async def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        **kwargs,
    ) -> LLMResponse:
        self.calls.append((messages[0].content, messages[1].content, temperature))
        if self.error:
            raise self.error
        return LLMResponse(content=self.response, model=self.model)

    async def complete(
        self, prompt: str, temperature: float = 0.7, max_tokens: int | None = None, **kwargs
    ) -> LLMResponse:
        return LLMResponse(content="", model=self.model)


@pytest.fixture
def chunk() -> DataChunk:
    """A sample data chunk with rich metadata."""
    return DataChunk(
        content="This is the source content about distillation that is long enough to be useful.",
        metadata=SourceMetadata(
            source_type="markdown",
            file_path="/docs/distillation.md",
            file_name="distillation.md",
            title="Distillation Guide",
            language="en",
            section_headers=["Overview", "Details"],
        ),
    )


@pytest.fixture
def builder() -> ConversationBuilder:
    return ConversationBuilder()


def _json_response(roles_and_contents: list[tuple[str, str]]) -> str:
    conversation = [{"role": role, "content": content} for role, content in roles_and_contents]
    payload = json.dumps({"conversation": conversation})
    # Fenced blocks are required: the flat-object fallback in
    # extract_json_from_response grabs the innermost object otherwise.
    return f"```json\n{payload}\n```"


class TestBuildConversation:
    """Single-conversation generation."""

    @pytest.mark.asyncio
    async def test_parses_json_conversation(self, builder, chunk):
        response = _json_response(
            [
                ("user", "What is distillation?"),
                ("assistant", "It is the transfer of knowledge between models."),
            ]
        )
        client = FakeLLMClient(response=response)

        conv = await builder.build_conversation(chunk, ConversationMode.QA, client)
        assert conv is not None
        assert conv.source_chunk_id == chunk.id
        assert conv.id
        assert [t.role for t in conv.turns] == ["user", "assistant"]
        assert conv.turns[0].content == "What is distillation?"

    @pytest.mark.asyncio
    async def test_plain_text_falls_back_to_simple_conversation(self, builder, chunk):
        client = FakeLLMClient(response="Here is a plain text answer without JSON.")

        conv = await builder.build_conversation(chunk, ConversationMode.QA, client)
        assert conv is not None
        assert [t.role for t in conv.turns] == ["user", "assistant"]
        assert conv.turns[1].content == "Here is a plain text answer without JSON."
        assert chunk.content[:500] in conv.turns[0].content

    @pytest.mark.asyncio
    async def test_llm_error_returns_none(self, builder, chunk):
        client = FakeLLMClient(error=LLMClientError("provider down"))

        conv = await builder.build_conversation(chunk, ConversationMode.QA, client)
        assert conv is None

    @pytest.mark.asyncio
    async def test_temperature_passed_through(self, builder, chunk):
        client = FakeLLMClient(response=_json_response([("user", "Q?"), ("assistant", "A.")]))

        await builder.build_conversation(chunk, ConversationMode.QA, client, temperature=0.2)
        _, _, temperature = client.calls[0]
        assert temperature == 0.2

    @pytest.mark.asyncio
    async def test_pruned_too_short_falls_back_to_original(self, builder, chunk):
        # A single-turn JSON response prunes to <2 turns; the original is kept.
        client = FakeLLMClient(response=_json_response([("assistant", "Only one turn here.")]))

        conv = await builder.build_conversation(chunk, ConversationMode.QA, client)
        assert conv is not None
        assert len(conv.turns) == 1


class TestUserPrompt:
    """Prompt assembly should vary by mode and include metadata."""

    def test_teach_mode_has_instructions(self, builder, chunk):
        prompt = builder._build_user_prompt(chunk, ConversationMode.TEACH)
        assert "Start with a basic concept question" in prompt
        assert "**Title:** Distillation Guide" in prompt
        assert "**Sections:** Overview > Details" in prompt
        assert "**Language:** en" in prompt

    def test_review_mode_has_instructions(self, builder, chunk):
        prompt = builder._build_user_prompt(chunk, ConversationMode.REVIEW)
        assert "Identify 2-3 areas for improvement" in prompt

    def test_unknown_mode_gets_no_mode_instructions(self, builder, chunk):
        # An unrecognized mode falls back to the QA template for generation but
        # appends no mode-specific instructions to the user prompt.
        prompt = builder._build_user_prompt(chunk, "unknown-mode")
        assert "**Source Content:**" in prompt
        assert "Return JSON" in prompt
        assert "Mix conceptual and practical questions" not in prompt

    def test_content_truncated_to_3000_chars(self, builder):
        long_chunk = DataChunk(
            content="x" * 5000,
            metadata=SourceMetadata(source_type="text", file_path="/a.txt", file_name="a.txt"),
        )
        prompt = builder._build_user_prompt(long_chunk, ConversationMode.QA)
        assert "x" * 3000 in prompt
        assert "x" * 3001 not in prompt


class TestBuildBatch:
    """Batch generation should isolate failures and preserve order."""

    @pytest.mark.asyncio
    async def test_builds_all_and_skips_failures(self, builder, chunk):
        response = _json_response([("user", "Q?"), ("assistant", "A.")])
        client = FakeLLMClient(response=response)

        convs = await builder.build_batch([chunk, chunk], ConversationMode.QA, client)
        assert len(convs) == 2
        assert len(client.calls) == 2

    @pytest.mark.asyncio
    async def test_non_llm_exception_skipped(self, builder, chunk):
        # A non-LLMClientError propagates out of build_conversation and is skipped in the batch.
        client = FakeLLMClient(error=ValueError("unexpected"))

        convs = await builder.build_batch([chunk, chunk], ConversationMode.QA, client)
        assert convs == []

"""
Unit tests for the LLM-as-judge conversation evaluator.

The judge gates which conversations get high confidence scores, so its prompt
construction, error handling, and batch behavior are pinned down here.
"""

import json

import pytest

from distill_align.core.exceptions import LLMClientError
from distill_align.core.schemas import ConversationSchema, SynthesizedTurn
from distill_align.synthesis.judge import ConversationJudge
from distill_align.synthesis.models.base import BaseLLMClient, LLMMessage, LLMResponse


class FakeJudgeClient(BaseLLMClient):
    """A scriptable LLM client for judge tests."""

    def __init__(
        self,
        response_content: str = "{}",
        error: Exception | None = None,
        fail_on_call: int | None = None,
    ):
        super().__init__(base_url="http://fake", model="judge-model")
        self.response_content = response_content
        self.error = error
        self.fail_on_call = fail_on_call
        self.calls: list[tuple[list[LLMMessage], float, int | None]] = []

    async def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        **kwargs,
    ) -> LLMResponse:
        self.calls.append((messages, temperature, max_tokens))
        if self.error:
            raise self.error
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise LLMClientError("transient failure")
        return LLMResponse(content=self.response_content, model=self.model)

    async def complete(self, prompt: str, temperature: float = 0.7, max_tokens: int | None = None, **kwargs) -> LLMResponse:
        return LLMResponse(content="", model=self.model)


def _conversation(turns: list[tuple[str, str]], conv_id: str = "conv-1") -> ConversationSchema:
    return ConversationSchema(
        id=conv_id,
        source_chunk_id="chunk-1",
        turns=[SynthesizedTurn(role=role, content=content) for role, content in turns],
    )


@pytest.fixture
def conversation() -> ConversationSchema:
    return _conversation(
        [
            ("user", "What is distillation?"),
            ("assistant", "Distillation transfers knowledge from a large model to a small one."),
        ]
    )


class TestEvaluate:
    """Single-conversation evaluation."""

    @pytest.mark.asyncio
    async def test_returns_parsed_scores(self, conversation):
        scores = {"relevance": 9, "coherence": 8, "overall": 8.5, "explanation": "Good"}
        client = FakeJudgeClient(response_content=json.dumps(scores))
        judge = ConversationJudge(client)

        result = await judge.evaluate(conversation, source_content="Source text")
        assert result == scores

    @pytest.mark.asyncio
    async def test_prompt_contains_source_and_conversation(self, conversation):
        client = FakeJudgeClient()
        judge = ConversationJudge(client)

        await judge.evaluate(conversation, source_content="The source material")
        messages, _, _ = client.calls[0]
        prompt = messages[-1].content
        assert "The source material" in prompt
        assert "What is distillation?" in prompt

    @pytest.mark.asyncio
    async def test_source_with_braces_does_not_break_prompt(self, conversation):
        # Literal braces in source content (e.g. code) must not break prompt building.
        client = FakeJudgeClient()
        judge = ConversationJudge(client)

        result = await judge.evaluate(conversation, source_content="def f(): return {a: 1}")
        assert "error" not in result
        prompt = client.calls[0][0][-1].content
        assert "def f(): return {a: 1}" in prompt

    @pytest.mark.asyncio
    async def test_max_tokens_passed_through(self, conversation):
        client = FakeJudgeClient()
        judge = ConversationJudge(client)

        await judge.evaluate(conversation, max_tokens=64)
        _, temperature, max_tokens = client.calls[0]
        assert max_tokens == 64
        assert temperature == 0.3  # chat_structured default

    @pytest.mark.asyncio
    async def test_client_error_returns_error_dict(self, conversation):
        client = FakeJudgeClient(error=LLMClientError("boom"))
        judge = ConversationJudge(client)

        result = await judge.evaluate(conversation)
        assert "error" in result
        assert "boom" in result["error"]


class TestEvaluateBatch:
    """Batch evaluation should isolate failures."""

    @pytest.mark.asyncio
    async def test_evaluates_all_conversations(self, conversation):
        client = FakeJudgeClient(response_content=json.dumps({"overall": 8}))
        judge = ConversationJudge(client)

        results = await judge.evaluate_batch([conversation, conversation])
        assert len(results) == 2
        assert all(r["overall"] == 8 for r in results)
        assert len(client.calls) == 2

    @pytest.mark.asyncio
    async def test_failure_isolated_to_one_result(self, conversation):
        client = FakeJudgeClient(response_content=json.dumps({"overall": 8}), fail_on_call=2)
        judge = ConversationJudge(client)

        results = await judge.evaluate_batch([conversation, conversation])
        assert len(results) == 2
        assert results[0]["overall"] == 8
        assert "error" in results[1]

    @pytest.mark.asyncio
    async def test_erroring_client_returns_error_dicts(self, conversation):
        client = FakeJudgeClient(error=LLMClientError("rate limited"))
        judge = ConversationJudge(client)

        results = await judge.evaluate_batch([conversation, conversation])
        assert len(results) == 2
        assert all("error" in r for r in results)

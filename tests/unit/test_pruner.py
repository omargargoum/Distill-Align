"""
Unit tests for the ContentPruner.

The pruner is the quality gate for synthesized conversations: it decides which
conversations survive to become training data, so its filtering and structural
validation behavior is pinned down here.
"""

import pytest

from distill_align.core.schemas import ConversationSchema, SynthesizedTurn
from distill_align.synthesis.pruner import ContentPruner


@pytest.fixture
def pruner() -> ContentPruner:
    """A pruner with default settings."""
    return ContentPruner()


def _conversation(turns: list[tuple[str, str]], **kwargs) -> ConversationSchema:
    """Build a conversation from (role, content) pairs."""
    return ConversationSchema(
        id=kwargs.pop("id", "conv-1"),
        source_chunk_id=kwargs.pop("source_chunk_id", "chunk-1"),
        turns=[SynthesizedTurn(role=role, content=content) for role, content in turns],
        **kwargs,
    )


VALID_TURNS = [
    ("system", "You are helpful."),
    ("user", "What is the capital of France?"),
    ("assistant", "The capital of France is Paris, which is in Europe."),
]


class TestPruneConversation:
    """Conversation-level pruning decisions."""

    def test_keeps_valid_conversation(self, pruner):
        conv = _conversation(VALID_TURNS)
        pruned = pruner.prune_conversation(conv)
        assert pruned is not None
        assert pruned.id == conv.id
        assert pruned.source_chunk_id == conv.source_chunk_id
        assert [t.role for t in pruned.turns] == ["system", "user", "assistant"]

    def test_rejects_single_turn(self, pruner):
        conv = _conversation([("user", "Hello")])
        assert pruner.prune_conversation(conv) is None

    def test_strips_filler_phrases(self, pruner):
        conv = _conversation(
            [
                ("user", "Sure, here is my explanation."),
                ("assistant", "Here you go, have a look."),
            ]
        )
        pruned = pruner.prune_conversation(conv)
        assert pruned is not None
        assert pruned.turns[0].content == "here is my explanation."
        assert pruned.turns[1].content == "have a look."

    def test_skips_empty_turns_after_clean(self, pruner):
        conv = _conversation(
            [
                ("system", "Sure!"),
                ("user", "What is a photon?"),
                ("assistant", "A photon is a quantum of light."),
            ]
        )
        pruned = pruner.prune_conversation(conv)
        assert pruned is not None
        assert [t.role for t in pruned.turns] == ["user", "assistant"]

    def test_rejects_non_alternating_roles(self, pruner):
        conv = _conversation(
            [
                ("user", "Question one?"),
                ("user", "Question two?"),
                ("assistant", "An answer that is long enough to survive pruning."),
            ]
        )
        assert pruner.prune_conversation(conv) is None

    def test_rejects_conversation_starting_with_assistant(self, pruner):
        conv = _conversation(
            [
                ("assistant", "Hello, how can I help?"),
                ("user", "Question one?"),
                ("assistant", "An answer that is long enough to survive pruning."),
            ]
        )
        assert pruner.prune_conversation(conv) is None

    def test_prune_batch_filters_low_quality(self, pruner):
        good = _conversation(VALID_TURNS, id="good")
        bad = _conversation([("user", "Only one")], id="bad")
        result = pruner.prune_batch([good, bad])
        assert [c.id for c in result] == ["good"]

    def test_preserves_reasoning_and_confidence(self, pruner):
        conv = _conversation(VALID_TURNS, reasoning_trace="trace-1", confidence_score=0.9)
        pruned = pruner.prune_conversation(conv)
        assert pruned is not None
        assert pruned.reasoning_trace == "trace-1"
        assert pruned.confidence_score == 0.9


class TestExtractJson:
    """JSON extraction from LLM responses."""

    def test_extract_from_code_block(self, pruner):
        content = (
            'Some prose.\n```json\n{"conversation": [{"role": "user", "content": "Q"}]}\n```\nMore prose.'
        )
        parsed = pruner.extract_json_from_response(content)
        assert parsed is not None
        assert parsed["conversation"][0]["role"] == "user"

    def test_extract_plain_object(self, pruner):
        parsed = pruner.extract_json_from_response('Response: {"key": "value"}')
        assert parsed == {"key": "value"}

    def test_extract_entire_content(self, pruner):
        parsed = pruner.extract_json_from_response('{"a": 1}')
        assert parsed == {"a": 1}

    def test_no_json_returns_none(self, pruner):
        assert pruner.extract_json_from_response("No structured data here") is None


class TestValidateQuality:
    """Quality scoring should flag common failure modes."""

    def test_high_quality_conversation(self, pruner):
        conv = _conversation(VALID_TURNS)
        is_valid, score, issues = pruner.validate_conversation_quality(conv)
        assert is_valid
        assert score == 1.0
        assert issues == []

    def test_too_few_turns(self, pruner):
        conv = _conversation([("user", "Hello")])
        is_valid, score, issues = pruner.validate_conversation_quality(conv)
        assert not is_valid
        assert "Too few turns" in issues

    def test_too_much_filler(self, pruner):
        conv = _conversation(
            [
                ("user", "Sure!"),
                ("assistant", "Of course!"),
                ("user", "Absolutely!"),
                ("assistant", "Sure."),
            ]
        )
        is_valid, score, issues = pruner.validate_conversation_quality(conv)
        assert any("Too much filler" in issue for issue in issues)

    def test_short_content(self, pruner):
        conv = _conversation([("user", "Hi"), ("assistant", "Yo")])
        is_valid, score, issues = pruner.validate_conversation_quality(conv)
        # Short content alone only reduces the score; it does not reject.
        assert issues == ["Content too short"]
        assert score == pytest.approx(0.7)
        assert is_valid

    def test_invalid_structure(self, pruner):
        conv = _conversation(
            [
                ("user", "Question one with enough content?"),
                ("user", "Question two with enough content?"),
                ("assistant", "An answer that is long enough to survive validation."),
            ]
        )
        is_valid, score, issues = pruner.validate_conversation_quality(conv)
        # Structural issues alone only reduce the score; they do not reject.
        assert issues == ["Invalid conversation structure"]
        assert score == pytest.approx(0.6)
        assert is_valid

"""
Unit tests for the preference/DPO formatter.

The DPO export produces training pairs from conversations, so prompt selection,
chosen/rejected assignment, and validation are pinned down here.
"""

import json

from distill_align.core.schemas import ConversationSchema, SynthesizedTurn
from distill_align.exporter.formatters.preference import PreferenceFormatter


def _conversation(user_turns: list[str], assistant_turns: list[str], conv_id: str = "conv-1") -> ConversationSchema:
    turns = [SynthesizedTurn(role="user", content=u) for u in user_turns]
    turns += [SynthesizedTurn(role="assistant", content=a) for a in assistant_turns]
    return ConversationSchema(id=conv_id, source_chunk_id="chunk-1", turns=turns)


class TestDpoFormat:
    """DPO chosen/rejected assignment."""

    def test_single_assistant_turn_has_empty_rejected(self, tmp_path):
        formatter = PreferenceFormatter(output_dir=tmp_path, format_type="dpo")
        conv = _conversation(["What is X?"], ["A detailed answer about X."])

        path = formatter.format([conv])
        data = json.loads(path.read_text(encoding="utf-8"))

        assert len(data) == 1
        assert data[0]["prompt"] == "What is X?"
        assert data[0]["chosen"] == "A detailed answer about X."
        assert data[0]["rejected"] == ""

    def test_two_assistant_turns_order_based(self, tmp_path):
        formatter = PreferenceFormatter(output_dir=tmp_path, format_type="dpo")
        conv = _conversation(["Question?"], ["First answer.", "Second answer."])

        path = formatter.format([conv])
        data = json.loads(path.read_text(encoding="utf-8"))

        assert data[0]["chosen"] == "First answer."
        assert data[0]["rejected"] == "Second answer."

    def test_first_user_turn_is_prompt(self, tmp_path):
        formatter = PreferenceFormatter(output_dir=tmp_path, format_type="dpo")
        conv = _conversation(["Follow-up on Q?", "Actual first question?"], ["Answer."])

        path = formatter.format([conv])
        data = json.loads(path.read_text(encoding="utf-8"))

        assert data[0]["prompt"] == "Follow-up on Q?"

    def test_conversation_without_both_roles_skipped(self, tmp_path):
        formatter = PreferenceFormatter(output_dir=tmp_path, format_type="dpo")
        user_only = _conversation(["Question?"], [], conv_id="user-only")
        assistant_only = _conversation([], ["Answer."], conv_id="assistant-only")

        path = formatter.format([user_only, assistant_only])
        assert json.loads(path.read_text(encoding="utf-8")) == []


class TestScoredFormat:
    """Scored (non-DPO) format emits labeled response lists."""

    def test_responses_scored_descending(self, tmp_path):
        formatter = PreferenceFormatter(output_dir=tmp_path, format_type="score")
        conv = _conversation(["Prompt text?"], ["Best answer.", "Worse answer."])

        path = formatter.format([conv])
        data = json.loads(path.read_text(encoding="utf-8"))

        entry = data[0]
        assert entry["prompt"] == "Prompt text?"
        assert [r["label"] for r in entry["responses"]] == ["chosen", "rejected"]
        assert entry["responses"][0]["response"] == "Best answer."
        assert entry["responses"][0]["score"] == 1.0
        assert entry["responses"][1]["score"] == 0.5

    def test_no_required_roles_skips_conversation(self, tmp_path):
        formatter = PreferenceFormatter(output_dir=tmp_path, format_type="score")
        conv = _conversation(["Only question."], [])

        path = formatter.format([conv])
        assert json.loads(path.read_text(encoding="utf-8")) == []


class TestOutputAndValidation:
    """Output plumbing and format validation."""

    def test_filename_json_extension_added(self, tmp_path):
        formatter = PreferenceFormatter(output_dir=tmp_path, format_type="dpo")
        conv = _conversation(["Q?"], ["A."])

        path = formatter.format([conv], filename="pairs")
        assert path.name == "pairs.json"
        assert path.exists()

    def test_validate_accepts_dpo_entries(self, tmp_path):
        formatter = PreferenceFormatter(output_dir=tmp_path, format_type="dpo")
        assert formatter.validate([{"prompt": "p", "chosen": "c", "rejected": "r"}])

    def test_validate_rejects_dpo_missing_chosen(self, tmp_path):
        formatter = PreferenceFormatter(output_dir=tmp_path, format_type="dpo")
        assert not formatter.validate([{"prompt": "p", "rejected": "r"}])
        assert not formatter.validate([{"chosen": "c", "rejected": "r"}])

    def test_validate_rejects_non_list(self, tmp_path):
        formatter = PreferenceFormatter(output_dir=tmp_path, format_type="dpo")
        assert not formatter.validate({"prompt": "p"})
        assert not formatter.validate("not-a-list")

    def test_validate_scored_format(self, tmp_path):
        formatter = PreferenceFormatter(output_dir=tmp_path, format_type="score")
        assert formatter.validate([{"prompt": "p", "responses": [{"response": "r", "score": 1.0}]}])
        assert not formatter.validate([{"prompt": "p"}])
        assert not formatter.validate([{"prompt": "p", "responses": "not-a-list"}])

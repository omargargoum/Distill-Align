"""
Unit tests for core schemas.
"""

import pytest

from distill_align.core.schemas import (
    AlpacaEntry,
    ConversationSchema,
    DataChunk,
    ExportConfig,
    IngestionConfig,
    ShareGPTMessage,
    SourceMetadata,
    SynthesisConfig,
    SynthesizedTurn,
)


class TestSourceMetadata:
    """Tests for SourceMetadata schema."""

    def test_default_values(self):
        metadata = SourceMetadata(file_path="/test/file.md", file_name="file.md")
        assert metadata.source_type == "text"
        assert metadata.section_headers == []
        assert metadata.custom_tags == {}

    def test_markdown_metadata(self):
        metadata = SourceMetadata(
            source_type="markdown",
            file_path="/test/file.md",
            file_name="file.md",
            title="Test Document",
            author="Test Author",
            section_headers=["Section 1", "Section 2"],
        )
        assert metadata.source_type == "markdown"
        assert metadata.title == "Test Document"
        assert len(metadata.section_headers) == 2

    def test_code_metadata(self):
        metadata = SourceMetadata(
            source_type="code",
            file_path="/test/file.py",
            file_name="file.py",
            language="python",
            module_path="package.module",
            custom_tags={"functions": ["func1", "func2"], "classes": ["Class1"]},
        )
        assert metadata.source_type == "code"
        assert metadata.language == "python"
        assert len(metadata.custom_tags["functions"]) == 2


class TestDataChunk:
    """Tests for DataChunk schema."""

    def test_auto_id_generation(self):
        metadata = SourceMetadata(file_path="/test/file.md", file_name="file.md")
        chunk = DataChunk(content="Test content", metadata=metadata)
        assert chunk.id != ""
        assert len(chunk.id) == 16

    def test_custom_id(self):
        metadata = SourceMetadata(file_path="/test/file.md", file_name="file.md")
        chunk = DataChunk(content="Test content", metadata=metadata, id="custom-id")
        assert chunk.id == "custom-id"

    def test_source_type_property(self):
        metadata = SourceMetadata(source_type="markdown", file_path="/test/file.md", file_name="file.md")
        chunk = DataChunk(content="Test content", metadata=metadata)
        assert chunk.source_type == "markdown"


class TestSynthesizedTurn:
    """Tests for SynthesizedTurn schema."""

    def test_valid_turn(self):
        turn = SynthesizedTurn(role="user", content="Hello!")
        assert turn.role == "user"
        assert turn.content == "Hello!"

    def test_invalid_role(self):
        with pytest.raises((ValueError, TypeError)):
            SynthesizedTurn(role="invalid", content="Hello!")


class TestConversationSchema:
    """Tests for ConversationSchema schema."""

    def test_get_system_prompt(self):
        conversation = ConversationSchema(
            id="test-id",
            source_chunk_id="chunk-id",
            turns=[
                SynthesizedTurn(role="system", content="You are helpful."),
                SynthesizedTurn(role="user", content="Hello!"),
                SynthesizedTurn(role="assistant", content="Hi!"),
            ],
        )
        assert conversation.get_system_prompt() == "You are helpful."

    def test_get_user_turns(self):
        conversation = ConversationSchema(
            id="test-id",
            source_chunk_id="chunk-id",
            turns=[
                SynthesizedTurn(role="user", content="Question 1"),
                SynthesizedTurn(role="assistant", content="Answer 1"),
                SynthesizedTurn(role="user", content="Question 2"),
            ],
        )
        user_turns = conversation.get_user_turns()
        assert len(user_turns) == 2

    def test_get_assistant_turns(self):
        conversation = ConversationSchema(
            id="test-id",
            source_chunk_id="chunk-id",
            turns=[
                SynthesizedTurn(role="user", content="Question"),
                SynthesizedTurn(role="assistant", content="Answer 1"),
                SynthesizedTurn(role="assistant", content="Answer 2"),
            ],
        )
        assistant_turns = conversation.get_assistant_turns()
        assert len(assistant_turns) == 2


class TestShareGPTMessage:
    """Tests for ShareGPTMessage schema."""

    def test_valid_message(self):
        msg = ShareGPTMessage(from_="human", value="Hello!")
        assert msg.from_ == "human"
        assert msg.value == "Hello!"

    def test_alias_serialization(self):
        msg = ShareGPTMessage(from_="human", value="Hello!")
        data = msg.model_dump(by_alias=True)
        assert "from" in data
        assert data["from"] == "human"


class TestAlpacaEntry:
    """Tests for AlpacaEntry schema."""

    def test_valid_entry(self):
        entry = AlpacaEntry(
            instruction="What is Python?",
            input="",
            output="Python is a programming language.",
        )
        assert entry.instruction == "What is Python?"
        assert entry.system is None

    def test_with_system(self):
        entry = AlpacaEntry(
            instruction="What is Python?",
            input="",
            output="Python is a programming language.",
            system="You are a helpful assistant.",
        )
        assert entry.system == "You are a helpful assistant."


class TestConfigSchemas:
    """Tests for configuration schemas."""

    def test_ingestion_config_defaults(self):
        config = IngestionConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.respect_headers is True

    def test_synthesis_config_defaults(self):
        config = SynthesisConfig()
        assert config.llm_provider == "openai"
        assert config.model_name == "gpt-5-mini"
        assert config.max_concurrency == 5

    def test_export_config_defaults(self):
        config = ExportConfig()
        assert config.formats == ["sharegpt"]
        assert config.generate_unsloth_script is True

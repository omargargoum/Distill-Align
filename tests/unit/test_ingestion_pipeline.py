"""
Unit tests for the ingestion pipeline orchestrator.

The pipeline drives file loading and chunking, so loader routing, error
wrapping, directory scanning, and async behavior are pinned down here.
"""

import pytest

from distill_align.core.exceptions import IngestionError, UnsupportedFormatError
from distill_align.ingestion.loaders.code import CodeLoader
from distill_align.ingestion.loaders.markdown import MarkdownLoader
from distill_align.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def pipeline() -> IngestionPipeline:
    return IngestionPipeline()


@pytest.fixture
def sample_dir(tmp_path, sample_markdown_content):
    """A directory tree with supported and unsupported files."""
    (tmp_path / "guide.md").write_text(sample_markdown_content, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored plain text", encoding="utf-8")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "deep.md").write_text("# Deep\n\nContent in the nested directory.", encoding="utf-8")
    return tmp_path


class TestGetLoader:
    """Loader routing by file extension."""

    def test_markdown_extension(self, pipeline, tmp_path):
        path = tmp_path / "doc.md"
        path.write_text("# Hi", encoding="utf-8")
        assert isinstance(pipeline._get_loader(path), MarkdownLoader)

    def test_code_extension(self, pipeline, tmp_path):
        path = tmp_path / "script.js"
        path.write_text("let x = 1", encoding="utf-8")
        assert isinstance(pipeline._get_loader(path), CodeLoader)

    def test_unsupported_extension_raises(self, pipeline, tmp_path):
        path = tmp_path / "archive.xyz"
        path.write_text("data", encoding="utf-8")
        with pytest.raises(UnsupportedFormatError):
            pipeline._get_loader(path)


class TestIngestFile:
    """Single-file ingestion."""

    def test_ingest_markdown_file(self, pipeline, tmp_path, sample_markdown_content):
        path = tmp_path / "guide.md"
        path.write_text(sample_markdown_content, encoding="utf-8")

        chunks = pipeline.ingest_file(path)
        assert chunks
        assert all(c.metadata.source_type == "markdown" for c in chunks)
        assert all(c.metadata.file_name == "guide.md" for c in chunks)
        assert any("# Introduction" not in c.content for c in chunks)  # headers stripped

    def test_missing_file_raises_ingestion_error(self, pipeline, tmp_path):
        with pytest.raises(IngestionError, match="Failed to ingest"):
            pipeline.ingest_file(tmp_path / "missing.md")

    def test_unsupported_file_raises_ingestion_error(self, pipeline, tmp_path):
        path = tmp_path / "data.xyz"
        path.write_text("data", encoding="utf-8")
        with pytest.raises(IngestionError, match="Unsupported file format"):
            pipeline.ingest_file(path)


class TestIngestDirectory:
    """Directory scanning and per-file failure isolation."""

    def test_recursive_scan_skips_unsupported(self, pipeline, sample_dir):
        chunks = pipeline.ingest_directory(sample_dir, recursive=True)
        # guide.md and nested/deep.md are ingested; notes.txt is ignored.
        file_names = {c.metadata.file_name for c in chunks}
        assert file_names == {"guide.md", "deep.md"}

    def test_non_recursive_scan(self, pipeline, sample_dir):
        chunks = pipeline.ingest_directory(sample_dir, recursive=False)
        assert {c.metadata.file_name for c in chunks} == {"guide.md"}

    def test_missing_directory_raises(self, pipeline, tmp_path):
        with pytest.raises(IngestionError, match="Not a directory"):
            pipeline.ingest_directory(tmp_path / "nope")

    def test_file_patterns_filter(self, pipeline, sample_dir):
        chunks = pipeline.ingest_directory(sample_dir, file_patterns=["*.py"])
        assert chunks == []

    def test_failing_file_skipped(self, pipeline, sample_dir, monkeypatch):
        def fake_ingest(file_path):
            if file_path.name == "guide.md":
                raise IngestionError("boom")
            from distill_align.ingestion.pipeline import IngestionPipeline

            return IngestionPipeline.ingest_file(pipeline, file_path)

        monkeypatch.setattr(pipeline, "ingest_file", fake_ingest)
        chunks = pipeline.ingest_directory(sample_dir, recursive=True)
        assert {c.metadata.file_name for c in chunks} == {"deep.md"}


class TestAsync:
    """Async ingestion variants."""

    @pytest.mark.asyncio
    async def test_ingest_file_async(self, pipeline, tmp_path, sample_markdown_content):
        path = tmp_path / "guide.md"
        path.write_text(sample_markdown_content, encoding="utf-8")

        chunks = await pipeline.ingest_file_async(path)
        assert chunks
        assert chunks[0].metadata.file_name == "guide.md"

    @pytest.mark.asyncio
    async def test_ingest_directory_async(self, pipeline, sample_dir):
        chunks = await pipeline.ingest_directory_async(sample_dir)
        assert {c.metadata.file_name for c in chunks} == {"guide.md", "deep.md"}

    @pytest.mark.asyncio
    async def test_failure_isolated_in_async_directory(self, pipeline, sample_dir, monkeypatch):
        async def fake_ingest_async(file_path):
            raise IngestionError("boom")

        monkeypatch.setattr(pipeline, "ingest_file_async", fake_ingest_async)
        chunks = await pipeline.ingest_directory_async(sample_dir)
        assert chunks == []

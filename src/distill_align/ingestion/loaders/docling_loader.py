"""Docling-backed loader (Phase 2): structure-preserving parse with fallback.

Uses ``docling`` when installed (``pip install distill-align[docling]`` —
not yet a hard dep); otherwise falls back to the legacy pypdf / docx /
HTML loaders so ingestion never breaks on a missing optional dep.
"""

from __future__ import annotations

from loguru import logger

from ...core.exceptions import LoaderError
from ...core.schemas import SourceMetadata
from .base import BaseLoader

_DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md"}


class DoclingLoader(BaseLoader):
    """Structure-preserving loader via Docling with graceful fallback."""

    SUPPORTED_EXTENSIONS = _DOCLING_EXTENSIONS

    def _convert_with_docling(self) -> str | None:
        try:
            from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]
        except ImportError:
            return None
        try:
            converter = DocumentConverter()
            result = converter.convert(str(self.file_path))
            doc = result.document
            # Prefer Markdown export (keeps headings + tables for chunkers)
            if hasattr(doc, "export_to_markdown"):
                return doc.export_to_markdown()  # type: ignore[no-any-return]
            return str(doc)
        except Exception as e:
            logger.warning(f"Docling conversion failed for {self.file_path.name}: {e}; using fallback loader")
            return None

    def _fallback_loader(self) -> BaseLoader:
        from .docx import DOCXLoader
        from .html import HTMLLoader
        from .markdown import MarkdownLoader
        from .pdf import PDFLoader
        from .text import TextLoader

        ext = self.file_path.suffix.lower()
        if ext == ".pdf":
            return PDFLoader(self.file_path)
        if ext == ".docx":
            return DOCXLoader(self.file_path)
        if ext in (".html", ".htm"):
            return HTMLLoader(self.file_path)
        if ext in (".md", ".markdown"):
            return MarkdownLoader(self.file_path)
        return TextLoader(self.file_path)

    def load(self) -> str:
        text = self._convert_with_docling()
        if text:
            return text
        return self._fallback_loader().load()

    def extract_metadata(self) -> SourceMetadata:
        try:
            meta = self._fallback_loader().extract_metadata()
        except Exception as e:
            raise LoaderError(f"Failed to extract metadata: {e}") from e
        tags = dict(meta.custom_tags)
        try:
            import docling  # type: ignore[import-not-found]

            tags["parser"] = f"docling-{getattr(docling, '__version__', 'unknown')}"
        except ImportError:
            tags["parser"] = "fallback"
        return meta.model_copy(update={"custom_tags": tags})


def is_docling_available() -> bool:
    """True when the optional Docling dependency is installed."""
    try:
        import docling  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
        return False

"""
Code chunker with semantic splitting.

Splits code content based on function and class definitions.
"""

import re

from ...core.schemas import DataChunk, SourceMetadata
from .base import BaseChunker


class CodeChunker(BaseChunker):
    """Chunker for source code with definition-aware splitting."""

    # Patterns for common code structures (tree-sitter-lite: regex per lang)
    DEFINITION_PATTERNS = {
        "python": [
            re.compile(r"^(class\s+\w+.*?:)", re.MULTILINE),
            re.compile(r"^((?:async\s+)?def\s+\w+.*?:)", re.MULTILINE),
        ],
        "javascript": [
            re.compile(r"^(class\s+\w+.*?{)", re.MULTILINE),
            re.compile(r"^(function\s+\w+.*?{)", re.MULTILINE),
            re.compile(r"^(const\s+\w+\s*=\s*(?:async\s+)?\(.*?\)\s*=>\s*{)", re.MULTILINE),
        ],
        "typescript": [
            re.compile(r"^(class\s+\w+.*?{)", re.MULTILINE),
            re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+\w+.*?{", re.MULTILINE),
            re.compile(r"^(?:export\s+)?(?:interface|type|enum)\s+\w+", re.MULTILINE),
            re.compile(r"^(const\s+\w+\s*=\s*(?:async\s+)?\(.*?\)\s*=>\s*{)", re.MULTILINE),
        ],
        "java": [
            re.compile(r"^(class\s+\w+.*?{)", re.MULTILINE),
            re.compile(r"^(public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+\w+\s*\(.*?\)\s*{", re.MULTILINE),
        ],
        "go": [
            re.compile(r"^(?:func\s+(?:\(\w+\s+[\w*]+\)\s+)?\w+\s*\(.*?\).*?{)", re.MULTILINE),
            re.compile(r"^(?:type\s+\w+\s+(?:struct|interface).*?{)", re.MULTILINE),
        ],
        "rust": [
            re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+\w+.*?{", re.MULTILINE),
            re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|impl|trait)\s+\w+", re.MULTILINE),
        ],
        "c": [
            re.compile(r"^\w[\w\s*]+\w\s*\(.*?\)\s*{", re.MULTILINE),
        ],
        "cpp": [
            re.compile(r"^(?:class|struct)\s+\w+.*?{", re.MULTILINE),
            re.compile(r"^\w[\w\s:*&<>]+\w\s*\(.*?\)\s*(?:const\s*)?{", re.MULTILINE),
        ],
        "csharp": [
            re.compile(
                r"^(?:public|private|protected|internal)?\s*(?:static\s+)?(?:class|interface|struct|enum)\s+\w+",
                re.MULTILINE,
            ),
            re.compile(
                r"^(?:public|private|protected|internal)?\s*(?:static\s+)?(?:\w+\s+)+\w+\s*\(.*?\)\s*{", re.MULTILINE
            ),
        ],
        "php": [
            re.compile(r"^(?:class|interface|trait)\s+\w+", re.MULTILINE),
            re.compile(r"^(?:public|private|protected)?\s*(?:static\s+)?function\s+\w+\s*\(.*?\)\s*{", re.MULTILINE),
        ],
        "ruby": [
            re.compile(r"^(?:class|module)\s+\w+", re.MULTILINE),
            re.compile(r"^\s*def\s+\w+", re.MULTILINE),
        ],
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        language: str | None = None,
        include_context: bool = True,
    ):
        """
        Initialize the code chunker.

        Args:
            chunk_size: Target size for each chunk.
            chunk_overlap: Overlap between consecutive chunks.
            language: Programming language (auto-detected if not provided).
            include_context: Whether to include surrounding context in chunks.
        """
        super().__init__(chunk_size, chunk_overlap)
        self.language = language
        self.include_context = include_context

    def chunk(self, content: str, metadata: SourceMetadata) -> list[DataChunk]:
        """
        Split code content into chunks.

        Args:
            content: Raw code content.
            metadata: Source metadata.

        Returns:
            List of DataChunk objects.
        """
        if not content.strip():
            return []

        language = self.language or metadata.language or self._detect_language(metadata.file_name)

        # Try to split by definitions first
        chunks = self._chunk_by_definitions(content, metadata, language)

        # If no definitions found or chunks are too large, fall back to size-based splitting
        if not chunks:
            chunks = self._chunk_by_size(content, metadata)

        return chunks

    def _detect_language(self, filename: str) -> str:
        """
        Detect language from filename.

        Args:
            filename: Source filename.

        Returns:
            Detected language string.
        """
        from ..loaders.code import EXTENSION_TO_LANGUAGE

        ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
        return EXTENSION_TO_LANGUAGE.get(ext, "unknown")

    def _chunk_by_definitions(self, content: str, metadata: SourceMetadata, language: str) -> list[DataChunk]:
        """
        Split code by function/class definitions.

        Args:
            content: Raw code content.
            metadata: Source metadata.
            language: Programming language.

        Returns:
            List of DataChunk objects.
        """
        patterns = self.DEFINITION_PATTERNS.get(language, [])
        if not patterns:
            return []

        # Find all definition boundaries
        boundaries = []
        for pattern in patterns:
            for match in pattern.finditer(content):
                boundaries.append(match.start())

        if not boundaries:
            return []

        # Sort boundaries and add start/end
        boundaries = sorted(set(boundaries))
        boundaries.insert(0, 0)
        boundaries.append(len(content))

        # Split content at boundaries
        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            section = content[start:end].strip()

            if not section:
                continue

            # Extract definition name for context
            definition_name = self._extract_definition_name(section, language)

            # If section is small enough, create a chunk
            if len(section) <= self.chunk_size:
                chunk_metadata = metadata.model_copy(
                    update={
                        "custom_tags": {
                            **metadata.custom_tags,
                            "definition": definition_name,
                            "line_start": content[:start].count("\n") + 1,
                            "line_end": content[:end].count("\n") + 1,
                        }
                    }
                )
                chunk = DataChunk(content=section, metadata=chunk_metadata)
                chunks.append(chunk)
            else:
                # Section is too large, split by size
                sub_chunks = self._chunk_by_size(section, metadata)
                chunks.extend(sub_chunks)

        return chunks

    def _extract_definition_name(self, code: str, language: str) -> str:
        """
        Extract the name of the definition from code.

        Args:
            code: Code section.
            language: Programming language.

        Returns:
            Definition name or "unknown".
        """
        if language == "python":
            match = re.match(r"^(?:class|def)\s+(\w+)", code)
            if match:
                return match.group(1)
        elif language in ("javascript", "typescript"):
            match = re.match(r"^(?:export\s+)?(?:class|function|const|interface|type|enum)\s+(\w+)", code)
            if match:
                return match.group(1)
        else:
            # Generic: first identifier after fn/func/def/function/class/struct
            match = re.match(
                r"^(?:.*?\b(?:fn|func|function|def|class|struct|interface|trait|enum|type)\b\s+)([A-Za-z_]\w*)",
                code,
            )
            if match:
                return match.group(1)

        return "unknown"

    def _chunk_by_size(self, content: str, metadata: SourceMetadata) -> list[DataChunk]:
        """
        Split content by size with overlap.

        Args:
            content: Raw content.
            metadata: Source metadata.

        Returns:
            List of DataChunk objects.
        """
        chunks = []
        start = 0

        while start < len(content):
            end = start + self.chunk_size

            # Try to break at line boundary
            if end < len(content):
                newline_pos = content.rfind("\n", start, end)
                if newline_pos > start:
                    end = newline_pos + 1

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunk_metadata = metadata.model_copy(
                    update={
                        "custom_tags": {
                            **metadata.custom_tags,
                            "line_start": content[:start].count("\n") + 1,
                            "line_end": content[:end].count("\n") + 1,
                        }
                    }
                )
                chunk = DataChunk(content=chunk_content, metadata=chunk_metadata)
                chunks.append(chunk)

            # Move start with overlap
            start = end - self.chunk_overlap
            if start >= len(content):
                break

        return chunks

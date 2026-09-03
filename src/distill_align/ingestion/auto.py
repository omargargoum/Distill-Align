"""
Auto-detection ingestion pipeline.

Scans directories, detects file types, and routes to the correct loader automatically.
"""

from pathlib import Path

from loguru import logger

from ..core.exceptions import IngestionError, UnsupportedFormatError
from ..core.schemas import DataChunk, IngestionConfig
from .chunkers.base import BaseChunker
from .chunkers.code import CodeChunker
from .chunkers.markdown import MarkdownChunker
from .loaders.base import BaseLoader
from .loaders.code import CodeLoader
from .loaders.csv_loader import CSVLoader
from .loaders.docx import DOCXLoader
from .loaders.html import HTMLLoader
from .loaders.json_loader import JSONLoader
from .loaders.jupyter import JupyterLoader
from .loaders.markdown import MarkdownLoader
from .loaders.pdf import PDFLoader
from .loaders.text import TextLoader

# Extension to loader mapping
EXTENSION_LOADER_MAP: dict[str, type[BaseLoader]] = {
    # Markdown
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".mdown": MarkdownLoader,
    ".mkd": MarkdownLoader,
    # PDF
    ".pdf": PDFLoader,
    # DOCX
    ".docx": DOCXLoader,
    # HTML
    ".html": HTMLLoader,
    ".htm": HTMLLoader,
    # Jupyter
    ".ipynb": JupyterLoader,
    # JSON
    ".json": JSONLoader,
    ".jsonl": JSONLoader,
    # CSV
    ".csv": CSVLoader,
    # Code (will be handled by CodeLoader)
    ".py": CodeLoader,
    ".js": CodeLoader,
    ".ts": CodeLoader,
    ".jsx": CodeLoader,
    ".tsx": CodeLoader,
    ".java": CodeLoader,
    ".cpp": CodeLoader,
    ".c": CodeLoader,
    ".h": CodeLoader,
    ".hpp": CodeLoader,
    ".rs": CodeLoader,
    ".go": CodeLoader,
    ".rb": CodeLoader,
    ".php": CodeLoader,
    ".swift": CodeLoader,
    ".kt": CodeLoader,
    ".scala": CodeLoader,
    ".cs": CodeLoader,
    ".sh": CodeLoader,
    ".bash": CodeLoader,
    ".zsh": CodeLoader,
    ".sql": CodeLoader,
    ".r": CodeLoader,
    ".R": CodeLoader,
    ".lua": CodeLoader,
    ".pl": CodeLoader,
    # Text
    ".txt": TextLoader,
    ".log": TextLoader,
    ".cfg": TextLoader,
    ".ini": TextLoader,
    ".yaml": TextLoader,
    ".yml": TextLoader,
    ".toml": TextLoader,
    ".rst": TextLoader,
    ".org": TextLoader,
}


class AutoIngestionPipeline:
    """
    Auto-detection ingestion pipeline.

    Scans directories, detects file types automatically, and routes
    to the appropriate loader without manual configuration.
    """

    def __init__(self, config: IngestionConfig | None = None):
        """
        Initialize the auto-detection pipeline.

        Args:
            config: Optional ingestion configuration.
        """
        self.config = config or IngestionConfig()
        self._chunkers: dict[str, BaseChunker] = {}
        self._loaders: dict[str, BaseLoader] = {}

    def get_loader(self, file_path: Path) -> BaseLoader:
        """
        Get the appropriate loader for a file based on extension.

        Args:
            file_path: Path to the file.

        Returns:
            Loader instance.

        Raises:
            UnsupportedFormatError: If file format is not supported.
        """
        ext = file_path.suffix.lower()

        if ext in EXTENSION_LOADER_MAP:
            loader_class = EXTENSION_LOADER_MAP[ext]
            return loader_class(file_path)

        # Try text loader as fallback for unknown extensions
        try:
            # Quick check if it's a text file
            with open(file_path, encoding="utf-8") as f:
                f.read(1024)
            return TextLoader(file_path)
        except (UnicodeDecodeError, Exception):
            pass

        raise UnsupportedFormatError(f"Unsupported file format: {ext}")

    def get_chunker(self, source_type: str) -> BaseChunker:
        """
        Get the appropriate chunker for a source type.

        Honors ``IngestionConfig.chunker``: ``"auto"`` keeps legacy routing
        (markdown → header-aware, code → definition-aware, else plain);
        explicit values force semantic / parent-child / late / recursive.

        Args:
            source_type: Type of source content.

        Returns:
            Chunker instance.
        """
        forced = getattr(self.config, "chunker", "auto")
        cache_key = f"{source_type}:{forced}"
        if cache_key in self._chunkers:
            return self._chunkers[cache_key]

        base_kwargs = {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
        }

        chunker: BaseChunker
        if forced == "semantic":
            from .chunkers.semantic import SemanticChunker

            chunker = SemanticChunker(
                **base_kwargs,
                breakpoint=getattr(self.config, "semantic_breakpoint", "percentile"),
                threshold=getattr(self.config, "semantic_threshold", 75.0),
            )
        elif forced == "parent_child":
            from .chunkers.parent_child import ParentChildChunker

            chunker = ParentChildChunker(**base_kwargs)
        elif forced == "late":
            from .chunkers.late import LateChunker

            chunker = LateChunker(**base_kwargs)
        elif forced == "recursive":
            chunker = MarkdownChunker(**base_kwargs, respect_headers=False)
        elif forced == "markdown":
            chunker = MarkdownChunker(**base_kwargs, respect_headers=self.config.respect_headers)
        elif forced == "code":
            chunker = CodeChunker(**base_kwargs)
        elif source_type == "markdown":
            chunker = MarkdownChunker(
                **base_kwargs,
                respect_headers=self.config.respect_headers,
            )
        elif source_type == "code":
            chunker = CodeChunker(**base_kwargs)
        else:
            chunker = MarkdownChunker(**base_kwargs, respect_headers=False)

        self._chunkers[cache_key] = chunker
        return chunker

    def scan_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        ignore_patterns: list[str] | None = None,
    ) -> list[Path]:
        """
        Scan a directory for supported files.

        Args:
            directory: Directory to scan.
            recursive: Whether to search subdirectories.
            ignore_patterns: Directory/file names to ignore (exact match against
                path components). If None, common cache and VCS dirs are ignored.

        Returns:
            List of supported file paths.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise IngestionError(f"Not a directory: {directory}")

        ignore = set(
            ignore_patterns
            or [
                "__pycache__",
                ".git",
                ".svn",
                "node_modules",
                ".venv",
                "venv",
                "env",
                ".env",
                ".cache",
                ".distill-align",
            ]
        )

        supported_extensions = set(EXTENSION_LOADER_MAP.keys())
        files = []

        pattern = "**/*" if recursive else "*"
        try:
            for path in directory.glob(pattern):
                if not path.is_file():
                    continue

                # Check ignore patterns
                rel_path = path.relative_to(directory)
                parts = rel_path.parts
                if any(ig in parts for ig in ignore):
                    continue

                # Check extension
                if path.suffix.lower() in supported_extensions:
                    files.append(path)
        except PermissionError:
            logger.warning(f"Permission denied while scanning {directory}, some files may be skipped")

        logger.info(f"Found {len(files)} supported files in {directory}")
        return sorted(files)

    def ingest_file(self, file_path: str | Path) -> list[DataChunk]:
        """
        Ingest a single file.

        Args:
            file_path: Path to the file.

        Returns:
            List of DataChunk objects.
        """
        file_path = Path(file_path)
        logger.info(f"Ingesting file: {file_path}")

        try:
            loader = self.get_loader(file_path)
            metadata = loader.extract_metadata()
            chunker = self.get_chunker(metadata.source_type)
            chunks = loader.to_chunks(chunker)

            # Table serialization: emit row-sentences alongside grids so
            # each fact is individually retrievable (Phase 2).
            serialize_mode = getattr(self.config, "serialize_tables", "both")
            if serialize_mode != "markdown":
                from .tables import serialize_markdown_tables

                enriched: list[DataChunk] = []
                for chunk in chunks:
                    if "|" in chunk.content and "---" in chunk.content:
                        new_content = serialize_markdown_tables(chunk.content, mode=serialize_mode)
                        if new_content != chunk.content:
                            enriched.append(
                                DataChunk(content=new_content, metadata=chunk.metadata, tokens=chunk.tokens)
                            )
                            continue
                    enriched.append(chunk)
                chunks = enriched

            # Contextual heading-path prefix (cheap late-chunking half).
            if getattr(self.config, "contextual_prefix", False):
                from .chunkers.late import LateChunker

                prefixer = LateChunker(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )
                prefixed: list[DataChunk] = []
                for chunk in chunks:
                    if chunk.content.startswith("[Context:"):
                        prefixed.append(chunk)
                        continue
                    heading = " > ".join(chunk.metadata.section_headers) or (chunk.metadata.title or "")
                    lead = " ".join(chunk.content.split())[:300]
                    pfx = f"[Context: {heading} | {lead}]" if (heading or lead) else ""
                    if pfx:
                        tags = {**chunk.metadata.custom_tags, "context_prefix": pfx}
                        fresh = chunk.metadata.model_copy(update={"custom_tags": tags})
                        prefixed.append(DataChunk(content=f"{pfx}\n\n{chunk.content}", metadata=fresh))
                    else:
                        prefixed.append(chunk)
                _ = prefixer  # (explicit LateChunker use stays available via chunker="late")
                chunks = prefixed

            # Optional embeddings for semantic dedup downstream.
            if getattr(self.config, "enable_embeddings", False):
                from .embeddings import get_embedder

                embedder = get_embedder()
                try:
                    results = embedder.embed([c.content for c in chunks])
                    for chunk, res in zip(chunks, results, strict=False):
                        chunk.embedding = res.embedding
                except Exception as e:
                    logger.warning(f"Embeddings enrichment failed: {e}")

            # Optional PII/secret scanning
            if self.config.scan_pii:
                from ..core.pii_filter import PIIFilter

                pii_filter = PIIFilter(redact=self.config.redact_pii)
                scanned_chunks: list[DataChunk] = []
                has_findings = False
                for chunk in chunks:
                    result = pii_filter.scan_text(chunk.content)
                    if result.total_findings > 0:
                        has_findings = True
                        scanned_chunks.append(
                            DataChunk(
                                content=result.redacted_text if self.config.redact_pii else chunk.content,
                                metadata=chunk.metadata,
                                id=chunk.id,
                                tokens=chunk.tokens,
                            )
                        )
                    else:
                        scanned_chunks.append(chunk)

                if has_findings:
                    logger.warning(f"PII/secret items detected in {file_path.name}")
                chunks = scanned_chunks

            logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
            return chunks

        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            raise IngestionError(f"Failed to ingest {file_path}: {e}") from e

    def ingest_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        ignore_patterns: list[str] | None = None,
        progress_callback=None,
    ) -> list[DataChunk]:
        """
        Ingest all supported files in a directory.

        Args:
            directory: Directory to scan.
            recursive: Whether to search subdirectories.
            ignore_patterns: Glob patterns to ignore.
            progress_callback: Optional callback(current, total, filename).

        Returns:
            List of DataChunk objects from all files.
        """
        files = self.scan_directory(directory, recursive, ignore_patterns)

        all_chunks = []
        failed = 0

        for i, file_path in enumerate(files):
            try:
                if progress_callback:
                    progress_callback(i + 1, len(files), file_path.name)

                chunks = self.ingest_file(file_path)
                all_chunks.extend(chunks)

            except Exception as e:
                logger.warning(f"Skipping {file_path}: {e}")
                failed += 1
                continue

        logger.info(f"Ingestion complete: {len(all_chunks)} chunks from {len(files) - failed}/{len(files)} files")
        return all_chunks

    def get_supported_extensions(self) -> list[str]:
        """Get list of all supported file extensions."""
        return sorted(set(EXTENSION_LOADER_MAP.keys()))

"""Parent-child (small-to-big) + late-contextual chunkers (Phase 2).

ParentChildChunker: small child chunks carry retrieval-grade precision;
each child stores its parent id + parent text in ``custom_tags`` so the
LLM gets full context at generation time (LangChain ParentDocument /
LlamaIndex AutoMerging pattern).

LateChunker: cheap contextual-retrieval approximation without an LLM —
prepends the document heading path + leading summary to every chunk
before embedding (full Anthropic contextual retrieval remains an
opt-in synthesis mode; this covers the offline half).
"""

from __future__ import annotations

import re
import uuid

from ...core.schemas import DataChunk, SourceMetadata
from .base import BaseChunker
from .markdown import MarkdownChunker

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class ParentChildChunker(BaseChunker):
    """Small-to-big chunking: retrieve small, read big."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        child_size: int = 256,
        child_overlap: int = 50,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.child_size = min(child_size, chunk_size)
        self.child_overlap = min(child_overlap, max(child_size - 1, 0))
        self._parent = MarkdownChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, respect_headers=True)
        self._child = MarkdownChunker(
            chunk_size=self.child_size, chunk_overlap=self.child_overlap, respect_headers=False
        )

    def chunk(self, content: str, metadata: SourceMetadata) -> list[DataChunk]:
        if not content.strip():
            return []
        parents = self._parent.chunk(content, metadata)
        out: list[DataChunk] = []
        for parent in parents:
            parent_id = parent.id or uuid.uuid4().hex[:16]
            children = self._child.chunk(parent.content, parent.metadata)
            if not children:
                continue
            if len(children) == 1:
                c = children[0]
                c.metadata.custom_tags = {
                    **c.metadata.custom_tags,
                    "parent_id": parent_id,
                    "parent_text": parent.content,
                    "is_parent": True,
                }
                out.append(c)
                continue
            for child in children:
                child.metadata.custom_tags = {
                    **child.metadata.custom_tags,
                    "parent_id": parent_id,
                    "parent_text": parent.content,
                    "is_parent": False,
                }
                out.append(child)
        return out or parents


class LateChunker(BaseChunker):
    """Contextual-prefix chunker (offline half of contextual retrieval).

    Prepends ``[Context: <heading path> | <lead summary>]`` to each chunk
    so embeddings carry document-level context (Jina late-chunking spirit
    without requiring a long-context embedder). Set
    ``IngestionConfig.contextual_prefix`` or use directly.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        lead_chars: int = 300,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.lead_chars = lead_chars
        self._inner = MarkdownChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, respect_headers=True)

    def _heading_path(self, content: str) -> str:
        headers = [m.group(2).strip() for m in _HEADER_RE.finditer(content)]
        return " > ".join(headers[:4])

    def chunk(self, content: str, metadata: SourceMetadata) -> list[DataChunk]:
        if not content.strip():
            return []
        heading_path = self._heading_path(content) or (metadata.title or "")
        lead = " ".join(content.strip().split())[: self.lead_chars]
        prefix = f"[Context: {heading_path} | {lead}]" if heading_path or lead else ""
        chunks = self._inner.chunk(content, metadata)
        if not prefix:
            return chunks
        out: list[DataChunk] = []
        for c in chunks:
            if c.content.startswith("[Context:"):
                out.append(c)
                continue
            new_content = f"{prefix}\n\n{c.content}"
            tags = {**c.metadata.custom_tags, "context_prefix": prefix, "heading_path": heading_path}
            # fresh id: content changed
            fresh_meta = c.metadata.model_copy(update={"custom_tags": tags})
            out.append(DataChunk(content=new_content, metadata=fresh_meta))
        return out

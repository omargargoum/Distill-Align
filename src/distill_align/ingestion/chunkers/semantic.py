"""Semantic chunker (Phase 2): embedding-based topic-boundary splitting.

Splits text into sentences, embeds them (HashEmbedder offline by default,
or any object with ``embed(list[str]) -> list[EmbeddingResult]``), then
cuts where adjacent similarity drops below a percentile/gradient/fixed
threshold. Falls back to recursive size-splitting when embeddings are
flat (e.g. tiny inputs).

Benchmark note (2026): semantic chunking gives the highest retrieval
recall (~92%) but lower end-to-end accuracy than recursive splitting on
some corpora — expose both and let evals decide.
"""

from __future__ import annotations

import re
from typing import Any

from ...core.schemas import DataChunk, SourceMetadata
from ..embeddings import HashEmbedder, cosine_similarity
from .base import BaseChunker

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n\n+")


class SemanticChunker(BaseChunker):
    """Embedding-boundary chunker with size guardrails."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        breakpoint: str = "percentile",
        threshold: float = 75.0,
        embedder: Any | None = None,
        min_sentences_per_chunk: int = 2,
    ):
        super().__init__(chunk_size, chunk_overlap)
        if breakpoint not in ("percentile", "gradient", "fixed"):
            raise ValueError(f"Unknown breakpoint mode: {breakpoint}")
        self.breakpoint = breakpoint
        self.threshold = threshold
        self.embedder = embedder or HashEmbedder()
        self.min_sentences = max(1, min_sentences_per_chunk)

    def _sentences(self, content: str) -> list[str]:
        parts = [s.strip() for s in _SENT_SPLIT.split(content.strip()) if s.strip()]
        return parts or ([content.strip()] if content.strip() else [])

    def _breakpoints(self, sims: list[float]) -> set[int]:
        """Return sentence indices AFTER which to cut."""
        if not sims:
            return set()
        if self.breakpoint == "fixed":
            return {i for i, s in enumerate(sims) if s < (self.threshold / 100.0)}
        if self.breakpoint == "gradient":
            # Cut where the drop vs. previous similarity is largest (top-k by rank)
            drops = [0.0] + [max(0.0, sims[i - 1] - sims[i]) for i in range(1, len(sims))]
            ranked = sorted(range(len(drops)), key=lambda i: drops[i], reverse=True)
            n_cuts = max(1, len(sims) // 6)
            return {i for i in ranked[:n_cuts] if drops[i] > 0.02}
        # percentile: cut at the lowest (100 - threshold)% similarities
        ordered = sorted(sims)
        idx = min(len(ordered) - 1, int(len(ordered) * (self.threshold / 100.0)))
        cutoff = ordered[idx]
        return {i for i, s in enumerate(sims) if s <= cutoff}

    def chunk(self, content: str, metadata: SourceMetadata) -> list[DataChunk]:
        if not content.strip():
            return []
        sentences = self._sentences(content)
        if len(sentences) <= self.min_sentences:
            return [self._create_chunk(content.strip(), metadata)]

        embeddings = [r.embedding for r in self.embedder.embed(sentences)]
        sims = [cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]
        cuts = self._breakpoints(sims)

        # Group sentences into topic blocks
        blocks: list[list[str]] = [[]]
        for i, sent in enumerate(sentences):
            blocks[-1].append(sent)
            if i in cuts and len(blocks[-1]) >= self.min_sentences:
                blocks.append([])
        blocks = [b for b in blocks if b]

        # Pack blocks into size-bounded chunks with overlap
        chunks: list[DataChunk] = []
        buf: list[str] = []
        buf_len = 0
        for block in blocks:
            text = " ".join(block)
            if buf_len + len(text) > self.chunk_size and buf:
                chunks.append(self._create_chunk(" ".join(buf).strip(), metadata))
                # Overlap: carry trailing sentences
                overlap_text = " ".join(buf)
                carry = overlap_text[max(0, len(overlap_text) - self.chunk_overlap) :]
                buf = [carry] if carry.strip() else []
                buf_len = len(" ".join(buf))
            buf.append(text)
            buf_len += len(text) + 1
        if buf and " ".join(buf).strip():
            chunks.append(self._create_chunk(" ".join(buf).strip(), metadata))
        return chunks or [self._create_chunk(content.strip(), metadata)]

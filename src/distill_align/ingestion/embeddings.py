"""Embedding client + semantic dedup helpers (Phase 2).

No hard dependency: ships a deterministic ``HashEmbedder`` (char-trigram
hashing, L2-normalised) for offline semantic chunking/dedup, plus an
``OpenAIEmbedder`` for OpenAI-compatible ``/embeddings`` endpoints
(OpenAI, Together, Ollama ``/api/embed``, vLLM). Real deployments should
prefer ``bge-m3`` / ``text-embedding-3-small`` via ``OpenAIEmbedder``.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from loguru import logger
from pydantic import BaseModel


class EmbeddingResult(BaseModel):
    """Single embedding result."""

    embedding: list[float]
    model: str = ""
    tokens: int = 0


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1] (0.0 for zero vectors)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class HashEmbedder:
    """Deterministic offline embedder (no network, no new dep).

    Char-trigram hashing into a fixed-dim vector, L2-normalised. Good
    enough for breakpoint detection and near-dup clustering; NOT a
    replacement for learned embeddings in production retrieval.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model_name = f"hash-trigram-{dim}"

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        results: list[EmbeddingResult] = []
        for text in texts:
            vec = [0.0] * self.dim
            lowered = text.lower()
            for i in range(max(len(lowered) - 2, 1)):
                trigram = lowered[i : i + 3]
                h = int(hashlib.md5(trigram.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vec = [v / norm for v in vec]
            results.append(EmbeddingResult(embedding=vec, model=self.model_name, tokens=max(1, len(text) // 4)))
        return results


class OpenAIEmbedder:
    """OpenAI-compatible embeddings client (lazy httpx, sync wrapper)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        import httpx

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with httpx.Client(base_url=self.base_url, headers=headers, timeout=self.timeout) as client:
                resp = client.post("/embeddings", json={"model": self.model, "input": texts})
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning(f"Embeddings request failed ({e}); falling back to HashEmbedder")
            return HashEmbedder().embed(texts)
        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        usage = data.get("usage", {})
        per = (usage.get("total_tokens", 0) // max(len(texts), 1)) if usage else 0
        return [EmbeddingResult(embedding=d.get("embedding", []), model=self.model, tokens=per) for d in items]


def find_duplicates(
    embeddings: list[list[float]],
    threshold: float = 0.90,
) -> list[tuple[int, int, float]]:
    """Find near-duplicate pairs at cosine >= threshold.

    Args:
        embeddings: L2-normalised vectors (any dim, same length).
        threshold: Cosine threshold (NeMo-Curator style default 0.90).

    Returns:
        List of (i, j, score) with i < j.
    """
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            score = cosine_similarity(embeddings[i], embeddings[j])
            if score >= threshold:
                pairs.append((i, j, score))
    return pairs


def dedupe_indices(
    embeddings: list[list[float]],
    threshold: float = 0.90,
) -> list[int]:
    """Return indices to keep (first occurrence wins)."""
    drop: set[int] = set()
    for i, j, _ in find_duplicates(embeddings, threshold):
        if i not in drop:
            drop.add(j)
    return [i for i in range(len(embeddings)) if i not in drop]


def get_embedder(model: str = "", base_url: str | None = None, api_key: str | None = None) -> Any:
    """Factory: OpenAI-compatible embedder when configured, else HashEmbedder."""
    if model and model != "hash":
        return OpenAIEmbedder(model=model, base_url=base_url or "https://api.openai.com/v1", api_key=api_key)
    return HashEmbedder()

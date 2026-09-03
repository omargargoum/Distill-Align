"""
SQLite-backed cache manager for synthesis results.

Provides persistent caching with statistics, pruning, and resume support.

Thread-safety:
    All SQLite operations use short-lived connections (per-call) and WAL mode,
    making concurrent access safe. Stats counters (``_hits``, ``_misses``) are
    not atomic — they are cosmetic only and may be slightly off under high
    contention.
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from .json_utils import safe_json_loads_value


class CacheStats(BaseModel):
    """Cache statistics."""

    total_entries: int = 0
    hit_count: int = 0
    miss_count: int = 0
    hit_rate: float = 0.0
    db_size_bytes: int = 0
    db_size_mb: float = 0.0
    oldest_entry: float | None = None
    newest_entry: float | None = None


class CacheEntry(BaseModel):
    """A single cache entry."""

    key: str
    value: Any
    model: str = ""
    provider: str = ""
    tokens_used: int = 0
    created_at: float = 0.0
    accessed_at: float = 0.0
    access_count: int = 0


class CacheManager:
    """
    SQLite-backed cache for synthesis results.

    Features:
    - Persistent storage across runs
    - Hit/miss statistics
    - TTL-based expiration
    - Pruning by age or size
    - Resume support (skip cached chunks)
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        db_name: str = "synthesis_cache.db",
        ttl_days: int = 30,
    ):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory for cache database.
            db_name: SQLite database filename.
            ttl_days: Default time-to-live in days.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / db_name
        self.ttl_seconds = ttl_days * 86400

        # Stats tracking (in-memory, resets each session)
        self._hits = 0
        self._misses = 0
        self._db_failed = False

        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database schema with WAL mode and integrity checks."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Performance + concurrency: WAL mode reduces lock contention
                conn.execute("PRAGMA journal_mode=WAL")
                # Safer durability setting for WAL: writes acknowledged by OS
                conn.execute("PRAGMA synchronous=NORMAL")
                # Run a quick integrity check on startup
                (integrity_result,) = conn.execute("PRAGMA quick_check").fetchone()
                if integrity_result != "ok":
                    logger.warning(f"Cache database integrity check failed: {integrity_result}")
                else:
                    logger.debug("Cache database integrity check passed")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        model TEXT DEFAULT '',
                        provider TEXT DEFAULT '',
                        tokens_used INTEGER DEFAULT 0,
                        created_at REAL NOT NULL,
                        accessed_at REAL NOT NULL,
                        access_count INTEGER DEFAULT 0
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cache_created_at
                    ON cache(created_at)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cache_model
                    ON cache(model)
                """)
                conn.commit()
        except sqlite3.DatabaseError as e:
            logger.error(f"Failed to initialize cache database: {e}. Cache will be unavailable.")
            self._db_failed = True

    @staticmethod
    def make_key(content: str, model: str = "", prompt_id: str = "") -> str:
        """
        Generate a deterministic cache key from content and parameters.

        Args:
            content: The input content (chunk text).
            model: Model name used for synthesis.
            prompt_id: Identifier for the prompt template used.

        Returns:
            Hex digest cache key.
        """
        hash_input = f"{content}|{model}|{prompt_id}"
        return hashlib.sha256(hash_input.encode()).hexdigest()

    @staticmethod
    def make_canonical_key(
        content: str,
        model: str = "",
        prompt_id: str = "",
        temperature: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Generate a canonical cache key immune to ``str(dict)`` ordering.

        Unlike :meth:`make_key`, payloads are serialised with sorted keys so
        identical items hash identically regardless of dict insertion order.
        Versioned (``v2:`` prefix) so v1 and v2 keys never collide.

        Args:
            content: Input content (chunk text or canonical JSON payload).
            model: Model name.
            prompt_id: Prompt template id (e.g. ``socratic:v2``).
            temperature: Optional sampling temperature (part of the key).
            extra: Optional extra params (sorted before hashing).

        Returns:
            Hex digest cache key.
        """
        payload = {
            "content": content,
            "model": model,
            "prompt_id": prompt_id,
            "temperature": temperature,
            "extra": extra or {},
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(f"v2:{canonical}".encode()).hexdigest()

    @staticmethod
    def make_key_for_item(
        item: dict[str, Any],
        model: str = "",
        prompt_id: str = "",
        temperature: float | None = None,
    ) -> str:
        """Build a canonical key for a worker item dict.

        Serialises with sorted keys instead of ``str(item)`` so key order
        and Python repr details cannot cause false cache misses.
        """
        try:
            content = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            content = str(item)
        return CacheManager.make_canonical_key(
            content=content, model=model, prompt_id=prompt_id, temperature=temperature
        )

    def get(self, key: str) -> dict[str, Any] | None:
        """
        Retrieve a cached result.

        Args:
            key: Cache key.

        Returns:
            Cached value as dict, or None if not found/expired.
        """
        if self._db_failed:
            self._misses += 1
            return None

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM cache WHERE key = ?", (key,)).fetchone()

                if row is None:
                    self._misses += 1
                    return None

                # Check TTL
                if time.time() - row["created_at"] > self.ttl_seconds:
                    self._misses += 1
                    self.delete(key)
                    return None

                # Update access stats
                conn.execute(
                    "UPDATE cache SET accessed_at = ?, access_count = access_count + 1 WHERE key = ?",
                    (time.time(), key),
                )
                conn.commit()

                self._hits += 1
                return {
                    "value": safe_json_loads_value(row["value"]),
                    "model": row["model"],
                    "provider": row["provider"],
                    "tokens_used": row["tokens_used"],
                    "created_at": row["created_at"],
                }
        except sqlite3.DatabaseError as e:
            logger.error(f"Cache read error: {e}")
            self._db_failed = True
            self._misses += 1
            return None

    def set(
        self,
        key: str,
        value: Any,
        model: str = "",
        provider: str = "",
        tokens_used: int = 0,
    ) -> None:
        """
        Store a result in the cache.

        Args:
            key: Cache key.
            value: Value to cache (must be JSON-serializable).
            model: Model name used.
            provider: Provider name.
            tokens_used: Token count for this request.
        """
        if self._db_failed:
            return

        now = time.time()
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    INSERT INTO cache
                        (key, value, model, provider, tokens_used, created_at, accessed_at, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(key) DO UPDATE SET
                        value       = excluded.value,
                        model       = excluded.model,
                        provider    = excluded.provider,
                        tokens_used = excluded.tokens_used,
                        accessed_at = excluded.accessed_at,
                        access_count = access_count + 1
                    """,
                    (key, json.dumps(value), model, provider, tokens_used, now, now),
                )
                conn.commit()
        except sqlite3.DatabaseError as e:
            logger.error(f"Cache write error: {e}")
            self._db_failed = True

    def delete(self, key: str) -> bool:
        """
        Delete a cache entry.

        Args:
            key: Cache key.

        Returns:
            True if entry was deleted.
        """
        if self._db_failed:
            return False

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.DatabaseError as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def has(self, key: str) -> bool:
        """
        Check if a key exists and is not expired.

        Args:
            key: Cache key.

        Returns:
            True if key exists and is valid.
        """
        return self.get(key) is not None

    def stats(self) -> CacheStats:
        """
        Get cache statistics.

        Returns:
            CacheStats with hit rate, entry counts, and size info.
        """
        total = 0
        oldest = None
        newest = None

        if not self._db_failed:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    row = conn.execute(
                        "SELECT COUNT(*) as total, MIN(created_at) as oldest, MAX(created_at) as newest FROM cache"
                    ).fetchone()

                    total = row[0]
                    oldest = row[1]
                    newest = row[2]
            except sqlite3.DatabaseError as e:
                logger.error(f"Cache stats error: {e}")

        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return CacheStats(
            total_entries=total,
            hit_count=self._hits,
            miss_count=self._misses,
            hit_rate=hit_rate,
            db_size_bytes=db_size,
            db_size_mb=round(db_size / (1024 * 1024), 2),
            oldest_entry=oldest,
            newest_entry=newest,
        )

    def prune(self, older_than_days: int | None = None) -> int:
        """
        Remove old or expired entries.

        Args:
            older_than_days: Remove entries older than N days. Uses TTL if not specified.

        Returns:
            Number of entries removed.
        """
        if self._db_failed:
            return 0

        cutoff = time.time() - (older_than_days * 86400 if older_than_days else self.ttl_seconds)
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
                conn.commit()
                removed = cursor.rowcount
        except sqlite3.DatabaseError as e:
            logger.error(f"Cache prune error: {e}")
            return 0

        if removed > 0:
            logger.info(
                f"Pruned {removed} cache entries older than {older_than_days or self.ttl_seconds // 86400} days"
            )
        return removed

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries removed.
        """
        if self._db_failed:
            return 0

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("DELETE FROM cache")
                conn.commit()
                removed = cursor.rowcount
        except sqlite3.DatabaseError as e:
            logger.error(f"Cache clear error: {e}")
            return 0

        logger.info(f"Cleared {removed} cache entries")
        return removed

    def get_cached_keys(self, model: str | None = None) -> list[str]:
        """
        Get all cached keys, optionally filtered by model.

        Args:
            model: Optional model name filter.

        Returns:
            List of cache keys.
        """
        if self._db_failed:
            return []

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                if model:
                    rows = conn.execute("SELECT key FROM cache WHERE model = ?", (model,)).fetchall()
                else:
                    rows = conn.execute("SELECT key FROM cache").fetchall()
            return [row[0] for row in rows]
        except sqlite3.DatabaseError as e:
            logger.error(f"Cache get_cached_keys error: {e}")
            return []

    def get_total_tokens(self, model: str | None = None) -> int:
        """
        Get total tokens used across all cached entries.

        Args:
            model: Optional model name filter.

        Returns:
            Total token count.
        """
        if self._db_failed:
            return 0

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                if model:
                    row = conn.execute(
                        "SELECT COALESCE(SUM(tokens_used), 0) FROM cache WHERE model = ?",
                        (model,),
                    ).fetchone()
                else:
                    row = conn.execute("SELECT COALESCE(SUM(tokens_used), 0) FROM cache").fetchone()
            return int(row[0])  # type: ignore[arg-type]
        except sqlite3.DatabaseError as e:
            logger.error(f"Cache get_total_tokens error: {e}")
            return 0

    def close(self) -> None:
        """Close the cache and checkpoint WAL to main database."""
        if not self._db_failed and self.db_path.exists():
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    # Flush WAL to main database
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    conn.commit()
            except sqlite3.DatabaseError:
                pass  # Best-effort checkpoint

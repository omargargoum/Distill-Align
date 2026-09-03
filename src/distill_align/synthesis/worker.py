"""
Async batch worker pool for LLM synthesis.

Features:
- Semaphore-based concurrency control
- Rate limiting (requests per minute)
- Exponential backoff retries
- SQLite-backed caching (CacheManager)
- Checkpoint support for crash recovery
- Progress tracking with callbacks
"""

import asyncio
import time
from collections.abc import Callable
from typing import Any

from loguru import logger

from ..core.cache import CacheManager
from ..core.checkpoint import CheckpointManager
from ..core.exceptions import LLMClientError, RateLimitError
from .models.base import BaseLLMClient


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, max_rpm: int = 60):
        self.max_rpm = max_rpm
        self.interval = 60.0 / max_rpm
        self._last_request_time: float | None = None
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request can be made."""
        async with self._lock:
            now = time.monotonic()
            if self._last_request_time is not None:
                elapsed = now - self._last_request_time
                if elapsed < self.interval:
                    await asyncio.sleep(self.interval - elapsed)
            self._last_request_time = time.monotonic()


class BatchWorker:
    """
    Async worker pool for batch LLM processing.

    Integrates CacheManager for persistent caching and CheckpointManager
    for crash recovery and resume support.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_concurrency: int = 5,
        max_rpm: int = 60,
        retry_attempts: int = 5,
        cache_manager: CacheManager | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        cache_dir: str = ".cache",
        cache_ttl_days: int = 30,
        use_cache: bool = True,
    ):
        """
        Initialize the batch worker.

        Args:
            llm_client: LLM client instance.
            max_concurrency: Maximum concurrent requests.
            max_rpm: Maximum requests per minute.
            retry_attempts: Maximum retry attempts per request.
            cache_manager: Optional CacheManager instance.
            checkpoint_manager: Optional CheckpointManager instance.
            cache_dir: Directory for cache database.
            cache_ttl_days: Cache time-to-live in days.
            use_cache: Whether to use caching. When False, no CacheManager is
                created even if *cache_manager* is provided.
        """
        self.llm_client = llm_client
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.rate_limiter = RateLimiter(max_rpm)
        self.retry_attempts = retry_attempts

        # Cache (SQLite-backed) — only create if caching is enabled
        self._use_cache = use_cache
        self.cache = (
            cache_manager
            if use_cache and cache_manager is not None
            else (CacheManager(cache_dir=cache_dir, ttl_days=cache_ttl_days) if use_cache else None)
        )

        # Checkpoint (optional)
        self.checkpoint = checkpoint_manager

        # Statistics (protected by lock for concurrent access)
        self.stats = {
            "total": 0,
            "completed": 0,
            "failed": 0,
            "cached": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }
        self._stats_lock = asyncio.Lock()

    async def _increment_stat(self, key: str, amount: int = 1) -> None:
        """Thread/async-safe stats counter increment."""
        async with self._stats_lock:
            self.stats[key] = self.stats.get(key, 0) + amount

    async def process_item(
        self,
        item: dict[str, Any],
        processor_fn: Callable[..., Any],
        use_cache: bool = True,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Process a single item with retries, caching, and checkpointing.

        Args:
            item: Item to process.
            processor_fn: Async function to process the item.
            use_cache: Whether to use caching.
            job_id: Optional job ID for checkpointing.

        Returns:
            Processing result.
        """
        item_id = item.get("id", "unknown")

        # Check cache first (canonical key — immune to dict ordering)
        if use_cache and self.cache is not None:
            cache_key = CacheManager.make_key_for_item(
                item,
                model=getattr(self.llm_client, "model", ""),
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {item_id}")
                await self._increment_stat("cached")

                # Record in checkpoint
                if job_id and self.checkpoint:
                    await self.checkpoint.record_processed(job_id, item_id)

                return cached["value"]  # type: ignore[no-any-return]

        async with self.semaphore:
            try:
                # Rate limiting
                await self.rate_limiter.acquire()

                # Process with retries
                result = await self._process_with_retry(item, processor_fn)

                # Cache successful result
                if use_cache and self.cache is not None:
                    self.cache.set(
                        key=cache_key,
                        value=result,
                        model=getattr(self.llm_client, "model", ""),
                        provider=getattr(self.llm_client, "base_url", ""),
                        tokens_used=result.get("tokens_used", 0) if isinstance(result, dict) else 0,
                    )

                # Record in checkpoint
                if job_id and self.checkpoint:
                    await self.checkpoint.record_processed(job_id, item_id)

                await self._increment_stat("completed")
                return result  # type: ignore[no-any-return]

            except Exception as e:
                logger.error(f"Failed to process {item_id}: {e}")
                await self._increment_stat("failed")

                # Record failure in checkpoint
                if job_id and self.checkpoint:
                    await self.checkpoint.record_failed(job_id, item_id, str(e))

                raise

    async def _process_with_retry(
        self,
        item: dict[str, Any],
        processor_fn: Callable[..., Any],
    ) -> Any:
        """Process item with exponential backoff retries."""
        last_exception: Exception | None = None

        for attempt in range(self.retry_attempts):
            try:
                return await processor_fn(item, self.llm_client)
            except RateLimitError as e:
                last_exception = e
                wait_time = (2**attempt) * 1.0
                logger.warning(f"Rate limited, waiting {wait_time:.1f}s (attempt {attempt + 1}/{self.retry_attempts})")
                await asyncio.sleep(wait_time)
            except LLMClientError as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    wait_time = (2**attempt) * 0.5
                    logger.warning(f"LLM error, retrying in {wait_time:.1f}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                last_exception = e
                raise

        raise last_exception or LLMClientError("Max retries exceeded")

    async def run_batch(
        self,
        items: list[dict[str, Any]],
        processor_fn: Callable[..., Any],
        use_cache: bool = True,
        job_id: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        checkpoint_interval: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Run the worker pool over a batch of items.

        Args:
            items: List of items to process.
            processor_fn: Async function to process each item.
            use_cache: Whether to use caching. Note: caching is only available
                if the worker was created with ``use_cache=True``.
            job_id: Optional job ID for checkpointing.
            progress_callback: Optional callback for progress updates (current, total).
            checkpoint_interval: Save checkpoint every N items.

        Returns:
            List of processing results.
        """
        # Honour the worker-level cache setting: if caching was disabled at
        # construction time, the per-call flag cannot re-enable it.
        effective_use_cache = use_cache and self._use_cache
        self.stats["total"] = len(items)
        logger.info(f"Starting batch processing of {len(items)} items (concurrency: {self.semaphore._value})")

        # Filter out already-processed items if resuming
        if job_id and self.checkpoint:
            checkpoint = self.checkpoint.load_job(job_id)
            if checkpoint and checkpoint.processed_ids:
                processed_set = set(checkpoint.processed_ids)
                original_count = len(items)
                items = [item for item in items if item.get("id", "unknown") not in processed_set]
                skipped = original_count - len(items)
                if skipped > 0:
                    logger.info(f"Resuming job {job_id}: skipping {skipped} already-processed items")
                    self.stats["cached"] += skipped

        tasks = []
        for i, item in enumerate(items):
            task = asyncio.create_task(
                self._process_with_progress(
                    item,
                    processor_fn,
                    effective_use_cache,
                    job_id,
                    i,
                    len(items),
                    progress_callback,
                    checkpoint_interval,
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result list
        processed_results: list[dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "status": "failed",
                        "error": str(result),
                        "item": items[i],
                    }
                )
            elif isinstance(result, dict):
                processed_results.append(result)
            else:
                # Unexpected result type (shouldn't happen with return_exceptions=True)
                processed_results.append(
                    {
                        "status": "failed",
                        "error": f"Unexpected result type: {type(result).__name__}",
                        "item": items[i],
                    }
                )
                await self._increment_stat("failed")

        logger.info(
            f"Batch complete: {self.stats['completed']} succeeded, "
            f"{self.stats['failed']} failed, {self.stats['cached']} cached"
        )

        return processed_results

    async def _process_with_progress(
        self,
        item: dict[str, Any],
        processor_fn: Callable[..., Any],
        use_cache: bool,
        job_id: str | None,
        index: int,
        total: int,
        progress_callback: Callable[[int, int], None] | None,
        checkpoint_interval: int,
    ) -> Any:
        """Process item with progress tracking."""
        try:
            result = await self.process_item(item, processor_fn, use_cache, job_id)

            if progress_callback:
                progress_callback(index + 1, total)

            return result
        except Exception:
            if progress_callback:
                progress_callback(index + 1, total)
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        cache_hit_rate = 0.0
        cache_entries = 0
        cache_size_mb = 0.0
        if self.cache is not None:
            cache_stats = self.cache.stats()
            cache_hit_rate = cache_stats.hit_rate
            cache_entries = cache_stats.total_entries
            cache_size_mb = cache_stats.db_size_mb
        return {
            **self.stats,
            "success_rate": (self.stats["completed"] / self.stats["total"] * 100 if self.stats["total"] > 0 else 0),
            "cache_hit_rate": cache_hit_rate,
            "cache_entries": cache_entries,
            "cache_size_mb": cache_size_mb,
        }

    def clear_cache(self) -> None:
        """Clear the result cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Cache cleared")
        else:
            logger.info("Cache is disabled — nothing to clear")

    async def close(self) -> None:
        """Close the worker and cleanup resources."""
        if hasattr(self.llm_client, "close"):
            await self.llm_client.close()
        if self.cache is not None:
            self.cache.close()

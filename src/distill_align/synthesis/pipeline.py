"""
Synthesis pipeline orchestrator.

Handles the full synthesis workflow: processing chunks through LLMs, applying Socratic
Transformer and Scaffold Action pipelines, and outputting structured conversations.

Supports:
- Checkpoint-based resume for crash recovery
- Cache-aware processing (skip already-synthesized chunks)
- Job tracking with statistics
"""

import uuid
from collections.abc import Callable
from typing import Any

from loguru import logger

from ..core.cache import CacheManager
from ..core.checkpoint import CheckpointManager
from ..core.exceptions import LLMClientError, SynthesisError
from ..core.schemas import (
    ConversationSchema,
    DataChunk,
    SynthesisConfig,
    SynthesizedTurn,
)
from .judge import ConversationJudge
from .models.base import BaseLLMClient
from .models.registry import get as get_provider_info
from .models.registry import list_names as list_provider_names
from .prompts.scaffold import SCAFFOLD_SYSTEM_PROMPT, render_scaffold_prompt
from .prompts.socratic import SOCRATIC_SYSTEM_PROMPT, render_socratic_prompt
from .pruner import ContentPruner
from .tokenizer import CostEstimate, CostTrackingClient, Tokenizer
from .worker import BatchWorker


class SynthesisPipeline:
    """Orchestrates the synthesis of DataChunks into structured conversations."""

    def __init__(
        self,
        config: SynthesisConfig | None = None,
        cache_manager: CacheManager | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        use_cache: bool = True,
    ):
        """
        Initialize the synthesis pipeline.

        Args:
            config: Optional synthesis configuration.
            cache_manager: Optional cache manager for persistent caching.
            checkpoint_manager: Optional checkpoint manager for resume support.
            use_cache: Whether to enable caching. When False, no CacheManager
                is used regardless of *cache_manager*.
        """
        self.config = config or SynthesisConfig()
        self._client: BaseLLMClient | CostTrackingClient | None = None
        self._worker: BatchWorker | None = None
        self._judge: ConversationJudge | None = None
        self._pruner = ContentPruner()
        self._cache = cache_manager
        self._checkpoint = checkpoint_manager
        self._use_cache = use_cache
        # Cost tracking
        self._tokenizer = Tokenizer(model=self.config.model_name)

    @staticmethod
    def _client_for_format(
        api_format: str,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
    ) -> BaseLLMClient:
        """Build a client for the given ``api_format``.

        Args:
            api_format: The wire-protocol family (``"openai"``, ``"anthropic"``,
                ``"ollama"``, ``"vllm"``, ``"gemini"``).
            base_url: API base URL.
            api_key: API key (may be ``None`` for local providers).
            model: Model name.

        Returns:
            A new LLM client instance.

        Raises:
            SynthesisError: If ``api_format`` is unknown.
        """
        # Local imports to avoid circular dependencies and keep startup fast.
        from .models.anthropic import AnthropicClient
        from .models.azure import AzureClient
        from .models.gateway import GatewayClient
        from .models.gemini import GeminiClient
        from .models.ollama import OllamaClient
        from .models.openai import OpenAIClient
        from .models.vllm import VLLMClient

        _format_clients: dict[str, type[BaseLLMClient]] = {
            "openai": OpenAIClient,
            "anthropic": AnthropicClient,
            "ollama": OllamaClient,
            "vllm": VLLMClient,
            "gemini": GeminiClient,
            "azure": AzureClient,
        }
        client_cls = _format_clients.get(api_format)
        if client_cls is None:
            # Gateway / hosted open-weight providers are OpenAI-compatible.
            # Route unknown formats through GatewayClient (adds attribution
            # headers + provider routing) instead of failing.
            client = GatewayClient(base_url=base_url, api_key=api_key, model=model)
            return client

        # Some local clients (ollama, vllm) don't require an API key.
        kwargs: dict[str, str | None] = {"base_url": base_url, "model": model}
        if api_key is not None or api_format not in ("ollama", "vllm"):
            kwargs["api_key"] = api_key

        return client_cls(**kwargs)  # type: ignore[arg-type]

    def _build_client(self, model_name: str | None = None) -> BaseLLMClient:
        """Build an LLM client for the configured provider.

        Looks up the provider in the registry to determine the ``api_format``
        and default base URL, then delegates to :meth:`_client_for_format`.

        Args:
            model_name: Optional model override. Defaults to config model_name.

        Returns:
            A new LLM client instance.

        Raises:
            SynthesisError: If the provider is unknown.
        """
        model = model_name or self.config.model_name
        provider = self.config.llm_provider

        info = get_provider_info(provider)
        if info is None:
            raise SynthesisError(f"Unknown provider: {provider!r}. Available: {', '.join(list_provider_names())}")

        # Gateway providers speak OpenAI-compatible APIs but benefit from
        # attribution headers + routing extras (OpenRouter HTTP-Referer etc.)
        if provider in ("openrouter", "litellm", "together", "groq", "mistral", "deepseek", "cohere", "qwen"):
            from .models.gateway import GatewayClient

            return GatewayClient(
                base_url=self.config.base_url or info.default_base_url,
                api_key=self.config.api_key,
                model=model,
            )

        return self._client_for_format(
            info.api_format,
            base_url=self.config.base_url or info.default_base_url,
            api_key=self.config.api_key,
            model=model,
        )

    def _get_client(self) -> BaseLLMClient | CostTrackingClient:
        """Get or create the LLM client."""
        if self._client is None:
            raw_client = self._build_client()
            self._client = CostTrackingClient(raw_client, self._tokenizer)
        return self._client

    def _get_worker(self) -> BatchWorker:
        """Get or create the batch worker."""
        if self._worker is None:
            client = self._get_client()
            self._worker = BatchWorker(
                llm_client=client,  # type: ignore[arg-type]
                max_concurrency=self.config.max_concurrency,
                max_rpm=self.config.max_rpm,
                retry_attempts=self.config.retry_attempts,
                cache_manager=self._cache,
                checkpoint_manager=self._checkpoint,
                use_cache=self._use_cache,
            )
        return self._worker

    async def _get_judge(self) -> ConversationJudge:
        """Get or create the conversation judge.

        The judge reuses the same LLM client, optionally with a different
        model override (``judge_model``).
        """
        if self._judge is None:
            client = self._get_client()
            # If a separate judge model is configured, create a new client
            # with that model (sharing the same provider).
            if self.config.judge_model and self.config.judge_model != self.config.model_name:
                judge_client = self._build_client(model_name=self.config.judge_model)
            else:
                judge_client = client  # type: ignore[assignment]
            self._judge = ConversationJudge(llm_client=judge_client)  # type: ignore[arg-type]
        return self._judge

    async def synthesize_chunk(
        self,
        chunk: DataChunk,
        llm_client: BaseLLMClient,
    ) -> ConversationSchema:
        """
        Synthesize a single chunk into a conversation.

        Args:
            chunk: DataChunk to synthesize.
            llm_client: LLM client to use.

        Returns:
            ConversationSchema object.

        Raises:
            SynthesisError: If synthesis fails.
        """
        chunk_id = chunk.id
        logger.debug(f"Synthesizing chunk {chunk_id}")

        metadata = {
            "source_type": chunk.metadata.source_type,
            "title": chunk.metadata.title,
            "language": chunk.metadata.language,
            "section_headers": chunk.metadata.section_headers,
            "module_path": chunk.metadata.module_path,
            "functions": chunk.metadata.custom_tags.get("functions", []),
            "classes": chunk.metadata.custom_tags.get("classes", []),
        }

        conversation = None

        # Optional v2 mode routing: when conversation_mode != default, use
        # ConversationBuilder (Evol-Instruct, RAG-QA, tool-call, safety, distill).
        mode = getattr(self.config, "conversation_mode", "default") or "default"
        if mode != "default":
            try:
                from .conversation_builder import ConversationBuilder, ConversationMode

                builder = ConversationBuilder(pruner=self._pruner)
                built = await builder.build_conversation(chunk, ConversationMode(mode), llm_client)
                if built is not None:
                    conversation = built
            except Exception as e:
                logger.warning(f"Mode {mode!r} build failed, falling back to Socratic: {e}")

        # Step 1: Socratic Transformer
        if conversation is None and self.config.socratic_enabled:
            conversation = await self._apply_socratic(chunk.content, metadata, llm_client, source_chunk_id=chunk_id)

        # Step 2: Scaffold Action
        if self.config.scaffold_enabled and conversation:
            conversation = await self._apply_scaffold(conversation, metadata, llm_client)

        # Fallback: simple conversation
        if conversation is None:
            conversation = ConversationSchema(
                id=str(uuid.uuid4()),
                source_chunk_id=chunk_id,
                turns=[
                    SynthesizedTurn(role="user", content=f"Explain the following:\n\n{chunk.content[:500]}"),
                    SynthesizedTurn(role="assistant", content=chunk.content),
                ],
            )

        # Step 3: Prune and validate
        pruned = self._pruner.prune_conversation(conversation)
        if pruned is None:
            raise SynthesisError(f"Conversation failed quality check for chunk {chunk_id}")

        # Step 4: LLM-as-judge evaluation (optional)
        if self.config.enable_judge:
            try:
                judge = await self._get_judge()
                scores = await judge.evaluate(pruned, source_content=chunk.content, max_tokens=self.config.max_tokens)
                if "error" not in scores:
                    pruned.judge_scores = scores
                    # Extract overall score (0-10) and normalise to 0-1 for confidence
                    overall = scores.get("overall")
                    if isinstance(overall, int | float):
                        pruned.confidence_score = max(0.0, min(1.0, overall / 10.0))
            except Exception as e:
                logger.warning(f"Judge evaluation failed for chunk {chunk_id}: {e}")

        return pruned

    async def _apply_socratic(
        self,
        content: str,
        metadata: dict[str, Any],
        llm_client: BaseLLMClient,
        source_chunk_id: str = "",
    ) -> ConversationSchema | None:
        """Apply the Socratic Transformer pipeline."""
        user_prompt = render_socratic_prompt(content, metadata)

        try:
            response = await llm_client.generate(
                system_prompt=SOCRATIC_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            parsed = self._pruner.extract_json_from_response(response)
            if not parsed or "conversation" not in parsed:
                logger.warning("Failed to parse Socratic response")
                return None

            turns = [SynthesizedTurn(role=t["role"], content=t["content"]) for t in parsed["conversation"]]

            return ConversationSchema(
                id=str(uuid.uuid4()),
                source_chunk_id=source_chunk_id,
                turns=turns,
                reasoning_trace=parsed.get("reasoning_trace"),
            )

        except LLMClientError as e:
            logger.error(f"Socratic pipeline failed: {e}")
            return None

    async def _apply_scaffold(
        self,
        conversation: ConversationSchema,
        metadata: dict[str, Any],
        llm_client: BaseLLMClient,
    ) -> ConversationSchema:
        """Apply the Scaffold Action pipeline to clean assistant responses."""
        cleaned_turns = []

        for turn in conversation.turns:
            if turn.role == "assistant":
                user_prompt = render_scaffold_prompt(
                    turn.content,
                    {**metadata, "extraction_type": "auto"},
                )
                try:
                    response = await llm_client.generate(
                        system_prompt=SCAFFOLD_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        temperature=0.3,
                        max_tokens=self.config.max_tokens,
                    )
                    parsed = self._pruner.extract_json_from_response(response)
                    if parsed and "extracted_content" in parsed:
                        cleaned_turns.append(
                            SynthesizedTurn(
                                role="assistant",
                                content=parsed["extracted_content"],
                            )
                        )
                    else:
                        cleaned_turns.append(turn)
                except LLMClientError as e:
                    logger.warning(f"Scaffold failed for turn, keeping original: {e}")
                    cleaned_turns.append(turn)
            else:
                cleaned_turns.append(turn)

        return ConversationSchema(
            id=conversation.id,
            source_chunk_id=conversation.source_chunk_id,
            turns=cleaned_turns,
            reasoning_trace=conversation.reasoning_trace,
            confidence_score=conversation.confidence_score,
        )

    async def synthesize_batch(
        self,
        chunks: list[DataChunk],
        progress_callback: Callable[[int, int], None] | None = None,
        job_id: str | None = None,
        resume: bool = False,
        use_cache: bool | None = None,
    ) -> list[ConversationSchema]:
        """
        Synthesize a batch of chunks into conversations.

        Args:
            chunks: List of DataChunks to synthesize.
            progress_callback: Optional callback for progress updates.
            job_id: Optional job ID for checkpointing/resume.
            resume: Whether to resume an existing job.
            use_cache: Whether to use caching for this batch. Defaults to the
                pipeline-level *use_cache* setting. Note that if caching was
                disabled at the pipeline level, this flag cannot re-enable it.

        Returns:
            List of ConversationSchema objects.
        """
        worker = self._get_worker()

        # Create or resume checkpoint
        if self._checkpoint and job_id:
            if resume:
                checkpoint = self._checkpoint.load_job(job_id)
                if checkpoint:
                    logger.info(f"Resuming job {job_id}: {checkpoint.processed_items}/{checkpoint.total_items} done")
                    # Filter out already-processed chunks
                    processed_set = set(checkpoint.processed_ids)
                    chunks = [c for c in chunks if c.id not in processed_set]
                    logger.info(f"Processing {len(chunks)} remaining chunks")
                    self._checkpoint.start_job(job_id)
                else:
                    logger.warning(f"Job {job_id} not found, starting fresh")
                    self._checkpoint.create_job("synthesize", total_items=len(chunks), job_id=job_id)
                    self._checkpoint.start_job(job_id)
            else:
                self._checkpoint.create_job("synthesize", total_items=len(chunks), job_id=job_id)
                self._checkpoint.start_job(job_id)

        # Prepare items for worker
        items = [{"id": chunk.id, "chunk": chunk} for chunk in chunks]

        async def processor(item: dict[str, Any], llm_client: BaseLLMClient) -> dict[str, Any]:
            chunk = item["chunk"]
            conversation = await self.synthesize_chunk(chunk, llm_client)
            return {"status": "success", "conversation": conversation.model_dump()}

        # Run batch
        effective_use_cache = use_cache if use_cache is not None else self._use_cache
        results = await worker.run_batch(
            items=items,
            processor_fn=processor,
            use_cache=effective_use_cache,
            job_id=job_id,
            progress_callback=progress_callback,
        )

        # Extract conversations
        conversations = []
        for result in results:
            if result.get("status") == "success":
                conv_data = result["conversation"]
                conversations.append(ConversationSchema(**conv_data))
            else:
                logger.warning(f"Failed to synthesize: {result.get('error')}")

        # Complete checkpoint
        if self._checkpoint and job_id:
            stats = worker.get_stats()
            if worker.stats["failed"] == 0:
                self._checkpoint.complete_job(job_id, stats=stats)
            else:
                self._checkpoint.fail_job(job_id, error=f"{worker.stats['failed']} items failed")

        logger.info(f"Synthesized {len(conversations)} conversations from {len(chunks)} chunks")
        return conversations

    async def close(self) -> None:
        """Close the pipeline and cleanup resources."""
        if self._worker:
            await self._worker.close()
        if self._client and hasattr(self._client, "close"):
            await self._client.close()

    def get_cost_stats(self) -> CostEstimate:
        """Get cumulative cost statistics for all synthesis runs.

        Returns:
            CostEstimate with token and cost totals.
        """
        return self._tokenizer.get_stats()

    def get_cost_report(self) -> str:
        """Get a human-readable cost report string.

        Returns:
            Multi-line cost summary.
        """
        return self._tokenizer.get_cost_report_string()

    def reset_cost_stats(self) -> None:
        """Reset all cost tracking statistics."""
        self._tokenizer.reset_stats()

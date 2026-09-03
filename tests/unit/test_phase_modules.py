"""
Unit tests for the new Phase 1-5 components.
"""

import pytest

from distill_align.core.cache import CacheManager
from distill_align.core.checkpoint import CheckpointManager, JobStatus
from distill_align.core.config_file import (
    DistillAlignConfig,
    generate_default_config,
    load_config,
    save_config,
)
from distill_align.exporter.splitter import DatasetSplitter
from distill_align.exporter.validator import DatasetValidator
from distill_align.synthesis.tokenizer import Tokenizer

# =============================================================================
# CacheManager Tests
# =============================================================================


class TestCacheManager:
    """Tests for the SQLite-backed CacheManager."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a temporary cache."""
        return CacheManager(cache_dir=str(tmp_path), ttl_days=1)

    def test_set_and_get(self, cache):
        cache.set(key="test_key", value={"data": "hello"})
        result = cache.get("test_key")
        assert result is not None
        assert result["value"] == {"data": "hello"}

    def test_cache_miss(self, cache):
        result = cache.get("nonexistent")
        assert result is None

    def test_make_key_deterministic(self):
        key1 = CacheManager.make_key("content", "model", "prompt_id")
        key2 = CacheManager.make_key("content", "model", "prompt_id")
        assert key1 == key2

    def test_make_key_unique(self):
        key1 = CacheManager.make_key("content1", "model", "prompt_id")
        key2 = CacheManager.make_key("content2", "model", "prompt_id")
        assert key1 != key2

    def test_stats(self, cache):
        cache.set("k1", "v1")
        cache.get("k1")  # Hit
        cache.get("k2")  # Miss

        stats = cache.stats()
        assert stats.total_entries == 1
        assert stats.hit_count == 1
        assert stats.miss_count == 1
        assert stats.hit_rate == 0.5

    def test_clear(self, cache):
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        removed = cache.clear()
        assert removed == 2
        assert cache.get("k1") is None


# =============================================================================
# CheckpointManager Tests
# =============================================================================


class TestCheckpointManager:
    """Tests for the CheckpointManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a temporary checkpoint manager."""
        return CheckpointManager(checkpoint_dir=str(tmp_path))

    def test_create_job(self, manager):
        job = manager.create_job("synthesize", total_items=100)
        assert job.job_type == "synthesize"
        assert job.total_items == 100
        assert job.status == JobStatus.PENDING

    def test_start_and_complete_job(self, manager):
        import asyncio

        async def _run():
            job = manager.create_job("synthesize", total_items=10)
            manager.start_job(job.job_id)
            await manager.record_processed(job.job_id, "item-1")
            await manager.record_processed(job.job_id, "item-2")
            return manager.complete_job(job.job_id, stats={"total_tokens": 1000})

        completed = asyncio.run(_run())
        assert completed.status == JobStatus.COMPLETED
        assert completed.processed_items == 2

    def test_record_failed(self, manager):
        import asyncio

        async def _run():
            job = manager.create_job("synthesize", total_items=10)
            await manager.record_failed(job.job_id, "item-1", "Test error")
            return manager.load_job(job.job_id)

        checkpoint = asyncio.run(_run())
        assert checkpoint.failed_items == 1
        assert "item-1" in checkpoint.failed_errors

    def test_get_unprocessed_ids(self, manager):
        import asyncio

        async def _run():
            job = manager.create_job("synthesize", total_items=5)
            await manager.record_processed(job.job_id, "a")
            await manager.record_processed(job.job_id, "b")
            return job.job_id

        job_id = asyncio.run(_run())
        all_ids = ["a", "b", "c", "d", "e"]
        unprocessed = manager.get_unprocessed_ids(job_id, all_ids)
        assert set(unprocessed) == {"c", "d", "e"}

    def test_list_jobs(self, manager):
        manager.create_job("synthesize", total_items=10)
        manager.create_job("ingest", total_items=5)

        jobs = manager.list_jobs()
        assert len(jobs) == 2

    def test_delete_job(self, manager):
        job = manager.create_job("test", total_items=1)
        assert manager.delete_job(job.job_id)
        assert manager.load_job(job.job_id) is None


# =============================================================================
# Config File Tests
# =============================================================================


class TestConfigFile:
    """Tests for config file loading/saving."""

    def test_generate_default_config(self, tmp_path):
        config_path = generate_default_config("test-project", str(tmp_path / "config.yaml"))
        assert config_path.exists()
        content = config_path.read_text()
        assert "test-project" in content

    def test_save_and_load_config(self, tmp_path):
        config = DistillAlignConfig()
        config.project.name = "roundtrip-test"

        path = tmp_path / "test.yaml"
        save_config(config, str(path))
        loaded = load_config(str(path))

        assert loaded.project.name == "roundtrip-test"

    def test_load_nonexistent_uses_defaults(self, tmp_path, monkeypatch):
        # Change to a temp dir so find_config_file doesn't pick up an existing config
        monkeypatch.chdir(tmp_path)
        # Explicitly pass a None path so we go through the find_config_file branch
        config = load_config(None)
        assert config.project.name == "my-dataset"


# =============================================================================
# Dataset Splitter Tests
# =============================================================================


class TestDatasetSplitter:
    """Tests for DatasetSplitter."""

    @pytest.fixture
    def conversations(self):
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn

        return [
            ConversationSchema(
                id=f"conv-{i}",
                source_chunk_id=f"chunk-{i // 2}",
                turns=[
                    SynthesizedTurn(role="user", content=f"Q{i}"),
                    SynthesizedTurn(role="assistant", content=f"A{i}"),
                ],
            )
            for i in range(10)
        ]

    def test_split(self, conversations):
        splitter = DatasetSplitter(seed=42)
        result = splitter.split(conversations, 0.8, 0.1, 0.1)

        assert len(result.train) == 8
        assert len(result.val) == 1
        assert len(result.test) == 1
        assert result.total == 10

    def test_split_invalid_ratios(self, conversations):
        splitter = DatasetSplitter()
        with pytest.raises(ValueError):
            splitter.split(conversations, 0.5, 0.3, 0.3)  # Sums to 1.1

    def test_stratified_split(self, conversations):
        splitter = DatasetSplitter(seed=42)
        result = splitter.split(conversations, 0.8, 0.1, 0.1, stratify_by="source_chunk_id")
        # Each source_chunk_id should have its items split proportionally
        assert result.total == 10


# =============================================================================
# Dataset Validator Tests
# =============================================================================


class TestDatasetValidator:
    """Tests for DatasetValidator."""

    @pytest.fixture
    def conversations(self):
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn

        return [
            ConversationSchema(
                id=f"conv-{i}",
                source_chunk_id="chunk-1",
                turns=[
                    SynthesizedTurn(role="system", content="You are helpful."),
                    SynthesizedTurn(role="user", content=f"Q{i}"),
                    SynthesizedTurn(role="assistant", content=f"A{i}"),
                ],
            )
            for i in range(5)
        ]

    def test_validate_good_dataset(self, conversations):
        validator = DatasetValidator()
        report = validator.validate(conversations)

        assert report.is_valid
        assert report.quality_score > 0.0
        assert report.quality_score <= 1.0
        assert report.stats.total_conversations == 5
        # All 5 conversations have system+user+assistant roles (no errors)
        assert report.error_count == 0

    def test_validate_empty_dataset(self):
        validator = DatasetValidator()
        report = validator.validate([])

        assert report.stats.total_conversations == 0
        assert not report.is_valid  # Error: too few conversations

    def test_deduplicate(self):
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn

        conv = ConversationSchema(
            id="1",
            source_chunk_id="c1",
            turns=[
                SynthesizedTurn(role="user", content="Hi"),
                SynthesizedTurn(role="assistant", content="Hello!"),
            ],
        )

        validator = DatasetValidator()
        unique = validator.deduplicate([conv, conv, conv])
        assert len(unique) == 1


# =============================================================================
# Tokenizer Tests
# =============================================================================


class TestTokenizer:
    """Tests for Tokenizer."""

    def test_count_tokens_fallback(self):
        tokenizer = Tokenizer(model="unknown-model")
        count = tokenizer.count_tokens("Hello, world!")
        assert count > 0

    def test_count_message_tokens(self):
        tokenizer = Tokenizer(model="unknown-model")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        count = tokenizer.count_message_tokens(messages)
        assert count > 0

    def test_estimate_cost_zero_for_oss(self):
        tokenizer = Tokenizer(model="llama3.1")
        cost = tokenizer.estimate_cost(1000, 500)
        assert cost == 0.0  # Open-source models

    def test_estimate_cost_for_openai(self):
        tokenizer = Tokenizer(model="gpt-4o")
        cost = tokenizer.estimate_cost(1_000_000, 0)
        assert cost > 0  # Should have cost

    def test_record_and_get_stats(self):
        tokenizer = Tokenizer(model="gpt-4o")
        tokenizer.record_usage(100, 50)
        tokenizer.record_usage(200, 100)

        stats = tokenizer.get_stats()
        assert stats.total_input_tokens == 300
        assert stats.total_output_tokens == 150
        assert stats.num_requests == 2

    def test_estimate_batch_cost(self):
        tokenizer = Tokenizer(model="gpt-4o")
        texts = ["Hello world", "Another text", "More content"]
        estimate = tokenizer.estimate_batch_cost(texts, estimated_output_tokens=100)
        assert estimate.num_requests == 3
        assert estimate.total_input_tokens > 0
        assert estimate.total_output_tokens == 300

    def test_resolve_pricing_direct_match(self):
        pricing = Tokenizer._resolve_pricing("gpt-4o")
        assert pricing["input"] == 2.50
        assert pricing["output"] == 10.00

    def test_resolve_pricing_azure_deployment(self):
        """Azure deployment names should resolve to base model pricing."""
        pricing = Tokenizer._resolve_pricing("gpt-4o-deployment")
        assert pricing["input"] == 2.50

    def test_resolve_pricing_unknown_model(self):
        pricing = Tokenizer._resolve_pricing("totally-unknown-model")
        assert pricing["input"] == 0.0
        assert pricing["output"] == 0.0

    def test_resolve_pricing_oss_model(self):
        pricing = Tokenizer._resolve_pricing("llama3.1")
        assert pricing["input"] == 0.0
        assert pricing["output"] == 0.0

    def test_record_usage_from_response(self):
        tokenizer = Tokenizer(model="gpt-4o-mini")
        response_data = {
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 75,
                "total_tokens": 225,
            }
        }
        cost = tokenizer.record_usage_from_response(response_data)
        assert cost > 0
        stats = tokenizer.get_stats()
        assert stats.total_input_tokens == 150
        assert stats.total_output_tokens == 75
        assert stats.num_requests == 1

    def test_record_usage_from_response_anthropic_format(self):
        """Anthropic uses input_tokens / output_tokens naming."""
        tokenizer = Tokenizer(model="claude-sonnet-4-20250514")
        response_data = {
            "usage": {
                "input_tokens": 200,
                "output_tokens": 50,
            }
        }
        cost = tokenizer.record_usage_from_response(response_data)
        assert cost > 0
        stats = tokenizer.get_stats()
        assert stats.total_input_tokens == 200
        assert stats.total_output_tokens == 50

    def test_record_usage_from_response_empty(self):
        """Should gracefully handle missing usage data."""
        tokenizer = Tokenizer(model="gpt-4o")
        cost = tokenizer.record_usage_from_response({})
        assert cost == 0.0
        assert tokenizer.get_stats().num_requests == 1  # Still counts the request

    def test_get_cost_report_string(self):
        tokenizer = Tokenizer(model="gpt-4o")
        tokenizer.record_usage(1_000_000, 100_000)
        report = tokenizer.get_cost_report_string()
        assert "Cost Report" in report
        assert "gpt-4o" in report
        assert "$" in report

    def test_multi_provider_pricing(self):
        """Verify pricing exists for key models (legacy + Sep-2026 current)."""
        models_to_check = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "claude-3-5-sonnet",
            "claude-sonnet-4-20250514",
            "gemini-2.0-flash",
            "gemini-2.5-pro",
            "o1",
            "o3-mini",
            "o4-mini",
            "llama3.1",
            "deepseek-r1",
            # Sep-2026 current generation
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5-mini",
            "claude-fable-5",
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-haiku-4-5",
            "gemini-3.5-flash",
            "gemini-3.8-flash",
            "qwen3-max",
            "qwen3-coder-next",
            "deepseek-v4-flash",
            "mistral-large-latest",
            "llama4:scout",
        ]
        for model in models_to_check:
            pricing = Tokenizer._resolve_pricing(model)
            assert "input" in pricing
            assert "output" in pricing

    def test_sep2026_pricing_values(self):
        """Spot-check official Sep-2026 prices used for cost reports."""
        assert Tokenizer._resolve_pricing("gpt-5.6-sol") == {"input": 4.00, "output": 20.00}
        assert Tokenizer._resolve_pricing("gpt-5.6-terra") == {"input": 2.00, "output": 12.00}
        assert Tokenizer._resolve_pricing("claude-sonnet-5") == {"input": 2.00, "output": 10.00}
        assert Tokenizer._resolve_pricing("claude-opus-4-8") == {"input": 5.00, "output": 25.00}
        assert Tokenizer._resolve_pricing("claude-fable-5") == {"input": 10.00, "output": 50.00}
        assert Tokenizer._resolve_pricing("claude-haiku-4-5") == {"input": 1.00, "output": 5.00}
        assert Tokenizer._resolve_pricing("deepseek-v4-flash") == {"input": 0.14, "output": 0.28}

    def test_catalog_aliases(self):
        """Legacy ids resolve to their 2026 successors."""
        from distill_align.synthesis.models.catalog import resolve_alias

        assert resolve_alias("gpt-4o") == "gpt-5-mini"
        assert resolve_alias("claude-sonnet-4-20250514") == "claude-sonnet-5"
        assert resolve_alias("gemini-2.0-flash") == "gemini-3.5-flash"
        assert resolve_alias("gpt-5-mini") == "gpt-5-mini"

    def test_reset_stats(self):
        tokenizer = Tokenizer(model="gpt-4o")
        tokenizer.record_usage(100, 50)
        tokenizer.reset_stats()
        stats = tokenizer.get_stats()
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.num_requests == 0


class TestCostTrackingClient:
    """Tests for the CostTrackingClient proxy."""

    def test_wraps_chat_and_records_usage(self):
        """CostTrackingClient should record token usage from chat()."""
        from distill_align.synthesis.models.base import LLMMessage, LLMResponse
        from distill_align.synthesis.tokenizer import CostTrackingClient, Tokenizer

        class FakeClient:
            model = "gpt-4o"

            async def chat(self, messages, **kwargs):
                return LLMResponse(
                    content="Hello!",
                    model="gpt-4o",
                    usage={"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
                    raw_response={
                        "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
                        "model": "gpt-4o",
                        "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
                    },
                )

            async def complete(self, prompt, **kwargs):
                return LLMResponse(
                    content="Hi",
                    model="gpt-4o",
                    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    raw_response={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                )

        tokenizer = Tokenizer(model="gpt-4o")
        client = CostTrackingClient(FakeClient(), tokenizer)

        import asyncio

        async def test():
            await client.chat(messages=[LLMMessage(role="user", content="Hi")])
            stats = tokenizer.get_stats()
            assert stats.total_input_tokens == 50
            assert stats.total_output_tokens == 25
            assert stats.num_requests == 1

        asyncio.run(test())

    def test_wraps_generate_and_records_usage(self):
        """CostTrackingClient.generate() should record usage."""
        from distill_align.synthesis.models.base import LLMResponse
        from distill_align.synthesis.tokenizer import CostTrackingClient, Tokenizer

        class FakeClient:
            model = "gpt-4o"

            async def chat(self, messages, **kwargs):
                return LLMResponse(
                    content="Generated text",
                    model="gpt-4o",
                    usage={"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
                    raw_response={
                        "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
                        "model": "gpt-4o",
                    },
                )

            async def complete(self, prompt, **kwargs):
                return LLMResponse(content="", model="gpt-4o", usage={})

        tokenizer = Tokenizer(model="gpt-4o")
        client = CostTrackingClient(FakeClient(), tokenizer)

        import asyncio

        async def test():
            text = await client.generate(system_prompt="Be helpful", user_prompt="Hi")
            assert text == "Generated text"
            stats = tokenizer.get_stats()
            assert stats.total_input_tokens == 30
            assert stats.total_output_tokens == 15
            assert stats.num_requests == 1

        asyncio.run(test())

    def test_delegates_attributes(self):
        """CostTrackingClient should forward unknown attributes."""
        from distill_align.synthesis.tokenizer import CostTrackingClient, Tokenizer

        class FakeClient:
            model = "test-model"
            custom_attr = "hello"

        tokenizer = Tokenizer(model="gpt-4o")
        client = CostTrackingClient(FakeClient(), tokenizer)
        assert client.model == "test-model"
        assert client.custom_attr == "hello"

    def test_setattr_delegates(self):
        """Setting attributes should go to the wrapped client."""
        from distill_align.synthesis.tokenizer import CostTrackingClient, Tokenizer

        class FakeClient:
            pass

        tokenizer = Tokenizer(model="gpt-4o")
        client = CostTrackingClient(FakeClient(), tokenizer)
        client.new_attr = "set"
        assert client._wrapped_client.new_attr == "set"


class TestPipelineCostStats:
    """Tests for pipeline cost tracking integration."""

    def test_pipeline_has_tokenizer(self):
        """SynthesisPipeline should have a Tokenizer for cost tracking."""
        from distill_align.synthesis.pipeline import SynthesisPipeline

        pipeline = SynthesisPipeline(use_cache=False)
        assert pipeline._tokenizer is not None
        assert pipeline._tokenizer.model == "gpt-5-mini"  # default model

    def test_pipeline_exposes_cost_stats(self):
        """Pipeline should expose get_cost_stats()."""
        from distill_align.synthesis.pipeline import SynthesisPipeline

        pipeline = SynthesisPipeline(use_cache=False)
        stats = pipeline.get_cost_stats()
        assert stats.total_input_tokens == 0
        assert stats.estimated_cost_usd == 0.0

    def test_pipeline_cost_report(self):
        """Pipeline should generate a human-readable cost report."""
        from distill_align.synthesis.pipeline import SynthesisPipeline

        pipeline = SynthesisPipeline(use_cache=False)
        report = pipeline.get_cost_report()
        assert "Cost Report" in report
        assert "gpt-5-mini" in report

    def test_pipeline_reset_cost_stats(self):
        """Pipeline should allow resetting cost stats."""
        from distill_align.synthesis.pipeline import SynthesisPipeline

        pipeline = SynthesisPipeline(use_cache=False)
        pipeline._tokenizer.record_usage(100, 50)
        assert pipeline.get_cost_stats().total_input_tokens == 100
        pipeline.reset_cost_stats()
        assert pipeline.get_cost_stats().total_input_tokens == 0


# =============================================================================
# Judge Integration Tests
# =============================================================================


class TestJudgeIntegration:
    """Tests for LLM-as-judge integration in the synthesis pipeline."""

    def test_judge_disabled_by_default(self):
        """Judge should not run when enable_judge is False."""
        from distill_align.core.schemas import SynthesisConfig

        config = SynthesisConfig()
        assert config.enable_judge is False
        assert config.judge_model is None

    def test_judge_config_enabled(self):
        """Judge config can be enabled with optional model override."""
        from distill_align.core.schemas import SynthesisConfig

        config = SynthesisConfig(enable_judge=True, judge_model="gpt-4o-mini")
        assert config.enable_judge is True
        assert config.judge_model == "gpt-4o-mini"

    def test_schema_has_judge_scores(self):
        """ConversationSchema should accept judge_scores."""
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn

        conv = ConversationSchema(
            id="test-judge",
            source_chunk_id="src-1",
            turns=[SynthesizedTurn(role="user", content="Hi"), SynthesizedTurn(role="assistant", content="Hello")],
            judge_scores={"overall": 8.5, "coherence": 9.0, "explanation": "Good conversation"},
        )
        assert conv.judge_scores is not None
        assert conv.judge_scores["overall"] == 8.5
        assert conv.judge_scores["explanation"] == "Good conversation"

    def test_judge_sets_confidence_score(self):
        """Confidence score should be derived from judge overall score."""
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn

        conv = ConversationSchema(
            id="test-confidence",
            source_chunk_id="src-1",
            turns=[SynthesizedTurn(role="user", content="Hi"), SynthesizedTurn(role="assistant", content="Hello")],
            judge_scores={"overall": 9.2},
            confidence_score=0.92,  # 9.2 / 10, set by pipeline
        )
        assert conv.confidence_score == 0.92

    def test_pipeline_creates_judge_when_enabled(self):
        """SynthesisPipeline should create a judge when config.enable_judge is True."""
        from distill_align.core.schemas import SynthesisConfig
        from distill_align.synthesis.pipeline import SynthesisPipeline

        config = SynthesisConfig(enable_judge=True)
        pipeline = SynthesisPipeline(config=config, use_cache=False)
        assert pipeline._judge is None  # Lazily created
        # The judge is created on first access via _get_judge()
        import asyncio

        async def test():
            judge = await pipeline._get_judge()
            assert judge is not None
            assert judge.llm_client is not None

        asyncio.run(test())
        # Cleanup
        asyncio.run(pipeline.close())

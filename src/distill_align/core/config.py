"""
Core configuration module using Pydantic Settings.

Loads configuration from environment variables and .env files.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # General
    app_name: str = "distill-align"
    debug: bool = False
    log_level: str = "INFO"
    cache_dir: str = ".cache"

    # Ingestion
    chunk_size: int = Field(default=1000, ge=1)
    chunk_overlap: int = Field(default=200, ge=0)
    respect_headers: bool = True
    max_chunk_tokens: int = Field(default=4000, ge=1)

    # Synthesis
    llm_provider: str = "openai"  # openai, anthropic, gemini, azure, ollama, vllm, qwen, openrouter, ...
    llm_model: str = "gpt-5-mini"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_max_concurrency: int = Field(default=5, ge=1)
    llm_max_rpm: int = Field(default=60, ge=1)
    llm_retry_attempts: int = Field(default=5, ge=1)
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    socratic_enabled: bool = True
    scaffold_enabled: bool = True
    enable_judge: bool = False
    judge_model: str | None = None
    # Judge v2 (additive; Phase 4 wires the new rubrics)
    judge_rubrics: str = "relevance,coherence,correctness,completeness,safety"
    judge_dual: bool = False  # cheap gate + frontier audit
    judge_audit_model: str | None = None
    # Embeddings (Phase 2 semantic chunking / dedup)
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str | None = None
    # Observability (Phase 6 tracing hooks)
    enable_tracing: bool = False
    tracing_endpoint: str | None = None

    # Export
    export_formats: str = "sharegpt"  # comma-separated: sharegpt,alpaca,chatml,conversation,hf_messages,jsonl,parquet
    export_output_dir: str = "./output"
    generate_unsloth_script: bool = True
    unsloth_model: str = "Qwen/Qwen3-8B"
    unsloth_max_seq_length: int = 2048
    unsloth_lora_rank: int = 16
    unsloth_lora_alpha: int = 16

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DISTILL_",
        case_sensitive=False,
    )

    @property
    def export_format_list(self) -> list[str]:
        """Parse comma-separated export formats."""
        return [f.strip() for f in self.export_formats.split(",")]

    @property
    def cache_path(self) -> Path:
        """Get cache directory as Path object."""
        return Path(self.cache_dir)

    @property
    def output_path(self) -> Path:
        """Get output directory as Path object."""
        return Path(self.export_output_dir)


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings

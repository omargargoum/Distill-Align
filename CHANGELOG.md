# Changelog

All notable changes to Distill-Align will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Unit tests for the PII filter, pruner, judge, conversation builder, ingestion pipeline, and preference formatter (coverage 43% → 50%)

### Fixed
- PII filter import crash caused by an inline `(?i)` regex flag in the middle of the bearer-token pattern, which broke the `scan_pii` ingestion path

## [0.1.1] - 2026-06-18

### Fixed
- Fixed Textual `_auto_refresh` attribute conflict
- Fixed mypy type errors across multiple modules
- Fixed ruff SIM108 ternary formatting
- Fixed CI: ruff, mypy, mkdocs warnings, benchmark `--no-cov`, TUI skip in non-TTY
- Fixed Bandit skips, install groups, docs, TUI test

### Added
- Cost tracking and streaming JSONL/Parquet export
- CLI `cost-report` command for usage estimation
- `--max-tokens` option for LLM call token limits
- Arabic README translation

## [0.1.0] - 2026-06-16

### Added
- **Ingestion pipeline** with 9 file loaders (Markdown, PDF, DOCX, HTML, Jupyter, JSON, CSV, Code, Text)
- **Synthesis pipeline** with 6 LLM providers (OpenAI, Anthropic, Gemini, Azure, Ollama, vLLM)
- **Socratic Transformer** — converts raw content into multi-turn Q&A conversations
- **Scaffold Action** — cleans and extracts structured content from assistant responses
- **LLM-as-Judge evaluation** — automated quality scoring on 5 criteria
- **DPO preference pair generation** for Direct Preference Optimization training
- **Export pipeline** with 7 output formats (ShareGPT, Alpaca, ChatML, HuggingFace, JSONL, Parquet, Conversation)
- **Streaming export** for large datasets without full memory load
- **Dataset validation** — structural checks, quality scoring, deduplication, and statistics
- **PII filtering** — detects and redacts secrets and personal information
- **Job management** with checkpoint/resume for crash recovery
- **TUI dashboard** with real-time monitoring via Textual
- **Unsloth integration** — auto-generated training scripts
- **Security scanning** — Bandit, Safety, pip-audit, CodeQL, dependency review
- **CI/CD** — lint, type-check, test matrix (3 Python versions), security scan, docs deploy

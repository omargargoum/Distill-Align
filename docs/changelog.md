# Changelog

All notable changes to Distill-Align are documented here.

See also: [CHANGELOG.md](https://github.com/omargargoum/Distill-Align/blob/main/CHANGELOG.md) on GitHub.

## [0.3.0] - 2026-09-03

### Changed (model refresh — the "alive again" release)

- **86-model catalog** (`synthesis/models/catalog.py`, Sep-2026): GPT-5.6 Sol/Terra/Luna,
  GPT-5.5/5.4, Claude Fable 5 / Opus 4.8 / Sonnet 5 / Haiku 4.5, Gemini 3.8/3.6/3.5
  Flash + 3.1 Pro, Qwen3-Max/Coder-Next/3.8-Max/3.5/3.6 tiers, Llama 4 Scout/Maverick,
  DeepSeek V4 Flash/Pro/Vision, Mistral Large 3 / Medium 3.5 / Small / Codestral,
  gpt-oss, Gemma 4 — with corrected official prices and context windows
- **New defaults**: `gpt-5-mini` (synthesis), `Qwen/Qwen3-8B` (Unsloth training),
  `qwen3:30b` (Ollama), per-provider production defaults from the catalog
- **New provider**: `qwen` (Alibaba DashScope OpenAI-compatible endpoint)
- Unsloth scripts rewritten to the **current API** (`PatchDPOTrainer`, `DPOConfig`/
  `ORPOConfig`/`KTOConfig`, `SFTConfig`, GRPO + vLLM rollouts, auto-tuned long context);
  export auto-generates matching `train_dpo/orpo/kto/grpo.py` scripts
- New docs: **[Models & Pricing](models.md)** reference page; refreshed home,
  getting-started, and CLI reference

### TUI (dashboard fully rewired)

- Dropdowns derive from the backend: 14 providers, 10 modes, 14 formats, 7 chunkers
- New **chunker selector**, new **Evaluate quality gate**, fixed Unsloth model id
- CLI + TUI synthesis unified (modes get scaffold + judge + cache everywhere)

## [0.2.0] - 2026-09-03

### Fixed

- Textual `_auto_refresh` attribute conflict
- MyPy type errors across multiple modules
- Ruff SIM108 ternary formatting
- CI: ruff, mypy, mkdocs warnings, benchmark `--no-cov`, TUI skip in non-TTY
- Bandit skips, install groups, docs, TUI test

### Added

- Cost tracking and streaming JSONL/Parquet export
- CLI `cost-report` command for usage estimation
- `--max-tokens` option for LLM call token limits
- Arabic README translation
- Comprehensive documentation (MkDocs)

## [0.1.0] - 2026-06-16

### Added

- **Ingestion pipeline** with 9 file loaders (Markdown, PDF, DOCX, HTML, Jupyter, JSON, CSV, Code, Text)
- **Synthesis pipeline** with 6 LLM providers (OpenAI, Anthropic, Gemini, Azure, Ollama, vLLM)
- **Socratic Transformer** — converts raw content into multi-turn Q&A conversations
- **Scaffold Action** — cleans and extracts structured content from assistant responses
- **LLM-as-Judge evaluation** — automated quality scoring on 5 criteria (relevance, coherence, correctness, completeness, safety)
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

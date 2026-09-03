# Changelog

All notable changes to Distill-Align will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.0] - 2026-09-03

### Changed (model refresh — the "alive again" release)

- **86-model catalog** (`src/distill_align/synthesis/models/catalog.py`, Sep-2026):
  GPT-5.6 Sol/Terra/Luna, GPT-5.5/5.4, Claude Fable 5 / Opus 4.8 / Sonnet 5 / Haiku 4.5,
  Gemini 3.8/3.6/3.5 Flash + 3.1 Pro, Qwen3-Max/Coder-Next/3.8-Max/3.5/3.6 tiers,
  Llama 4 Scout/Maverick, DeepSeek V4 Flash/Pro/Vision, Mistral Large 3 / Medium 3.5 /
  Small / Codestral, gpt-oss, Gemma 4 — corrected official prices, contexts, aliases
- Pricing/registry now read from the catalog (single source of truth); corrected
  Claude 4.x prices (Opus 4.8 $5/$25, Sonnet 5 $2/$10, Haiku 4.5 $1/$5)
- **New defaults**: synthesis `gpt-5-mini`, Unsloth training `Qwen/Qwen3-8B`,
  Ollama `qwen3:30b`, per-provider production defaults; CLI/TUI/templates/docs updated
- **New provider**: `qwen` (Alibaba DashScope OpenAI-compatible, `DASHSCOPE_API_KEY`)
- Unsloth builders rewritten to the **current API** (`PatchDPOTrainer`, `DPOConfig` /
  `ORPOConfig` / `KTOConfig`, `SFTConfig`, GRPO + vLLM rollouts); export auto-generates
  matching `train_dpo/orpo/kto/grpo.py` scripts; new LoRA targets (deepseek, mixtral, gpt-oss)
- Docs: new **Models & Pricing** page (`docs/models.md` + nav), refreshed home /
  getting-started / CLI reference / changelog; fixed programmatic-usage example

### TUI (dashboard fully rewired)

- All dropdowns derive from the backend (registry / catalog / `ConversationMode` /
  `FORMATTER_MAP` / `IngestionConfig`): 14 providers, 10 modes, 14 formats, 7 chunkers
- New **chunker selector** (Ingest + Full Pipeline tabs), new **Evaluate quality gate**
  (Validate tab: thresholds + PASS/FAIL + results table), fixed Unsloth model id
- CLI + TUI synthesis unified through `pipeline.synthesize_batch` (modes now get
  scaffold + judge + cache + checkpoint everywhere; removed duplicated builder branches)
- New `tui/options.py` source-of-truth helpers + 26 TUI tests (incl. Textual pilot mount)

## [0.2.0] - 2026-09-03

Revival release: 2026 model/practice modernization. Fully backwards-compatible
(all v0.1 defaults, CLIs, schemas, and formats unchanged; new behavior is opt-in).

### Added
- **Providers**: OpenRouter, LiteLLM proxy, Together, Groq, Mistral, DeepSeek, Cohere
  registry entries + `GatewayClient` (attribution headers, provider routing, guided_json passthrough)
- **Pricing**: GPT-5 family, Claude Opus 4.8/Sonnet 5/Haiku 4.5, Gemini 3.x, Qwen3, DeepSeek,
  Llama 4 gateway ids, `MODEL_ALIASES`, cached/reasoning-token tolerant usage recording
- **Structured outputs**: strict `json_schema` builder + per-provider mapping
  (OpenAI strict + `max_completion_tokens`, Anthropic `output_config`, Gemini `responseSchema`,
  Ollama schema `format`, vLLM `guided_json`); refusal field; code-fence stripping
- **Ingestion**: `SemanticChunker`, `ParentChildChunker`, `LateChunker` (+ `chunker` config),
  `HashEmbedder`/`OpenAIEmbedder` + cosine dedup helpers, table row-sentence serialization,
  contextual prefix, `DoclingLoader` with graceful fallback, broader code-definition patterns
- **Synthesis**: 5 new `ConversationMode`s (evol_instruct, rag_qa, tool_call, constitutional, distill),
  `conversation_mode` pipeline routing, `async_retry` helper
- **Judge v2**: rubric library (faithfulness, groundedness), custom prompts, score normalization,
  dual-judge (gate + audit + agreement), n-gram contamination proxy
- **Eval**: `synthesis/eval.py` harness (valid rate, confidence, judge mean, contamination gate)
  + `distill-align evaluate` CLI command
- **Export**: kto/grpo/agent/rag_qa/dpo/orpo formats, score-aware DPO ordering, `PreferenceGenerator`
  kto/orpo/grpo converters, shared `BaseFormatter.load`, Unsloth DPO/ORPO/KTO + GRPO script builders
- **Platform (lite)**: `serve.py` (FastAPI factory + MCP tools), `distill-align serve` command,
  OTel-compatible `telemetry.span` hook; new extras: pii/embeddings/gateway/eval/docling/serve/mcp
- **PII**: Anthropic/OpenAI/OpenRouter key patterns, Luhn validation for credit cards,
  optional Presidio NER pass (`distill-align[pii]`)
- **Cache**: canonical `make_canonical_key`/`make_key_for_item` (immune to dict ordering);
  worker uses canonical keys (legacy `make_key` kept)

### Fixed
- `ExportPipeline.validate_export` now works for all JSON formats (shared `BaseFormatter.load`)
- Gateway provider ids (`openai/gpt-5-mini`, `meta-llama/...:free`) resolve pricing correctly
- Reasoning/cached token details no longer double-count usage totals

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

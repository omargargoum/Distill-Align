# Distill-Align

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/omargargoum/Distill-Align/actions/workflows/ci.yml/badge.svg)](https://github.com/omargargoum/Distill-Align/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/distill-align.svg)](https://pypi.org/project/distill-align/)
[![Security](https://github.com/omargargoum/Distill-Align/actions/workflows/security-scan.yml/badge.svg)](https://github.com/omargargoum/Distill-Align/actions/workflows/security-scan.yml)

> **Distill-Align: The Structured Reasoning Extraction Factory**
>
> A CLI/Python framework that automates the generation of high-quality, fine-tuning datasets from raw domain data. It utilizes high-context frontier reasoning models as teachers, captures their deep thinking traces, and filters/prunes those traces into highly structured instruction-following formats optimized for fine-tuning.

🌐 **العربية**: [README.ar.md](README.ar.md) — الترجمة العربية لهذا الدليل.

📖 **Full documentation**: [omargargoum.github.io/Distill-Align](https://omargargoum.github.io/Distill-Align/)

---

## Features

- **Smart Ingestion**: Async chunking pipelines with semantic-aware splitting for Markdown and Code (PDF, DOCX, HTML, CSV, JSON, Jupyter notebooks, and web pages also supported). v0.2 adds **semantic**, **parent-child (small-to-big)**, and **late-contextual** chunkers, table row-sentence serialization, and an optional Docling parser.
- **Multi-Provider Synthesis**: Supports **OpenAI**, **Ollama**, **vLLM**, **Anthropic Claude**, **Google Gemini**, and **Azure OpenAI** backends with async worker pools — plus 2026 gateways (**OpenRouter, LiteLLM, Together, Groq, Mistral, DeepSeek, Cohere**) and strict structured outputs per provider.
- **Socratic Transformer**: Converts raw reasoning into structured, multi-turn conversational Q&A. v0.2 adds **Evol-Instruct**, **RAG-QA (grounded)**, **tool-call/agent traces**, **Constitutional safety**, and **distillation-trace** modes.
- **Scaffold Action Pruner**: Strips conversational filler to extract pure tool-calling or structural output.
- **LLM-as-Judge Evaluation** (optional): Automated quality scoring of generated conversations using a separate judge model, with confidence scores normalized 0–1. v0.2 adds **faithfulness/groundedness rubrics**, **dual-judge (gate + audit)**, and a `distill-align evaluate` CI gate with contamination checks.
- **Preference Generation**: Create pairs for **DPO** — plus **KTO** (unpaired labels), **ORPO** (single-stage), and **GRPO**-ready groups with verifiable-reward stubs.
- **Multiple Export Formats**: ShareGPT, Alpaca, ChatML, HuggingFace messages (JSONL/JSON), KTO, GRPO, agent trajectories, RAG-QA, streaming JSON Lines, and **Apache Parquet**.
- **Streaming Export**: Export large datasets without loading them entirely into memory using iterative producers.
- **Cost Tracking**: Pay-as-you-go cost estimation and tracking across all providers with per-request token accounting.
- **Unsloth Integration**: Auto-generates optimized `train.py` scripts for Unsloth fine-tuning.
- **Rich TUI**: Interactive terminal dashboard for monitoring pipeline execution.

## Installation

```bash
pip install distill-align

# With optional dependencies
pip install distill-align[parquet]   # Parquet export support
pip install distill-align[hub]       # HuggingFace Hub integration
pip install distill-align[all]       # All extras
```

## Package Management

### Update

```bash
pip install --upgrade distill-align
```

### Uninstall

```bash
pip uninstall distill-align
```

### Verify Installation

```bash
distill-align --version
distill-align --help
```

## Docker

A production-ready Docker image is available:

```bash
# Build locally
docker build -t distill-align .

# Run
docker run --rm -v "$(pwd):/app" distill-align --help

# Synthesize with mounted volumes
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  -e OPENAI_API_KEY="sk-..." \
  distill-align synthesize \
    --input /app/data/chunks.json \
    --output /app/output/conversations.json \
    --provider openai \
    --model gpt-5-mini
```

## Configuration

Distill-Align can be configured via **three layers** (highest priority first):

1. **CLI arguments** — flags passed at runtime
2. **Environment variables** — prefixed with `DISTILL_`
3. **Config file** — YAML/TOML file generated via `distill-align init`

Generate a config file to get started:

```bash
distill-align init
```

> 📖 Full configuration reference: [docs/configuration.md](docs/configuration.md)

### Quick Env-Only Setup

```bash
export DISTILL_LLM_PROVIDER=openai
export DISTILL_LLM_MODEL=gpt-5-mini
export DISTILL_LLM_API_KEY=sk-...
export DISTILL_LOG_LEVEL=INFO
```

## Environment Variables (API Keys)

| Variable                  | Required For      | Description                        |
|---------------------------|-------------------|------------------------------------|
| `OPENAI_API_KEY`          | OpenAI / Azure    | OpenAI API key                     |
| `ANTHROPIC_API_KEY`       | Anthropic         | Anthropic API key                  |
| `GOOGLE_API_KEY`          | Google Gemini     | Google AI Studio API key           |
| `AZURE_OPENAI_API_KEY`    | Azure OpenAI      | Azure OpenAI resource key          |
| `AZURE_OPENAI_ENDPOINT`   | Azure OpenAI      | Azure OpenAI endpoint URL          |
| `DASHSCOPE_API_KEY`       | Qwen              | Alibaba DashScope API key          |
| `DEEPSEEK_API_KEY`        | DeepSeek          | DeepSeek API key                   |
| `MISTRAL_API_KEY`         | Mistral           | Mistral La Plateforme key          |
| `OPENROUTER_API_KEY`      | OpenRouter        | OpenRouter gateway key             |
| `DISTILL_LLM_API_KEY`     | Any provider      | Generic override (takes precedence) |

## Quick Start

```bash
# Ingest and process data
distill-align ingest --source ./my-docs --output ./chunks.json

# Synthesize conversations (with judge evaluation)
distill-align synthesize \
    --input ./chunks.json \
    --output ./conversations.json \
    --provider openai \
    --model gpt-5-mini \
    --judge \
    --judge-model gpt-5-nano

# Export to training format
distill-align export \
    --input ./conversations.json \
    --format hf_messages \
    --output ./dataset

# Generate preference pairs for DPO training
distill-align export \
    --input ./conversations.json \
    --format preference \
    --output ./dpo-pairs

# Zero-LLM quality gate (CI-friendly) with contamination check
distill-align evaluate \
    --input ./conversations.json \
    --threshold 0.5

# Serve the REST API (requires the serve extra)
distill-align serve --port 8000

# Launch TUI
distill-align tui
```

## Supported Providers

| Provider    | SDK-Free | Structured Output | Auth                          |
|-------------|----------|-------------------|-------------------------------|
| OpenAI      | ✓        | ✓ (strict)        | API key                       |
| Anthropic   | ✓        | ✓ (`output_config`, GA) | API key                 |
| Google Gemini | ✓      | ✓ (`responseSchema`) | API key                     |
| Azure OpenAI | ✓       | ✓                 | API key or Entra ID (OAuth2)  |
| Ollama      | ✓        | ✓ (schema `format`) | None (local)                |
| vLLM        | ✓        | ✓ (`guided_json`) | None / API key                |
| OpenRouter  | ✓        | ✓ (gateway)       | API key                       |
| LiteLLM proxy | ✓      | ✓ (gateway)       | API key / none (self-host)    |
| Together/Groq/Mistral/DeepSeek/Cohere | ✓ | ✓ (gateway) | API key          |

## Export Formats

| Format             | Extension | Description                                    |
|--------------------|-----------|------------------------------------------------|
| `hf_messages`      | `.jsonl`  | HuggingFace messages format (JSONL recommended) |
| `jsonl`            | `.jsonl`  | Generic JSON Lines (streaming-capable)         |
| `parquet`          | `.parquet`| Columnar format (requires `pyarrow`)           |
| `sharegpt`         | `.json`   | ShareGPT conversation format                   |
| `alpaca`           | `.json`   | Alpaca instruction format                      |
| `chatml`           | `.json`   | ChatML markup format                           |
| `conversation`     | `.json`   | Raw conversation schema export                 |
| `preference`       | `.json`   | DPO preference pairs (requires judge scores)   |
| `dpo` / `orpo`     | `.json`   | DPO / ORPO triples (score-aware when judged)   |
| `kto`              | `.json`   | KTO unpaired rows (`prompt/completion/label`)  |
| `grpo`             | `.json`   | GRPO groups (`prompt/completions/rewards`)     |
| `agent`            | `.json`   | Agent / tool-call trajectories                 |
| `rag_qa`           | `.json`   | Grounded RAG-QA rows (doubles as retrieval eval) |

## Project Structure

This project follows a **Modular Monolith** architecture.

```text
distill-align/
├── src/distill_align/    # Core application package
│   ├── core/             # Config, schemas, logging, caching, checkpointing
│   ├── ingestion/        # Data loaders and chunkers (PDF, DOCX, HTML, code, etc.)
│   ├── synthesis/        # LLM clients, worker pool, prompts, judge, cost tracking
│   │   └── models/       # Provider-specific clients (OpenAI, Anthropic, Gemini, Azure, Ollama, vLLM)
│   ├── exporter/         # Formatters, validator, splitter, preference generator
│   │   └── formatters/   # Output format converters (JSONL, Parquet, ShareGPT, Alpaca, etc.)
│   ├── tui/              # Textual terminal UI
│   └── cli/              # Typer CLI entry points
├── tests/                # Pytest suite
└── docs/                 # Documentation (MkDocs)
```

## Development

1. Clone the repository
2. Install dependencies with Poetry: `poetry install`
3. Install dev dependencies: `poetry install --with dev`
4. Run tests: `poetry run pytest`
5. Run linting: `poetry run ruff check src/`

## License

MIT License - see [LICENSE](LICENSE) for details.

# Distill-Align

**The Structured Reasoning Extraction Factory**

Generate high-quality fine-tuning datasets from raw domain data using frontier reasoning models as teachers.

<div class="grid cards" markdown>

- :material-download: **Install** — `pip install distill-align`
- :material-github: **Source** — [github.com/omargargoum/Distill-Align](https://github.com/omargargoum/Distill-Align)
- :material-tag: **Version** — 0.3.0
- :material-license: **License** — MIT

</div>

---

## What It Does

Distill-Align transforms raw content (documents, code, PDFs, web pages) into structured conversation datasets optimized for fine-tuning LLMs. It captures the reasoning traces of frontier models and refines them into clean, multi-turn Q&A formats.

```
Raw Files  ──▶  Ingestion  ──▶  Synthesis  ──▶  Export
(PDF, MD,       (chunking)     (LLM teacher)   (ShareGPT,
 Code, HTML)                                   Alpaca, ...)
```

## Quick Example

```bash
# 1. Install
pip install distill-align

# 2. Set API key
export OPENAI_API_KEY=sk-...

# 3. Ingest your data
distill-align ingest --source ./docs --output chunks.json

# 4. Generate conversations
distill-align synthesize --input chunks.json --provider openai --model gpt-5-mini

# 5. Export for training
distill-align export --input conversations.json --format sharegpt --split
```

## Core Features

| Feature | Description |
|---------|-------------|
| **9 File Loaders** | Markdown, PDF, DOCX, HTML, Jupyter, JSON, CSV, Code (11 language families), Text (+ Docling) |
| **14 LLM Providers** | OpenAI, Anthropic, Gemini, Azure, Ollama, vLLM, Qwen, OpenRouter, LiteLLM, Together, Groq, Mistral, DeepSeek, Cohere |
| **86-Model Catalog** | Sep-2026 IDs, pricing, context windows — [Models & Pricing](models.md) |
| **Socratic Transformer** | Converts raw content into guided multi-turn Q&A (+ Evol-Instruct, RAG-QA, tool-call, safety, distill modes) |
| **Scaffold Action** | Strips filler, extracts clean structured output |
| **LLM-as-Judge** | Automated quality scoring (7 rubrics, dual-judge, CI `evaluate` gate) |
| **14 Export Formats** | ShareGPT, Alpaca, ChatML, HuggingFace, JSONL, Parquet, DPO/ORPO/KTO/GRPO, agent, RAG-QA |
| **Job Checkpoints** | Resume failed synthesis jobs from last checkpoint |
| **Cost Tracking** | Estimate costs across all providers |
| **TUI Dashboard** | Real-time interactive monitoring |

## Documentation

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting Started](getting-started.md)**

    ---

    Installation, quick start, and first pipeline run.

-   :material-cog: **[Configuration](configuration.md)**

    ---

    Config file, environment variables, and advanced options.

-   :material-database: **[Models & Pricing](models.md)**

    ---

    Every supported model, Sep-2026 prices, and how to pick a teacher.

-   :material-console: **[CLI Reference](cli-reference.md)**

    ---

    All commands, flags, and examples.

-   :material-pipe: **Pipelines**

    ---

    [Ingestion](pipelines/ingestion.md) · [Synthesis](pipelines/synthesis.md) · [Export](pipelines/export.md)

-   :material-book-open-variant: **Guides**

    ---

    [Best Practices](guides/best-practices.md) · [Contributing](guides/contributing.md) · [Troubleshooting](guides/troubleshooting.md)

-   :material-history: **[Changelog](changelog.md)**

    ---

    Release history and changes.

</div>

## Supported Providers

| Provider | API Key | Structured Output | Local |
|----------|---------|-------------------|-------|
| OpenAI | `OPENAI_API_KEY` | ✓ (strict) | — |
| Anthropic | `ANTHROPIC_API_KEY` | ✓ (`output_config`, GA) | — |
| Google Gemini | `GEMINI_API_KEY` | ✓ (`responseSchema`) | — |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | ✓ | — |
| Alibaba Qwen | `DASHSCOPE_API_KEY` | ✓ (OpenAI-compatible) | — |
| DeepSeek | `DEEPSEEK_API_KEY` | ✓ (OpenAI-compatible) | — |
| Mistral | `MISTRAL_API_KEY` | ✓ (OpenAI-compatible) | — |
| OpenRouter / LiteLLM / Together / Groq / Cohere | provider key | ✓ (gateway) | — |
| Ollama | None | ✓ (schema `format`) | ✓ |
| vLLM | None (or API key) | ✓ (`guided_json`) | ✓ |

Full model list: [Models & Pricing](models.md).

## License

MIT License — see [LICENSE](https://github.com/omargargoum/Distill-Align/blob/main/LICENSE) for details.

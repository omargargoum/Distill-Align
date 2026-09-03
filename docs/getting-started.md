# Getting Started

This guide walks you through installing Distill-Align and running your first end-to-end pipeline.

## Prerequisites

- **Python 3.11** or higher
- An LLM API key (OpenAI, Anthropic, etc.) or a local model (Ollama, vLLM)

## Installation

### From PyPI (Recommended)

```bash
pip install distill-align
```

With optional extras:

```bash
pip install distill-align[parquet]   # Apache Parquet export support
pip install distill-align[hub]       # HuggingFace Hub integration`n`npip install distill-align[eval]      # Eval harness (DeepEval + RAGAS)`n`npip install distill-align[serve]     # REST API server`n`npip install distill-align[gateway]   # LiteLLM routing
pip install distill-align[all]       # All optional dependencies
```

### From Source (Development)

```bash
git clone https://github.com/omargargoum/Distill-Align.git
cd Distill-Align
poetry install --with dev
```

### Using Docker

```bash
docker build -t distill-align .
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  distill-align ingest --source /app/data --output /app/output/chunks.json
```

Or use Docker Compose:

```bash
docker compose run distill-align status
```

### Package Management

**Update:**

```bash
pip install --upgrade distill-align
```

**Uninstall:**

```bash
pip uninstall distill-align
```

**Verify Installation:**

```bash
distill-align --version
distill-align --help
```

## Quick Start (5 Steps)

### Step 1: Initialize a Config

```bash
distill-align init
```

This creates `distill-align.yaml` in your current directory:

```yaml
project:
  name: "my-dataset"

ingestion:
  sources:
    - path: ./data
      type: auto

synthesis:
  provider: openai
  model: gpt-5-mini
  max_concurrency: 5

export:
  formats: [sharegpt, alpaca]
  output_dir: ./output
```

### Step 2: Set Your API Key

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Gemini
export GEMINI_API_KEY=AI...

# No key needed for Ollama/vLLM
```

### Step 3: Ingest Your Data

```bash
# Ingest a directory (auto-detects file types)
distill-align ingest --source ./docs --output chunks.json

# Ingest a single file
distill-align ingest --source ./README.md --output chunks.json

# Custom chunk size
distill-align ingest --source ./docs --output chunks.json --chunk-size 2000 --overlap 200
```

**What happens here:** Your files are read, split into semantic chunks, and saved as a JSON array. Each chunk contains the content, metadata (source type, file path, language), and a unique ID.

### Step 4: Synthesize Conversations

```bash
# With OpenAI
distill-align synthesize \
    --input chunks.json \
    --provider openai \
    --model gpt-5-mini \
    --output conversations.json

# With Ollama (local, no API key)
distill-align synthesize \
    --input chunks.json \
    --provider ollama \
    --model qwen3:30b \
    --base-url http://localhost:11434 \
    --output conversations.json

# With Anthropic
distill-align synthesize \
    --input chunks.json \
    --provider anthropic \
    --model claude-sonnet-5 \
    --output conversations.json
```

**What happens here:** Each chunk is sent to the LLM, which creates a multi-turn teaching conversation. The Socratic Transformer generates Q&A pairs, the Scaffold Action cleans the output, and the Pruner removes low-quality results.

!!! tip "Use `--judge` for quality control"
    Add `--judge` to automatically score each conversation on relevance, coherence, correctness, completeness, and safety:

    ```bash
    distill-align synthesize \
        --input chunks.json \
        --provider openai --model gpt-5-mini \
        --judge --judge-model gpt-5-nano \
        --output conversations.json
    ```

### Step 5: Export for Training

```bash
# Export to multiple formats with train/val/test split
distill-align export \
    --input conversations.json \
    --format sharegpt,alpaca,chatml \
    --split \
    --card \
    --output-dir ./output
```

**What happens here:** Conversations are validated, optionally split (90/5/5 by default), formatted to your chosen export format, and saved. A dataset card (README.md) is also generated.

## What You Get

After the full pipeline, your output directory will contain:

```
./output/
├── sharegpt_train.json      # Training data
├── sharegpt_val.json        # Validation data
├── sharegpt_test.json       # Test data
├── alpaca_train.json
├── alpaca_val.json
├── alpaca_test.json
├── chatml_train.json
├── chatml_val.json
├── chatml_test.json
└── README.md                # Dataset card
```

## Resume Failed Jobs

If synthesis is interrupted (network error, Ctrl+C, etc.), resume from the last checkpoint:

```bash
# First run creates a job ID
distill-align synthesize --input chunks.json --job-id my-training-run

# Resume after failure
distill-align synthesize --input chunks.json --job-id my-training-run --resume

# List all jobs
distill-align jobs list
```

## Validate a Dataset

Check quality before training:

```bash
distill-align validate --input conversations.json
```

Output:

```
Validation Report
==================================================
Valid:    Yes
Quality:  0.95 / 1.00
Errors:   0
Warnings: 2

Statistics:
  Conversations:  100
  Total Turns:    312
  Avg Turns/Conv: 3.1
  Est. Tokens:    45,230
  Duplicates:     0
  Filler Ratio:   0.02
```

Deduplicate automatically:

```bash
distill-align validate --input conversations.json --dedupe --output clean.json
```

## Using the TUI

Launch the interactive terminal dashboard:

```bash
distill-align tui
```

Features:

- 📊 Real-time stats (conversations created, cache hits, errors)
- 📋 Job management (list, resume, delete)
- 💾 Cache inspector (view, prune, clear)
- 🔧 Configuration viewer
- 📝 Live logs
- 🧠 All 10 synthesis modes, 14 providers, 14 export formats, 7 chunkers
- 📊 Evaluate quality gate (PASS/FAIL with thresholds, no LLM calls)
- Press `m` for expert options (chunker, concurrency, judge, dataset card)

## Programmatic Usage

Use Distill-Align as a Python library:

```python
import asyncio
from distill_align.core.schemas import SynthesisConfig
from distill_align.exporter.pipeline import ExportPipeline
from distill_align.ingestion.auto import AutoIngestionPipeline
from distill_align.synthesis.pipeline import SynthesisPipeline


async def main():
    # Ingest
    pipeline = AutoIngestionPipeline()
    chunks = pipeline.ingest_directory("./data")
    print(f"Ingested {len(chunks)} chunks")

    # Synthesize
    synth = SynthesisPipeline(SynthesisConfig(llm_provider="openai", model_name="gpt-5-mini"))
    conversations = await synth.synthesize_batch(chunks)
    print(f"Generated {len(conversations)} conversations")

    # Export
    export = ExportPipeline()
    files = export.export(conversations, formats=["sharegpt", "alpaca"])
    for name, path in files.items():
        print(f"{name}: {path}")


asyncio.run(main())
```

## Next Steps

- [Configuration](configuration.md) — Advanced config options and environment variables
- [CLI Reference](cli-reference.md) — All commands and flags
- [Pipelines](pipelines/ingestion.md) — How each pipeline stage works
- [Best Practices](guides/best-practices.md) — Tips for quality datasets
- [Examples](https://github.com/omargargoum/Distill-Align/tree/main/examples) — Runnable example scripts

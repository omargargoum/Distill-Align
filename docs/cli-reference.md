# CLI Reference

## Global Options

These options are available for all commands:

```
--log-level, -l   Set logging level (DEBUG, INFO, WARNING, ERROR)
--log-file        Path to log file
--config, -c      Path to config file
--help            Show help
```

## Commands

### `distill-align init`

Initialize a new project with a config file.

```bash
distill-align init [OPTIONS]
  --path, -p    Output config path (default: distill-align.yaml)
  --name, -n    Project name (default: my-dataset)
```

### `distill-align ingest`

Ingest files and split into semantic chunks.

```bash
distill-align ingest SOURCE [OPTIONS]
  SOURCE                     Source file or directory path
  --output, -o               Output file path (default: ./chunks.json)
  --chunk-size, -s           Chunk size in characters (default: 1000)
  --overlap                  Chunk overlap in characters (default: 200)
  --recursive/--no-recursive Search subdirectories (default: recursive)
  --auto/--no-auto           Auto-detect file types (default: auto)
```

### `distill-align synthesize`

Synthesize chunks into structured conversations.

```bash
distill-align synthesize INPUT [OPTIONS]
  INPUT                  Input chunks JSON file
  --output, -o           Output file path (default: ./conversations.json)
  --provider, -p         LLM provider (openai, anthropic, gemini, azure, ollama, vllm,
                         qwen, openrouter, litellm, together, groq, mistral, deepseek,
                         cohere) (default: openai)
  --model, -m            Model name (default: gpt-5-mini; see Models & Pricing)
  --base-url             API base URL
  --api-key              API key (or use OPENAI_API_KEY env var)
  --concurrency, -c      Max concurrent requests (default: 5)
  --rpm                  Max requests per minute (default: 60)
  --job-id               Job ID for resume support
  --resume               Resume a previous job
  --no-cache             Disable caching
  --no-checkpoint        Disable checkpointing
  --prompts              Custom prompts directory
  --mode                 Conversation mode: default, teach, debug, review, qa, explain,
                         evol_instruct, rag_qa, tool_call, constitutional, distill
  --judge                Enable LLM-as-judge evaluation
  --judge-model          Model for judge (defaults to --model)

After completion, a cost summary panel shows estimated token usage and API cost.
```

### `distill-align export`

Export conversations to training formats.

```bash
distill-align export INPUT [OPTIONS]
  INPUT                Input conversations JSON file
  --format, -f         Export formats (comma-separated): sharegpt, alpaca, chatml,
                        conversation, hf_messages, jsonl, parquet, preference, dpo,
                        orpo, kto, grpo, agent, rag_qa (default: sharegpt)
  --output-dir, -o     Output directory (default: ./output)
  --model              Unsloth model name
  --no-unsloth         Skip Unsloth script generation
  --split              Split into train/val/test
  --card               Generate dataset card

Note: parquet format requires `pip install distill-align[parquet]` or `pip install pyarrow`.
```

### `distill-align validate`

Validate and analyze a dataset.

```bash
distill-align validate INPUT [OPTIONS]
  INPUT              Input conversations JSON file
  --dedupe/--no-dedupe   Remove duplicates (default: dedupe)
  --output, -o       Save report to file
```

### `distill-align evaluate`

Evaluate dataset quality without LLM calls (CI-friendly gate).

```bash
distill-align evaluate INPUT [OPTIONS]
  INPUT              Input conversations JSON file
  --threshold        Min mean confidence 0-1 (default: 0.5)
  --min-valid-rate   Min valid-turn rate 0-1 (default: 0.8)
  --reference        Eval-set text file for contamination check
  --output, -o       Save report to file
```

### `distill-align serve`

Serve the REST API (`pip install distill-align[serve]`).

```bash
distill-align serve [--host 127.0.0.1] [--port 8000]
```

### `distill-align status`

Check configuration and connections.

```bash
distill-align status
```

### `distill-align jobs`

Manage synthesis jobs.

```bash
# List all jobs
distill-align jobs list [--status STATUS] [--type TYPE] [--limit N]

# Resume a job (just shows instructions)
distill-align jobs resume JOB_ID

# Delete a job checkpoint
distill-align jobs delete JOB_ID [--force]

# Clean up old job checkpoints
distill-align jobs cleanup [--days N]
```

### `distill-align config`

View and manage configuration.

```bash
# Show current configuration
distill-align config show

# Show config file path
distill-align config path
```

### `distill-align tui`

Launch the interactive terminal UI.

```bash
distill-align tui
```

### `distill-align version`

Show version information.

```bash
distill-align version
```

## Examples

### Full Pipeline

```bash
# Initialize
distill-align init

# Ingest
distill-align ingest --source ./data --output chunks.json

# Synthesize
distill-align synthesize --input chunks.json --output conversations.json

# Export with splitting and card
distill-align export --input conversations.json --format sharegpt,alpaca --split --card
```

### With Local Ollama

```bash
distill-align synthesize \
    --input chunks.json \
    --provider ollama \
    --model qwen3:30b \
    --base-url http://localhost:11434 \
    --concurrency 2 \
    --rpm 30
```

### Frontier Teacher + Cheap Judge

```bash
distill-align synthesize \
    --input chunks.json \
    --provider anthropic \
    --model claude-sonnet-5 \
    --judge --judge-model claude-haiku-4-5 \
    --output conversations.json

distill-align evaluate --input conversations.json --threshold 0.6
```

### Resume a Failed Job

```bash
distill-align synthesize \
    --input chunks.json \
    --job-id synth_20240101_120000_abc123 \
    --resume
```

### Validate Before Training

```bash
distill-align validate --input conversations.json --output validation_report.txt
```

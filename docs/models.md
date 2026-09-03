# Models & Pricing

> Refreshed **September 2026**. This page is generated from the in-code catalog
> (`distill_align.synthesis.models.catalog`) — the single source of truth the
> CLI, cost tracker, and provider registry all read from. Prices are USD per
> **1M tokens** (short-context standard tier). `official` = provider pricing
> page · `indicative` = gateway/vendor list price, verify before invoicing ·
> `free` = self-hosted open weights.

## How to pick a teacher model

| Workload | Reach for | Why |
|---|---|---|
| Balanced production synthesis | `gpt-5-mini`, `claude-sonnet-5`, `gemini-3.5-flash` | Best quality-per-dollar; strict structured outputs |
| Hardest reasoning / long agents | `gpt-5.6-sol`, `claude-fable-5`, `claude-opus-4-8` | Frontier traces; verify with dual-judge |
| High-volume / cheap judges | `gpt-5-nano`, `claude-haiku-4-5`, `gemini-3.5-flash-lite` | Judge-tier pricing; 5–15× cheaper eval loops |
| Code-heavy corpora | `qwen3-coder-next`, `deepseek-v4-flash`, `gpt-5.6-terra` | Agentic-coding specialists |
| Private / offline / free | `qwen3:30b`, `llama4:scout` (Ollama) | No per-token cost; data never leaves |
| Training student (Unsloth) | `Qwen/Qwen3-8B` | 110K-ctx GRPO on H100; QLoRA-friendly |

```bash
# Frontier quality
distill-align synthesize --input chunks.json --provider anthropic --model claude-sonnet-5

# Cheap + fast (default)
distill-align synthesize --input chunks.json --provider openai --model gpt-5-mini

# Free local generation
ollama pull qwen3:30b
distill-align synthesize --input chunks.json --provider ollama --model qwen3:30b

# Qwen commercial API (thinking mode supported via DashScope)
export DASHSCOPE_API_KEY=sk-...
distill-align synthesize --input chunks.json --provider qwen --model qwen3-max
```

## OpenAI

| Model | In / 1M | Out / 1M | Context | Status |
|---|---|---|---|---|
| `gpt-5.6-sol` (flagship) | $4.00 | $20.00 | 1M | current (promo thru Nov 21 2026) |
| `gpt-5.6-terra` (balanced) | $2.00 | $12.00 | 1M | current |
| `gpt-5.6-luna` (volume) | $0.20 | $1.20 | 1M | current |
| `gpt-5.5` | $5.00 | $30.00 | 400K | current |
| `gpt-5.4` | $2.50 | $15.00 | 272K | current |
| `gpt-5.4-mini` | $0.75 | $4.50 | 272K | current |
| `gpt-5.4-nano` | $0.20 | $1.25 | 272K | current |
| `gpt-5` / `gpt-5.1` | $1.25 | $10.00 | 272K/400K | current |
| `gpt-5-mini` (default) | $0.25 | $2.00 | 272K | current |
| `gpt-5-nano` | $0.05 | $0.40 | 272K | current |
| `gpt-5-codex` | $1.25 | $10.00 | 400K | current |
| `o3` (reasoning) | $2.00 | $8.00 | 200K | current |
| `gpt-4o`, `gpt-4o-mini`, `gpt-4.1*` | as listed | as listed | 128K/1M | legacy |

Notes: newer reasoning models only accept `max_completion_tokens` — handled
automatically. Safety refusals arrive in a separate field and are surfaced
instead of empty content. Open-weight `gpt-oss:20b` / `gpt-oss:120b`
(`ollama pull gpt-oss:20b`, 16GB) are free to self-host.

## Anthropic Claude

| Model | In / 1M | Out / 1M | Context | Status |
|---|---|---|---|---|
| `claude-fable-5` (most capable) | $10.00 | $50.00 | 1M | current |
| `claude-opus-4-8` (agentic coding) | $5.00 | $25.00 | 1M | current |
| `claude-sonnet-5` (default) | $2.00 | $10.00 | 1M | current |
| `claude-haiku-4-5` (+ dated `-20251001`) | $1.00 | $5.00 | 200K | current |
| `claude-sonnet-4-6`, `claude-opus-4-7` | $3/$5 | $15/$25 | 1M | legacy |
| `claude-sonnet-4-20250514` | $3.00 | $15.00 | 200K | legacy |

Structured outputs are GA via `output_config` (no beta header). Prompt caching
cuts input cost up to 90% on repeated prefixes. `claude-sonnet-4-6` and
earlier 4.x IDs are legacy — `claude-sonnet-5` is the production default.

## Google Gemini

| Model | In / 1M | Out / 1M | Context | Status |
|---|---|---|---|---|
| `gemini-3.8-flash` (latest) | $0.10 | $0.40 | 1M | current (indicative) |
| `gemini-3.6-flash` | $0.10 | $0.40 | 1M | current (indicative) |
| `gemini-3.5-flash` (default, GA) | $0.10 | $0.40 | 1M | current (indicative) |
| `gemini-3.5-flash-lite` (subagents) | $0.075 | $0.30 | 1M | current (indicative) |
| `gemini-3.1-pro` (flagship) | $1.25 | $10.00 | 1M | current (indicative) |
| `gemini-2.5-pro` / `gemini-2.5-flash` | $1.25/$0.15 | $10.00/$0.60 | 1M | legacy |
| `gemini-2.0-flash` | $0.10 | $0.40 | 1M | legacy — **shut down Jun 1 2026** |

Native `responseSchema` constrained decoding. Open-weight `gemma4`
(`ollama pull gemma4:e4b`) covers vision + tool calling locally.

## Alibaba Qwen

| Model | In / 1M | Out / 1M | Context | Status |
|---|---|---|---|---|
| `qwen3-max` (flagship, DashScope) | $0.36 | $1.43 | 256K | current (indicative) |
| `qwen3-coder-next` (agentic coding) | $0.20 | $1.50 | 256K | current (indicative) |
| `qwen3.8-max` (coding + cowork) | $0.55 | $3.50 | 1M | current (indicative) |
| `qwen3.6-plus` / `qwen3.5-plus` | $0.50/$0.40 | $3.00/$2.40 | 1M | current (indicative) |
| `qwen3.5-flash` | $0.10 | $0.40 | 1M | current (indicative) |
| `qwen3.5-27b` / `-122b-a10b` / `-397b-a17b` | $0.29–$0.55 | $1.83–$3.50 | 256K | current (indicative) |
| `qwen3:30b` (Ollama, Apache 2.0) | free | free | 256K | local default |
| `qwen3-coder:30b` (Ollama) | free | free | 256K | local |
| `Qwen/Qwen3-8B` (HF, training) | free | free | 256K | local (Unsloth default) |

All Qwen3 models ship **thinking / non-thinking** dual modes in one
checkpoint (`enable_thinking` / `/think` / `/no_think`). Provider id: `qwen`
(DashScope OpenAI-compatible endpoint, `DASHSCOPE_API_KEY`).

## Meta Llama 4 (open weights, MoE)

| Model | Params | Context | Run it |
|---|---|---|---|
| Scout (fast, multimodal) | 109B / 17B active | **10M** | `ollama pull llama4:scout` · `meta-llama/Llama-4-Scout-17B-16E-Instruct` (vLLM) |
| Maverick (most capable) | 400B / 17B active | 1M | `ollama pull llama4:maverick` · `meta-llama/Llama-4-Maverick-17B-128E-Instruct` (vLLM) |
| Behemoth (~2T) | TBD | TBD | paused May 2026 — not available |

Scout Q4 fits ~24GB VRAM; Maverick needs multi-GPU. Pair with
**Llama Guard 4** for input/output safety when self-hosting.

## DeepSeek

| Model | In / 1M | Out / 1M | Context | Status |
|---|---|---|---|---|
| `deepseek-v4-flash` (MIT weights) | $0.14 | $0.28 | 1M | current |
| `deepseek-v4-flash-vision` (API-only exp.) | $0.15 | $0.29 | 1M | current |
| `deepseek-v4-pro` (1.6T MoE) | $0.50 | $2.00 | 1M | current (indicative out) |
| `deepseek-chat` / `deepseek-reasoner` | $0.27/$0.55 | $1.10/$2.19 | 128K | current (indicative) |
| `deepseek-r1` (Ollama) | free | free | 128K | local reasoning |

V4-Flash-0731 (Terminal-Bench 82.7) is the draft-and-verify value pick;
weights: `deepseek-ai/DeepSeek-V4-Flash-0731` (MIT).

## Mistral

| Model | In / 1M | Out / 1M | Status |
|---|---|---|---|
| `mistral-large-latest` (Large 3, Apache 2.0) | $0.50 | $1.50 | current |
| `mistral-medium-3.5` (agentic + coding) | $0.40 | $2.00 | current (indicative) |
| `mistral-small-latest` | $0.15 | $0.60 | current |
| `codestral-latest` (code) | $0.30 | $0.90 | current (indicative) |
| `ministral-3-8b` / `-14b` (Apache 2.0) | free | free | local |

EU data residency (GDPR-native) is Mistral's enterprise edge.

## Gateways & local runtimes

| Provider | Base URL | Default model | Key |
|---|---|---|---|
| `openrouter` | `https://openrouter.ai/api/v1` | `qwen/qwen3-30b-a3b` | `OPENROUTER_API_KEY` |
| `litellm` (self-host proxy) | `http://localhost:4000/v1` | `gpt-5-mini` | `LITELLM_API_KEY` |
| `together` | `https://api.together.xyz/v1` | Llama-4-Scout | `TOGETHER_API_KEY` |
| `groq` (fastest inference) | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| `cohere` | `https://api.cohere.ai/compatibility/v1` | `command-r-plus` | `COHERE_API_KEY` |
| `ollama` | `http://localhost:11434` | `qwen3:30b` | none |
| `vllm` | `http://localhost:8000/v1` | Llama-4-Scout | none |

Ollama ≥ v0.33 supports schema-constrained structured output and the
Qwen3.8 / Llama 4 / gpt-oss families — upgrade with
`curl https://ollama.ai/install.sh | sh`.

## Legacy aliases

Old IDs keep working for cost accounting and are auto-suggested forward —
never silently rewritten:

| Legacy | Successor |
|---|---|
| `gpt-4o`, `gpt-4o-mini` | `gpt-5-mini` |
| `gpt-4.1` | `gpt-5.6-terra` |
| `claude-sonnet-4-20250514`, `claude-sonnet-4-6` | `claude-sonnet-5` |
| `claude-opus-4-7` | `claude-opus-4-8` |
| `gemini-2.0-flash`, `gemini-2.5-flash` | `gemini-3.5-flash` |
| `llama3.1` | `llama4:scout` |
| `llama3.2` | `qwen3:30b` |

## Training students (Unsloth)

| Recipe | Dataset format | Script | Notes |
|---|---|---|---|
| SFT (QLoRA) | `sharegpt` / `chatml` / `hf_messages` | `train.py` | `SFTConfig`, `adamw_8bit`, Qwen3-8B default |
| DPO | `dpo` / `preference` | `train_dpo.py` | `DPOConfig`, β=0.1, LR 5e-7, 1 epoch |
| ORPO (single-stage, no ref model) | `orpo` | `train_orpo.py` | ~1.2× SFT VRAM |
| KTO (thumbs-up/down) | `kto` | `train_kto.py` | boolean `label` rows |
| GRPO (verifiable rewards) | `grpo` | `train_grpo.py` | vLLM rollouts, auto-tuned long context |

```bash
pip install --upgrade --no-cache-dir unsloth unsloth_zoo
distill-align export --input conversations.json --format dpo,grpo --output-dir ./output
# → ./output/train_dpo.py + ./output/train_grpo.py alongside the datasets
```

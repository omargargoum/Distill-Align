# Distill-Align

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/omargargoum/Distill-Align/actions/workflows/ci.yml/badge.svg)](https://github.com/omargargoum/Distill-Align/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/distill-align.svg)](https://pypi.org/project/distill-align/)
[![Security](https://github.com/omargargoum/Distill-Align/actions/workflows/security-scan.yml/badge.svg)](https://github.com/omargargoum/Distill-Align/actions/workflows/security-scan.yml)

> **Distill-Align: مصنع الاستدلال المنظَّم**
>
> إطار عمل سطر أوامر / بايثون يُؤتمت عملية إنشاء مجموعات بيانات عالية الجودة لضبط النماذج اللغوية (fine-tuning) انطلاقًا من البيانات الخام. يستخدم نماذج الاستدلال المتطورة كمُعلّمين، ويلتقط آثار تفكيرهم العميق، ثم يُرشّحها ويُهذّبها إلى صيغ تعليمية منظمة ومناسبة لضبط النماذج.

📖 **الوثائق الكاملة**: [omargargoum.github.io/Distill-Align](https://omargargoum.github.io/Distill-Align/)

🌐 **English**: [README.md](README.md) — English version of this guide.

---

## الميزات

- **استيراد ذكي**: تقسيم دلالي وأبوي (parent-child) وسياقي متأخر (late-contextual) لمستندات Markdown و Code (يدعم أيضًا PDF و DOCX و HTML و CSV و JSON و Jupyter notebook وصفحات الويب، مع محلل Docling الاختياري).
- **توليد عبر مزودات متعددة**: يدعم **OpenAI** و **Ollama** و **vLLM** و **Anthropic Claude** و **Google Gemini** و **Azure OpenAI** و **Qwen** وبوابات **OpenRouter و Together و Groq و Mistral و DeepSeek و Cohere** مع مخرجات هيكلية صارمة لكل مزود.
- **محوّل سقراطي (Socratic Transformer)**: يحوّل الاستدلال الخام إلى حوارات سؤال وجواب متعددة الأدوار ومنظّمة، مع أنماط Evol-Instruct و RAG-QA و tool-call و Constitutional و distillation.
- **مُهذّب Scaffold Action**: يزيل الحشو اللغوي لاستخراج المخرجات الهيكلية النقية.
- **تقييم LLM كحَكَم (اختياري)**: تقييم آلي على 7 معايير (منها faithfulness) مع حَكَم مزدوج وبوابة جودة `evaluate`، ونتائج ثقة من 0 إلى 1.
- **توليد تفضيلات**: أزواج **DPO** وصفوف **KTO** و **ORPO** ومجموعات **GRPO** للتدريب التفضيلي والتعزيزي.
- **صيغ تصدير متعددة**: ShareGPT، Alpaca، ChatML، HuggingFace messages (JSONL/JSON)، KTO، GRPO، agent، RAG-QA، JSON Lines المتدفق، و **Apache Parquet**.
- **تصدير متدفق (Streaming)**: تصدير مجموعات بيانات كبيرة دون تحميلها بالكامل في الذاكرة باستخدام منتجات تكرارية.
- **تتبّع التكاليف**: تقدير التكاليف أثناء الاستخدام عبر جميع المزودات مع محاسبة الرموز لكل طلب.
- **تكامل مع Unsloth**: يولّد نصوص `train.py` مُحسّنة لضبط النماذج باستخدام Unsloth.
- **واجهة طرفية غنية (TUI)**: لوحة تحكم تفاعلية لمراقبة تنفيذ سير العمل.

## التثبيت

```bash
pip install distill-align

# مع الاعتماديات الاختيارية
pip install distill-align[parquet]   # دعم تصدير Parquet
pip install distill-align[hub]       # تكامل مع HuggingFace Hub
pip install distill-align[all]       # جميع الإضافات
```

## إدارة الحزمة

### التحديث

```bash
pip install --upgrade distill-align
```

### الإزالة

```bash
pip uninstall distill-align
```

### التحقق من التثبيت

```bash
distill-align --version
distill-align --help
```

## Docker

صورة Docker جاهزة للإنتاج:

```bash
# بناء محلي
docker build -t distill-align .

# تشغيل
docker run --rm -v "$(pwd):/app" distill-align --help

# توليد الحوارات مع مجلدات محملة
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

## الإعدادات

يمكن تهيئة Distill-Align عبر **ثلاث طبقات** (الأعلى أولوية أولاً):

1. **وسائط سطر الأوامر (CLI)** — تُمرر عند التشغيل
2. **متغيرات البيئة** — مسبوقة بالبادئة `DISTILL_`
3. **ملف الإعدادات** — بصيغة YAML أو TOML ويُولّد عبر `distill-align init`

لتوليد ملف إعدادات مبدئي:

```bash
distill-align init
```

> 📖 دليل الإعدادات الكامل: [docs/configuration.md](docs/configuration.md)

### إعداد سريع باستخدام المتغيرات فقط

```bash
export DISTILL_LLM_PROVIDER=openai
export DISTILL_LLM_MODEL=gpt-5-mini
export DISTILL_LLM_API_KEY=sk-...
export DISTILL_LOG_LEVEL=INFO
```

## متغيرات البيئة (مفاتيح API)

| المتغير                | مطلوب لـ           | الوصف                            |
|------------------------|---------------------|----------------------------------|
| `OPENAI_API_KEY`       | OpenAI / Azure      | مفتاح OpenAI API                 |
| `ANTHROPIC_API_KEY`    | Anthropic           | مفتاح Anthropic API              |
| `GOOGLE_API_KEY`       | Google Gemini       | مفتاح Google AI Studio API       |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI        | مفتاح مورد Azure OpenAI          |
| `AZURE_OPENAI_ENDPOINT`| Azure OpenAI        | رابط نقطة نهاية Azure OpenAI     |
| `DISTILL_LLM_API_KEY`  | أي مزود             | تجاوز عام (له الأولوية)          |

## البداية السريعة

```bash
# استيراد ومعالجة البيانات
distill-align ingest --source ./my-docs --output ./chunks.json

# توليد الحوارات (مع تقييم الحَكَم)
distill-align synthesize \
    --input ./chunks.json \
    --output ./conversations.json \
    --provider openai \
    --model gpt-5-mini \
    --judge \
    --judge-model gpt-5-nano

# التصدير إلى صيغة التدريب
distill-align export \
    --input ./conversations.json \
    --format hf_messages \
    --output ./dataset

# توليد أزواج تفضيل لتدريب DPO
distill-align export \
    --input ./conversations.json \
    --format preference \
    --output ./dpo-pairs

# تشغيل الواجهة التفاعلية
distill-align tui
```

## المزودات المدعومة

| المزود       | بدون SDK | مخرجات منظمة | التوثيق                      |
|--------------|----------|---------------|------------------------------|
| OpenAI       | ✓        | ✓             | مفتاح API                    |
| Anthropic    | ✓        | ✓ (وضع JSON)  | مفتاح API                    |
| Google Gemini| ✓        | ✓ (نوع MIME)  | مفتاح API                    |
| Azure OpenAI | ✓        | ✓             | مفتاح API أو Entra ID (OAuth2) |
| Ollama       | ✓        | —             | لا شيء (محلي)                |
| vLLM         | ✓        | ✓ (متوافق مع OpenAI) | لا شيء / مفتاح API      |

## صيغ التصدير

| الصيغة              | الامتداد      | الوصف                                           |
|---------------------|---------------|-------------------------------------------------|
| `hf_messages`       | `.jsonl`      | صيغة رسائل HuggingFace (JSONL موصى به)          |
| `jsonl`             | `.jsonl`      | JSON Lines عام (يدعم التدفق)                    |
| `parquet`           | `.parquet`    | صيغة عمودية (يتطلب `pyarrow`)                   |
| `sharegpt`          | `.json`       | صيغة محادثات ShareGPT                           |
| `alpaca`            | `.json`       | صيغة تعليمات Alpaca                             |
| `chatml`            | `.json`       | صيغة ترميز ChatML                               |
| `conversation`      | `.json`       | تصدير مخطط المحادثة الخام                        |
| `preference`        | `.json`       | أزواج تفضيل DPO (يتطلب تقييم الحَكَم)           |

## هيكل المشروع

يتبع المشروع نمط **الوحدة الأحادية المعيارية (Modular Monolith)**.

```text
distill-align/
├── src/distill_align/    # حزمة التطبيق الأساسية
│   ├── core/             # الإعدادات، المخططات، التسجيل، التخزين المؤقت، نقاط التفتيش
│   ├── ingestion/        # أدوات تحميل البيانات وتقسيمها (PDF، DOCX، HTML، كود، إلخ)
│   ├── synthesis/        # عملاء LLM، مجمّع العمل، الاستدعاءات، الحَكَم، تتبّع التكاليف
│   │   └── models/       # عملاء خاصون بكل مزود (OpenAI، Anthropic، Gemini، Azure، Ollama، vLLM)
│   ├── exporter/         # أدوات التنسيق، المدقّق، المقسم، مولد التفضيلات
│   │   └── formatters/   # محوّلات صيغ الإخراج (JSONL، Parquet، ShareGPT، Alpaca، إلخ)
│   ├── tui/              # واجهة المستخدم النصية (Textual)
│   └── cli/              # نقاط الدخول عبر Typer
├── tests/                # اختبارات Pytest
└── docs/                 # التوثيق (MkDocs)
```

## التطوير

1. استنساخ المستودع
2. ثبت الاعتماديات بواسطة Poetry: `poetry install`
3. ثبت اعتماديات التطوير: `poetry install --with dev`
4. شغّل الاختبارات: `poetry run pytest`
5. شغّل التدقيق اللغوي: `poetry run ruff check src/`

## الترخيص

رخصة MIT — راجع ملف [LICENSE](LICENSE) للتفاصيل.

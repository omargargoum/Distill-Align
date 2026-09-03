"""
LLM-as-judge for evaluating conversation quality.

Uses a separate LLM call to score generated conversations on multiple criteria.
"""

import json
from typing import Any

from loguru import logger

from ..core.schemas import ConversationSchema
from .models.base import BaseLLMClient, LLMMessage

JUDGE_PROMPT = """You are an expert evaluator of LLM training data. Evaluate the following conversation on these criteria:

1. **Relevance** (0-10): How well does the conversation relate to the source content?
2. **Coherence** (0-10): Is the conversation logical and well-structured?
3. **Correctness** (0-10): Is the information factually accurate?
4. **Completeness** (0-10): Does the conversation cover the key points?
5. **Safety** (0-10): Is the content safe and appropriate?

Source content:
{source_content}

Conversation:
{conversation_text}

Respond with a JSON object only:
{"relevance": <score>, "coherence": <score>, "correctness": <score>, "completeness": <score>, "safety": <score>, "overall": <average>, "explanation": "<brief explanation>"}}
"""

# Phase 4: extended rubric library (G-Eval style). Each entry maps a rubric
# name → one-line criterion used to build custom judge prompts.
JUDGE_RUBRICS: dict[str, str] = {
    "relevance": "How well does the conversation relate to the source content?",
    "coherence": "Is the conversation logical and well-structured?",
    "correctness": "Is the information factually accurate and consistent with the source?",
    "completeness": "Does the conversation cover the key points of the source?",
    "safety": "Is the content safe, non-harmful, and appropriate for training?",
    "faithfulness": "Is every factual claim in the assistant turns grounded in the source content (no hallucinations)?",
    "groundedness": "For RAG-QA rows: is the answer supported by the cited chunk, with abstention when unanswerable?",
}

DEFAULT_RUBRICS = ("relevance", "coherence", "correctness", "completeness", "safety")
EXTENDED_RUBRICS = (*DEFAULT_RUBRICS, "faithfulness", "groundedness")


def build_judge_prompt(
    source_content: str,
    conversation_text: str,
    rubrics: tuple[str, ...] | list[str] = DEFAULT_RUBRICS,
) -> str:
    """Build a G-Eval-style rubric prompt for the given rubric set."""
    lines = [
        "You are an expert evaluator of LLM training data. Evaluate the following conversation on these criteria:",
        "",
    ]
    for i, name in enumerate(rubrics, 1):
        criterion = JUDGE_RUBRICS.get(name, name)
        lines.append(f"{i}. **{name.capitalize()}** (0-10): {criterion}")
    lines += [
        "",
        "Source content:",
        source_content or "No source content provided",
        "",
        "Conversation:",
        conversation_text,
        "",
        "Respond with a JSON object only, with one 0-10 score per rubric plus overall + explanation:",
    ]
    keys = ", ".join(f'"{r}": <score>' for r in rubrics)
    lines.append("{" + keys + ', "overall": <average>, "explanation": "<brief explanation>"}')
    return "\n".join(lines)


def normalize_scores(scores: dict[str, Any]) -> dict[str, Any]:
    """Clamp numeric rubric scores to [0, 10] and set overall if missing."""
    out = dict(scores)
    nums = []
    for k, v in list(out.items()):
        if k in ("explanation", "error"):
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            clamped = max(0.0, min(10.0, float(v)))
            out[k] = round(clamped, 2)
            if k != "overall":
                nums.append(clamped)
    if "overall" not in out and nums:
        out["overall"] = round(sum(nums) / len(nums), 2)
    return out


def ngram_overlap(a: str, b: str, n: int = 8) -> float:
    """Contamination proxy: fraction of *a*'s n-grams found in *b* (0-1)."""
    toks_a = a.lower().split()
    toks_b = b.lower().split()
    if not toks_a or not toks_b:
        return 0.0
    # Back off for short texts so small eval rows still produce a signal.
    n_eff = max(1, min(n, len(toks_a), len(toks_b)))

    def ngrams(toks: list[str]) -> set[str]:
        return {" ".join(toks[i : i + n_eff]) for i in range(max(len(toks) - n_eff + 1, 0))}

    grams_a = ngrams(toks_a)
    if not grams_a:
        return 0.0
    grams_b = ngrams(toks_b)
    return len(grams_a & grams_b) / len(grams_a)


class ConversationJudge:
    """LLM-as-judge for evaluating conversation quality."""

    def __init__(self, llm_client: BaseLLMClient, rubrics: tuple[str, ...] | list[str] | None = None):
        """Initialize the judge with an LLM client.

        Args:
            llm_client: LLM client used for evaluation.
            rubrics: Optional rubric subset (defaults to the 5 legacy criteria
                so existing callers are unaffected).
        """
        self.llm_client = llm_client
        self.rubrics: tuple[str, ...] = tuple(rubrics) if rubrics else DEFAULT_RUBRICS

    async def evaluate(
        self,
        conversation: ConversationSchema,
        source_content: str | None = None,
        max_tokens: int | None = None,
        rubrics: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single conversation.

        Args:
            conversation: The conversation to evaluate.
            source_content: Optional source content for context.
            max_tokens: Optional max tokens for the evaluation call.
            rubrics: Optional per-call rubric override.

        Returns:
            Dictionary with evaluation scores.
        """
        conversation_text = json.dumps(
            [{"role": t.role, "content": t.content} for t in conversation.turns],
            indent=2,
        )

        active = tuple(rubrics) if rubrics else self.rubrics
        if active == DEFAULT_RUBRICS:
            # Legacy path: identical prompt bytes to v0.1 (replace, not format,
            # to survive literal braces in source code).
            prompt = JUDGE_PROMPT.replace(
                "{source_content}",
                source_content or "No source content provided",
            ).replace(
                "{conversation_text}",
                conversation_text,
            )
        else:
            prompt = build_judge_prompt(source_content or "", conversation_text, active)

        try:
            # Use chat_structured for reliable JSON parsing via response_format
            messages = [
                LLMMessage(role="system", content="You are a quality evaluator."),
                LLMMessage(role="user", content=prompt),
            ]
            result = await self.llm_client.chat_structured(messages, max_tokens=max_tokens)
            return normalize_scores(result)

        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return {"error": str(e)}

    async def evaluate_dual(
        self,
        conversation: ConversationSchema,
        source_content: str | None = None,
        audit_judge: BaseLLMClient | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Dual-judge: cheap gate score + optional frontier audit score.

        Returns ``{"gate": {...}, "audit": {...} | None, "agreement": float | None}``.
        """
        from .models.base import BaseLLMClient as _Base

        gate = await self.evaluate(conversation, source_content, max_tokens)
        audit: dict[str, Any] | None = None
        agreement: float | None = None
        if audit_judge is not None:
            auditor = ConversationJudge(llm_client=audit_judge, rubrics=self.rubrics)  # type: ignore[arg-type]
            audit = await auditor.evaluate(conversation, source_content, max_tokens)
            try:
                g = float(gate.get("overall", 0))
                a = float(audit.get("overall", 0))
                agreement = round(1.0 - abs(g - a) / 10.0, 3)
            except (TypeError, ValueError):
                agreement = None
        _ = _Base  # keep import meaningful for type-checkers
        return {"gate": gate, "audit": audit, "agreement": agreement}

    async def evaluate_batch(
        self,
        conversations: list[ConversationSchema],
        source_contents: list[str | None] | None = None,
        max_concurrency: int = 5,
    ) -> list[dict[str, Any]]:
        """Evaluate a batch of conversations.

        Args:
            conversations: List of conversations to evaluate.
            source_contents: Optional source contents for each conversation.
            max_concurrency: Maximum concurrent evaluations.

        Returns:
            List of evaluation result dictionaries.
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrency)

        async def evaluate_one(
            conv: ConversationSchema,
            source: str | None,
        ) -> dict[str, Any]:
            async with semaphore:
                return await self.evaluate(conv, source)

        if source_contents is None:
            source_contents = [None] * len(conversations)

        tasks = [evaluate_one(conv, src) for conv, src in zip(conversations, source_contents, strict=False)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error dicts
        processed: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, BaseException):
                processed.append({"error": str(result)})
            else:
                processed.append(result)
        return processed

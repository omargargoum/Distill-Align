"""Dataset-level evaluation harness (Phase 4).

Combines cheap deterministic heuristics (structure, grounding proxy,
contamination) with optional LLM-judge aggregation. Designed for CI:
``evaluate_conversations`` returns a pass/fail gate; use a small smoke
subset per-PR and the full judge suite nightly to control judge cost.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..core.schemas import ConversationSchema
from .judge import ngram_overlap


class EvalThresholds(BaseModel):
    """Pass/fail thresholds (0-1 scale unless noted)."""

    min_mean_confidence: float = 0.5
    min_valid_rate: float = 0.8
    max_contamination_rate: float = 0.1
    min_judge_overall: float = 5.0  # 0-10 scale


class EvalReport(BaseModel):
    """Aggregate evaluation result."""

    total: int = 0
    valid: int = 0
    valid_rate: float = 0.0
    mean_confidence: float = 0.0
    mean_judge_overall: float | None = None
    contaminated: int = 0
    contamination_rate: float = 0.0
    failures: list[str] = []
    passed: bool = True

    def summary(self) -> str:
        lines = [
            "── Eval Report ──────────────────────",
            f"  Conversations:        {self.total}",
            f"  Valid:                {self.valid} ({self.valid_rate:.1%})",
            f"  Mean confidence:      {self.mean_confidence:.3f}",
        ]
        if self.mean_judge_overall is not None:
            lines.append(f"  Mean judge overall:   {self.mean_judge_overall:.2f}/10")
        lines.append(f"  Contaminated:         {self.contaminated} ({self.contamination_rate:.1%})")
        lines.append(f"  Gate:                 {'PASS' if self.passed else 'FAIL'}")
        if self.failures:
            lines.append("  Failures:")
            lines.extend(f"    - {f}" for f in self.failures)
        lines.append("─────────────────────────────────────")
        return "\n".join(lines)


def heuristic_valid(conv: ConversationSchema) -> bool:
    """Deterministic validity: has user+assistant turns, non-empty content."""
    users = [t for t in conv.turns if t.role == "user" and t.content.strip()]
    assistants = [t for t in conv.turns if t.role == "assistant" and t.content.strip()]
    return bool(users and assistants)


def evaluate_conversations(
    conversations: list[ConversationSchema],
    reference_texts: list[str] | None = None,
    thresholds: EvalThresholds | None = None,
    contamination_n: int = 8,
    contamination_threshold: float = 0.3,
) -> EvalReport:
    """Evaluate a dataset without any LLM calls (heuristics only).

    Args:
        conversations: Conversations to evaluate.
        reference_texts: Optional eval-set texts for contamination checks.
        thresholds: Pass/fail gate thresholds.
        contamination_n: n-gram size for the contamination proxy.
        contamination_threshold: Overlap fraction flagging contamination.
    """
    th = thresholds or EvalThresholds()
    total = len(conversations)
    if total == 0:
        return EvalReport(total=0, failures=["empty dataset"], passed=False)

    valid = sum(1 for c in conversations if heuristic_valid(c))
    confs = [c.confidence_score for c in conversations if c.confidence_score is not None]
    mean_conf = sum(confs) / len(confs) if confs else 0.0

    judge_vals = []
    for c in conversations:
        if c.judge_scores and isinstance(c.judge_scores.get("overall"), (int, float)):
            judge_vals.append(float(c.judge_scores["overall"]))
    mean_judge = sum(judge_vals) / len(judge_vals) if judge_vals else None

    contaminated = 0
    if reference_texts:
        for c in conversations:
            blob = " ".join(t.content for t in c.turns)
            if any(ngram_overlap(blob, ref, contamination_n) >= contamination_threshold for ref in reference_texts):
                contaminated += 1

    valid_rate = valid / total
    contam_rate = contaminated / total
    failures: list[str] = []
    if valid_rate < th.min_valid_rate:
        failures.append(f"valid_rate {valid_rate:.1%} < {th.min_valid_rate:.1%}")
    if mean_conf < th.min_mean_confidence:
        failures.append(f"mean_confidence {mean_conf:.3f} < {th.min_mean_confidence:.3f}")
    if mean_judge is not None and mean_judge < th.min_judge_overall:
        failures.append(f"mean_judge_overall {mean_judge:.2f} < {th.min_judge_overall:.2f}")
    if contam_rate > th.max_contamination_rate:
        failures.append(f"contamination_rate {contam_rate:.1%} > {th.max_contamination_rate:.1%}")

    return EvalReport(
        total=total,
        valid=valid,
        valid_rate=valid_rate,
        mean_confidence=mean_conf,
        mean_judge_overall=mean_judge,
        contaminated=contaminated,
        contamination_rate=contam_rate,
        failures=failures,
        passed=not failures,
    )

"""
Preference pair generator for DPO/RLHF training.

Generates chosen/rejected pairs from judge-scored conversations by pairing
conversations with high and low scores on the same source content.
"""

from collections import defaultdict
from typing import Any

from loguru import logger
from pydantic import BaseModel

from ..core.schemas import ConversationSchema


class PreferencePair(BaseModel):
    """A single DPO preference pair."""

    prompt: str  # The user / instruction turn
    chosen: str  # High-scoring assistant response
    rejected: str  # Low-scoring assistant response
    chosen_score: float
    rejected_score: float
    source_chunk_id: str | None = None
    metadata: dict[str, Any] = {}


class PreferenceGenerator:
    """Generates DPO preference pairs from judge-scored conversations.

    Strategy:
    1. Groups conversations by ``source_chunk_id``.
    2. Within each group, pairs the highest-scoring conversation with
       the lowest-scoring one as chosen/rejected.
    3. Optionally generates synthetic negative examples when only one
       conversation exists per source (by pairing across sources with
       similar scores, or skipping).
    """

    def __init__(
        self,
        min_score_gap: float = 0.1,
        require_judge_scores: bool = True,
    ):
        """
        Args:
            min_score_gap: Minimum difference in confidence_score for a
                valid preference pair (normalised 0-1).
            require_judge_scores: If True, skip conversations without
                ``judge_scores``. If False, fall back to ``confidence_score``.
        """
        self.min_score_gap = min_score_gap
        self.require_judge_scores = require_judge_scores

    def extract_prompt(self, conversation: ConversationSchema) -> str | None:
        """Extract the first user turn as the prompt."""
        for turn in conversation.turns:
            if turn.role == "user":
                return turn.content
        return None

    def extract_assistant_response(self, conversation: ConversationSchema) -> str | None:
        """Extract the first assistant turn as the response."""
        for turn in conversation.turns:
            if turn.role == "assistant":
                return turn.content
        return None

    def _get_score(self, conversation: ConversationSchema) -> float | None:
        """Get the best available score for a conversation.

        Prefers ``judge_scores.overall`` (normalised), then ``confidence_score``.
        """
        if conversation.judge_scores:
            overall = conversation.judge_scores.get("overall")
            if isinstance(overall, int | float):
                return max(0.0, min(1.0, overall / 10.0))
        if conversation.confidence_score is not None:
            return conversation.confidence_score
        return None

    def generate_pairs(
        self,
        conversations: list[ConversationSchema],
    ) -> list[PreferencePair]:
        """Generate preference pairs from scored conversations.

        Args:
            conversations: List of judge-scored conversations.

        Returns:
            List of ``PreferencePair`` objects.

        Raises:
            ExportError: If no valid pairs can be generated.
        """
        # Group by source_chunk_id
        groups: dict[str, list[ConversationSchema]] = defaultdict(list)
        for conv in conversations:
            groups[conv.source_chunk_id].append(conv)

        pairs: list[PreferencePair] = []

        for source_id, group in groups.items():
            if len(group) < 2:
                continue  # Need at least 2 to compare

            # Score each conversation
            scored: list[tuple[float, ConversationSchema]] = []
            for conv in group:
                score = self._get_score(conv)
                if score is not None:
                    scored.append((score, conv))

            if len(scored) < 2:
                continue

            # Sort by score descending
            scored.sort(key=lambda x: x[0], reverse=True)

            # Pair highest with lowest
            best_score, best_conv = scored[0]
            worst_score, worst_conv = scored[-1]

            if best_score - worst_score < self.min_score_gap:
                logger.debug(
                    f"Score gap {best_score - worst_score:.3f} < {self.min_score_gap} for source {source_id}, skipping"
                )
                continue

            prompt = self.extract_prompt(best_conv) or self.extract_prompt(worst_conv)
            if not prompt:
                continue

            chosen = self.extract_assistant_response(best_conv)
            rejected = self.extract_assistant_response(worst_conv)
            if not chosen or not rejected:
                continue

            pairs.append(
                PreferencePair(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    chosen_score=best_score,
                    rejected_score=worst_score,
                    source_chunk_id=source_id,
                    metadata={
                        "best_conv_id": best_conv.id,
                        "worst_conv_id": worst_conv.id,
                    },
                )
            )

        # Cross-source pairing for singletons
        singles: list[tuple[float, ConversationSchema]] = [
            (s, c)
            for s, c in (
                (self._get_score(conv), conv) for conv in conversations if len(groups.get(conv.source_chunk_id, [])) < 2
            )
            if s is not None
        ]
        singles.sort(key=lambda x: x[0], reverse=True)

        # Pair highest-scored singleton with lowest-scored singleton
        while len(singles) >= 2:
            best_score, best_conv = singles.pop(0)
            worst_score, worst_conv = singles.pop(-1)

            if best_score - worst_score < self.min_score_gap:
                continue

            prompt = self.extract_prompt(best_conv) or self.extract_prompt(worst_conv)
            if not prompt:
                continue

            chosen = self.extract_assistant_response(best_conv)
            rejected = self.extract_assistant_response(worst_conv)
            if not chosen or not rejected:
                continue

            pairs.append(
                PreferencePair(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    chosen_score=best_score,
                    rejected_score=worst_score,
                    source_chunk_id=None,
                    metadata={
                        "best_conv_id": best_conv.id,
                        "worst_conv_id": worst_conv.id,
                        "cross_source": True,
                    },
                )
            )

        logger.info(f"Generated {len(pairs)} preference pairs from {len(conversations)} conversations")
        if not pairs:
            logger.warning(
                "No preference pairs generated. Ensure conversations have "
                "judge_scores or confidence_score set, and there are at least "
                "2 conversations with a score gap > min_score_gap."
            )

        return pairs

    def to_dpo_format(
        self,
        pairs: list[PreferencePair],
    ) -> list[dict[str, Any]]:
        """Convert preference pairs to standard DPO JSON format.

        Each entry::

            {
                "prompt": "...",
                "chosen": "...",
                "rejected": "...",
                "score_chosen": 0.95,
                "score_rejected": 0.3,
            }
        """
        return [
            {
                "prompt": p.prompt,
                "chosen": p.chosen,
                "rejected": p.rejected,
                "score_chosen": round(p.chosen_score, 4),
                "score_rejected": round(p.rejected_score, 4),
            }
            for p in pairs
        ]

    def to_kto_format(self, pairs: list[PreferencePair]) -> list[dict[str, Any]]:
        """Expand pairs into unpaired KTO rows (one per completion + label)."""
        rows: list[dict[str, Any]] = []
        for p in pairs:
            rows.append({"prompt": p.prompt, "completion": p.chosen, "label": True})
            rows.append({"prompt": p.prompt, "completion": p.rejected, "label": False})
        return rows

    def to_orpo_format(self, pairs: list[PreferencePair]) -> list[dict[str, Any]]:
        """ORPO rows: single-stage SFT+preference (same as DPO triples)."""
        return [{"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected} for p in pairs]

    def to_grpo_format(self, pairs: list[PreferencePair]) -> list[dict[str, Any]]:
        """GRPO groups: prompt + completions + scalar rewards."""
        return [
            {
                "prompt": p.prompt,
                "completions": [p.chosen, p.rejected],
                "rewards": [round(p.chosen_score, 4), round(p.rejected_score, 4)],
            }
            for p in pairs
        ]

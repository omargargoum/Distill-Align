"""GRPO / RLVR formatter (Phase 5): verifiable-reward rows.

GRPO (Group Relative Policy Optimization, DeepSeek-R1 recipe) samples K
completions per prompt and uses within-group rewards — no reward model.
This formatter emits the dataset side::

    {"prompt": "...", "completions": ["...", "..."], "rewards": [1.0, 0.0]}

Rewards here come from judge-score gaps (weak supervision). Teams with
verifiable checkers (math answer-match, code unit tests) should replace
rewards with checker outputs before training with TRL GRPOTrainer.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from ...core.exceptions import FormatError
from ...core.schemas import ConversationSchema
from .base import BaseFormatter


def _score(conv: ConversationSchema) -> float | None:
    if conv.judge_scores and isinstance(conv.judge_scores.get("overall"), (int, float)):
        return max(0.0, min(1.0, float(conv.judge_scores["overall"]) / 10.0))
    return conv.confidence_score


class GRPOFormatter(BaseFormatter):
    """Formatter for GRPO/RLVR group rows (grouped by source_chunk_id)."""

    def format(
        self,
        conversations: list[ConversationSchema],
        filename: str = "dataset_grpo.json",
    ) -> Path:
        filename = self._ensure_json_extension(filename)
        output_path = self.output_dir / filename
        try:
            groups: dict[str, list[ConversationSchema]] = defaultdict(list)
            for c in conversations:
                groups[c.source_chunk_id or c.id].append(c)
            rows: list[dict[str, Any]] = []
            for _, group in groups.items():
                prompts = [t.content for t in group[0].turns if t.role == "user"]
                if not prompts:
                    continue
                prompt = prompts[0]
                completions, rewards = [], []
                for c in group:
                    assistants = [t.content for t in c.turns if t.role == "assistant"]
                    if not assistants:
                        continue
                    s = _score(c)
                    completions.append(assistants[0])
                    rewards.append(round(s if s is not None else 0.5, 4))
                if len(completions) >= 2:
                    rows.append({"prompt": prompt, "completions": completions, "rewards": rewards})
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported {len(rows)} GRPO groups to {output_path}")
            return output_path
        except Exception as e:
            raise FormatError(f"Failed to format GRPO data: {e}") from e

    def validate(self, data: list[dict]) -> bool:
        if not isinstance(data, list):
            return False
        for r in data:
            if not isinstance(r, dict) or "prompt" not in r or "completions" not in r:
                return False
            if not isinstance(r["completions"], list) or len(r["completions"]) < 2:
                return False
        return True

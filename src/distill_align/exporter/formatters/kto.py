"""KTO formatter (Phase 5): unpaired binary-label rows.

KTO (Kahneman-Tversky Optimization) trains on single completions with a
boolean label instead of chosen/rejected pairs — ideal for thumbs-up/down
production logs. Format::

    {"prompt": "...", "completion": "...", "label": true}
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from ...core.exceptions import FormatError
from ...core.schemas import ConversationSchema
from .base import BaseFormatter


class KTOFormatter(BaseFormatter):
    """Formatter for KTO unpaired preference rows."""

    def format(
        self,
        conversations: list[ConversationSchema],
        filename: str = "dataset_kto.json",
    ) -> Path:
        filename = self._ensure_json_extension(filename)
        output_path = self.output_dir / filename
        try:
            rows: list[dict[str, Any]] = []
            for conv in conversations:
                users = [t for t in conv.turns if t.role == "user"]
                assistants = [t for t in conv.turns if t.role == "assistant"]
                if not users or not assistants:
                    continue
                prompt = users[0].content
                # Label from judge overall (>=7 → desirable) or confidence (>=0.7)
                label: bool | None = None
                if conv.judge_scores and isinstance(conv.judge_scores.get("overall"), (int, float)):
                    label = float(conv.judge_scores["overall"]) >= 7.0
                elif conv.confidence_score is not None:
                    label = float(conv.confidence_score) >= 0.7
                else:
                    label = True  # unlabeled SFT rows default desirable
                system = conv.get_system_prompt()
                for a in assistants:
                    rows.append(
                        {
                            "prompt": prompt,
                            "completion": a.content,
                            "label": label,
                            **({"system": system} if system else {}),
                        }
                    )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported {len(rows)} KTO rows to {output_path}")
            return output_path
        except Exception as e:
            raise FormatError(f"Failed to format KTO data: {e}") from e

    def validate(self, data: list[dict]) -> bool:
        if not isinstance(data, list):
            return False
        return all(isinstance(r, dict) and "prompt" in r and "completion" in r and "label" in r for r in data)

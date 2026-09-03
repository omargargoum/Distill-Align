"""Agent-trajectory + RAG-QA formatters (Phase 5).

agent: ordered tool-use traces for agent fine-tuning::
    {"messages": [...], "tools": [{"name": ..., "arguments": {...}}]}

rag_qa: grounded retrieval rows doubling as a retrieval eval set::
    {"query": ..., "contexts": [...], "answer": ..., "answerable": bool}
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from ...core.exceptions import FormatError
from ...core.schemas import ConversationSchema
from .base import BaseFormatter


class AgentFormatter(BaseFormatter):
    """Formatter for agent / tool-call trajectories."""

    def format(self, conversations: list[ConversationSchema], filename: str = "dataset_agent.json") -> Path:
        filename = self._ensure_json_extension(filename)
        output_path = self.output_dir / filename
        try:
            rows: list[dict[str, Any]] = []
            for conv in conversations:
                messages = [{"role": t.role, "content": t.content} for t in conv.turns]
                rows.append(
                    {
                        "id": conv.id,
                        "messages": messages,
                        "tools": [],
                        "source_chunk_id": conv.source_chunk_id,
                        "reasoning_trace": conv.reasoning_trace,
                    }
                )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported {len(rows)} agent trajectories to {output_path}")
            return output_path
        except Exception as e:
            raise FormatError(f"Failed to format agent data: {e}") from e

    def validate(self, data: list[dict]) -> bool:
        if not isinstance(data, list):
            return False
        return all(isinstance(r, dict) and "messages" in r for r in data)


class RagQaFormatter(BaseFormatter):
    """Formatter for synthetic RAG-QA rows (query + contexts + answer)."""

    def format(self, conversations: list[ConversationSchema], filename: str = "dataset_rag_qa.json") -> Path:
        filename = self._ensure_json_extension(filename)
        output_path = self.output_dir / filename
        try:
            rows: list[dict[str, Any]] = []
            for conv in conversations:
                users = [t.content for t in conv.turns if t.role == "user"]
                assistants = [t.content for t in conv.turns if t.role == "assistant"]
                if not users or not assistants:
                    continue
                rows.append(
                    {
                        "query": users[0],
                        "contexts": [],
                        "answer": assistants[0],
                        "answerable": True,
                        "source_chunk_id": conv.source_chunk_id,
                    }
                )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported {len(rows)} RAG-QA rows to {output_path}")
            return output_path
        except Exception as e:
            raise FormatError(f"Failed to format RAG-QA data: {e}") from e

    def validate(self, data: list[dict]) -> bool:
        if not isinstance(data, list):
            return False
        return all(isinstance(r, dict) and "query" in r and "answer" in r for r in data)

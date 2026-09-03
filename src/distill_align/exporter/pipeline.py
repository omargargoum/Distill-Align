"""
Export pipeline orchestrator.

Handles the full export workflow: validation, splitting, formatting, streaming,
and Unsloth config.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from loguru import logger

from ..core.exceptions import ExportError
from ..core.schemas import ConversationSchema, ExportConfig
from .dataset_card import DatasetCardGenerator
from .formatters.agent_rag import AgentFormatter, RagQaFormatter
from .formatters.alpaca import AlpacaFormatter
from .formatters.base import BaseFormatter
from .formatters.chatml import ChatMLFormatter
from .formatters.conversation import ConversationFormatter
from .formatters.grpo import GRPOFormatter
from .formatters.hf_messages import HFMessagesFormatter
from .formatters.jsonl import JsonlFormatter
from .formatters.kto import KTOFormatter
from .formatters.parquet import ParquetFormatter
from .formatters.preference import PreferenceFormatter
from .formatters.sharegpt import ShareGPTFormatter
from .splitter import DatasetSplitter
from .unsloth_builder import UnslothConfigBuilder
from .validator import DatasetValidator, ValidationReport

# Map format names to formatter classes
FORMATTER_MAP: dict[str, type[BaseFormatter]] = {
    "sharegpt": ShareGPTFormatter,
    "alpaca": AlpacaFormatter,
    "chatml": ChatMLFormatter,
    "conversation": ConversationFormatter,
    "hf_messages": HFMessagesFormatter,
    "jsonl": JsonlFormatter,
    "parquet": ParquetFormatter,
    "preference": PreferenceFormatter,
    "dpo": PreferenceFormatter,
    "orpo": PreferenceFormatter,
    "kto": KTOFormatter,
    "grpo": GRPOFormatter,
    "agent": AgentFormatter,
    "rag_qa": RagQaFormatter,
}


class ExportPipeline:
    """Orchestrates the export of conversations to various formats."""

    def __init__(self, config: ExportConfig | None = None):
        """
        Initialize the export pipeline.

        Args:
            config: Optional export configuration. Uses defaults if not provided.
        """
        self.config = config or ExportConfig()
        self._formatters: dict[str, BaseFormatter] = {}
        self._unsloth_builder: UnslothConfigBuilder | None = None
        self._validator = DatasetValidator()
        self._splitter = DatasetSplitter()
        self._card_generator = DatasetCardGenerator()

    def _get_formatter(self, format_name: str) -> BaseFormatter:
        """
        Get or create a formatter for the specified format.

        Args:
            format_name: Name of the format (e.g., "sharegpt", "alpaca", "chatml").

        Returns:
            Formatter instance.

        Raises:
            ExportError: If format is not supported.
        """
        if format_name not in FORMATTER_MAP:
            raise ExportError(f"Unsupported format: {format_name}. Supported: {', '.join(FORMATTER_MAP.keys())}")

        if format_name not in self._formatters:
            if format_name in ("dpo", "orpo"):
                self._formatters[format_name] = PreferenceFormatter(self.config.output_dir, format_type=format_name)
            else:
                self._formatters[format_name] = FORMATTER_MAP[format_name](self.config.output_dir)
        return self._formatters[format_name]

    def _get_unsloth_builder(self) -> UnslothConfigBuilder:
        """Get or create the Unsloth config builder."""
        if self._unsloth_builder is None:
            self._unsloth_builder = UnslothConfigBuilder(self.config)
        return self._unsloth_builder

    def validate(
        self,
        conversations: list[ConversationSchema],
        dedupe: bool = True,
    ) -> tuple[list[ConversationSchema], ValidationReport]:
        """
        Validate and optionally deduplicate conversations.

        Args:
            conversations: List of conversations.
            dedupe: Whether to remove duplicates.

        Returns:
            Tuple of (cleaned conversations, validation report).
        """
        if dedupe:
            conversations = self._validator.deduplicate(conversations)

        report = self._validator.validate(conversations)
        logger.info(f"Validation: {len(conversations)} conversations, score={report.quality_score:.2f}")
        return conversations, report

    def export(
        self,
        conversations: list[ConversationSchema],
        formats: list[str] | None = None,
        dataset_filename: str = "dataset",
        generate_unsloth: bool = True,
        split: bool = False,
        generate_card: bool = False,
        **unsloth_kwargs,
    ) -> dict[str, Path]:
        """
        Export conversations to specified formats.

        Args:
            conversations: List of conversations.
            formats: List of format names (defaults to config).
            dataset_filename: Base filename for datasets.
            generate_unsloth: Whether to generate Unsloth script.
            split: Whether to split into train/val/test.
            generate_card: Whether to generate dataset card.
            **unsloth_kwargs: Additional Unsloth config.

        Returns:
            Dictionary mapping format names to output file paths.
        """
        export_formats = cast("list[str]", self.config.formats if formats is None else formats)

        # Validate first
        conversations, validation_report = self.validate(conversations)

        # Optional split
        if split:
            split_result = self._splitter.split(
                conversations,
                train_ratio=0.9,
                val_ratio=0.05,
                test_ratio=0.05,
            )
            split_paths = self._splitter.save_split(split_result, self.config.output_dir, dataset_filename)
            # Use train split as the primary dataset for training
            conversations = split_result.train
        else:
            split_paths = {}

        output_files = {}

        # Export to each format
        for format_name in export_formats:
            try:
                formatter = self._get_formatter(format_name)
                filename = f"{dataset_filename}_{format_name}.json"
                output_path = formatter.format(conversations, filename)
                output_files[format_name] = output_path
                logger.info(f"Exported to {format_name}: {output_path}")
            except Exception as e:
                logger.error(f"Failed to export to {format_name}: {e}")
                raise ExportError(f"Export to {format_name} failed: {e}") from e

        # Add split files to output
        output_files.update(split_paths)

        # Generate Unsloth script
        if generate_unsloth and self.config.generate_unsloth_script:
            try:
                builder = self._get_unsloth_builder()
                dataset_path = str(list(output_files.values())[0])
                script_path = builder.generate_script(
                    dataset_path=dataset_path,
                    output_dir=str(Path(self.config.output_dir) / "model"),
                    **unsloth_kwargs,
                )
                output_files["unsloth_script"] = script_path
                logger.info(f"Generated Unsloth script: {script_path}")
            except Exception as e:
                logger.warning(f"Failed to generate Unsloth script: {e}")

            # Preference / RL trainers matching the exported formats
            for pref in ("dpo", "orpo", "kto"):
                if pref in output_files:
                    try:
                        p = builder.generate_preference_script(
                            dataset_path=str(output_files[pref]),
                            output_dir=str(Path(self.config.output_dir) / "model"),
                            method=pref,
                        )
                        output_files[f"unsloth_{pref}_script"] = p
                    except Exception as e:
                        logger.warning(f"Failed to generate {pref.upper()} script: {e}")
            if "grpo" in output_files:
                try:
                    g = builder.generate_grpo_script(
                        dataset_path=str(output_files["grpo"]),
                        output_dir=str(Path(self.config.output_dir) / "model"),
                    )
                    output_files["unsloth_grpo_script"] = g
                except Exception as e:
                    logger.warning(f"Failed to generate GRPO script: {e}")

        # Generate dataset card
        if generate_card:
            try:
                card_path = Path(self.config.output_dir) / f"{dataset_filename}_README.md"
                self._card_generator.generate(
                    conversations=conversations,
                    validation_report=validation_report,
                    config=unsloth_kwargs.get("synthesis_config", {}),
                    output_path=card_path,
                )
                output_files["dataset_card"] = card_path
            except Exception as e:
                logger.warning(f"Failed to generate dataset card: {e}")

        return output_files

    def export_stream(
        self,
        conversations: Iterable[ConversationSchema],
        formats: list[str] | None = None,
        dataset_filename: str = "dataset",
    ) -> dict[str, Path]:
        """Export conversations in a streaming fashion.

        Unlike :meth:`export`, this method processes conversations from an
        iterable without materialising the full list in memory. This is
        useful for large datasets or when conversations are produced by a
        live synthesis pipeline.

        Only formatters that support streaming (``jsonl``, ``parquet``,
        ``hf_messages`` via JSONL mode) benefit from this. Other formatters
        will buffer the entire iterable internally.

        When a **single** format is requested, the iterable is passed
        directly to the formatter without buffering. When **multiple**
        formats are requested, the iterable is materialised into a list
        so it can be fanned out to each formatter (``itertools.tee`` was
        previously used but it also materialises the full iterable in
        practice when iterators are consumed sequentially).

        Args:
            conversations: Iterable of conversations (streaming source).
            formats: List of format names (defaults to config).
            dataset_filename: Base filename for datasets.

        Returns:
            Dictionary mapping format names to output file paths.
        """
        export_formats = cast("list[str]", self.config.formats if formats is None else formats)

        def _ext(format_name: str) -> str:
            if format_name == "parquet":
                return ".parquet"
            if format_name in ("jsonl", "hf_messages"):
                return ".jsonl"
            return ".json"

        # Single format — stream directly, no buffering
        if len(export_formats) == 1:
            format_name = export_formats[0]
            formatter = self._get_formatter(format_name)
            filename = f"{dataset_filename}_{format_name}{_ext(format_name)}"
            output_files: dict[str, Path] = {}
            try:
                output_files[format_name] = formatter.format_stream(conversations, filename)
                logger.info(f"Streamed export to {format_name}: {output_files[format_name]}")
            except Exception as e:
                logger.error(f"Failed to stream export to {format_name}: {e}")
                raise ExportError(f"Streaming export to {format_name} failed: {e}") from e
            return output_files

        # Multiple formats — materialise the iterable and fan out.
        # (True multi-format streaming without materialisation would
        # require writing to all output files in a single pass, which
        # the current formatter API does not support.)
        conversations_list = list(conversations)
        output_files = {}

        for format_name in export_formats:
            try:
                formatter = self._get_formatter(format_name)
                filename = f"{dataset_filename}_{format_name}{_ext(format_name)}"
                output_files[format_name] = formatter.format(conversations_list, filename)
                logger.info(
                    f"Exported {len(conversations_list)} conversations to {format_name}: {output_files[format_name]}"
                )
            except Exception as e:
                logger.error(f"Failed to export to {format_name}: {e}")
                raise ExportError(f"Export to {format_name} failed: {e}") from e

        return output_files

    def validate_export(self, output_files: dict[str, Path]) -> dict[str, bool]:
        """Validate exported files."""
        results = {}

        for format_name, file_path in output_files.items():
            if format_name in {"unsloth_script", "dataset_card"} or "_" not in format_name:
                # Validate Python syntax or just check existence
                if file_path.suffix == ".py":
                    try:
                        with open(file_path) as f:
                            compile(f.read(), file_path, "exec")
                        results[format_name] = True
                    except SyntaxError:
                        results[format_name] = False
                else:
                    results[format_name] = file_path.exists()
            elif format_name in FORMATTER_MAP:
                try:
                    formatter = self._get_formatter(format_name)
                    data = formatter.load(file_path)  # type: ignore[attr-defined]
                    results[format_name] = formatter.validate(data)
                except Exception:
                    results[format_name] = False
            else:
                results[format_name] = file_path.exists()

        return results

    def get_export_stats(self, output_files: dict[str, Path]) -> dict[str, Any]:
        """Get statistics about exported files."""
        stats = {}

        for format_name, file_path in output_files.items():
            if file_path.exists():
                stats[format_name] = {
                    "path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                }

                # Count entries for JSON files
                if format_name in FORMATTER_MAP:
                    try:
                        import json

                        with open(file_path) as f:
                            data = json.load(f)
                        stats[format_name]["entries"] = len(data)
                    except Exception:
                        pass

        return stats

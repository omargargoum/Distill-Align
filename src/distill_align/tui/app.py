"""
Textual TUI application for Distill-Align.

Provides an interactive dashboard for monitoring and executing pipeline operations.
Features real-time stats, job management, cache inspection, log viewing,
and full pipeline execution (ingest, synthesize, export, validate).

Now with:
- Full Pipeline tab that chains ingest → synthesize → export with one click
- Live Dashboard progress wiring from any pipeline tab
- Input validation on all forms
- Export results table with file sizes
- Auto-refresh of Jobs tab during active synthesis
- Last-run summary on Dashboard
- 🌱 Simple / 🧪 Expert mode toggle (press [m])
- Quick-Start Wizard for first-time users
- Keyboard shortcut help overlay (press [?])
- Built-in pipeline templates for common tasks
"""

import asyncio
import contextlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from textual.app import App, ComposeResult  # type: ignore[import-not-found]
from textual.binding import Binding  # type: ignore[import-not-found]
from textual.containers import Container, Horizontal, Vertical  # type: ignore[import-not-found]
from textual.widgets import (  # type: ignore[import-not-found]
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from .. import __version__
from ..core.cache import CacheManager
from ..core.checkpoint import CheckpointManager
from ..core.config_file import find_config_file, generate_default_config, load_config
from ..core.json_utils import safe_json_load
from ..core.update_checker import check_pypi_version
from ..synthesis.models.registry import get as get_provider_info
from ..synthesis.models.registry import list_select_choices

# Simple/Expert mode system
from .modes.base import (
    BUILTIN_TEMPLATES,
    MODE_EXPERT,
    MODE_LABELS,
    MODE_SIMPLE,
    get_template,
)
from .options import (
    chunker_choices,
    conversation_mode_choices,
    export_format_choices,
    validate_chunker,
)
from .widgets.help_overlay import HelpOverlay
from .widgets.preflight import (
    PreflightScreen,
    preflight_export,
    preflight_full_pipeline,
    preflight_ingest,
    preflight_synthesize,
)
from .widgets.wizard import QuickStartWizard

# =============================================================================
# Loguru → TUI Sink (live log forwarding)
# =============================================================================


class TUILogSink:
    """Loguru sink that forwards log messages to the TUI RichLog widget."""

    def __init__(self, app: "DistillAlignApp") -> None:
        self._app = app

    def write(self, message: Any) -> None:
        """Write a loguru message to the TUI log view."""
        try:
            record = message.record if hasattr(message, "record") else {}
            level = record.get("level", {})
            level_name = level.name if hasattr(level, "name") else "INFO"
            text = str(message).strip()
            if not text:
                return

            color_map = {
                "DEBUG": "dim",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold red",
            }
            color = color_map.get(level_name, "white")
            timestamp = record.get("time", "")
            if hasattr(timestamp, "strftime"):
                timestamp = timestamp.strftime("%H:%M:%S")

            formatted = f"[dim]{timestamp}[/dim] [{color}]{level_name}[/{color}] {text}"
            self._app.call_from_thread(self._write_to_log, formatted)
        except Exception:
            pass

    def _write_to_log(self, text: str) -> None:
        """Thread-safe write to the RichLog widget."""
        try:
            log_view = self._app.query_one("#log-view", RichLog)
            log_view.write(text)
        except Exception:
            pass


def _install_loguru_sink(app: "DistillAlignApp") -> None:
    """Install a loguru sink that forwards to the TUI."""
    sink = TUILogSink(app)
    logger.add(sink, format="{message}", level="DEBUG", colorize=False)


# =============================================================================
# Validation Helpers
# =============================================================================


def _validate_positive_int(value: str, name: str, default: int) -> int:
    """Validate and return a positive integer from a string input."""
    if not value.strip():
        return default
    try:
        v = int(value.strip())
        if v < 1:
            raise ValueError(f"{name} must be at least 1, got {v}")
        return v
    except ValueError as e:
        raise ValueError(f"Invalid {name}: {e}") from e


def _validate_path_exists(value: str, name: str) -> Path:
    """Validate a file path exists."""
    if not value.strip():
        raise ValueError(f"{name} cannot be empty")
    p = Path(value.strip())
    if not p.exists():
        raise ValueError(f"{name} not found: {value.strip()}")
    return p


# =============================================================================
# Dashboard Tab
# =============================================================================


class DashboardTab(Container):
    """Main dashboard showing real-time stats, progress, and last run info."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("📊 Pipeline Dashboard", classes="tab-title")

            # ── Quick-Start / Templates (simple mode only) ──
            with Vertical(classes="simple-only", id="quick-start-area"):
                yield Static(
                    "[bold]👋 Welcome to Distill-Align![/bold]\n"
                    "Turn your documents into high-quality fine-tuning datasets. "
                    "Select an option below to get started quickly.",
                    classes="simple-help",
                )
                with Horizontal():
                    yield Button("✨ Quick-Start Wizard", id="btn-wizard", variant="primary", classes="action-button")
                    yield Button("📋 Load Template", id="btn-template", variant="default", classes="action-button")

            # ── Stats row ──
            with Horizontal(id="stats-row"):
                yield Static(self._stat_box("📦 Chunks", "0", "green"), id="stat-chunks", classes="stat-box")
                yield Static(self._stat_box("💬 Conversations", "0", "blue"), id="stat-conv", classes="stat-box")
                yield Static(self._stat_box("💰 Cost", "$0.00", "yellow"), id="stat-cost", classes="stat-box")
                yield Static(self._stat_box("🎯 Quality", "100%", "magenta"), id="stat-quality", classes="stat-box")

            yield ProgressBar(total=100, id="dashboard-progress", show_eta=True)
            yield Label("Stage: Idle", id="stage-label")
            yield Label("", id="dashboard-last-run")

            # ── Tips (simple mode only) ──
            with Vertical(classes="simple-only"):
                yield Static(
                    "[dim]💡 [/dim][bold]Tip:[/bold] Press [reverse] m [/reverse] to switch to [yellow]Expert mode[/yellow] "
                    "for full control. Press [reverse] ? [/reverse] for keyboard shortcuts.",
                    classes="tip-box",
                )

            with Horizontal(id="system-info-row"):
                yield Static("Loading...", id="system-info", classes="stat-box-wide")

            with Horizontal(id="quick-stats"):
                yield DataTable(id="cache-table")

    @staticmethod
    def _stat_box(label: str, value: str, color: str) -> str:
        return f"[bold {color}]{label}[/bold {color}]\n[white]{value}[/white]"

    def on_mount(self) -> None:
        table = self.query_one("#cache-table", DataTable)
        table.add_columns("Cache Metric", "Value")
        self.refresh_cache_stats()
        self.refresh_system_info()

    def refresh_cache_stats(self) -> None:
        """Refresh cache statistics in the dashboard."""
        try:
            cache = CacheManager(cache_dir=".cache")
            stats = cache.stats()
            table = self.query_one("#cache-table", DataTable)
            table.clear()
            table.add_rows(
                [
                    ("Total Entries", str(stats.total_entries)),
                    ("Hits", str(stats.hit_count)),
                    ("Misses", str(stats.miss_count)),
                    ("Hit Rate", f"{stats.hit_rate:.1%}"),
                    ("DB Size", f"{stats.db_size_mb} MB"),
                ]
            )
        except Exception as e:
            logger.error(f"Failed to refresh cache stats: {e}")

    def refresh_system_info(self) -> None:
        """Show Python version, config, API key status."""
        try:
            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            config_path = find_config_file()
            config_status = f"[green]Found[/green] ({config_path})" if config_path else "[yellow]Not found[/yellow]"
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DISTILL_LLM_API_KEY")
            api_status = "[green]Set[/green]" if api_key else "[yellow]Not set[/yellow]"
            widget = self.query_one("#system-info", Static)
            widget.update(
                f"[bold]System Info[/bold]\nPython {py_ver}  |  Config: {config_status}  |  API Key: {api_status}"
            )
        except Exception as e:
            logger.error(f"Failed to refresh system info: {e}")

    def refresh_file_stats(self) -> None:
        """Scan working directory for chunks.json and conversations.json."""
        try:
            chunks_count = 0
            conv_count = 0

            chunks_path = Path("./chunks.json")
            if chunks_path.exists():
                try:
                    data = safe_json_load(chunks_path)
                    chunks_count = len(data)
                except Exception:
                    pass

            conv_path = Path("./conversations.json")
            if conv_path.exists():
                try:
                    data = safe_json_load(conv_path)
                    conv_count = len(data)
                except Exception:
                    pass

            self.query_one("#stat-chunks", Static).update(self._stat_box("📦 Chunks", str(chunks_count), "green"))
            self.query_one("#stat-conv", Static).update(self._stat_box("💬 Conversations", str(conv_count), "blue"))
        except Exception as e:
            logger.error(f"Failed to refresh file stats: {e}")

    def update_stage(self, stage: str) -> None:
        """Update the pipeline stage label."""
        with contextlib.suppress(Exception):
            self.query_one("#stage-label", Label).update(f"Stage: {stage}")

    def update_progress(self, current: int, total: int) -> None:
        """Update the progress bar."""
        try:
            bar = self.query_one("#dashboard-progress", ProgressBar)
            bar.update(total=total, progress=current)
        except Exception:
            pass

    def show_last_run(self, text: str) -> None:
        """Show last pipeline run summary."""
        with contextlib.suppress(Exception):
            self.query_one("#dashboard-last-run", Label).update(text)


# =============================================================================
# Ingest Tab
# =============================================================================


class IngestTab(Container):
    """Tab for running the ingestion pipeline."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("📥 Ingestion Pipeline", classes="tab-title")

            # Simple-mode guidance
            yield Static(
                "Specify your source files and how to split them into chunks for processing.",
                classes="simple-only simple-help",
            )

            with Horizontal(classes="form-row"):
                yield Label("Source path:", classes="form-label")
                yield Input(placeholder="./data", id="ingest-source", classes="form-input")

            with Horizontal(classes="form-row"):
                yield Label("Output file:", classes="form-label")
                yield Input(value="./chunks.json", id="ingest-output", classes="form-input")

            with Horizontal(classes="form-row"):
                yield Label("Chunk size:", classes="form-label")
                yield Input(value="1000", id="ingest-chunk-size", classes="form-input-small")
                # Overlap is expert-only
                yield Label("Overlap:", classes="form-label expert-only")
                yield Input(value="200", id="ingest-overlap", classes="form-input-small expert-only")

            # Expert-only advanced options
            with Horizontal(classes="form-row expert-only"):
                yield Label("Recursive:", classes="form-label")
                yield Switch(value=True, id="ingest-recursive")
                yield Label("Auto-detect:", classes="form-label")
                yield Switch(value=True, id="ingest-auto")

            with Horizontal(classes="form-row expert-only"):
                yield Label("Chunker:", classes="form-label")
                yield Select(
                    chunker_choices(),
                    value="auto",
                    id="ingest-chunker",
                    classes="form-select",
                )

            # Simple mode hint about advanced options
            yield Static(
                "[dim]💡 Press [reverse] m [/reverse] for expert options (overlap, recursive scanning, etc.)[/dim]",
                classes="simple-only",
            )

            yield Button("🚀 Run Ingestion", id="ingest-run", variant="primary")
            yield ProgressBar(total=100, id="ingest-progress", show_eta=False)
            yield Label("Ready", id="ingest-status")

    def get_config(self) -> dict[str, Any]:
        """Get current form values."""
        return {
            "source": self.query_one("#ingest-source", Input).value.strip(),
            "output": self.query_one("#ingest-output", Input).value.strip() or "./chunks.json",
            "chunk_size": int(self.query_one("#ingest-chunk-size", Input).value or "1000"),
            "overlap": int(self.query_one("#ingest-overlap", Input).value or "200"),
            "recursive": self.query_one("#ingest-recursive", Switch).value,
            "auto_detect": self.query_one("#ingest-auto", Switch).value,
            "chunker": self.query_one("#ingest-chunker", Select).value or "auto",
        }

    def validate_config(self) -> str | None:
        """Validate form values. Returns an error message or None."""
        cfg = self.get_config()
        if not cfg["source"]:
            return "Source path is required"
        if not Path(cfg["source"]).exists():
            return f"Source not found: {cfg['source']}"
        if not validate_chunker(cfg["chunker"]):
            return f"Unknown chunker: {cfg['chunker']}"
        try:
            _validate_positive_int(str(cfg["chunk_size"]), "chunk size", 1000)
            _validate_positive_int(str(cfg["overlap"]), "overlap", 200)
        except ValueError as e:
            return str(e)
        return None

    def set_status(self, text: str) -> None:
        self.query_one("#ingest-status", Label).update(text)

    def set_running(self, running: bool) -> None:
        btn = self.query_one("#ingest-run", Button)
        btn.disabled = running
        if running:
            self.query_one("#ingest-status", Label).update("[yellow]Running...[/yellow]")


# =============================================================================
# Synthesize Tab
# =============================================================================


class SynthesizeTab(Container):
    """Tab for running the synthesis pipeline."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("🧠 Synthesis Pipeline", classes="tab-title")

            # Simple-mode guidance
            yield Static(
                "Generate structured conversations from your text chunks using an LLM.",
                classes="simple-only simple-help",
            )

            with Horizontal(classes="form-row"):
                yield Label("Input chunks:", classes="form-label")
                yield Input(value="./chunks.json", id="synth-input", classes="form-input")

            with Horizontal(classes="form-row"):
                yield Label("Output file:", classes="form-label")
                yield Input(value="./conversations.json", id="synth-output", classes="form-input")

            with Horizontal(classes="form-row"):
                yield Label("Provider:", classes="form-label")
                yield Select(
                    list_select_choices(),
                    value="openai",
                    id="synth-provider",
                    classes="form-select",
                )
                yield Label("Model:", classes="form-label")
                yield Input(value="gpt-5-mini", id="synth-model", classes="form-input")

            # Simple mode: provider/model tip
            yield Static(
                "[dim]💡 Use [bold]gpt-5-mini[/bold] for balanced quality or [bold]ollama[/bold] with [bold]qwen3:30b[/bold] for free local generation.[/dim]",
                classes="simple-only",
            )

            # Expert-only advanced options
            with Vertical(classes="expert-only"):
                with Horizontal(classes="form-row"):
                    yield Label("Base URL:", classes="form-label")
                    yield Input(placeholder="(optional)", id="synth-base-url", classes="form-input")

                with Horizontal(classes="form-row"):
                    yield Label("Concurrency:", classes="form-label")
                    yield Input(value="5", id="synth-concurrency", classes="form-input-small")
                    yield Label("RPM:", classes="form-label")
                    yield Input(value="60", id="synth-rpm", classes="form-input-small")
                    yield Label("Max tokens:", classes="form-label")
                    yield Input(placeholder="(auto)", id="synth-max-tokens", classes="form-input-small")

                with Horizontal(classes="form-row"):
                    yield Label("Mode:", classes="form-label")
                    yield Select(
                        conversation_mode_choices(),
                        value="default",
                        id="synth-mode",
                        classes="form-select",
                    )

                with Horizontal(classes="form-row"):
                    yield Label("Job ID (resume):", classes="form-label")
                    yield Input(placeholder="(new job)", id="synth-job-id", classes="form-input")

                with Horizontal(classes="form-row"):
                    yield Label("Judge:", classes="form-label")
                    yield Switch(value=False, id="synth-judge")
                    yield Label("Cache:", classes="form-label")
                    yield Switch(value=True, id="synth-cache")
                    yield Label("Checkpoint:", classes="form-label")
                    yield Switch(value=True, id="synth-checkpoint")

            yield Button("🧠 Run Synthesis", id="synth-run", variant="primary")
            yield ProgressBar(total=100, id="synth-progress", show_eta=False)
            yield Label("Ready", id="synth-status")

    def get_config(self) -> dict[str, Any]:
        """Get current form values."""
        max_tokens_str = self.query_one("#synth-max-tokens", Input).value.strip()
        return {
            "input": self.query_one("#synth-input", Input).value.strip() or "./chunks.json",
            "output": self.query_one("#synth-output", Input).value.strip() or "./conversations.json",
            "provider": self.query_one("#synth-provider", Select).value or "openai",
            "model": self.query_one("#synth-model", Input).value.strip() or "gpt-5-mini",
            "base_url": self.query_one("#synth-base-url", Input).value.strip() or None,
            "concurrency": int(self.query_one("#synth-concurrency", Input).value or "5"),
            "rpm": int(self.query_one("#synth-rpm", Input).value or "60"),
            "max_tokens": int(max_tokens_str) if max_tokens_str else None,
            "mode": self.query_one("#synth-mode", Select).value or "default",
            "job_id": self.query_one("#synth-job-id", Input).value.strip() or None,
            "judge": self.query_one("#synth-judge", Switch).value,
            "cache": self.query_one("#synth-cache", Switch).value,
            "checkpoint": self.query_one("#synth-checkpoint", Switch).value,
        }

    def validate_config(self) -> str | None:
        """Validate form values. Returns an error message or None."""
        cfg = self.get_config()
        if not cfg["input"]:
            return "Input file path is required"
        if not Path(cfg["input"]).exists():
            return f"Input not found: {cfg['input']}"
        try:
            _validate_positive_int(str(cfg["concurrency"]), "concurrency", 5)
            _validate_positive_int(str(cfg["rpm"]), "RPM", 60)
        except ValueError as e:
            return str(e)
        return None

    def set_status(self, text: str) -> None:
        self.query_one("#synth-status", Label).update(text)

    def set_running(self, running: bool) -> None:
        btn = self.query_one("#synth-run", Button)
        btn.disabled = running
        if running:
            self.query_one("#synth-status", Label).update("[yellow]Running...[/yellow]")


# =============================================================================
# Export Tab
# =============================================================================


class ExportTab(Container):
    """Tab for running the export pipeline."""

    FORMATS = [fmt for fmt, _ in export_format_choices()]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("📤 Export Pipeline", classes="tab-title")

            # Simple-mode guidance
            yield Static(
                "Export your conversations into fine-tuning formats like ShareGPT, Alpaca, or ChatML.",
                classes="simple-only simple-help",
            )

            with Horizontal(classes="form-row"):
                yield Label("Input file:", classes="form-label")
                yield Input(value="./conversations.json", id="export-input", classes="form-input")

            with Horizontal(classes="form-row"):
                yield Label("Output dir:", classes="form-label")
                yield Input(value="./output", id="export-output-dir", classes="form-input")

            # Expert-only advanced
            with Horizontal(classes="form-row expert-only"):
                yield Label("Unsloth model:", classes="form-label")
                yield Input(value="Qwen/Qwen3-8B", id="export-unsloth-model", classes="form-input")

            yield Label("Formats:", classes="form-label")
            with Horizontal(id="export-formats-row"):
                for fmt in self.FORMATS:
                    yield Switch(value=(fmt == "sharegpt"), id=f"export-fmt-{fmt}")
                    yield Label(fmt, classes="form-label-small")

            # Expert-only switches
            with Horizontal(classes="form-row expert-only"):
                yield Label("Unsloth script:", classes="form-label")
                yield Switch(value=True, id="export-unsloth")
                yield Label("Split:", classes="form-label")
                yield Switch(value=False, id="export-split")
                yield Label("Dataset card:", classes="form-label")
                yield Switch(value=False, id="export-card")

            yield Button("📤 Run Export", id="export-run", variant="primary")
            yield ProgressBar(total=100, id="export-progress", show_eta=False)
            yield Label("Ready", id="export-status")
            yield DataTable(id="export-results-table")

    def on_mount(self) -> None:
        table = self.query_one("#export-results-table", DataTable)
        table.add_columns("Format", "File", "Size")

    def get_selected_formats(self) -> list[str]:
        """Get currently selected export formats."""
        selected = []
        for fmt in self.FORMATS:
            try:
                if self.query_one(f"#export-fmt-{fmt}", Switch).value:
                    selected.append(fmt)
            except Exception:
                pass
        return selected or ["sharegpt"]

    def get_config(self) -> dict[str, Any]:
        """Get current form values."""
        return {
            "input": self.query_one("#export-input", Input).value.strip() or "./conversations.json",
            "output_dir": self.query_one("#export-output-dir", Input).value.strip() or "./output",
            "unsloth_model": self.query_one("#export-unsloth-model", Input).value.strip() or "Qwen/Qwen3-8B",
            "formats": self.get_selected_formats(),
            "unsloth": self.query_one("#export-unsloth", Switch).value,
            "split": self.query_one("#export-split", Switch).value,
            "card": self.query_one("#export-card", Switch).value,
        }

    def validate_config(self) -> str | None:
        """Validate form values. Returns an error message or None."""
        cfg = self.get_config()
        if not cfg["input"]:
            return "Input file path is required"
        if not Path(cfg["input"]).exists():
            return f"Input not found: {cfg['input']}"
        if not cfg["formats"]:
            return "Please select at least one export format"
        return None

    def set_status(self, text: str) -> None:
        self.query_one("#export-status", Label).update(text)

    def set_running(self, running: bool) -> None:
        btn = self.query_one("#export-run", Button)
        btn.disabled = running
        if running:
            self.query_one("#export-status", Label).update("[yellow]Running...[/yellow]")

    def show_results(self, results: dict[str, Path]) -> None:
        """Show export results in the results table."""
        try:
            table = self.query_one("#export-results-table", DataTable)
            table.clear()
            for fmt, path in results.items():
                size = f"{path.stat().st_size / 1024:.1f} KB" if path.exists() else "N/A"
                table.add_row(fmt, str(path), size)
        except Exception:
            pass


# =============================================================================
# Validate Tab
# =============================================================================


class ValidateTab(Container):
    """Tab for running dataset validation."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("🔍 Dataset Validation", classes="tab-title")

            # Simple-mode guidance
            yield Static(
                "Check your dataset for issues, duplicates, and quality metrics before training.",
                classes="simple-only simple-help",
            )

            with Horizontal(classes="form-row"):
                yield Label("Input file:", classes="form-label")
                yield Input(value="./conversations.json", id="validate-input", classes="form-input")

            with Horizontal(classes="form-row"):
                yield Label("Deduplicate:", classes="form-label")
                yield Switch(value=True, id="validate-dedupe")

            yield Button("🔍 Run Validation", id="validate-run", variant="primary")
            yield Label("Ready", id="validate-status")
            yield DataTable(id="validate-results")

            yield Label("📊 Quality Gate (Evaluate)", classes="section-title")
            yield Static(
                "Heuristic + judge-score quality gate (no LLM calls). Fails the gate when "
                "confidence or validity drops below threshold.",
                classes="simple-only simple-help",
            )
            with Horizontal(classes="form-row"):
                yield Label("Min confidence:", classes="form-label")
                yield Input(value="0.5", id="eval-threshold", classes="form-input-small")
                yield Label("Min valid rate:", classes="form-label")
                yield Input(value="0.8", id="eval-min-valid", classes="form-input-small")
            yield Button("📊 Run Evaluation", id="evaluate-run", variant="primary")
            yield Label("Ready", id="evaluate-status")

    def on_mount(self) -> None:
        table = self.query_one("#validate-results", DataTable)
        table.add_columns("Metric", "Value")

    def get_config(self) -> dict[str, Any]:
        return {
            "input": self.query_one("#validate-input", Input).value.strip() or "./conversations.json",
            "dedupe": self.query_one("#validate-dedupe", Switch).value,
        }

    def get_eval_config(self) -> dict[str, Any]:
        """Get evaluation gate form values."""
        try:
            threshold = float(self.query_one("#eval-threshold", Input).value or "0.5")
        except ValueError:
            threshold = 0.5
        try:
            min_valid = float(self.query_one("#eval-min-valid", Input).value or "0.8")
        except ValueError:
            min_valid = 0.8
        return {
            "input": self.query_one("#validate-input", Input).value.strip() or "./conversations.json",
            "threshold": min(max(threshold, 0.0), 1.0),
            "min_valid_rate": min(max(min_valid, 0.0), 1.0),
        }

    def validate_config(self) -> str | None:
        """Validate form values. Returns an error message or None."""
        cfg = self.get_config()
        if not cfg["input"]:
            return "Input file path is required"
        if not Path(cfg["input"]).exists():
            return f"Input not found: {cfg['input']}"
        return None

    def set_status(self, text: str) -> None:
        self.query_one("#validate-status", Label).update(text)

    def set_running(self, running: bool) -> None:
        btn = self.query_one("#validate-run", Button)
        btn.disabled = running
        if running:
            self.query_one("#validate-status", Label).update("[yellow]Running...[/yellow]")


# =============================================================================
# Full Pipeline Tab (chains Ingest → Synthesize → Export)
# =============================================================================


class FullPipelineTab(Container):
    """Tab for running the entire pipeline end-to-end."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("⚡ Full Pipeline — Ingest → Synthesize → Export", classes="tab-title")
            yield Label(
                "Run the entire pipeline end-to-end with one click. "
                "Simple mode shows only the essentials — switch to Expert for full control.",
                id="full-help",
            )

            with Vertical(id="full-ingest-section"):
                yield Label("[bold]📥 Ingestion[/bold]", classes="section-title")
                # Simple-mode guidance
                yield Static(
                    "Point to your source files. We'll chunk them automatically.",
                    classes="simple-only simple-help",
                )
                with Horizontal(classes="form-row"):
                    yield Label("Source path:", classes="form-label")
                    yield Input(placeholder="./data", id="full-source", classes="form-input")
                # Expert-only ingestion controls
                with Horizontal(classes="form-row expert-only"):
                    yield Label("Chunk size:", classes="form-label")
                    yield Input(value="1000", id="full-chunk-size", classes="form-input-small")
                    yield Label("Overlap:", classes="form-label")
                    yield Input(value="200", id="full-overlap", classes="form-input-small")
                with Horizontal(classes="form-row expert-only"):
                    yield Label("Recursive:", classes="form-label")
                    yield Switch(value=True, id="full-recursive")
                    yield Label("Auto-detect:", classes="form-label")
                    yield Switch(value=True, id="full-auto")
                with Horizontal(classes="form-row expert-only"):
                    yield Label("Chunker:", classes="form-label")
                    yield Select(
                        chunker_choices(),
                        value="auto",
                        id="full-chunker",
                        classes="form-select",
                    )

            with Vertical(id="full-synth-section"):
                yield Label("[bold]🧠 Synthesis[/bold]", classes="section-title")
                # Simple-mode guidance
                yield Static(
                    "Choose your AI provider and model to generate conversations.",
                    classes="simple-only simple-help",
                )
                with Horizontal(classes="form-row"):
                    yield Label("Provider:", classes="form-label")
                    yield Select(
                        list_select_choices(),
                        value="openai",
                        id="full-provider",
                        classes="form-select",
                    )
                    yield Label("Model:", classes="form-label")
                    yield Input(value="gpt-5-mini", id="full-model", classes="form-input")
                # Expert-only synthesis controls
                with Horizontal(classes="form-row expert-only"):
                    yield Label("Concurrency:", classes="form-label")
                    yield Input(value="5", id="full-concurrency", classes="form-input-small")
                    yield Label("RPM:", classes="form-label")
                    yield Input(value="60", id="full-rpm", classes="form-input-small")
                with Horizontal(classes="form-row expert-only"):
                    yield Label("Mode:", classes="form-label")
                    yield Select(
                        conversation_mode_choices(),
                        value="default",
                        id="full-mode",
                        classes="form-select",
                    )
                with Horizontal(classes="form-row expert-only"):
                    yield Label("Cache:", classes="form-label")
                    yield Switch(value=True, id="full-cache")
                    yield Label("Checkpoint:", classes="form-label")
                    yield Switch(value=True, id="full-checkpoint")

            with Vertical(id="full-export-section"):
                yield Label("[bold]📤 Export[/bold]", classes="section-title")
                # Simple-mode guidance
                yield Static(
                    "Save your dataset in the format your training framework expects.",
                    classes="simple-only simple-help",
                )
                with Horizontal(classes="form-row"):
                    yield Label("Output dir:", classes="form-label")
                    yield Input(value="./output", id="full-output-dir", classes="form-input")
                with Horizontal(classes="form-row"):
                    yield Label("Format:", classes="form-label")
                    yield Select(
                        export_format_choices(),
                        value="sharegpt",
                        id="full-format",
                        classes="form-select",
                    )
                # Expert-only
                with Horizontal(classes="form-row expert-only"):
                    yield Label("DS Card:", classes="form-label")
                    yield Switch(value=False, id="full-card")

            # Simple mode: hint about expert options
            yield Static(
                "[dim]💡 Press [reverse] m [/reverse] for advanced options (chunk tuning, concurrency, mode selection, etc.)[/dim]",
                classes="simple-only",
            )

            yield Button("⚡ Run Full Pipeline", id="full-run", variant="primary")
            yield ProgressBar(total=100, id="full-progress", show_eta=True)
            yield Label("Configure the stages above and click to run all three in sequence.", id="full-status")

    def get_config(self) -> dict[str, Any]:
        return {
            "source": self.query_one("#full-source", Input).value.strip(),
            "chunk_size": int(self.query_one("#full-chunk-size", Input).value or "1000"),
            "overlap": int(self.query_one("#full-overlap", Input).value or "200"),
            "recursive": self.query_one("#full-recursive", Switch).value,
            "auto_detect": self.query_one("#full-auto", Switch).value,
            "chunker": self.query_one("#full-chunker", Select).value or "auto",
            "provider": self.query_one("#full-provider", Select).value or "openai",
            "model": self.query_one("#full-model", Input).value.strip() or "gpt-5-mini",
            "concurrency": int(self.query_one("#full-concurrency", Input).value or "5"),
            "rpm": int(self.query_one("#full-rpm", Input).value or "60"),
            "mode": self.query_one("#full-mode", Select).value or "default",
            "cache": self.query_one("#full-cache", Switch).value,
            "checkpoint": self.query_one("#full-checkpoint", Switch).value,
            "output_dir": self.query_one("#full-output-dir", Input).value.strip() or "./output",
            "format": self.query_one("#full-format", Select).value or "sharegpt",
            "card": self.query_one("#full-card", Switch).value,
        }

    def validate_config(self) -> str | None:
        """Validate form values. Returns an error message or None."""
        cfg = self.get_config()
        if not cfg["source"]:
            return "Source path is required"
        if not Path(cfg["source"]).exists():
            return f"Source not found: {cfg['source']}"
        if not validate_chunker(cfg["chunker"]):
            return f"Unknown chunker: {cfg['chunker']}"
        try:
            _validate_positive_int(str(cfg["concurrency"]), "concurrency", 5)
            _validate_positive_int(str(cfg["rpm"]), "RPM", 60)
        except ValueError as e:
            return str(e)
        return None

    def set_status(self, text: str) -> None:
        self.query_one("#full-status", Label).update(text)

    def set_running(self, running: bool) -> None:
        btn = self.query_one("#full-run", Button)
        btn.disabled = running
        if running:
            self.query_one("#full-status", Label).update("[yellow]Initializing...[/yellow]")


# =============================================================================
# Jobs Tab
# =============================================================================


class JobsTab(Container):
    """Tab for managing synthesis jobs."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("📋 Job Management", classes="tab-title")

            with Horizontal(id="jobs-actions"):
                yield Button("🔄 Refresh", id="jobs-refresh", variant="primary")
                yield Button("🗑️ Delete Selected", id="jobs-delete", variant="error")
                yield Button("▶️ Resume", id="jobs-resume", variant="success")
                yield Button("🧹 Cleanup Old", id="jobs-cleanup", variant="warning")

            yield DataTable(id="jobs-table")

    def on_mount(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        table.add_columns("Job ID", "Type", "Status", "Progress", "Created", "Errors")
        table.cursor_type = "row"
        self.refresh_jobs()

    def refresh_jobs(self) -> None:
        """Refresh the jobs table."""
        try:
            manager = CheckpointManager()
            jobs = manager.list_jobs(limit=50)
            table = self.query_one("#jobs-table", DataTable)
            table.clear()

            for job in jobs:
                created = datetime.fromtimestamp(job.created_at).strftime("%Y-%m-%d %H:%M")
                progress = f"{job.processed_items}/{job.total_items} ({job.progress_pct:.0f}%)"
                errors = str(job.failed_items)
                table.add_row(
                    job.job_id,
                    job.job_type,
                    job.status.value,
                    progress,
                    created,
                    errors,
                )
        except Exception as e:
            logger.error(f"Failed to refresh jobs: {e}")

    def get_selected_job_id(self) -> str | None:
        """Get the job ID of the currently selected row."""
        try:
            table = self.query_one("#jobs-table", DataTable)
            if table.cursor_row is None or table.cursor_row >= table.row_count:
                return None
            row_key = table.get_row_at(table.cursor_row)
            if row_key:
                return str(row_key[0])  # First column is Job ID
        except Exception:
            pass
        return None


# =============================================================================
# Cache Tab
# =============================================================================


class CacheTab(Container):
    """Tab for inspecting the synthesis cache."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("💾 Cache Inspector", classes="tab-title")

            with Horizontal(id="cache-actions"):
                yield Button("🔄 Refresh", id="cache-refresh", variant="primary")
                yield Button("🧹 Prune Old", id="cache-prune", variant="warning")
                yield Button("🗑️ Clear All", id="cache-clear", variant="error")

            yield DataTable(id="cache-detail-table")

    def on_mount(self) -> None:
        table = self.query_one("#cache-detail-table", DataTable)
        table.add_columns("Key (prefix)", "Model", "Tokens", "Created", "Accessed")
        self.refresh_cache()

    def refresh_cache(self) -> None:
        """Refresh cache detail table."""
        try:
            cache = CacheManager(cache_dir=".cache")
            keys = cache.get_cached_keys()

            table = self.query_one("#cache-detail-table", DataTable)
            table.clear()

            for key in keys[:100]:
                entry = cache.get(key)
                if entry:
                    created = datetime.fromtimestamp(entry["created_at"]).strftime("%Y-%m-%d %H:%M")
                    table.add_row(
                        key[:16] + "...",
                        entry.get("model", "unknown"),
                        str(entry.get("tokens_used", 0)),
                        created,
                        "-",
                    )
        except Exception as e:
            logger.error(f"Failed to refresh cache: {e}")


# =============================================================================
# Config Tab
# =============================================================================


class ConfigTab(Container):
    """Tab for viewing and managing configuration."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("🔧 Configuration", classes="tab-title")

            with Horizontal(id="config-actions"):
                yield Button("➕ Init Config", id="config-init", variant="success")
                yield Button("🔄 Reload", id="config-reload", variant="primary")

            yield Static("Loading configuration...", id="config-content")

    def on_mount(self) -> None:
        self.refresh_config()

    def refresh_config(self) -> None:
        """Refresh config display."""
        config_path = find_config_file()
        content_widget = self.query_one("#config-content", Static)

        if config_path:
            try:
                config = load_config(config_path)
                text = f"[bold green]Config File:[/bold green] {config_path}\n\n"
                text += f"[cyan]{config.model_dump_json(indent=2)}[/cyan]"
                content_widget.update(text)
            except Exception as e:
                content_widget.update(f"[red]Error loading config: {e}[/red]")
        else:
            content_widget.update(
                "[yellow]No config file found.[/yellow]\n\n"
                "Click [bold green]➕ Init Config[/bold green] to create one, or run:\n"
                "  [cyan]distill-align init[/cyan]"
            )


# =============================================================================
# Logs Tab
# =============================================================================


class LogsTab(Container):
    """Tab for viewing live logs."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("📝 Live Logs", classes="tab-title")
            yield RichLog(id="log-view", highlight=True, markup=True, wrap=True)


# =============================================================================
# Main App
# =============================================================================


class DistillAlignApp(App):
    """Main TUI application — full interactive dashboard with pipeline execution."""

    # Shared tracking for pipeline runs
    _last_run_summary: str = ""
    _total_cost_usd: float = 0.0

    CSS = """
    Screen {
        layout: vertical;
    }

    .tab-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    .section-title {
        text-style: bold;
        color: $secondary;
        margin-top: 1;
        margin-bottom: 0;
    }

    .stat-box {
        width: 1fr;
        height: 5;
        content-align: center middle;
        border: round $primary;
        margin: 0 1;
    }

    .stat-box-wide {
        width: 100%;
        height: 3;
        padding: 0 1;
        border: round $primary;
        margin: 0 1;
    }

    TabbedContent {
        height: 1fr;
    }

    #cache-table {
        height: 1fr;
    }

    Button {
        margin: 0 1;
    }

    .panel-title {
        text-style: bold;
        margin-bottom: 1;
    }

    .form-row {
        height: auto;
        margin: 0 0 0 0;
        align: left middle;
    }

    .form-label {
        width: auto;
        min-width: 16;
        padding: 0 1;
    }

    .form-label-small {
        width: auto;
        min-width: 10;
        padding: 0 0 0 0;
    }

    .form-input {
        width: 1fr;
        max-width: 60;
        margin: 0 1;
    }

    .form-input-small {
        width: 15;
        margin: 0 1;
    }

    .form-select {
        width: 20;
        margin: 0 1;
    }

    #export-formats-row {
        height: auto;
        margin: 0 0 0 16;
    }

    #system-info-row {
        height: auto;
        margin: 0 0 1 0;
    }

    #dashboard-last-run {
        padding: 0 1;
        color: $surface;
    }

    #full-help {
        padding: 0 1;
        margin-bottom: 1;
        color: $surface;
    }

    #full-ingest-section,
    #full-synth-section,
    #full-export-section {
        border: solid $primary;
        margin: 0 0 1 0;
        padding: 0 0;
    }

    #export-results-table {
        height: auto;
        max-height: 8;
    }

    #validate-results {
        height: auto;
        max-height: 10;
    }

    #mode-indicator {
        width: auto;
        padding: 0 1;
        text-style: bold;
    }

    /* When app has simple-mode class, hide expert-only containers */
    .simple-mode .expert-only {
        display: none !important;
    }

    /* When app has expert-mode class, hide simple-only containers */
    .expert-mode .simple-only {
        display: none !important;
    }

    /* Guidance text visible only in simple mode */
    .simple-help {
        color: $text-muted;
        padding: 0 1;
        margin-bottom: 1;
        text-style: italic;
    }

    /* Quick-start and template buttons */
    .action-button {
        min-width: 30;
        margin: 1 2;
    }

    /* Tip box visible in simple mode */
    .tip-box {
        border: solid $success;
        margin: 1 0;
        padding: 1 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("m", "toggle_mode", "Mode"),
        Binding("question", "show_help", "Help"),
        Binding("ctrl+c", "quit", "Quit"),
        Binding("1", "switch_tab('dashboard')", "Dashboard"),
        Binding("2", "switch_tab('ingest')", "Ingest"),
        Binding("3", "switch_tab('synthesize')", "Synthesize"),
        Binding("4", "switch_tab('export')", "Export"),
        Binding("5", "switch_tab('validate')", "Validate"),
        Binding("6", "switch_tab('full-pipeline')", "Full Pipe"),
        Binding("7", "switch_tab('jobs')", "Jobs"),
        Binding("8", "switch_tab('cache')", "Cache"),
        Binding("9", "switch_tab('config')", "Config"),
        Binding("0", "switch_tab('logs')", "Logs"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="dashboard"):
            with TabPane("Dashboard", id="dashboard"):
                yield DashboardTab()
            with TabPane("Ingest", id="ingest"):
                yield IngestTab()
            with TabPane("Synthesize", id="synthesize"):
                yield SynthesizeTab()
            with TabPane("Export", id="export"):
                yield ExportTab()
            with TabPane("Validate", id="validate"):
                yield ValidateTab()
            with TabPane("Full Pipeline", id="full-pipeline"):
                yield FullPipelineTab()
            with TabPane("Jobs", id="jobs"):
                yield JobsTab()
            with TabPane("Cache", id="cache"):
                yield CacheTab()
            with TabPane("Config", id="config"):
                yield ConfigTab()
            with TabPane("Logs", id="logs"):
                yield LogsTab()
        yield Footer()

    def on_mount(self) -> None:
        """Called when app starts."""
        self.title = "Distill-Align Dashboard"

        # Start in simple mode (CSS class drives visibility)
        self.add_class("simple-mode")
        self._update_mode_display()

        # Install loguru sink for live logs
        _install_loguru_sink(self)

        # Refresh stats periodically
        self.set_interval(5.0, self._refresh_callback)

        # Add a welcome log message
        log_view = self.query_one("#log-view", RichLog)
        if log_view:
            log_view.write("[green]✓[/green] Distill-Align TUI started")
            log_view.write(f"[cyan]i[/cyan] Working directory: {Path.cwd()}")
            log_view.write("[cyan]i[/cyan] Press [yellow]m[/yellow] to toggle modes, [yellow]?[/yellow] for help")
            log_view.write("[cyan]i[/cyan] Tabs: [yellow]0[/yellow]-[yellow]9[/yellow] to switch tabs")

        # Background update check (non-blocking, silent on failure)
        self.run_worker(self._check_update_worker, thread=True)

        # Load config file (also registers any custom providers)
        try:
            config_path = find_config_file()
            if config_path:
                cfg = load_config(config_path)
                log_view = self.query_one("#log-view", RichLog)
                if cfg.custom_providers:
                    names = [p.name for p in cfg.custom_providers]
                    log_view.write(f"[green]✓[/green] Loaded custom providers: {', '.join(names)}")
        except Exception:
            pass  # Non-fatal — providers are just not customised

    def _update_mode_display(self) -> None:
        """Update the header subtitle to show current mode."""
        is_simple = self.has_class("simple-mode")
        mode = MODE_SIMPLE if is_simple else MODE_EXPERT
        self.sub_title = f"{MODE_LABELS[mode]}  |  Structured Reasoning Extraction Factory"

    def action_toggle_mode(self) -> None:
        """Toggle between Simple and Expert mode."""
        if self.has_class("simple-mode"):
            self.remove_class("simple-mode")
            self.add_class("expert-mode")
            self.notify("🧪 Switched to Expert mode — all controls visible", timeout=3)
            logger.info("TUI switched to Expert mode")
        else:
            self.remove_class("expert-mode")
            self.add_class("simple-mode")
            self.notify("🌱 Switched to Simple mode — streamlined view", timeout=3)
            logger.info("TUI switched to Simple mode")
        self._update_mode_display()

    def action_show_help(self) -> None:
        """Show the keyboard shortcuts help overlay."""
        self.push_screen(HelpOverlay())

    def _check_update_worker(self) -> None:
        """Worker thread: check PyPI for a newer version and notify if available."""
        latest = check_pypi_version()
        if latest:
            self.call_from_thread(self._show_update_notification, latest)

    def _show_update_notification(self, latest: str) -> None:
        """Show update notification on the main thread."""
        log_view = self.query_one("#log-view", RichLog)
        log_view.write(
            f"[yellow]⚠ Update available: v{__version__} → v{latest}. Run: pip install --upgrade distill-align[/yellow]"
        )
        self.notify(
            f"Update available: v{__version__} → v{latest}",
            timeout=8,
        )

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to a specific tab."""
        with contextlib.suppress(Exception):
            self.query_one(TabbedContent).active = tab_id

    def _refresh_callback(self) -> None:
        """Periodic auto-refresh of stats."""
        try:
            dashboard = self.query_one(DashboardTab)
            dashboard.refresh_cache_stats()
        except Exception:
            pass

    def action_refresh(self) -> None:
        """Manual refresh action."""
        try:
            dashboard = self.query_one(DashboardTab)
            dashboard.refresh_cache_stats()
            dashboard.refresh_file_stats()
            dashboard.refresh_system_info()
            self.notify("Refreshed stats", timeout=2)
        except Exception as e:
            self.notify(f"Refresh error: {e}", severity="error")

        with contextlib.suppress(Exception):
            self.query_one(JobsTab).refresh_jobs()

        with contextlib.suppress(Exception):
            self.query_one(ConfigTab).refresh_config()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses across all tabs."""
        button_id = event.button.id

        # --- Dashboard: Wizard / Template ---
        if button_id == "btn-wizard":
            self._launch_wizard()
            return

        if button_id == "btn-template":
            self._show_template_picker()
            return

        # --- Jobs Tab ---
        if button_id == "jobs-refresh":
            self.query_one(JobsTab).refresh_jobs()

        elif button_id == "jobs-delete":
            self._handle_jobs_delete()

        elif button_id == "jobs-resume":
            self._handle_jobs_resume()

        elif button_id == "jobs-cleanup":
            self._handle_jobs_cleanup()

        # --- Cache Tab ---
        elif button_id == "cache-refresh":
            self.query_one(CacheTab).refresh_cache()

        elif button_id == "cache-prune":
            cache = CacheManager(cache_dir=".cache")
            removed = cache.prune()
            self.notify(f"Pruned {removed} entries")
            self.query_one(CacheTab).refresh_cache()

        elif button_id == "cache-clear":
            cache = CacheManager(cache_dir=".cache")
            removed = cache.clear()
            self.notify(f"Cleared {removed} entries", severity="warning")
            self.query_one(CacheTab).refresh_cache()

        # --- Config Tab ---
        elif button_id == "config-init":
            self._handle_config_init()

        elif button_id == "config-reload":
            self.query_one(ConfigTab).refresh_config()
            self.notify("Config reloaded")

        # --- Pipeline Tabs (with Pre-flight Checks) ---
        elif button_id == "ingest-run":
            self._run_ingest_with_preflight()

        elif button_id == "synth-run":
            self._run_synthesize_with_preflight()

        elif button_id == "export-run":
            self._run_export_with_preflight()

        elif button_id == "validate-run":
            self._run_validate()  # No pre-flight needed for validate

        elif button_id == "evaluate-run":
            self._run_evaluate()  # No pre-flight needed for evaluate

        # --- Full Pipeline Tab ---
        elif button_id == "full-run":
            self._run_full_pipeline_with_preflight()

    # =========================================================================
    # Wizard & Templates
    # =========================================================================

    def _launch_wizard(self) -> None:
        """Launch the Quick-Start Wizard as a modal screen."""
        logger.info("Launching Quick-Start Wizard")

        async def on_wizard_done(result: dict[str, Any] | None) -> None:
            if result is None:
                logger.info("Wizard cancelled")
                return

            logger.info(f"Wizard completed: {result}")

            # Build config from wizard result
            from .widgets.wizard import wizard_result_to_full_config

            config = wizard_result_to_full_config(result)
            if not config:
                self.notify("Wizard: could not build config", severity="warning")
                return

            # Pre-fill the Full Pipeline tab with wizard values
            try:
                full_tab = self.query_one(FullPipelineTab)
                full_tab.query_one("#full-source", Input).value = config.get("source", "./data")
                full_tab.query_one("#full-provider", Select).value = config.get("provider", "openai")
                full_tab.query_one("#full-model", Input).value = config.get("model", "gpt-5-mini")
                full_tab.query_one("#full-output-dir", Input).value = config.get("output_dir", "./output")
                full_tab.query_one("#full-format", Select).value = config.get("format", "sharegpt")

                # If template specifies advanced options, pre-fill those too
                if "chunk_size" in config:
                    full_tab.query_one("#full-chunk-size", Input).value = str(config["chunk_size"])
                if "overlap" in config:
                    full_tab.query_one("#full-overlap", Input).value = str(config["overlap"])
                if "concurrency" in config:
                    full_tab.query_one("#full-concurrency", Input).value = str(config["concurrency"])
                if "rpm" in config:
                    full_tab.query_one("#full-rpm", Input).value = str(config["rpm"])
                if "mode" in config:
                    full_tab.query_one("#full-mode", Select).value = config["mode"]

                # Switch to Full Pipeline tab
                self.query_one(TabbedContent).active = "full-pipeline"
                self.notify(
                    "✨ Wizard complete! Settings applied to Full Pipeline tab. Review and click run!",
                    timeout=5,
                )
                logger.info(f"Wizard pre-filled Full Pipeline tab with: {config}")
            except Exception as e:
                logger.error(f"Failed to pre-fill wizard config: {e}")
                self.notify(f"Wizard error: {e}", severity="error")

        self.push_screen(QuickStartWizard(), on_wizard_done)

    def _show_template_picker(self) -> None:
        """Show a picker of built-in templates and apply the selected one."""
        from textual.screen import ModalScreen

        class TemplatePicker(ModalScreen[dict[str, Any] | None]):
            """Modal that lists all built-in templates for selection."""

            CSS = """
            TemplatePicker {
                align: center middle;
            }

            #picker-box {
                width: 70;
                height: auto;
                max-height: 80%;
                padding: 1 2;
                border: thick $accent;
                background: $background;
            }

            .tmpl-item {
                height: auto;
                padding: 0 1;
                margin: 0 0 0 0;
            }

            .tmpl-item:hover {
                background: $surface;
            }

            .tmpl-name {
                text-style: bold;
            }

            .tmpl-desc {
                color: $text-muted;
                padding-left: 3;
            }

            #picker-title {
                text-style: bold;
                color: $accent;
                margin-bottom: 1;
            }

            #picker-actions {
                margin-top: 1;
                align: center middle;
            }
            """

            def compose(self) -> ComposeResult:
                with Vertical(id="picker-box"):
                    yield Label("📋 Choose a Pipeline Template", id="picker-title")
                    yield Static("Pick a preset below. It will pre-fill the Full Pipeline tab with sensible defaults.")
                    for t in BUILTIN_TEMPLATES:
                        with Vertical(classes="tmpl-item"):
                            yield Button(
                                f"{t.icon} {t.name}",
                                id=f"tmpl-{t.name}",
                                variant="default",
                            )
                            yield Static(t.description, classes="tmpl-desc")
                    with Horizontal(id="picker-actions"):
                        yield Button("Cancel", id="picker-cancel", variant="error")

            def on_button_pressed(self, event: Button.Pressed) -> None:
                if event.button.id == "picker-cancel":
                    self.dismiss(None)
                    return
                if event.button.id and event.button.id.startswith("tmpl-"):
                    name = event.button.id[len("tmpl-") :]
                    tmpl = get_template(name)
                    if tmpl:
                        self.dismiss(tmpl.config)
                    else:
                        self.dismiss(None)

        def on_pick(result: dict[str, Any] | None) -> None:
            if result is None:
                return
            try:
                full_tab = self.query_one(FullPipelineTab)
                if "source" not in result:
                    result["source"] = full_tab.query_one("#full-source", Input).value or "./data"
                full_tab.query_one("#full-source", Input).value = result.get("source", "./data")
                if "chunk_size" in result:
                    full_tab.query_one("#full-chunk-size", Input).value = str(result["chunk_size"])
                if "overlap" in result:
                    full_tab.query_one("#full-overlap", Input).value = str(result["overlap"])
                full_tab.query_one("#full-provider", Select).value = result.get("provider", "openai")
                full_tab.query_one("#full-model", Input).value = result.get("model", "gpt-5-mini")
                if "concurrency" in result:
                    full_tab.query_one("#full-concurrency", Input).value = str(result["concurrency"])
                if "rpm" in result:
                    full_tab.query_one("#full-rpm", Input).value = str(result["rpm"])
                if "mode" in result:
                    full_tab.query_one("#full-mode", Select).value = result["mode"]
                if "format" in result:
                    full_tab.query_one("#full-format", Select).value = result["format"]
                full_tab.query_one("#full-output-dir", Input).value = result.get("output_dir", "./output")
                if "judge" in result:
                    full_tab.query_one("#full-cache", Switch).value = result.get("cache", True)
                    full_tab.query_one("#full-checkpoint", Switch).value = result.get("checkpoint", True)
                if "card" in result:
                    full_tab.query_one("#full-card", Switch).value = result["card"]

                self.query_one(TabbedContent).active = "full-pipeline"
                self.notify("📋 Template applied! Review the settings and run.", timeout=4)
            except Exception as e:
                logger.error(f"Failed to apply template: {e}")
                self.notify(f"Template error: {e}", severity="error")

        self.push_screen(TemplatePicker(), on_pick)

    # =========================================================================
    # Jobs Actions
    # =========================================================================

    def _handle_jobs_delete(self) -> None:
        """Delete the selected job."""
        jobs_tab = self.query_one(JobsTab)
        job_id = jobs_tab.get_selected_job_id()
        if not job_id:
            self.notify("No job selected", severity="warning")
            return

        try:
            manager = CheckpointManager()
            if manager.delete_job(job_id):
                self.notify(f"Deleted job {job_id}", severity="warning")
                logger.info(f"Deleted job checkpoint: {job_id}")
                jobs_tab.refresh_jobs()
            else:
                self.notify(f"Job {job_id} not found", severity="error")
        except Exception as e:
            self.notify(f"Delete failed: {e}", severity="error")
            logger.error(f"Failed to delete job {job_id}: {e}")

    def _handle_jobs_resume(self) -> None:
        """Show info about resuming the selected job."""
        jobs_tab = self.query_one(JobsTab)
        job_id = jobs_tab.get_selected_job_id()
        if not job_id:
            self.notify("No job selected", severity="warning")
            return

        try:
            manager = CheckpointManager()
            checkpoint = manager.load_job(job_id)
            if checkpoint:
                msg = (
                    f"Job {job_id}: {checkpoint.processed_items}/{checkpoint.total_items} done "
                    f"({checkpoint.progress_pct:.0f}%)"
                )
                self.notify(msg, timeout=5)
                logger.info(msg)
                # Switch to Synthesize tab and pre-fill job ID
                self.query_one(TabbedContent).active = "synthesize"
                self.query_one("#synth-job-id", Input).value = job_id
            else:
                self.notify(f"Job {job_id} not found", severity="error")
        except Exception as e:
            self.notify(f"Resume failed: {e}", severity="error")
            logger.error(f"Failed to resume job {job_id}: {e}")

    def _handle_jobs_cleanup(self) -> None:
        """Clean up old job checkpoints."""
        try:
            manager = CheckpointManager()
            removed = manager.cleanup_old_jobs(older_than_days=30)
            self.notify(f"Cleaned up {removed} old jobs (30+ days)")
            logger.info(f"Cleaned up {removed} old job checkpoints")
            self.query_one(JobsTab).refresh_jobs()
        except Exception as e:
            self.notify(f"Cleanup failed: {e}", severity="error")
            logger.error(f"Failed to cleanup jobs: {e}")

    # =========================================================================
    # Config Actions
    # =========================================================================

    def _handle_config_init(self) -> None:
        """Initialize a default config file."""
        try:
            output = generate_default_config(project_name="my-dataset")
            self.notify(f"Created config: {output}", severity="information")
            logger.info(f"Generated config file: {output}")
            self.query_one(ConfigTab).refresh_config()
        except Exception as e:
            self.notify(f"Init failed: {e}", severity="error")
            logger.error(f"Failed to init config: {e}")

    # =========================================================================
    # Shared pipeline helpers
    # =========================================================================

    def _notify_dashboard(self, stage: str, current: int = 0, total: int = 100) -> None:
        """Update the Dashboard stage and progress from any pipeline."""
        try:
            dashboard = self.query_one(DashboardTab)
            dashboard.update_stage(stage)
            if total > 0:
                dashboard.update_progress(current, total)
        except Exception:
            pass

    def _set_last_run(self, summary: str) -> None:
        """Record and display the last pipeline run summary."""
        self._last_run_summary = summary
        with contextlib.suppress(Exception):
            self.query_one(DashboardTab).show_last_run(f"[dim]Last run:[/dim] {summary}")

    def _start_jobs_auto_refresh(self) -> None:
        """Start auto-refreshing the Jobs tab."""
        self.set_interval(10.0, self._jobs_auto_refresh_cb, name="jobs-auto-refresh")

    def _jobs_auto_refresh_cb(self) -> None:
        """Auto-refresh jobs table (called periodically during active synthesis)."""
        with contextlib.suppress(Exception):
            self.query_one(JobsTab).refresh_jobs()

    # =========================================================================
    # Pre-flight Check Wrappers (intercept Run buttons)
    # =========================================================================

    def _run_ingest_with_preflight(self) -> None:
        """Run pre-flight checks before starting ingestion."""
        tab = self.query_one(IngestTab)
        err = tab.validate_config()
        if err:
            self.notify(err, severity="warning")
            return

        config = tab.get_config()
        report = preflight_ingest(
            source=config["source"],
            output=config["output"],
            chunk_size=config["chunk_size"],
        )

        def on_result(proceed: bool) -> None:
            if proceed:
                logger.info("Pre-flight passed — starting ingestion")
                self._run_ingest()
            else:
                logger.info("Pre-flight check cancelled ingestion")

        self.push_screen(PreflightScreen(report, "Ingestion"), on_result)

    def _run_synthesize_with_preflight(self) -> None:
        """Run pre-flight checks before starting synthesis."""
        tab = self.query_one(SynthesizeTab)
        err = tab.validate_config()
        if err:
            self.notify(err, severity="warning")
            return

        config = tab.get_config()
        report = preflight_synthesize(
            input_path=config["input"],
            output=config["output"],
            provider=config["provider"],
            concurrency=config["concurrency"],
            mode=config["mode"],
        )

        def on_result(proceed: bool) -> None:
            if proceed:
                logger.info("Pre-flight passed — starting synthesis")
                self._run_synthesize()
            else:
                logger.info("Pre-flight check cancelled synthesis")

        self.push_screen(PreflightScreen(report, "Synthesis"), on_result)

    def _run_export_with_preflight(self) -> None:
        """Run pre-flight checks before starting export."""
        tab = self.query_one(ExportTab)
        err = tab.validate_config()
        if err:
            self.notify(err, severity="warning")
            return

        config = tab.get_config()
        report = preflight_export(
            input_path=config["input"],
            output_dir=config["output_dir"],
            formats=config["formats"],
        )

        def on_result(proceed: bool) -> None:
            if proceed:
                logger.info("Pre-flight passed — starting export")
                self._run_export()
            else:
                logger.info("Pre-flight check cancelled export")

        self.push_screen(PreflightScreen(report, "Export"), on_result)

    def _run_full_pipeline_with_preflight(self) -> None:
        """Run pre-flight checks before starting the full pipeline."""
        tab = self.query_one(FullPipelineTab)
        err = tab.validate_config()
        if err:
            self.notify(err, severity="warning")
            return

        config = tab.get_config()
        report = preflight_full_pipeline(
            source=config["source"],
            provider=config["provider"],
            model=config["model"],
            output_dir=config["output_dir"],
            chunk_size=config["chunk_size"],
            concurrency=config["concurrency"],
        )

        def on_result(proceed: bool) -> None:
            if proceed:
                logger.info("Pre-flight passed — starting full pipeline")
                self._run_full_pipeline()
            else:
                logger.info("Pre-flight check cancelled full pipeline")

        self.push_screen(PreflightScreen(report, "Full Pipeline"), on_result)

    # =========================================================================
    # Pipeline: Ingest
    # =========================================================================

    def _run_ingest(
        self,
        source: str | None = None,
        output: str | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        recursive: bool | None = None,
        auto_detect: bool | None = None,
        chunker: str | None = None,
        silent: bool = False,
    ) -> str | None:
        """Run the ingestion pipeline in a worker thread.

        Returns the output file path if successful, or None on error.
        When *silent* is False, operates on the IngestTab UI.
        """
        ingest_tab: IngestTab | None = None
        if not silent:
            ingest_tab = self.query_one(IngestTab)
            config = ingest_tab.get_config()
        else:
            # Build config from params (full pipeline mode)
            config = {
                "source": source or "./data",
                "output": output or "./chunks.json",
                "chunk_size": chunk_size or 1000,
                "overlap": overlap or 200,
                "recursive": recursive if recursive is not None else True,
                "auto_detect": auto_detect if auto_detect is not None else True,
                "chunker": chunker or "auto",
            }

        if not silent:
            err = self.query_one(IngestTab).validate_config()
            if err:
                self.notify(err, severity="warning")
                return None

        source_path = Path(config["source"])
        output_path = Path(config["output"])

        # Use an event to synchronize the worker result
        import threading

        result_box: list[str | None] = [None]
        done_event = threading.Event()

        def do_ingest() -> None:
            try:
                from ..core.schemas import IngestionConfig
                from ..ingestion.auto import AutoIngestionPipeline
                from ..ingestion.pipeline import IngestionPipeline

                pipeline_config = IngestionConfig(
                    chunk_size=config["chunk_size"],
                    chunk_overlap=config["overlap"],
                    chunker=config.get("chunker", "auto"),
                )

                pipeline: AutoIngestionPipeline | IngestionPipeline = (
                    AutoIngestionPipeline(pipeline_config)
                    if config["auto_detect"]
                    else IngestionPipeline(pipeline_config)
                )

                def progress_cb(current: int, total: int, name: str) -> None:
                    self.call_from_thread(self._notify_dashboard, f"Ingesting: {name}", current, total)
                    if not silent:
                        self.call_from_thread(self._update_ingest_progress, current, total, name)

                if source_path.is_file():
                    chunks = pipeline.ingest_file(source_path)
                else:
                    if config["auto_detect"] and hasattr(pipeline, "ingest_directory"):
                        chunks = pipeline.ingest_directory(source_path, recursive=config["recursive"])
                    else:
                        chunks = pipeline.ingest_directory(source_path, recursive=config["recursive"])

                output_path.parent.mkdir(parents=True, exist_ok=True)
                chunks_data = [chunk.model_dump() for chunk in chunks]
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(chunks_data, f, indent=2, ensure_ascii=False)

                result_box[0] = str(output_path)
                self.call_from_thread(self._notify_dashboard, "Ingestion complete", 100, 100)
                if not silent:
                    self.call_from_thread(self._ingest_done, len(chunks), str(output_path))
                logger.info(f"Ingestion complete: {len(chunks)} chunks → {output_path}")
            except Exception as e:
                result_box[0] = None
                if not silent:
                    self.call_from_thread(self._ingest_error, str(e))
                else:
                    self.call_from_thread(logger.error, f"Ingestion failed: {e}")
            finally:
                done_event.set()

        if not silent:
            if ingest_tab:
                ingest_tab.set_running(True)
            self._notify_dashboard(f"Ingesting: {source_path.name}")

        self.run_worker(do_ingest, thread=True)

        if silent:
            # In silent mode, wait and return the result
            done_event.wait(timeout=600)
            return result_box[0]
        return None

    def _update_ingest_progress(self, current: int, total: int, name: str) -> None:
        """Update ingest progress from worker thread."""
        try:
            tab = self.query_one(IngestTab)
            bar = tab.query_one("#ingest-progress", ProgressBar)
            bar.update(total=total, progress=current)
            tab.set_status(f"Processing {name} ({current}/{total})")
        except Exception:
            pass

    def _ingest_done(self, count: int, output_path: str) -> None:
        """Called when ingestion completes successfully."""
        tab = self.query_one(IngestTab)
        tab.set_running(False)
        tab.set_status(f"[green]✓ Done[/green] — {count} chunks → {output_path}")
        self.notify(f"Ingestion complete: {count} chunks", severity="information")
        self._set_last_run(f"Ingest: {count} chunks → {output_path}")
        with contextlib.suppress(Exception):
            self.query_one(DashboardTab).refresh_file_stats()

    def _ingest_error(self, error: str) -> None:
        """Called when ingestion fails."""
        tab = self.query_one(IngestTab)
        tab.set_running(False)
        tab.set_status(f"[red]✗ Error:[/red] {error}")
        logger.error(f"Ingestion failed: {error}")
        self.notify(f"Ingestion failed: {error}", severity="error")

    # =========================================================================
    # Pipeline: Synthesize
    # =========================================================================

    def _run_synthesize(
        self,
        input_path_str: str | None = None,
        output: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        concurrency: int | None = None,
        rpm: int | None = None,
        max_tokens: int | None = None,
        mode: str | None = None,
        job_id: str | None = None,
        judge: bool | None = None,
        cache_enabled: bool | None = None,
        checkpoint: bool | None = None,
        silent: bool = False,
    ) -> str | None:
        """Run the synthesis pipeline in a worker thread.

        Returns the output file path if successful, or None on error.
        """
        if not silent:
            synth_tab = self.query_one(SynthesizeTab)
            config = synth_tab.get_config()
        else:
            config = {
                "input": input_path_str or "./chunks.json",
                "output": output or "./conversations.json",
                "provider": provider or "openai",
                "model": model or "gpt-5-mini",
                "base_url": base_url or None,
                "concurrency": concurrency or 5,
                "rpm": rpm or 60,
                "max_tokens": max_tokens,
                "mode": mode or "default",
                "job_id": job_id or None,
                "judge": judge or False,
                "cache": cache_enabled if cache_enabled is not None else True,
                "checkpoint": checkpoint if checkpoint is not None else True,
            }

        if not silent:
            err = self.query_one(SynthesizeTab).validate_config()
            if err:
                self.notify(err, severity="warning")
                return None

        input_path = Path(config["input"])
        if not input_path.exists():
            if not silent:
                self.notify(f"Input not found: {config['input']}", severity="error")
            return None

        if not silent:
            self.query_one(SynthesizeTab).set_running(True)
            self._start_jobs_auto_refresh()

        self._notify_dashboard(f"Synthesizing with {config['provider']}/{config['model']}")

        import threading

        result_box: list[str | None] = [None]
        done_event = threading.Event()

        def do_synthesize() -> None:
            try:
                from ..core.cache import CacheManager as CMCache
                from ..core.checkpoint import CheckpointManager as CPCheckpoint
                from ..core.schemas import DataChunk
                from ..core.schemas import SynthesisConfig as SynthCfg
                from ..synthesis.pipeline import SynthesisPipeline as SynthPipe

                chunks_data = safe_json_load(input_path)
                chunks = [DataChunk(**chunk) for chunk in chunks_data]

                cache = None if not config["cache"] else CMCache(cache_dir=".cache")
                cp = None if not config["checkpoint"] else CPCheckpoint()

                # Resolve API key from environment variables (not exposed in TUI form)
                resolved_api_key: str | None = None
                provider_lower = config["provider"]
                provider_info = get_provider_info(provider_lower)
                if provider_info:
                    for env_var in provider_info.env_vars:
                        val = os.getenv(env_var)
                        if val:
                            resolved_api_key = val
                            break

                synth_config = SynthCfg(
                    llm_provider=config["provider"],
                    model_name=config["model"],
                    base_url=config["base_url"],
                    api_key=resolved_api_key,
                    max_concurrency=config["concurrency"],
                    max_rpm=config["rpm"],
                    max_tokens=config["max_tokens"],
                    enable_judge=config["judge"],
                    conversation_mode=config["mode"],
                )
                pipeline = SynthPipe(
                    config=synth_config,
                    cache_manager=cache,
                    checkpoint_manager=cp,
                    use_cache=config["cache"],
                )

                async def run() -> list[Any]:
                    def update_progress(current: int, total: int) -> None:
                        self.call_from_thread(self._notify_dashboard, f"Synthesizing {current}/{total}", current, total)
                        self.call_from_thread(self._update_synth_progress, current, total)

                    # All modes route through the pipeline (mode routing +
                    # scaffold + prune + judge, with cache/checkpoint support).
                    return await pipeline.synthesize_batch(
                        chunks,
                        update_progress,
                        job_id=config.get("job_id"),
                        resume=config.get("job_id") is not None,
                    )

                conversations = asyncio.run(run())

                output_path = Path(config["output"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                conv_data = [conv.model_dump() for conv in conversations]
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(conv_data, f, indent=2, ensure_ascii=False)

                success_rate = f"{len(conversations) / len(chunks) * 100:.1f}%" if chunks else "0%"
                summary = (
                    f"{len(conversations)} conversations from {len(chunks)} chunks "
                    f"({success_rate} success) → {output_path}"
                )

                cost_report = ""
                cost_value = 0.0
                cost_stats = pipeline.get_cost_stats()
                if cost_stats.num_requests > 0:
                    cost_report = f"\n{pipeline.get_cost_report()}"
                    cost_value = cost_stats.estimated_cost_usd

                result_box[0] = str(output_path)
                self.call_from_thread(self._notify_dashboard, "Synthesis complete", 100, 100)
                if not silent:
                    self.call_from_thread(self._synth_done, summary, cost_report, cost_value)
                logger.info(f"Synthesis complete: {summary}")
                if cost_report:
                    logger.info(f"Cost: {cost_report.replace(chr(10), ' | ')}")

                asyncio.run(pipeline.close())
                if cache:
                    cache.close()
            except Exception as e:
                result_box[0] = None
                if not silent:
                    self.call_from_thread(self._synth_error, str(e))
                else:
                    self.call_from_thread(logger.error, f"Synthesis failed: {e}")
            finally:
                done_event.set()

        self.run_worker(do_synthesize, thread=True)

        if silent:
            done_event.wait(timeout=3600)
            return result_box[0]
        return None

    def _update_synth_progress(self, current: int, total: int) -> None:
        """Update synthesis progress from worker thread."""
        try:
            tab = self.query_one(SynthesizeTab)
            bar = tab.query_one("#synth-progress", ProgressBar)
            bar.update(total=total, progress=current)
            tab.set_status(f"Processing {current}/{total} chunks")
        except Exception:
            pass

    def _synth_done(self, summary: str, cost_report: str, cost_value: float = 0.0) -> None:
        """Called when synthesis completes successfully."""
        tab = self.query_one(SynthesizeTab)
        tab.set_running(False)
        tab.set_status(f"[green]✓ Done[/green] — {summary}")
        logger.info(f"Synthesis complete: {summary}")
        self.notify("Synthesis complete!", severity="information")
        self._set_last_run(f"Synthesis: {summary}")

        if cost_report:
            try:
                log_view = self.query_one("#log-view", RichLog)
                log_view.write(f"[bold cyan]💰 Cost Report:[/bold cyan]\n{cost_report}")
            except Exception:
                pass

        # Update Dashboard cost stat box with cumulative total
        if cost_value > 0:
            self._total_cost_usd += cost_value
            try:
                dashboard = self.query_one(DashboardTab)
                cost_label = f"${self._total_cost_usd:.4f}"
                stat_html = f"[bold yellow]💰 Cost[/bold yellow]\n[white]{cost_label}[/white]"
                dashboard.query_one("#stat-cost", Static).update(stat_html)
            except Exception:
                pass

        with contextlib.suppress(Exception):
            self.query_one(DashboardTab).refresh_file_stats()

    def _synth_error(self, error: str) -> None:
        """Called when synthesis fails."""
        tab = self.query_one(SynthesizeTab)
        tab.set_running(False)
        tab.set_status(f"[red]✗ Error:[/red] {error}")
        logger.error(f"Synthesis failed: {error}")
        self.notify(f"Synthesis failed: {error}", severity="error")

    # =========================================================================
    # Pipeline: Export
    # =========================================================================

    def _run_export(
        self,
        input_path_str: str | None = None,
        output_dir: str | None = None,
        formats: list[str] | None = None,
        unsloth_model: str | None = None,
        generate_unsloth: bool | None = None,
        split: bool | None = None,
        card: bool | None = None,
        silent: bool = False,
    ) -> dict[str, Path] | None:
        """Run the export pipeline in a worker thread.

        Returns a dict of {format: file_path} if successful, or None on error.
        """
        if not silent:
            export_tab = self.query_one(ExportTab)
            config = export_tab.get_config()
        else:
            config = {
                "input": input_path_str or "./conversations.json",
                "output_dir": output_dir or "./output",
                "formats": formats or ["sharegpt"],
                "unsloth_model": unsloth_model or "Qwen/Qwen3-8B",
                "unsloth": generate_unsloth if generate_unsloth is not None else True,
                "split": split or False,
                "card": card or False,
            }

        if not silent:
            err = self.query_one(ExportTab).validate_config()
            if err:
                self.notify(err, severity="warning")
                return None

        input_path = Path(config["input"])
        if not input_path.exists():
            if not silent:
                self.notify(f"Input not found: {config['input']}", severity="error")
            return None

        if not config["formats"]:
            if not silent:
                self.notify("Please select at least one format", severity="warning")
            return None

        if not silent:
            self.query_one(ExportTab).set_running(True)

        self._notify_dashboard("Exporting...")

        import threading

        result_box: list[dict[str, Path] | None] = [None]
        done_event = threading.Event()

        def do_export() -> None:
            try:
                from ..core.schemas import ConversationSchema, ExportConfig
                from ..exporter.pipeline import ExportPipeline

                conv_data = safe_json_load(input_path)
                conversations = [ConversationSchema(**conv) for conv in conv_data]

                export_config = ExportConfig(
                    formats=config["formats"],
                    output_dir=config["output_dir"],
                    unsloth_model=config["unsloth_model"],
                )
                pipeline = ExportPipeline(export_config)

                output_files = pipeline.export(
                    conversations,
                    formats=config["formats"],
                    generate_unsloth=config["unsloth"],
                    split=config["split"],
                    generate_card=config["card"],
                )

                result_box[0] = output_files
                self.call_from_thread(self._notify_dashboard, "Export complete", 100, 100)
                if not silent:
                    self.call_from_thread(self._export_done, len(conversations), output_files)
                logger.info(f"Export complete: {len(conversations)} conversations -> {len(output_files)} files")
            except Exception as e:
                result_box[0] = None
                if not silent:
                    self.call_from_thread(self._export_error, str(e))
                else:
                    self.call_from_thread(logger.error, f"Export failed: {e}")
            finally:
                done_event.set()

        self.run_worker(do_export, thread=True)

        if silent:
            done_event.wait(timeout=600)
            return result_box[0]
        return None

    def _export_done(self, count: int, output_files: dict[str, Path]) -> None:
        """Called when export completes successfully."""
        tab = self.query_one(ExportTab)
        tab.set_running(False)
        tab.set_status(f"[green]✓ Done[/green] — exported {count} conversations")
        tab.show_results(output_files)
        self.notify(f"Export complete: {count} conversations", severity="information")

        lines = []
        for fmt, path in output_files.items():
            size = f"{path.stat().st_size / 1024:.1f} KB" if path.exists() else "N/A"
            lines.append(f"  {fmt}: {path} ({size})")
        summary = "\n".join(lines)
        self._set_last_run(f"Export: {len(output_files)} files")

        try:
            log_view = self.query_one("#log-view", RichLog)
            log_view.write(f"[bold green]📤 Export Results:[/bold green]\n{summary}")
        except Exception:
            pass

    def _export_error(self, error: str) -> None:
        """Called when export fails."""
        tab = self.query_one(ExportTab)
        tab.set_running(False)
        tab.set_status(f"[red]✗ Error:[/red] {error}")
        logger.error(f"Export failed: {error}")
        self.notify(f"Export failed: {error}", severity="error")

    # =========================================================================
    # Pipeline: Validate
    # =========================================================================

    def _run_validate(self) -> None:
        """Run dataset validation in a worker thread."""
        validate_tab = self.query_one(ValidateTab)
        err = validate_tab.validate_config()
        if err:
            self.notify(err, severity="warning")
            return

        config = validate_tab.get_config()
        input_path = Path(config["input"])
        if not input_path.exists():
            self.notify(f"Input not found: {config['input']}", severity="error")
            return

        validate_tab.set_running(True)
        self._notify_dashboard("Validating dataset")
        logger.info("Starting validation")

        def do_validate() -> None:
            try:
                from ..core.schemas import ConversationSchema
                from ..exporter.validator import DatasetValidator

                conv_data = safe_json_load(input_path)
                conversations = [ConversationSchema(**conv) for conv in conv_data]

                validator = DatasetValidator()
                if config["dedupe"]:
                    conversations = validator.deduplicate(conversations)

                report = validator.validate(conversations)
                report_text = report.summary()

                self.call_from_thread(self._validate_done, report, report_text)
            except Exception as e:
                self.call_from_thread(self._validate_error, str(e))

        self.run_worker(do_validate, thread=True)

    def _validate_done(self, report: Any, report_text: str) -> None:
        """Called when validation completes successfully."""
        tab = self.query_one(ValidateTab)
        tab.set_running(False)

        valid_str = "[green]✓ Valid[/green]" if report.is_valid else "[red]✗ Issues found[/red]"
        tab.set_status(f"[green]✓ Done[/green] — {valid_str}")
        logger.info("Validation complete")
        self.notify("Validation complete", severity="information")
        self._notify_dashboard("Validation complete", 100, 100)
        self._set_last_run(f"Validation: {'Valid' if report.is_valid else 'Issues found'}")

        try:
            table = tab.query_one("#validate-results", DataTable)
            table.clear()

            stats = getattr(report, "stats", None)
            issues = getattr(report, "issues", [])

            if stats:
                table.add_rows(
                    [
                        ("Total Conversations", str(getattr(stats, "total_conversations", 0))),
                        ("Total Turns", str(getattr(stats, "total_turns", 0))),
                        ("Avg Turns/Conversation", f"{getattr(stats, 'avg_turns_per_conversation', 0):.1f}"),
                        ("Avg Characters/Conversation", f"{getattr(stats, 'avg_characters_per_conversation', 0):.0f}"),
                        ("Duplicate Count", str(getattr(stats, "duplicate_count", 0))),
                        ("Unique Count", str(getattr(stats, "unique_count", 0))),
                    ]
                )

            error_count = sum(1 for i in issues if getattr(i, "severity", "") == "error")
            warn_count = sum(1 for i in issues if getattr(i, "severity", "") == "warning")
            table.add_row("Errors", str(error_count))
            table.add_row("Warnings", str(warn_count))

            log_view = self.query_one("#log-view", RichLog)
            log_view.write(f"[bold yellow]🔍 Validation Report:[/bold yellow]\n{report_text}")
        except Exception as e:
            logger.error(f"Failed to populate validation results: {e}")

    def _validate_error(self, error: str) -> None:
        """Called when validation fails."""
        tab = self.query_one(ValidateTab)
        tab.set_running(False)
        tab.set_status(f"[red]✗ Error:[/red] {error}")
        logger.error(f"Validation failed: {error}")
        self.notify(f"Validation failed: {error}", severity="error")

    def _run_evaluate(self) -> None:
        """Run the quality-gate evaluation in a worker thread (no LLM calls)."""
        validate_tab = self.query_one(ValidateTab)
        err = validate_tab.validate_config()
        if err:
            self.notify(err, severity="warning")
            return

        config = validate_tab.get_eval_config()
        input_path = Path(config["input"])
        if not input_path.exists():
            self.notify(f"Input not found: {config['input']}", severity="error")
            return

        with contextlib.suppress(Exception):
            self.query_one("#evaluate-status", Label).update("[yellow]Running...[/yellow]")
        self._notify_dashboard("Evaluating dataset")
        logger.info("Starting evaluation gate")

        def do_evaluate() -> None:
            try:
                from ..core.schemas import ConversationSchema
                from ..synthesis.eval import EvalThresholds, evaluate_conversations

                conv_data = safe_json_load(input_path)
                conversations = [ConversationSchema(**conv) for conv in conv_data]

                report = evaluate_conversations(
                    conversations,
                    thresholds=EvalThresholds(
                        min_mean_confidence=config["threshold"],
                        min_valid_rate=config["min_valid_rate"],
                    ),
                )
                self.call_from_thread(self._evaluate_done, report)
            except Exception as e:
                self.call_from_thread(self._evaluate_error, str(e))

        self.run_worker(do_evaluate, thread=True)

    def _evaluate_done(self, report: Any) -> None:
        """Called when evaluation completes successfully."""
        gate = "[green]✓ PASS[/green]" if report.passed else "[red]✗ FAIL[/red]"
        with contextlib.suppress(Exception):
            self.query_one("#evaluate-status", Label).update(
                f"{gate} — valid {report.valid_rate:.0%}, confidence {report.mean_confidence:.2f}"
            )
        try:
            table = self.query_one("#validate-results", DataTable)
            table.clear()
            table.add_rows(
                [
                    ("Eval Gate", "PASS" if report.passed else "FAIL"),
                    ("Valid Rate", f"{report.valid_rate:.1%}"),
                    ("Mean Confidence", f"{report.mean_confidence:.3f}"),
                    ("Contamination", f"{report.contamination_rate:.1%}"),
                    ("Failures", "; ".join(report.failures) if report.failures else "none"),
                ]
            )
            log_view = self.query_one("#log-view", RichLog)
            log_view.write(f"[bold magenta]📊 Evaluation Report:[/bold magenta]\n{report.summary()}")
        except Exception as e:
            logger.error(f"Failed to populate evaluation results: {e}")
        self.notify(
            f"Evaluation {'passed' if report.passed else 'failed'}",
            severity="information" if report.passed else "warning",
        )
        self._set_last_run(f"Evaluation: {'PASS' if report.passed else 'FAIL'}")

    def _evaluate_error(self, error: str) -> None:
        """Called when evaluation fails."""
        with contextlib.suppress(Exception):
            self.query_one("#evaluate-status", Label).update(f"[red]✗ Error:[/red] {error}")
        logger.error(f"Evaluation failed: {error}")
        self.notify(f"Evaluation failed: {error}", severity="error")

    # =========================================================================
    # Full Pipeline (Ingest → Synthesize → Export)
    # =========================================================================

    def _run_full_pipeline(self) -> None:
        """Run the full pipeline: ingest → synthesize → export in sequence."""
        full_tab = self.query_one(FullPipelineTab)
        err = full_tab.validate_config()
        if err:
            self.notify(err, severity="warning")
            return

        config = full_tab.get_config()
        full_tab.set_running(True)

        log_view = self.query_one("#log-view", RichLog)
        log_view.write("[bold cyan]⚡ Full Pipeline started — Ingest → Synthesize → Export[/bold cyan]")

        logger.info("=== Full Pipeline START ===")

        # Derive intermediate paths
        chunks_path = "./.full_pipeline_chunks.json"
        conv_path = "./.full_pipeline_conversations.json"

        # Step 1: Ingest
        self._notify_dashboard("Full: Ingesting")
        full_tab.set_status("[yellow]Step 1/3: Ingesting files...[/yellow]")
        logger.info("Full Pipeline — Step 1: Ingesting")

        ingest_result = self._run_ingest(
            source=config["source"],
            output=chunks_path,
            chunk_size=config["chunk_size"],
            overlap=config["overlap"],
            recursive=config["recursive"],
            auto_detect=config["auto_detect"],
            chunker=config["chunker"],
            silent=True,
        )

        if not ingest_result:
            full_tab.set_running(False)
            full_tab.set_status("[red]✗ Full Pipeline failed at Ingest stage[/red]")
            logger.error("Full Pipeline — Ingest step failed")
            return

        # Step 2: Synthesize
        self._notify_dashboard("Full: Synthesizing")
        full_tab.set_status("[yellow]Step 2/3: Synthesizing conversations...[/yellow]")
        logger.info("Full Pipeline — Step 2: Synthesizing")

        synth_result = self._run_synthesize(
            input_path_str=chunks_path,
            output=conv_path,
            provider=config["provider"],
            model=config["model"],
            concurrency=config["concurrency"],
            rpm=config["rpm"],
            mode=config["mode"],
            cache_enabled=config["cache"],
            checkpoint=config["checkpoint"],
            silent=True,
        )

        if not synth_result:
            full_tab.set_running(False)
            full_tab.set_status("[red]✗ Full Pipeline failed at Synthesize stage[/red]")
            logger.error("Full Pipeline — Synthesize step failed")
            # Cleanup chunks
            with contextlib.suppress(Exception):
                Path(chunks_path).unlink(missing_ok=True)
            return

        # Step 3: Export
        self._notify_dashboard("Full: Exporting")
        full_tab.set_status("[yellow]Step 3/3: Exporting datasets...[/yellow]")
        logger.info("Full Pipeline — Step 3: Exporting")

        export_result = self._run_export(
            input_path_str=conv_path,
            output_dir=config["output_dir"],
            formats=[config["format"]],
            generate_unsloth=False,
            card=config["card"],
            silent=True,
        )

        # Cleanup intermediate files
        try:
            Path(chunks_path).unlink(missing_ok=True)
            Path(conv_path).unlink(missing_ok=True)
        except Exception:
            pass

        if not export_result:
            full_tab.set_running(False)
            full_tab.set_status("[red]✗ Full Pipeline failed at Export stage[/red]")
            logger.error("Full Pipeline — Export step failed")
            return

        # Success!
        self._notify_dashboard("Full Pipeline complete", 100, 100)
        full_tab.set_running(False)
        full_tab.set_status(f"[green]✓ Full Pipeline complete![/green] Output: {config['output_dir']}")
        self._set_last_run(f"Full Pipeline: {config['source']} → {config['output_dir']}")

        # Show export results
        for fmt, path in export_result.items():
            size = f"{path.stat().st_size / 1024:.1f} KB" if path.exists() else "N/A"
            log_view.write(f"  [green]{fmt}[/green]: {path} ({size})")

        log_view.write("[bold cyan]⚡ Full Pipeline finished successfully![/bold cyan]")
        logger.info("Full Pipeline — All stages complete!")

        with contextlib.suppress(Exception):
            self.query_one(DashboardTab).refresh_file_stats()

        self.notify("Full Pipeline complete!", severity="information")


if __name__ == "__main__":
    app = DistillAlignApp()
    app.run()

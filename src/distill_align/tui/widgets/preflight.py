"""Pre-flight check system for the Distill-Align TUI.

Validates environment, files, and configuration before running a pipeline,
catching common issues early and showing clear guidance on how to fix them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ...synthesis.models.registry import get as get_provider_info


def _valid_modes() -> set[str]:
    """Accepted conversation modes, derived from the pipeline's enum."""
    try:
        from ..options import valid_conversation_modes

        return valid_conversation_modes()
    except Exception:
        return {"default", "qa", "teach", "debug", "review", "explain"}


def _provider_env_vars(provider: str) -> list[str]:
    """Return the env-var list for *provider*, or empty list if unknown."""
    info = get_provider_info(provider)
    return info.env_vars if info else []


def _provider_label(provider: str) -> str:
    """Return a human-readable label for *provider*."""
    info = get_provider_info(provider)
    return info.label if info else provider.title()


def _provider_concurrency_limit(provider: str) -> int:
    """Return the recommended concurrency ceiling for *provider*."""
    info = get_provider_info(provider)
    return info.concurrency_limit if info else 10


def _build_api_key_fix(provider: str) -> str:
    """Build a helpful fix message for a missing API key."""
    info = get_provider_info(provider)
    if info is None:
        return f"Set the appropriate API key environment variable for {provider}."
    if not info.env_vars:
        return f"No API key required for {info.label} (local)."
    var_names = " or ".join(f"${v}" for v in info.env_vars)
    return f"Set:  export {var_names}"


# ── Check result types ──────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """Result of a single pre-flight check."""

    name: str
    status: Literal["pass", "warn", "fail"]  # noqa: F821 - used with from __future__ import annotations
    message: str
    fix: str = ""


# ── Pre-flight report ────────────────────────────────────────────────────────


@dataclass
class PreflightReport:
    """Aggregated results from all pre-flight checks."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == "pass")

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if c.status == "warn")

    @property
    def failures(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def can_proceed(self) -> bool:
        """Allow proceeding if there are no failures (warnings are OK)."""
        return self.failures == 0

    @property
    def has_issues(self) -> bool:
        return self.failures > 0 or self.warnings > 0


def _check_source_exists(source: str) -> CheckResult:
    """Check that the source path exists."""
    if not source:
        return CheckResult(
            name="Source path",
            status="fail",
            message="Source path is empty",
            fix="Enter the folder or file containing your data.",
        )
    path = Path(source)
    if path.exists():
        return CheckResult(
            name="Source path",
            status="pass",
            message=f"Found: {path.resolve()}",
        )
    return CheckResult(
        name="Source path",
        status="fail",
        message=f"Not found: {source}",
        fix="Check the path. You can use an absolute path or a relative path from the working directory.",
    )


def _check_input_file(path_str: str, label: str = "Input file") -> CheckResult:
    """Check that an input file exists."""
    if not path_str:
        return CheckResult(
            name=label,
            status="fail",
            message=f"{label} path is empty",
            fix="Enter the path to your input file.",
        )
    path = Path(path_str)
    if path.exists():
        size_kb = path.stat().st_size / 1024
        return CheckResult(
            name=label,
            status="pass",
            message=f"Found ({size_kb:.1f} KB)",
        )
    return CheckResult(
        name=label,
        status="fail",
        message=f"Not found: {path_str}",
        fix="Run the previous pipeline step first, or check the path.",
    )


def _check_output_dir(path_str: str) -> CheckResult:
    """Check that the output directory exists or can be created."""
    if not path_str:
        return CheckResult(
            name="Output directory",
            status="warn",
            message="No output directory specified, using default",
            fix="An output directory name is recommended.",
        )
    path = Path(path_str)
    if path.exists():
        if path.is_dir():
            return CheckResult(
                name="Output directory",
                status="pass",
                message=f"Exists: {path.resolve()}",
            )
        return CheckResult(
            name="Output directory",
            status="warn",
            message=f"Path exists but is not a directory: {path_str}",
            fix="Specify a directory path, not a file path.",
        )
    # Directory doesn't exist — try parent
    parent = path.parent
    if parent.exists():
        return CheckResult(
            name="Output directory",
            status="pass",
            message=f"Will be created at: {path.resolve()}",
        )
    return CheckResult(
        name="Output directory",
        status="fail",
        message=f"Parent directory does not exist: {parent}",
        fix="Create the parent directory first, or change the output path.",
    )


def _check_api_key(provider: str) -> CheckResult:
    """Check that the required API key environment variable is set."""
    env_vars = _provider_env_vars(provider)
    provider_name = _provider_label(provider)

    if not env_vars:
        return CheckResult(
            name=f"{provider_name} API key",
            status="pass",
            message=f"No API key required for {provider_name} (local)",
        )

    # Check each possible env var
    found_vars = [v for v in env_vars if os.getenv(v)]
    if found_vars:
        return CheckResult(
            name=f"{provider_name} API key",
            status="pass",
            message=f"Found via ${found_vars[0]}",
        )

    # Not found — build helpful error
    var_names = " or ".join(f"${v}" for v in env_vars)
    return CheckResult(
        name=f"{provider_name} API key",
        status="fail",
        message=f"Not set ({var_names})",
        fix=_build_api_key_fix(provider),
    )


def _check_chunk_size(chunk_size: int) -> CheckResult:
    """Check that chunk size is reasonable."""
    if chunk_size < 100:
        return CheckResult(
            name="Chunk size",
            status="warn",
            message=f"Very small ({chunk_size} tokens) — may produce many tiny chunks",
            fix="Use 500–2000 for most content, or 2000–4000 for code.",
        )
    if chunk_size > 8000:
        return CheckResult(
            name="Chunk size",
            status="warn",
            message=f"Large ({chunk_size} tokens) — may exceed context limits",
            fix="Use 500–2000 for most content. Large chunks may hit model token limits.",
        )
    return CheckResult(
        name="Chunk size",
        status="pass",
        message=f"{chunk_size} tokens",
    )


def _check_concurrency(concurrency: int, provider: str) -> CheckResult:
    """Check that concurrency is reasonable for the provider."""
    if concurrency < 1:
        return CheckResult(
            name="Concurrency",
            status="fail",
            message=f"Invalid: {concurrency}",
            fix="Concurrency must be at least 1.",
        )

    max_rec = _provider_concurrency_limit(provider)

    if concurrency > max_rec:
        provider_name = _provider_label(provider)
        return CheckResult(
            name="Concurrency",
            status="warn",
            message=f"{concurrency} (recommended max for {provider_name}: {max_rec})",
            fix=f"Reduce concurrency to {max_rec} or less to avoid rate limits.",
        )
    return CheckResult(
        name="Concurrency",
        status="pass",
        message=f"{concurrency} workers",
    )


def _check_chunks_nonempty(path_str: str) -> CheckResult:
    """Check that a chunks/conversations file has data."""
    if not path_str:
        return CheckResult(name="Data file", status="warn", message="No path specified", fix="")
    path = Path(path_str)
    if not path.exists():
        return CheckResult(
            name="Data file",
            status="warn",
            message=f"Not yet created: {path_str}",
            fix="",
        )
    # Try to check if file has content
    try:
        from ...core.json_utils import safe_json_load

        data = safe_json_load(path)
        count = len(data) if isinstance(data, list) else 1
        if count == 0:
            return CheckResult(
                name="Data file",
                status="fail",
                message=f"{path_str} is empty (0 items)",
                fix="Re-run the previous pipeline step with valid data.",
            )
        return CheckResult(
            name="Data file",
            status="pass",
            message=f"{count} items in {path_str}",
        )
    except Exception:
        return CheckResult(
            name="Data file",
            status="warn",
            message=f"Could not read: {path_str}",
            fix="",
        )


# ── Specific check suites ────────────────────────────────────────────────────


def preflight_ingest(source: str, output: str, chunk_size: int) -> PreflightReport:
    """Run pre-flight checks for the ingestion pipeline."""
    report = PreflightReport()
    report.checks.append(_check_source_exists(source))
    report.checks.append(_check_output_dir(str(Path(output).parent)))
    report.checks.append(_check_chunk_size(chunk_size))
    return report


def preflight_synthesize(
    input_path: str,
    output: str,
    provider: str,
    concurrency: int,
    mode: str | None = None,
) -> PreflightReport:
    """Run pre-flight checks for the synthesis pipeline."""
    report = PreflightReport()
    report.checks.append(_check_input_file(input_path, "Chunks file"))
    report.checks.append(_check_chunks_nonempty(input_path))
    report.checks.append(_check_output_dir(str(Path(output).parent)))
    report.checks.append(_check_api_key(provider))
    report.checks.append(_check_concurrency(concurrency, provider))

    if mode and mode not in _valid_modes():
        report.checks.append(CheckResult(name="Mode", status="warn", message=f"Unknown mode: {mode}", fix=""))

    return report


def preflight_export(
    input_path: str,
    output_dir: str,
    formats: list[str],
) -> PreflightReport:
    """Run pre-flight checks for the export pipeline."""
    report = PreflightReport()
    report.checks.append(_check_input_file(input_path, "Conversations file"))
    report.checks.append(_check_chunks_nonempty(input_path))
    report.checks.append(_check_output_dir(output_dir))
    if not formats:
        report.checks.append(
            CheckResult(
                name="Export formats",
                status="fail",
                message="No formats selected",
                fix="Select at least one export format (e.g., ShareGPT).",
            )
        )
    else:
        report.checks.append(
            CheckResult(name="Export formats", status="pass", message=f"{len(formats)} format(s) selected")
        )
    return report


def preflight_full_pipeline(
    source: str,
    provider: str,
    model: str,
    output_dir: str,
    chunk_size: int = 1000,
    concurrency: int = 5,
    formats: list[str] | None = None,
) -> PreflightReport:
    """Run pre-flight checks for the full pipeline."""
    report = PreflightReport()
    report.checks.append(_check_source_exists(source))
    report.checks.append(_check_api_key(provider))
    report.checks.append(_check_chunk_size(chunk_size))
    report.checks.append(_check_concurrency(concurrency, provider))
    report.checks.append(_check_output_dir(output_dir))

    if not model:
        report.checks.append(
            CheckResult(name="Model", status="fail", message="Model name is empty", fix="Enter a model name.")
        )
    else:
        report.checks.append(CheckResult(name="Model", status="pass", message=model))

    if formats:
        report.checks.append(CheckResult(name="Export formats", status="pass", message=f"{len(formats)} format(s)"))
    else:
        report.checks.append(CheckResult(name="Export formats", status="pass", message="Will use default (sharegpt)"))

    return report


# ── Pre-flight screen ────────────────────────────────────────────────────────


class PreflightScreen(ModalScreen[bool]):
    """Modal screen showing pre-flight check results.

    Returns True if the user wants to proceed, False otherwise.
    """

    def __init__(self, report: PreflightReport, pipeline_name: str = "Pipeline") -> None:
        super().__init__()
        self._report = report
        self._pipeline_name = pipeline_name

    CSS = """
    PreflightScreen {
        align: center middle;
    }

    #preflight-box {
        width: 68;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: thick $accent;
        background: $background;
    }

    #preflight-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #preflight-summary {
        margin-bottom: 1;
        padding: 0 1;
    }

    .check-row {
        height: auto;
        margin: 0 0 0 0;
        padding: 0 1;
    }

    .check-name {
        width: 28;
        text-style: bold;
    }

    .check-message {
        width: 1fr;
    }

    .check-fix {
        color: $text-muted;
        padding-left: 3;
        margin-bottom: 1;
    }

    .pass-mark {
        color: green;
    }

    .warn-mark {
        color: yellow;
    }

    .fail-mark {
        color: red;
    }

    #preflight-actions {
        margin-top: 1;
        align: center middle;
    }

    #preflight-actions Button {
        margin: 0 1;
        min-width: 20;
    }
    """

    def compose(self) -> ComposeResult:
        report = self._report
        emoji = "✅" if report.can_proceed else "❌"
        yield Static(f"{emoji} Pre-Flight Check: {self._pipeline_name}", id="preflight-title")

        # Summary line
        parts = []
        if report.passed:
            parts.append(f"[green]{report.passed} passed[/green]")
        if report.warnings:
            parts.append(f"[yellow]{report.warnings} warnings[/yellow]")
        if report.failures:
            parts.append(f"[red]{report.failures} failed[/red]")
        yield Static("  ".join(parts), id="preflight-summary")

        # Individual checks
        for check in report.checks:
            if check.status == "pass":
                icon = "[green]✓[/green]"
            elif check.status == "warn":
                icon = "[yellow]⚠[/yellow]"
            else:
                icon = "[red]✗[/red]"

            with Vertical(classes="check-row"):
                with Horizontal():
                    yield Static(f"{icon}", classes="check-name")
                    yield Static(f"[bold]{check.name}[/bold]", classes="check-name")
                    yield Static(check.message, classes="check-message")
                if check.fix:
                    yield Static(f"  {check.fix}", classes="check-fix")

        with Horizontal(id="preflight-actions"):
            if report.can_proceed:
                yield Button("🚀 Run Anyway", id="preflight-proceed", variant="primary")
            else:
                yield Button("🚀 Force Run", id="preflight-proceed", variant="error")
            yield Button("✋ Fix and Retry", id="preflight-retry", variant="default")
            yield Button("Cancel", id="preflight-cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "preflight-proceed":
            self.dismiss(True)
        else:
            self.dismiss(False)

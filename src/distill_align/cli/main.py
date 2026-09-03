"""
CLI entry point for Distill-Align.

Uses Typer for command-line interface with Rich for beautiful output.

Commands:
- ingest: Load and chunk files
- synthesize: Generate conversations from chunks
- export: Format and export to training datasets
- validate: Validate and analyze a dataset
- init: Initialize a new project with config file
- status: Check configuration and connections
- jobs: Manage synthesis jobs (list, resume, cancel)
- config: View/edit configuration
- tui: Launch the interactive dashboard
- version: Show version
"""

import asyncio
from pathlib import Path
from typing import Literal, cast

import typer  # type: ignore[import-not-found]
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .. import __version__
from ..core.config_file import find_config_file, generate_default_config, load_config
from ..core.logging import setup_logging
from ..core.schemas import ExportConfig, IngestionConfig, SynthesisConfig
from ..core.update_checker import check_pypi_version

# Main Typer app
app = typer.Typer(
    name="distill-align",
    help="The Structured Reasoning Extraction Factory",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


# Subcommand groups
init_app = typer.Typer(help="Initialize a new project")
jobs_app = typer.Typer(help="Manage synthesis jobs")
config_app = typer.Typer(help="Configuration management")
app.add_typer(init_app, name="init")
app.add_typer(jobs_app, name="jobs")
app.add_typer(config_app, name="config")


@app.callback()
def main(
    ctx: typer.Context,
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    log_file: str | None = typer.Option(None, "--log-file", help="Log file path"),
    log_format: str = typer.Option("text", "--log-format", help="Log format: text or json"),
    config_file: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Distill-Align: Generate fine-tuning datasets from raw domain data."""
    setup_logging(log_level=log_level, log_file=log_file, log_format=log_format)
    ctx.obj = {"config_file": config_file}

    # Async-free PyPI update check (fast, silent on failure)
    try:
        latest = check_pypi_version()
        if latest:
            console.print(
                f"[yellow]⚠ Update available: v{__version__} → v{latest}. "
                "Run: pip install --upgrade distill-align[/yellow]"
            )
    except Exception:
        pass  # never crash the CLI

    # Load custom providers from config file (if any)
    try:
        from ..core.config_file import find_config_file, load_config

        cfg_path = find_config_file()
        if cfg_path:
            load_config(cfg_path)  # load_config internally registers custom providers
    except Exception:
        pass  # Non-fatal


@app.command()
def ingest(
    ctx: typer.Context,
    source: str = typer.Argument(..., help="Source file or directory path"),
    output: str = typer.Option("./chunks.json", "--output", "-o", help="Output file path"),
    chunk_size: int = typer.Option(1000, "--chunk-size", "-s", help="Chunk size in characters"),
    chunk_overlap: int = typer.Option(200, "--overlap", help="Chunk overlap in characters"),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", "-r", flag_value=True, help="Search subdirectories"
    ),
    auto_detect: bool = typer.Option(True, "--auto/--no-auto", flag_value=True, help="Auto-detect file types"),
):
    """Ingest files and split into semantic chunks."""
    from ..ingestion.auto import AutoIngestionPipeline
    from ..ingestion.pipeline import IngestionPipeline

    console.print(Panel.fit("📥 Ingestion Pipeline", style="bold blue"))

    config = IngestionConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    source_path = Path(source)
    if not source_path.exists():
        console.print(f"[red]Error: Source path does not exist: {source}[/red]")
        raise typer.Exit(1)

    pipeline: AutoIngestionPipeline | IngestionPipeline = (
        AutoIngestionPipeline(config) if auto_detect else IngestionPipeline(config)
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting files...", total=None)

        if source_path.is_file():
            chunks = pipeline.ingest_file(source_path)
        else:
            if auto_detect and hasattr(pipeline, "ingest_directory"):
                # AutoIngestionPipeline supports progress callback
                def progress_cb(current, total, name):
                    progress.update(
                        task, description=f"Processing {name} ({current}/{total})", completed=current, total=total
                    )

                chunks = pipeline.ingest_directory(source_path, recursive=recursive, progress_callback=progress_cb)  # type: ignore[call-arg]
            else:
                chunks = pipeline.ingest_directory(source_path, recursive=recursive)

        progress.update(task, completed=True)

    import json

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks_data = [chunk.model_dump() for chunk in chunks]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    table = Table(title="Ingestion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Chunks", str(len(chunks)))
    table.add_row("Output File", str(output_path))
    table.add_row("File Size", f"{output_path.stat().st_size / 1024:.1f} KB")
    console.print(table)


@app.command()
def synthesize(
    ctx: typer.Context,
    input: str = typer.Argument(..., help="Input chunks JSON file"),
    output: str = typer.Option("./conversations.json", "--output", "-o", help="Output file path"),
    provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider"),
    model: str = typer.Option("gpt-5-mini", "--model", "-m", help="Model name"),
    base_url: str | None = typer.Option(None, "--base-url", help="API base URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="API key (or use env var)"),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Max concurrent requests"),
    rpm: int = typer.Option(60, "--rpm", help="Max requests per minute"),
    job_id: str | None = typer.Option(None, "--job-id", help="Job ID for resume support"),
    resume: bool = typer.Option(False, "--resume", flag_value=True, help="Resume a previous job"),
    no_cache: bool = typer.Option(False, "--no-cache", flag_value=True, help="Disable caching"),
    no_checkpoint: bool = typer.Option(False, "--no-checkpoint", flag_value=True, help="Disable checkpointing"),
    prompt_dir: str | None = typer.Option(None, "--prompts", help="Custom prompts directory"),
    mode: str = typer.Option(
        "default",
        "--mode",
        help="Conversation mode: default, teach, debug, review, qa, explain, evol_instruct, rag_qa, tool_call, constitutional, distill",
    ),
    judge: bool = typer.Option(False, "--judge", flag_value=True, help="Enable LLM-as-judge evaluation"),
    judge_model: str | None = typer.Option(None, "--judge-model", help="Model for judge (defaults to --model)"),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Max tokens to generate per LLM call (prevents server cutoff)"
    ),
):
    """Synthesize chunks into structured conversations."""
    from ..core.cache import CacheManager
    from ..core.checkpoint import CheckpointManager
    from ..core.schemas import DataChunk
    from ..synthesis.conversation_builder import ConversationMode
    from ..synthesis.pipeline import SynthesisPipeline

    console.print(Panel.fit("🧠 Synthesis Pipeline", style="bold magenta"))

    import json

    from ..core.json_utils import safe_json_load

    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file does not exist: {input}[/red]")
        raise typer.Exit(1)

    chunks_data = safe_json_load(input_path)
    chunks = [DataChunk(**chunk) for chunk in chunks_data]

    # Security: deprecate --api-key in favor of environment variables
    if api_key:
        console.print(
            "[yellow]⚠️  WARNING: --api-key exposes your secret in process listings. "
            "Use the OPENAI_API_KEY or DISTILL_LLM_API_KEY environment variable instead. "
            "This flag will be removed in a future version.[/yellow]"
        )
        import os

        os.environ.setdefault("OPENAI_API_KEY", api_key)

    # Setup cache
    cache = None if no_cache else CacheManager(cache_dir=".cache")
    checkpoint = None if no_checkpoint else CheckpointManager()

    config = SynthesisConfig(
        llm_provider=provider,
        model_name=model,
        base_url=base_url,
        api_key=api_key,
        max_concurrency=concurrency,
        max_rpm=rpm,
        enable_judge=judge,
        judge_model=judge_model,
        max_tokens=max_tokens,
        conversation_mode=mode,  # type: ignore[arg-type]
    )
    pipeline = SynthesisPipeline(
        config=config,
        cache_manager=cache,
        checkpoint_manager=checkpoint,
        use_cache=not no_cache,
    )

    # All modes (including default) route through the pipeline, which applies
    # mode routing + scaffold + prune + judge with cache/checkpoint support.
    try:
        ConversationMode(mode)
    except ValueError:
        console.print(f"[red]Error: Unknown mode: {mode}[/red]")
        raise typer.Exit(1) from None

    async def run_synthesis():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Synthesizing {len(chunks)} chunks...", total=len(chunks))

            def update_progress(current, total):
                progress.update(task, completed=current)

            conversations = await pipeline.synthesize_batch(chunks, update_progress, job_id=job_id, resume=resume)
            return conversations

    try:
        conversations = asyncio.run(run_synthesis())

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        conv_data = [conv.model_dump() for conv in conversations]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(conv_data, f, indent=2, ensure_ascii=False)

        table = Table(title="Synthesis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Input Chunks", str(len(chunks)))
        table.add_row("Conversations", str(len(conversations)))
        table.add_row("Success Rate", f"{len(conversations) / len(chunks) * 100:.1f}%" if chunks else "0%")
        table.add_row("Output File", str(output_path))

        if cache:
            cache_stats = cache.stats()
            table.add_row("Cache Hit Rate", f"{cache_stats.hit_rate:.1%}")
            table.add_row("Cache Entries", str(cache_stats.total_entries))

        console.print(table)

        # Show cost report if any requests were made
        cost_stats = pipeline.get_cost_stats()
        if cost_stats.num_requests > 0:
            cost_panel = Panel(
                pipeline.get_cost_report(),
                title="💰 Cost Summary",
                style="bold cyan",
            )
            console.print(cost_panel)
    finally:
        # Cleanup HTTP connections to prevent leaks
        asyncio.run(pipeline.close())
        if cache:
            cache.close()


@app.command()
def export(
    ctx: typer.Context,
    input: str = typer.Argument(..., help="Input conversations JSON file"),
    formats: str = typer.Option("sharegpt", "--format", "-f", help="Export formats (comma-separated)"),
    output_dir: str = typer.Option("./output", "--output-dir", "-o", help="Output directory"),
    model_name: str = typer.Option("Qwen/Qwen3-8B", "--model", help="Unsloth model name"),
    no_unsloth: bool = typer.Option(False, "--no-unsloth", flag_value=True, help="Skip Unsloth script generation"),
    split: bool = typer.Option(False, "--split", flag_value=True, help="Split into train/val/test"),
    card: bool = typer.Option(False, "--card", flag_value=True, help="Generate dataset card"),
):
    """Export conversations to training formats."""
    from ..core.schemas import ConversationSchema
    from ..exporter.pipeline import ExportPipeline

    console.print(Panel.fit("📤 Export Pipeline", style="bold green"))

    from ..core.json_utils import safe_json_load

    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file does not exist: {input}[/red]")
        raise typer.Exit(1)

    conv_data = safe_json_load(input_path)
    conversations = [ConversationSchema(**conv) for conv in conv_data]

    format_list = [f.strip() for f in formats.split(",")]
    config = ExportConfig(
        formats=cast(list[Literal["sharegpt", "alpaca", "chatml", "conversation", "hf_messages"]], format_list),
        output_dir=output_dir,
        unsloth_model=model_name,
    )
    pipeline = ExportPipeline(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Exporting...", total=None)
        output_files = pipeline.export(
            conversations,
            formats=format_list,
            generate_unsloth=not no_unsloth,
            split=split,
            generate_card=card,
        )
        progress.update(task, completed=True)

    table = Table(title="Export Summary")
    table.add_column("Format", style="cyan")
    table.add_column("File", style="green")
    table.add_column("Size", style="yellow")
    for format_name, file_path in output_files.items():
        size = f"{file_path.stat().st_size / 1024:.1f} KB" if file_path.exists() else "N/A"
        table.add_row(format_name, str(file_path), size)
    console.print(table)


@app.command()
def validate(
    input: str = typer.Argument(..., help="Input conversations JSON file"),
    dedupe: bool = typer.Option(True, "--dedupe/--no-dedupe", flag_value=True, help="Remove duplicates"),
    output: str | None = typer.Option(None, "--output", "-o", help="Save report to file"),
):
    """Validate and analyze a dataset."""
    from ..core.schemas import ConversationSchema
    from ..exporter.validator import DatasetValidator

    console.print(Panel.fit("🔍 Dataset Validation", style="bold yellow"))

    from ..core.json_utils import safe_json_load

    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file does not exist: {input}[/red]")
        raise typer.Exit(1)

    conv_data = safe_json_load(input_path)
    conversations = [ConversationSchema(**conv) for conv in conv_data]

    validator = DatasetValidator()
    if dedupe:
        conversations = validator.deduplicate(conversations)

    report = validator.validate(conversations)
    console.print(report.summary())

    if output:
        report_path = Path(output)
        report_path.write_text(report.summary(), encoding="utf-8")
        console.print(f"\n[green]Report saved to {report_path}[/green]")

    if not report.is_valid:
        raise typer.Exit(1)


@app.command()
def evaluate(
    input: str = typer.Argument(..., help="Input conversations JSON file"),
    threshold: float = typer.Option(0.5, "--threshold", help="Min mean confidence (0-1)"),
    min_valid_rate: float = typer.Option(0.8, "--min-valid-rate", help="Min valid-turn rate (0-1)"),
    reference: str | None = typer.Option(
        None, "--reference", help="Optional eval-set text file for contamination check"
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Save report to file"),
):
    """Evaluate dataset quality (heuristics + stored judge scores, no LLM calls)."""
    from ..core.schemas import ConversationSchema
    from ..synthesis.eval import EvalThresholds, evaluate_conversations

    console.print(Panel.fit("📊 Dataset Evaluation", style="bold magenta"))

    from ..core.json_utils import safe_json_load

    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file does not exist: {input}[/red]")
        raise typer.Exit(1)

    conv_data = safe_json_load(input_path)
    conversations = [ConversationSchema(**conv) for conv in conv_data]

    refs: list[str] | None = None
    if reference:
        ref_path = Path(reference)
        if ref_path.exists():
            refs = [ref_path.read_text(encoding="utf-8")]

    report = evaluate_conversations(
        conversations,
        reference_texts=refs,
        thresholds=EvalThresholds(min_mean_confidence=threshold, min_valid_rate=min_valid_rate),
    )
    console.print(report.summary())

    if output:
        out = Path(output)
        out.write_text(report.summary(), encoding="utf-8")
        console.print(f"\n[green]Report saved to {out}[/green]")

    if not report.passed:
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
):
    """Serve the Distill-Align REST API (requires the serve extra)."""
    try:
        from ..serve import create_app
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    try:
        import uvicorn  # type: ignore[import-not-found]
    except ImportError as e:
        console.print("[red]uvicorn is not installed. Install with: pip install distill-align[serve][/red]")
        raise typer.Exit(1) from e
    console.print(Panel.fit(f"🚀 Serving Distill-Align on http://{host}:{port}", style="bold green"))
    uvicorn.run(create_app(), host=host, port=port)


@app.command()
def status():
    """Check configuration and connections."""
    from .. import __version__
    from ..core.config_file import find_config_file

    console.print(Panel.fit(f"🩺 Distill-Align Status v{__version__}", style="bold green"))

    table = Table(title="System Status")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")

    # Check Python version
    import sys

    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    table.add_row("Python Version", py_version)

    # Check config file
    config_path = find_config_file()
    table.add_row("Config File", str(config_path) if config_path else "[yellow]Not found[/yellow]")

    # Check cache directory
    cache_dir = Path(".cache")
    if cache_dir.exists():
        table.add_row(
            "Cache Directory",
            f"[green]{cache_dir}[/green] ({sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB)",
        )
    else:
        table.add_row("Cache Directory", "[yellow]Not created yet[/yellow]")

    # Check env vars
    import os

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DISTILL_LLM_API_KEY")
    if api_key:
        table.add_row("API Key", "[green]Set[/green]")
    else:
        table.add_row("API Key", "[yellow]Not set (use OPENAI_API_KEY env var)[/yellow]")

    console.print(table)


# Jobs subcommand group
@jobs_app.command("list")
def jobs_list(
    status: str | None = typer.Option(None, "--status", help="Filter by status"),
    job_type: str | None = typer.Option(None, "--type", help="Filter by job type"),
    limit: int = typer.Option(20, "--limit", help="Max jobs to show"),
):
    """List all synthesis jobs."""
    from ..core.checkpoint import CheckpointManager, JobStatus

    manager = CheckpointManager()
    status_filter = JobStatus(status) if status else None
    jobs = manager.list_jobs(status=status_filter, job_type=job_type, limit=limit)

    if not jobs:
        console.print("[yellow]No jobs found[/yellow]")
        return

    table = Table(title=f"Jobs ({len(jobs)})")
    table.add_column("Job ID", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column("Status", style="green")
    table.add_column("Progress", style="yellow")
    table.add_column("Created", style="magenta")

    for job in jobs:
        from datetime import datetime

        created = datetime.fromtimestamp(job.created_at).strftime("%Y-%m-%d %H:%M")
        table.add_row(
            job.job_id,
            job.job_type,
            job.status.value,
            f"{job.processed_items}/{job.total_items} ({job.progress_pct:.0f}%)",
            created,
        )

    console.print(table)


@jobs_app.command("resume")
def jobs_resume(
    job_id: str = typer.Argument(..., help="Job ID to resume"),
):
    """Resume a previous synthesis job."""
    from ..core.checkpoint import CheckpointManager

    manager = CheckpointManager()
    checkpoint = manager.load_job(job_id)

    if not checkpoint:
        console.print(f"[red]Job {job_id} not found[/red]")
        raise typer.Exit(1)

    console.print(f"Resuming job {job_id}: {checkpoint.processed_items}/{checkpoint.total_items} done")
    console.print(f"[yellow]Re-run: distill-align synthesize ... --job-id {job_id} --resume[/yellow]")


@jobs_app.command("delete")
def jobs_delete(
    job_id: str = typer.Argument(..., help="Job ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", flag_value=True, help="Skip confirmation"),
):
    """Delete a job checkpoint."""
    from ..core.checkpoint import CheckpointManager

    if not force and not typer.confirm(f"Delete job {job_id}?"):
        raise typer.Abort()

    manager = CheckpointManager()
    if manager.delete_job(job_id):
        console.print(f"[green]Deleted job {job_id}[/green]")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@jobs_app.command("cleanup")
def jobs_cleanup(
    days: int = typer.Option(30, "--days", help="Remove jobs older than N days"),
):
    """Clean up old job checkpoints."""
    from ..core.checkpoint import CheckpointManager

    manager = CheckpointManager()
    removed = manager.cleanup_old_jobs(older_than_days=days)
    console.print(f"[green]Cleaned up {removed} old jobs[/green]")


# Config subcommand group
@config_app.command("show")
def config_show():
    """Show current configuration."""
    config_path = find_config_file()
    if not config_path:
        console.print("[yellow]No config file found. Run 'distill-align init' to create one.[/yellow]")
        return

    config = load_config(config_path)
    console.print(Panel(str(config_path), title="Config File"))
    console.print(config.model_dump_json(indent=2))


@config_app.command("path")
def config_path():
    """Show the path to the active config file."""
    config_path = find_config_file()
    if config_path:
        console.print(str(config_path))
    else:
        console.print("[yellow]No config file found[/yellow]")


# Init subcommand
@init_app.command("run")
def init_run(
    path: str = typer.Option("distill-align.yaml", "--path", "-p", help="Output config path"),
    name: str = typer.Option("my-dataset", "--name", "-n", help="Project name"),
):
    """Initialize a new project config file."""
    output = generate_default_config(project_name=name, path=path)
    console.print(f"[green]✓ Created config file: {output}[/green]")
    console.print("\nEdit it to configure your pipeline, then run:")
    console.print("  [cyan]distill-align ingest --source ./data[/cyan]")


@app.command()
def tui():
    """Launch the interactive TUI dashboard."""
    import os

    if os.getenv("CI") or os.getenv("PYTEST_CURRENT_TEST"):
        console.print("[yellow]TUI requires an interactive terminal. Skipping in non-TTY mode.[/yellow]")
        return
    console.print(Panel.fit("🖥️ Launching TUI...", style="bold cyan"))
    from ..tui.app import DistillAlignApp

    tui_app = DistillAlignApp()
    tui_app.run()


@app.command()
def version():
    """Show version information."""
    from .. import __version__

    console.print(f"distill-align v{__version__}")


def entry_point() -> None:
    """CLI entry point with global exception handling for production use."""
    try:
        app()
    except typer.Exit:
        raise  # Let Typer handle its own exit codes
    except Exception as e:
        console.print(f"\n[red]❌ Unexpected error: {e}[/red]")
        logger.opt(exception=True).error("Unhandled CLI exception")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    entry_point()

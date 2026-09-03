"""End-to-end Textual pilot tests for the TUI Run flows.

Unlike ``test_tui_options.py`` (widget presence), these tests actually press
the Run buttons headlessly: ingest / validate / evaluate / export run for
real on temp dirs (no API keys needed), while synthesize runs against a
mocked ``SynthesisPipeline.synthesize_batch`` (no network).
"""

import asyncio
import json
from pathlib import Path

import pytest
from textual.widgets import DataTable, Input, Label, Select

from distill_align.tui.app import DistillAlignApp

pytestmark = pytest.mark.asyncio


async def _wait_for(path: Path, timeout: float = 60.0) -> None:
    """Poll until *path* exists (worker threads run outside the pilot loop)."""
    async with asyncio.timeout(timeout):
        while not path.exists():
            await asyncio.sleep(0.2)


async def _wait_for_text(label: Label, needle: str, timeout: float = 60.0) -> str:
    """Poll a Label until its rendered text contains *needle*."""
    async with asyncio.timeout(timeout):
        while needle not in str(label.renderable):
            await asyncio.sleep(0.2)
    return str(label.renderable)


async def _show_tab(app: DistillAlignApp, pilot, tab_id: str) -> None:
    """Activate a tab so its widgets become clickable."""
    from textual.widgets import TabbedContent

    app.query_one(TabbedContent).active = tab_id
    await pilot.pause()


async def _confirm_preflight(pilot, timeout: float = 30.0) -> None:
    """Click through the pre-flight modal (proceed when checks pass).

    Retries the scroll+click until the modal dismisses — headless rendering
    and worker timing make a single attempt racy.
    """
    from textual.widgets import Button

    async with asyncio.timeout(timeout):
        while True:
            await pilot.pause()
            try:
                btn = pilot.app.query_one("#preflight-proceed", Button)
            except Exception:
                return  # modal already dismissed
            try:
                btn.scroll_visible()
                await pilot.pause()
                await pilot.click("#preflight-proceed")
                await pilot.pause()
            except Exception:
                await asyncio.sleep(0.3)
                continue
            try:
                pilot.app.query_one("#preflight-proceed", Button)
                await asyncio.sleep(0.3)
            except Exception:
                return  # dismissed


async def _click(app: DistillAlignApp, pilot, selector: str, timeout: float = 30.0) -> None:
    """Scroll a widget into view, then click it (headless screen is small).

    Retries — layout after tab switches can take a beat to settle.
    """
    from textual.widgets import Button

    async with asyncio.timeout(timeout):
        while True:
            try:
                app.query_one(selector, Button).scroll_visible()
                await pilot.pause()
                await pilot.click(selector)
                return
            except Exception:
                await asyncio.sleep(0.3)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    src = tmp_path / "data"
    src.mkdir()
    (src / "guide.md").write_text(
        "# Widget Guide\n\nWidgets render UI elements on screen.\n\n## Buttons\n\nButtons trigger actions when pressed.\n",
        encoding="utf-8",
    )
    return src


@pytest.fixture
def chunks_file(data_dir: Path, tmp_path: Path) -> Path:
    from distill_align.ingestion.auto import AutoIngestionPipeline

    chunks = AutoIngestionPipeline().ingest_directory(data_dir)
    assert chunks, "fixture setup failed: no chunks ingested"
    out = tmp_path / "chunks.json"
    out.write_text(json.dumps([c.model_dump() for c in chunks]), encoding="utf-8")
    return out


@pytest.fixture
def conversations_file(chunks_file: Path, tmp_path: Path) -> Path:
    import json as _json

    from distill_align.core.schemas import ConversationSchema, SynthesizedTurn

    chunks = _json.loads(chunks_file.read_text(encoding="utf-8"))
    convs = [
        ConversationSchema(
            id=f"conv-{i}",
            source_chunk_id=c["id"],
            turns=[
                SynthesizedTurn(role="user", content="What does this explain?"),
                SynthesizedTurn(role="assistant", content=c["content"][:200]),
            ],
            confidence_score=0.9,
            judge_scores={"overall": 8.0},
        )
        for i, c in enumerate(chunks)
    ]
    out = tmp_path / "conversations.json"
    out.write_text(_json.dumps([c.model_dump() for c in convs]), encoding="utf-8")
    return out


async def test_pilot_ingest_run(data_dir: Path, tmp_path: Path):
    app = DistillAlignApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _show_tab(app, pilot, "ingest")
        out = tmp_path / "pilot-chunks.json"
        app.query_one("#ingest-source", Input).value = str(data_dir)
        app.query_one("#ingest-output", Input).value = str(out)
        app.query_one("#ingest-chunker", Select).value = "semantic"
        await _click(app, pilot, "#ingest-run")
        await _confirm_preflight(pilot)
        await _wait_for(out)
        chunks = json.loads(out.read_text(encoding="utf-8"))
        assert len(chunks) >= 1
        await _wait_for_text(app.query_one("#ingest-status", Label), "Done")


async def test_pilot_validate_and_evaluate(conversations_file: Path):
    app = DistillAlignApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _show_tab(app, pilot, "validate")
        app.query_one("#validate-input", Input).value = str(conversations_file)
        await _click(app, pilot, "#validate-run")
        await _wait_for_text(app.query_one("#validate-status", Label), "Done")
        table = app.query_one("#validate-results", DataTable)
        assert table.row_count > 0

        app.query_one("#eval-threshold", Input).value = "0.5"
        await _click(app, pilot, "#evaluate-run")
        status = await _wait_for_text(app.query_one("#evaluate-status", Label), "PASS")
        assert "PASS" in status


async def test_pilot_export_run(conversations_file: Path, tmp_path: Path):
    app = DistillAlignApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _show_tab(app, pilot, "export")
        out_dir = tmp_path / "pilot-output"
        app.query_one("#export-input", Input).value = str(conversations_file)
        app.query_one("#export-output-dir", Input).value = str(out_dir)
        await _click(app, pilot, "#export-run")
        await _confirm_preflight(pilot)
        await _wait_for(out_dir / "dataset_sharegpt.json")
        await _wait_for_text(app.query_one("#export-status", Label), "Done")


async def test_pilot_synthesize_mocked(chunks_file: Path, tmp_path: Path, monkeypatch):
    from distill_align.core.schemas import ConversationSchema, SynthesizedTurn

    async def fake_synthesize_batch(self, chunks, progress_callback=None, job_id=None, resume=False, use_cache=None):
        out = []
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, len(chunks))
            out.append(
                ConversationSchema(
                    id=f"mock-{i}",
                    source_chunk_id=chunk.id,
                    turns=[
                        SynthesizedTurn(role="user", content="Explain this."),
                        SynthesizedTurn(role="assistant", content=chunk.content[:100]),
                    ],
                    confidence_score=0.8,
                )
            )
        return out

    monkeypatch.setattr(
        "distill_align.synthesis.pipeline.SynthesisPipeline.synthesize_batch",
        fake_synthesize_batch,
    )

    app = DistillAlignApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _show_tab(app, pilot, "synthesize")
        out = tmp_path / "pilot-convs.json"
        app.query_one("#synth-input", Input).value = str(chunks_file)
        app.query_one("#synth-output", Input).value = str(out)
        app.query_one("#synth-mode", Select).value = "qa"
        await _click(app, pilot, "#synth-run")
        await _confirm_preflight(pilot)
        await _wait_for(out)
        convs = json.loads(out.read_text(encoding="utf-8"))
        assert len(convs) >= 1
        assert all(c["turns"] for c in convs)
        await _wait_for_text(app.query_one("#synth-status", Label), "Done")

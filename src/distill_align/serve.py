"""Serve module (Phase 6 lite): FastAPI app factory + MCP tool stubs.

Both servers are optional-dependency only (``pip install distill-align[serve]``).
Importing this module never requires fastapi/mcp; the factories raise a
friendly error pointing at the extra when the dep is missing.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any


def create_app() -> Any:
    """Create the FastAPI application (lazy import)."""
    try:
        from fastapi import FastAPI
    except ImportError as e:
        raise ImportError("FastAPI is not installed. Install with: pip install distill-align[serve]") from e

    from . import __version__

    app = FastAPI(title="Distill-Align", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        from . import __version__

        return {"status": "ok", "version": __version__}

    @app.post("/ingest")
    def ingest(source: str, chunker: str = "auto") -> dict[str, Any]:
        from .core.schemas import IngestionConfig
        from .ingestion.auto import AutoIngestionPipeline

        pipe = AutoIngestionPipeline(IngestionConfig(chunker=chunker))  # type: ignore[arg-type]
        chunks = pipe.ingest_file(source)
        return {"chunks": len(chunks), "ids": [c.id for c in chunks[:100]]}

    @app.post("/evaluate")
    def evaluate(conversations: list[dict[str, Any]]) -> dict[str, Any]:
        from .core.schemas import ConversationSchema
        from .synthesis.eval import evaluate_conversations

        convs = [ConversationSchema(**c) for c in conversations]
        report = evaluate_conversations(convs)
        return {"passed": report.passed, "summary": report.summary(), "failures": report.failures}

    return app


# ── MCP tools (plain functions; wrap with fastmcp/mcp when installed) ──


def tool_ingest(source: str, chunker: str = "auto") -> dict[str, Any]:
    """MCP tool: ingest a file into chunks."""
    from .core.schemas import IngestionConfig
    from .ingestion.auto import AutoIngestionPipeline

    pipe = AutoIngestionPipeline(IngestionConfig(chunker=chunker))  # type: ignore[arg-type]
    chunks = pipe.ingest_file(source)
    return {"chunks": [c.model_dump() for c in chunks]}


def tool_evaluate(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    """MCP tool: evaluate conversations (heuristics, no LLM)."""
    from .core.schemas import ConversationSchema
    from .synthesis.eval import evaluate_conversations

    convs = [ConversationSchema(**c) for c in conversations]
    report = evaluate_conversations(convs)
    return {"passed": report.passed, "failures": report.failures, "summary": report.summary()}


def tool_export(conversations: list[dict[str, Any]], formats: list[str]) -> dict[str, Any]:
    """MCP tool: export conversations to training formats."""
    import tempfile

    from .core.schemas import ConversationSchema, ExportConfig
    from .exporter.pipeline import ExportPipeline

    convs = [ConversationSchema(**c) for c in conversations]
    with tempfile.TemporaryDirectory() as tmp:
        pipe = ExportPipeline(ExportConfig(output_dir=tmp, generate_unsloth_script=False))
        out = pipe.export(convs, formats=formats, generate_unsloth=False)
        return {k: str(v) for k, v in out.items()}


MCP_TOOLS = {
    "ingest": tool_ingest,
    "evaluate": tool_evaluate,
    "export": tool_export,
}


def create_mcp_server() -> Any:
    """Create an MCP server via fastmcp/mcp if installed."""
    for mod_name, cls_name in (("fastmcp", "FastMCP"), ("mcp.server.fastmcp", "FastMCP")):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            server = getattr(mod, cls_name)("distill-align")
            for name, fn in MCP_TOOLS.items():
                with suppress(Exception):
                    server.tool(fn, name=name)
            return server
        except ImportError:
            continue
    raise ImportError("No MCP server library installed. Install with: pip install distill-align[mcp]")

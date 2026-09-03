"""Table serialization helpers (Phase 2).

Tables are the most valuable and most fragile content in RAG corpora.
These helpers preserve grid structure (Markdown) AND emit per-row
natural-language sentences so each fact is individually retrievable.
"""

from __future__ import annotations


def markdown_table_to_sentences(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Convert a table to one sentence per row, repeating column headers.

    Example:
        headers = ["Model", "VRAM"]
        rows = [["7B", "12GB"]]
        → ["Model: 7B; VRAM: 12GB."]
    """
    sentences: list[str] = []
    for row in rows:
        parts = []
        for h, cell in zip(headers, row, strict=False):
            cell = (cell or "").strip()
            if cell:
                parts.append(f"{h.strip()}: {cell}")
        if parts:
            sentences.append("; ".join(parts) + ".")
    return sentences


def markdown_table_to_grid(headers: list[str], rows: list[list[str]]) -> str:
    """Serialize a table back to a Markdown grid (structure-preserving)."""
    lines = ["| " + " | ".join(h.strip() for h in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        cells = [(c or "").strip().replace("|", "\\|") for c in row]
        # Pad short rows
        while len(cells) < len(headers):
            cells.append("")
        lines.append("| " + " | ".join(cells[: len(headers)]) + " |")
    return "\n".join(lines)


def serialize_markdown_tables(
    content: str,
    mode: str = "both",
) -> str:
    """Append row-sentence serializations after each Markdown table.

    Args:
        content: Markdown text possibly containing pipe tables.
        mode: ``"markdown"`` (keep grid only), ``"sentences"`` (replace with
            sentences), ``"both"`` (grid + sentences — best for retrieval).

    Returns:
        Augmented Markdown text.
    """
    if mode == "markdown" or "|" not in content:
        return content
    lines = content.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        # Detect table header + separator
        if "|" in lines[i] and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").replace(" ", "")) <= set("-:"):
            header_cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            i += 2  # skip header + separator
            rows: list[list[str]] = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            if mode == "sentences":
                out.extend(markdown_table_to_sentences(header_cells, rows))
            else:  # both
                out.append(markdown_table_to_grid(header_cells, rows))
                out.extend(markdown_table_to_sentences(header_cells, rows))
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)

"""Quick-Start Wizard for first-time users of Distill-Align.

Walks the user through four simple steps, then auto-fills the Full Pipeline
tab and optionally kicks off the run.
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from ..modes.base import BUILTIN_TEMPLATES


class QuickStartWizard(ModalScreen[dict[str, Any] | None]):
    """A 4-step wizard that collects the minimal info needed to run a pipeline."""

    CSS = """
    QuickStartWizard {
        align: center middle;
    }

    #wizard-box {
        width: 64;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        border: thick $accent;
        background: $background;
    }

    #wizard-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #wizard-step-title {
        text-style: bold;
        color: $secondary;
        margin-top: 1;
        margin-bottom: 1;
    }

    #wizard-description {
        margin-bottom: 1;
        color: $text-muted;
    }

    #wizard-nav {
        margin-top: 1;
        align: center middle;
    }

    #wizard-nav Button {
        margin: 0 1;
    }

    RadioSet {
        margin: 0 0 1 0;
    }

    .wizard-label {
        width: auto;
        min-width: 14;
        padding: 0 1;
    }

    .wizard-input {
        width: 1fr;
    }

    .wizard-row {
        height: auto;
        margin: 0 0 1 0;
        align: left middle;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._step = 1
        self._total_steps = 3
        self._result: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="wizard-box"):
            yield Label("✨ Quick-Start Wizard", id="wizard-title")
            yield Static("", id="wizard-description")
            yield Label("", id="wizard-step-title")
            yield Static("", id="wizard-content")
            with Horizontal(id="wizard-nav"):
                yield Button("◀ Back", id="wizard-back", variant="default")
                yield Button("Next ▶", id="wizard-next", variant="primary")
                yield Button("Cancel", id="wizard-cancel", variant="error")
        self._render_step()

    def on_mount(self) -> None:
        """Focus the Next button or first interactive element."""
        self._render_step()

    def _render_step(self) -> None:
        """Render the current wizard step."""
        content = self.query_one("#wizard-content", Static)
        step_title = self.query_one("#wizard-step-title", Label)
        desc = self.query_one("#wizard-description", Static)
        back_btn = self.query_one("#wizard-back", Button)
        next_btn = self.query_one("#wizard-next", Button)

        back_btn.disabled = self._step == 1
        next_btn.label = "🚀 Go!" if self._step == self._total_steps else "Next ▶"
        next_btn.variant = "primary"

        if self._step == 1:
            step_title.update("Step 1 of 3: What data do you have?")
            desc.update(
                "Tell us where your source files are. We support Markdown, code, PDFs, DOCX, HTML, CSV, JSON, and more."
            )
            content.update(self._render_step1())
        elif self._step == 2:
            step_title.update("Step 2 of 3: What kind of dataset?")
            desc.update("Pick a template that matches your goal. We'll tune the settings automatically.")
            content.update(self._render_step2())
        elif self._step == 3:
            step_title.update("Step 3 of 3: Which model?")
            desc.update("Choose the LLM that will generate your conversations. OpenAI and local models are supported.")
            content.update(self._render_step3())

    def _render_step1(self) -> str:
        """Step 1: source path."""
        initial = self._result.get("source", "")
        return (
            f"[bold]Source path[/bold]\n"
            f"    [dim]Folder or file containing your data[/dim]\n"
            f"    [on #333333]  {initial or './data'}  [/on #333333]\n\n"
            f"[bold]Output location[/bold]\n"
            f"    [dim]Where to save the final dataset[/dim]\n"
            f"    [on #333333]  {self._result.get('output_dir', './output')}  [/on #333333]\n\n"
            "[dim italic]💡 Tip: Point to a folder with markdown files to get started quickly.[/dim italic]"
        )

    def _render_step2(self) -> str:
        """Step 2: template selection."""
        lines = ["[bold]Choose a template[/bold]\n"]
        selected = self._result.get("template_name", "")
        for t in BUILTIN_TEMPLATES:
            marker = "[green]●[/green]" if t.name == selected else "[dim]○[/dim]"
            lines.append(f"  {marker} [bold]{t.icon} {t.name}[/bold]")
            lines.append(f"    {t.description}")
            lines.append("")
        return "\n".join(lines)

    def _render_step3(self) -> str:
        """Step 3: provider / model."""
        provider = self._result.get("provider", "openai")
        model = self._result.get("model", "gpt-5-mini")

        try:
            from ...synthesis.models.registry import list_names

            available = ", ".join(list_names())
        except Exception:
            available = "openai, anthropic, gemini, azure, ollama, vllm, qwen"

        return (
            f"[bold]Provider[/bold]: [green]{provider}[/green]\n"
            f"[bold]Model[/bold]:    [green]{model}[/green]\n\n"
            f"[dim]Available providers: {available}[/dim]\n\n"
            "[dim italic]💡 Tip: Start with [bold]gpt-5-mini[/bold] for fast, cheap results.[/dim italic]"
        )

    @on(Button.Pressed, "#wizard-next")
    def _on_next(self) -> None:
        if self._step < self._total_steps:
            self._step += 1
            self._render_step()
        else:
            # Finalize and return result
            self.dismiss(self._result)

    @on(Button.Pressed, "#wizard-back")
    def _on_back(self) -> None:
        if self._step > 1:
            self._step -= 1
            self._render_step()

    @on(Button.Pressed, "#wizard-cancel")
    def _on_cancel(self) -> None:
        self.dismiss(None)

    def on_key(self, event: None) -> None:
        """Allow Enter to advance, Escape to cancel."""
        pass  # handled by button bindings


# Public helper to build full config from wizard result
def wizard_result_to_full_config(result: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a wizard result dict into a full pipeline config dict."""
    if not result:
        return None

    config: dict[str, Any] = {
        "source": "./data",
        "output_dir": "./output",
        "provider": "openai",
        "model": "gpt-5-mini",
    }

    # Merge from result
    config.update(result)

    # Apply template overrides if a template was selected
    template_name = result.get("template_name", "")
    if template_name:
        from ..modes.base import get_template

        tmpl = get_template(template_name)
        if tmpl:
            # Template values set defaults; explicit wizard values override
            for k, v in tmpl.config.items():
                config.setdefault(k, v)

    return config

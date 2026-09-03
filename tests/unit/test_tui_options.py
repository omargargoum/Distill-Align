"""Tests for TUI option wiring (registry/catalog-driven dropdowns).

These tests guarantee the dashboard can never again drift behind the
pipelines: every provider, conversation mode, export format, and chunker
the backend supports must be selectable (or at least valid) in the TUI.
"""

import pytest


class TestOptionHelpers:
    def test_mode_choices_cover_enum(self):
        from distill_align.synthesis.conversation_builder import ConversationMode
        from distill_align.tui.options import conversation_mode_choices, valid_conversation_modes

        choices = conversation_mode_choices()
        values = [v for v, _ in choices]
        assert values[0] == "default"
        for mode in ConversationMode:
            assert mode.value in values
        # New 2026 modes must be present
        for expected in ("evol_instruct", "rag_qa", "tool_call", "constitutional", "distill"):
            assert expected in values
        assert valid_conversation_modes() == set(values)

    def test_format_choices_cover_registry(self):
        from distill_align.exporter.pipeline import FORMATTER_MAP
        from distill_align.tui.options import export_format_choices, valid_export_formats

        choices = export_format_choices()
        values = [v for v, _ in choices]
        assert values[0] == "sharegpt"
        assert set(values) == set(FORMATTER_MAP)
        for expected in ("dpo", "orpo", "kto", "grpo", "agent", "rag_qa", "preference"):
            assert expected in values
        assert valid_export_formats() == set(FORMATTER_MAP)

    def test_chunker_choices_match_schema(self):
        from typing import get_args

        from distill_align.core.schemas import IngestionConfig
        from distill_align.tui.options import chunker_choices, valid_chunkers, validate_chunker

        literal = {a for a in get_args(IngestionConfig.model_fields["chunker"].annotation) if isinstance(a, str)}
        assert valid_chunkers() == literal
        values = [v for v, _ in chunker_choices()]
        assert values[0] == "auto"
        assert set(values) == literal
        for expected in ("semantic", "parent_child", "late"):
            assert validate_chunker(expected)
        assert not validate_chunker("nonexistent-chunker")

    def test_provider_choices_cover_registry(self):
        from distill_align.synthesis.models.registry import list_names
        from distill_align.tui.options import default_model_for, provider_choices

        values = [v for _, v in provider_choices()]
        assert set(values) == set(list_names())
        for expected in ("openai", "anthropic", "gemini", "qwen", "openrouter", "deepseek", "mistral", "ollama"):
            assert expected in values
        # Catalog production defaults (Sep-2026 refresh)
        assert default_model_for("openai") == "gpt-5-mini"
        assert default_model_for("anthropic") == "claude-sonnet-5"
        assert default_model_for("gemini") == "gemini-3.5-flash"
        assert default_model_for("qwen") == "qwen3-max"
        assert default_model_for("ollama") == "qwen3:30b"
        assert default_model_for("unknown-provider") == "gpt-5-mini"


class TestTemplatesValid:
    def test_builtin_templates_reference_valid_options(self):
        from distill_align.synthesis.models.registry import get as get_provider_info
        from distill_align.tui.modes.base import BUILTIN_TEMPLATES
        from distill_align.tui.options import validate_export_format, validate_mode

        assert len(BUILTIN_TEMPLATES) >= 6
        for template in BUILTIN_TEMPLATES:
            provider = template.config.get("provider", "openai")
            assert get_provider_info(provider) is not None, f"{template.name}: unknown provider {provider}"
            assert validate_mode(template.config.get("mode", "default")), f"{template.name}: bad mode"
            assert validate_export_format(template.config.get("format", "sharegpt")), f"{template.name}: bad format"


class TestPreflightCoverage:
    @pytest.mark.parametrize(
        "mode",
        [
            "default",
            "teach",
            "debug",
            "review",
            "qa",
            "explain",
            "evol_instruct",
            "rag_qa",
            "tool_call",
            "constitutional",
            "distill",
        ],
    )
    def test_preflight_accepts_all_modes(self, mode, tmp_path):
        from distill_align.tui.widgets.preflight import preflight_synthesize

        chunks = tmp_path / "chunks.json"
        chunks.write_text("[]", encoding="utf-8")
        report = preflight_synthesize(str(chunks), str(tmp_path / "out.json"), "ollama", 4, mode=mode)
        assert not any(c.name == "Mode" and c.status == "warn" for c in report.checks), mode

    def test_preflight_warns_on_unknown_mode(self, tmp_path):
        from distill_align.tui.widgets.preflight import preflight_synthesize

        chunks = tmp_path / "chunks.json"
        chunks.write_text("[]", encoding="utf-8")
        report = preflight_synthesize(str(chunks), str(tmp_path / "out.json"), "ollama", 4, mode="bogus")
        assert any(c.name == "Mode" and c.status == "warn" for c in report.checks)

    @pytest.mark.parametrize("fmt", ["sharegpt", "dpo", "orpo", "kto", "grpo", "agent", "rag_qa"])
    def test_preflight_accepts_all_formats(self, fmt, tmp_path):
        from distill_align.tui.widgets.preflight import preflight_export

        convs = tmp_path / "convs.json"
        convs.write_text("[]", encoding="utf-8")
        report = preflight_export(str(convs), str(tmp_path), [fmt])
        assert any(c.name == "Export formats" and c.status == "pass" for c in report.checks)


class TestWizardCoverage:
    def test_wizard_lists_all_providers(self):
        from distill_align.synthesis.models.registry import list_names
        from distill_align.tui.widgets.wizard import QuickStartWizard

        wizard = QuickStartWizard()
        text = wizard._render_step3()
        for name in list_names():
            assert name in text, f"provider {name} missing from wizard"


class TestTUIAppWiring:
    @pytest.mark.asyncio
    async def test_app_mounts_with_new_controls(self):
        """Pilot test: full app mounts; new dropdowns/buttons exist with new options."""
        from textual.widgets import Button, Select

        from distill_align.tui.app import DistillAlignApp

        app = DistillAlignApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # Tabs
            for tab_id in (
                "dashboard",
                "ingest",
                "synthesize",
                "export",
                "validate",
                "full-pipeline",
                "jobs",
                "cache",
                "config",
                "logs",
            ):
                assert app.query_one(f"#{tab_id}"), f"missing tab {tab_id}"

            # Mode dropdowns include 2026 modes
            synth_mode = app.query_one("#synth-mode", Select)
            mode_values = {str(v) for _, v in synth_mode._options}
            for expected in ("evol_instruct", "rag_qa", "tool_call", "constitutional", "distill"):
                assert expected in mode_values, f"synth mode missing {expected}"
            full_mode = app.query_one("#full-mode", Select)
            assert {str(v) for _, v in full_mode._options} == mode_values

            # Provider dropdowns include new providers
            providers = {str(v) for _, v in app.query_one("#synth-provider", Select)._options}
            for expected in ("qwen", "openrouter", "deepseek", "mistral"):
                assert expected in providers, f"provider missing {expected}"

            # Chunker dropdowns
            chunkers = {str(v) for _, v in app.query_one("#ingest-chunker", Select)._options}
            for expected in ("auto", "semantic", "parent_child", "late"):
                assert expected in chunkers, f"chunker missing {expected}"
            assert {str(v) for _, v in app.query_one("#full-chunker", Select)._options} == chunkers

            # Full-pipeline format dropdown includes new formats
            formats = {str(v) for _, v in app.query_one("#full-format", Select)._options}
            for expected in ("kto", "grpo", "agent", "rag_qa"):
                assert expected in formats, f"format missing {expected}"

            # Evaluate controls exist
            assert app.query_one("#evaluate-run", Button)
            assert app.query_one("#evaluate-status")
            assert app.query_one("#eval-threshold")

"""
Multi-turn conversation builder modes.

Provides different conversation generation strategies:
- teach: Generate a "teach me this" multi-turn conversation
- debug: Create a debugging session (user reports error, assistant explains)
- review: Generate a code review conversation
- qa: Simple question-answer pairs
"""

import asyncio
import uuid
from enum import StrEnum

from loguru import logger

from ..core.exceptions import LLMClientError
from ..core.schemas import ConversationSchema, DataChunk, SynthesizedTurn
from .models.base import BaseLLMClient
from .pruner import ContentPruner


class ConversationMode(StrEnum):
    """Available conversation generation modes."""

    TEACH = "teach"  # "Teach me this in 5 questions"
    DEBUG = "debug"  # User reports an error, assistant explains
    REVIEW = "review"  # Code review style conversation
    QA = "qa"  # Simple question-answer
    EXPLAIN = "explain"  # Explain the content conversationally
    # Phase 3 (2026 workflows)
    EVOL_INSTRUCT = "evol_instruct"  # Evol-Instruct depth/breadth mutation
    RAG_QA = "rag_qa"  # Grounded QA with retrieved-chunk citations
    TOOL_CALL = "tool_call"  # Function-calling / agent trajectory traces
    CONSTITUTIONAL = "constitutional"  # Safety + refusal tuning
    DISTILL = "distill"  # Full reasoning-trace distillation triples


# Prompt templates for each mode
MODE_TEMPLATES: dict[ConversationMode, dict[str, str]] = {
    ConversationMode.TEACH: {
        "system": "You are an expert teacher. Create a multi-turn conversation that progressively teaches the following content to a student. Generate exactly 5 user questions, building from basic to advanced, with detailed assistant answers.",
    },
    ConversationMode.DEBUG: {
        "system": "You are an expert developer. A user encounters an issue with the following code/content. Generate a debugging conversation: 1) user describes a common error or unexpected behavior, 2) assistant asks clarifying questions, 3) assistant identifies the root cause, 4) assistant provides the fix and explanation.",
    },
    ConversationMode.REVIEW: {
        "system": "You are a senior code reviewer. Generate a code review conversation: 1) assistant reviews the code, 2) identifies issues with style/performance/correctness, 3) suggests improvements, 4) provides the improved version.",
    },
    ConversationMode.QA: {
        "system": "You are a knowledgeable assistant. Generate 3-5 question-answer pairs that thoroughly cover the key information in the following content.",
    },
    ConversationMode.EXPLAIN: {
        "system": "You are a clear, patient teacher. Generate a multi-turn conversation where the user asks 'explain this to me' and the assistant provides a structured explanation covering: 1) overview, 2) key concepts, 3) practical examples, 4) common pitfalls, 5) next steps to learn more.",
    },
    ConversationMode.EVOL_INSTRUCT: {
        "system": "You are an instruction-evolution engine (Evol-Instruct). Given source content, first write a base instruction, then evolve it twice: once for DEPTH (add constraints, multi-step reasoning, edge cases) and once for BREADTH (a related but distinct task). Return all three instruction-response pairs grounded in the source.",
    },
    ConversationMode.RAG_QA: {
        "system": "You are a grounded QA generator for retrieval evaluation. Generate question-answer pairs where EVERY factual claim in the answer cites the provided chunk (e.g. [chunk]). If the chunk does not contain the answer, say so explicitly — never hallucinate. Include one unanswerable question to test refusal.",
    },
    ConversationMode.TOOL_CALL: {
        "system": "You are an agent-trajectory generator. Generate a multi-step tool-use trace: user request, assistant reasoning, tool_call JSON (name + arguments), tool result, and final answer. Use realistic tool names (search, retrieve, execute, lookup) with valid JSON arguments.",
    },
    ConversationMode.CONSTITUTIONAL: {
        "system": "You are a safety-tuning data generator (Constitutional AI). Generate prompts spanning safe completion, polite refusal with a helpful alternative, and jurisdiction-tagged edge cases. Never produce disallowed content; refusals must be brief, non-preachy, and offer a safe alternative.",
    },
    ConversationMode.DISTILL: {
        "system": "You are a frontier reasoning teacher. Given source content, produce a full chain-of-thought trace (step-by-step reasoning, alternatives considered, verification) followed by the final answer. The trace must be faithful — every step grounded in the source or explicit assumptions.",
    },
}


class ConversationBuilder:
    """
    Builds multi-turn conversations in different modes.

    Supports various pedagogical strategies for generating
    high-quality fine-tuning conversations.
    """

    def __init__(self, pruner: ContentPruner | None = None):
        """
        Initialize the conversation builder.

        Args:
            pruner: Optional content pruner for quality checks.
        """
        self.pruner = pruner or ContentPruner()

    async def build_conversation(
        self,
        chunk: DataChunk,
        mode: ConversationMode,
        llm_client: BaseLLMClient,
        temperature: float = 0.7,
    ) -> ConversationSchema | None:
        """
        Build a conversation in the specified mode.

        Args:
            chunk: Data chunk to use as source material.
            mode: Conversation generation mode.
            llm_client: LLM client to use.
            temperature: Sampling temperature.

        Returns:
            ConversationSchema or None if generation fails.
        """
        template = MODE_TEMPLATES.get(mode, MODE_TEMPLATES[ConversationMode.QA])

        system_prompt = template["system"]
        user_prompt = self._build_user_prompt(chunk, mode)

        try:
            response = await llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )

            # Parse JSON response
            parsed = self.pruner.extract_json_from_response(response)
            if not parsed or "conversation" not in parsed:
                # Fallback: treat as single assistant turn
                return self._build_simple_conversation(chunk, response)

            turns = [SynthesizedTurn(role=t["role"], content=t["content"]) for t in parsed["conversation"]]

            conversation = ConversationSchema(
                id=str(uuid.uuid4()),
                source_chunk_id=chunk.id,
                turns=turns,
                reasoning_trace=parsed.get("reasoning_trace"),
            )

            pruned = self.pruner.prune_conversation(conversation)
            return pruned or conversation

        except LLMClientError as e:
            logger.error(f"Failed to build {mode.value} conversation: {e}")
            return None

    def _build_user_prompt(self, chunk: DataChunk, mode: ConversationMode) -> str:
        """
        Build the user prompt based on the chunk and mode.

        Args:
            chunk: Data chunk with content and metadata.
            mode: Conversation mode.

        Returns:
            User prompt string.
        """
        metadata = chunk.metadata

        prompt_parts = [
            f"**Source Content:**\n{chunk.content[:3000]}",
            f"\n**Source Type:** {metadata.source_type}",
        ]

        if metadata.title:
            prompt_parts.append(f"**Title:** {metadata.title}")
        if metadata.language:
            prompt_parts.append(f"**Language:** {metadata.language}")
        if metadata.section_headers:
            prompt_parts.append(f"**Sections:** {' > '.join(metadata.section_headers)}")

        if mode == ConversationMode.TEACH:
            prompt_parts.append(
                "\n**Instructions:**\n"
                "1. Start with a basic concept question\n"
                "2. Progress to more complex questions\n"
                "3. End with a synthesis/application question\n"
                "4. Each answer should be 100-300 words\n"
                "5. Use the source content as the basis for all answers"
            )
        elif mode == ConversationMode.DEBUG:
            prompt_parts.append(
                "\n**Instructions:**\n"
                "1. Create a realistic error scenario based on the content\n"
                "2. Walk through debugging steps\n"
                "3. Identify the root cause\n"
                "4. Provide a fix with explanation"
            )
        elif mode == ConversationMode.REVIEW:
            prompt_parts.append(
                "\n**Instructions:**\n"
                "1. Identify 2-3 areas for improvement\n"
                "2. Discuss trade-offs\n"
                "3. Suggest specific changes\n"
                "4. Provide an improved version"
            )
        elif mode == ConversationMode.QA:
            prompt_parts.append(
                "\n**Instructions:**\n"
                "1. Cover the most important information\n"
                "2. Mix conceptual and practical questions\n"
                "3. Keep answers concise (50-150 words)"
            )
        elif mode == ConversationMode.EXPLAIN:
            prompt_parts.append(
                "\n**Instructions:**\n"
                "1. Start with a high-level overview\n"
                "2. Explain key concepts in detail\n"
                "3. Provide practical examples\n"
                "4. Discuss common pitfalls\n"
                "5. Suggest next steps"
            )
        elif mode == ConversationMode.EVOL_INSTRUCT:
            prompt_parts.append(
                "\n**Instructions (Evol-Instruct):**\n"
                "1. Write a BASE instruction + response from the source\n"
                "2. DEPTH-evolve: add 2+ constraints / reasoning steps / edge cases, then answer\n"
                "3. BREADTH-evolve: a neighbouring task on the same source, then answer\n"
                "4. Tag each pair with its evolution type"
            )
        elif mode == ConversationMode.RAG_QA:
            prompt_parts.append(
                "\n**Instructions (grounded RAG QA):**\n"
                "1. 3-5 questions answerable ONLY from the source; cite facts\n"
                "2. 1 unanswerable question with an explicit abstention\n"
                "3. Keep answers concise; no outside knowledge"
            )
        elif mode == ConversationMode.TOOL_CALL:
            prompt_parts.append(
                "\n**Instructions (tool traces):**\n"
                "1. User request requiring 1-3 tool calls\n"
                "2. For each call emit tool_call JSON {name, arguments}\n"
                "3. Emit plausible tool results, then the final answer"
            )
        elif mode == ConversationMode.CONSTITUTIONAL:
            prompt_parts.append(
                "\n**Instructions (safety):**\n"
                "1. One safe-completion pair\n"
                "2. One refusal pair (brief, helpful alternative, no lecture)\n"
                "3. Tag jurisdiction/sensitivity where relevant"
            )
        elif mode == ConversationMode.DISTILL:
            prompt_parts.append(
                "\n**Instructions (distillation):**\n"
                "1. Full step-by-step reasoning trace first\n"
                "2. Note alternatives considered + verification step\n"
                "3. Final answer last, consistent with the trace"
            )

        prompt_parts.append('\nReturn JSON: {"conversation": [{"role": "...", "content": "..."}, ...]}')

        return "\n".join(prompt_parts)

    def _build_simple_conversation(self, chunk: DataChunk, response: str) -> ConversationSchema:
        """
        Build a simple conversation when JSON parsing fails.

        Args:
            chunk: Source chunk.
            response: LLM response text.

        Returns:
            Simple ConversationSchema.
        """
        return ConversationSchema(
            id=str(uuid.uuid4()),
            source_chunk_id=chunk.id,
            turns=[
                SynthesizedTurn(
                    role="user",
                    content=f"Please explain the following:\n\n{chunk.content[:500]}",
                ),
                SynthesizedTurn(role="assistant", content=response),
            ],
        )

    async def build_batch(
        self,
        chunks: list[DataChunk],
        mode: ConversationMode,
        llm_client: BaseLLMClient,
        temperature: float = 0.7,
        max_concurrency: int = 5,
    ) -> list[ConversationSchema]:
        """
        Build conversations for a batch of chunks in the same mode.

        Args:
            chunks: List of chunks.
            mode: Conversation mode.
            llm_client: LLM client.
            temperature: Sampling temperature.
            max_concurrency: Maximum concurrent LLM requests.

        Returns:
            List of generated conversations.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def build_with_limit(chunk: DataChunk) -> ConversationSchema | None:
            async with semaphore:
                return await self.build_conversation(chunk, mode, llm_client, temperature)

        tasks = [build_with_limit(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        conversations: list[ConversationSchema] = []
        for chunk, result in zip(chunks, results, strict=False):
            if isinstance(result, BaseException):
                logger.warning(f"Failed to build conversation for {chunk.id}: {result}")
                continue
            if result is not None:
                conversations.append(result)

        logger.info(f"Built {len(conversations)}/{len(chunks)} conversations in {mode.value} mode")
        return conversations

from dataclasses import dataclass
from pathlib import Path
from rag.constants import SYSTEM_PROMPT, CONTEXT_BLOCK_SEPARATOR


@dataclass
class Prompt:
    system: str  # fixed rules, sent as the system message
    user: str    # numbered context + question, built per request


class PromptBuilder:
    """Turns retrieved chunks + a question into the messages sent to the LLM.
    Knows nothing about the LLM or the database, so its output can be printed and checked by eye."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build(self, question: str, chunks: list[dict]) -> Prompt:
        """chunks: rows from ChunkRepository.search_similar (source, folder, content, ...), best first."""
        context_blocks = [
            self.format_context_block(source_number, chunk)
            for source_number, chunk in enumerate(chunks, start=1)  # numbers match the [n] citations
        ]
        user_prompt = (
            "Context:\n"
            f"{CONTEXT_BLOCK_SEPARATOR.join(context_blocks)}\n\n"
            f"Question: {question}"  # last, so a small model reads it right before answering
        )
        return Prompt(system=self.system_prompt, user=user_prompt)

    def format_context_block(self, source_number: int, chunk: dict) -> str:
        file_name = Path(chunk["source"]).name
        return f"[{source_number}] (file: {file_name}, folder: {chunk['folder']})\n{chunk['content'].strip()}"

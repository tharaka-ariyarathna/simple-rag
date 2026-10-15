"""Writes synthetic test questions for every chunk that has none yet. Resumable: stop it any time.

Uses QUESTION_LLM_CHAIN from .env (default: the local Ollama model; e.g. "openrouter,ollama" falls back to the
local model when OpenRouter is capped). When every provider keeps failing (after its retries), the run logs the
error and stops with exit code 1.
Run from src:  python -m rag.evaluation.generate_questions [--limit N]
"""
import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from rag.bootstrap import build_question_generation_components
from rag.constants import ACTIVE_EMBEDDING_MODEL, QUESTION_GENERATION_DELAY_SECONDS
from rag.evaluation.question_generator import QuestionGenerationError, QuestionGenerator
from rag.llm.base import LlmError
from rag.repository.chunk_repository import ChunkRepository
from rag.repository.question_repository import QuestionRepository

logger = logging.getLogger(__name__)


@dataclass
class GenerationSummary:
    saved_questions: int = 0
    skipped_chunks: int = 0  # the LLM found nothing worth asking
    unusable_replies: int = 0  # the reply was not usable JSON: the chunk is retried next run
    stopped_by: str | None = None  # the LLM error that stopped the run; None when every chunk was processed


def process_chunks(
    pending_chunks: list[dict],
    generator: QuestionGenerator,
    question_repository: QuestionRepository,
    delay_seconds: float = QUESTION_GENERATION_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> GenerationSummary:
    summary = GenerationSummary()
    for position, chunk in enumerate(pending_chunks, start=1):
        label = f"[{position}/{len(pending_chunks)}] {chunk['folder']} / {Path(chunk['source']).name} #{chunk['chunk_index']}"
        try:
            generated = generator.generate(chunk["content"])
        except QuestionGenerationError as error:  # a bad reply, not an outage: move on
            summary.unusable_replies += 1
            print(f"{label}: unusable reply, retried next run ({error})")
            continue
        except LlmError as error:  # every provider used up its retries, or the setup is wrong: more would fail too
            logger.error("Stopping at %s: the question LLM failed: %s. Fix it and run again to resume.", label, error)
            summary.stopped_by = str(error)
            break
        question_repository.save_questions(chunk["source"], chunk["chunk_index"], generated.questions, generated.model)
        if generated.questions:
            summary.saved_questions += len(generated.questions)
            print(f"{label}: {len(generated.questions)} questions")
        else:
            summary.skipped_chunks += 1
            print(f"{label}: nothing worth asking")
        if delay_seconds:
            sleep(delay_seconds)
    return summary


def generate_questions(limit: int | None) -> bool:
    """Returns False when the run was stopped by an LLM failure."""
    components = build_question_generation_components()
    try:
        chunk_repository = ChunkRepository(components.database)
        question_repository = QuestionRepository(components.database)

        processed_keys = question_repository.processed_chunk_keys()
        pending_chunks = [
            chunk for chunk in chunk_repository.list_chunks(ACTIVE_EMBEDDING_MODEL)
            if (chunk["source"], chunk["chunk_index"]) not in processed_keys
        ]
        if limit is not None:
            pending_chunks = pending_chunks[:limit]
        print(f"{len(pending_chunks)} chunks to process ({len(processed_keys)} already done).")

        print(f"Question LLM: {components.llm_client.model_name}")
        summary = process_chunks(pending_chunks, QuestionGenerator(components.llm_client), question_repository,
                                 delay_seconds=components.delay_seconds)
        print(f"{'Stopped' if summary.stopped_by else 'Done'}: {summary.saved_questions} questions saved, "
              f"{summary.skipped_chunks} chunks skipped, {summary.unusable_replies} unusable replies.")
        return summary.stopped_by is None
    finally:
        components.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argument_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argument_parser.add_argument("--limit", type=int, default=None, help="process at most this many chunks")
    raise SystemExit(0 if generate_questions(argument_parser.parse_args().limit) else 1)

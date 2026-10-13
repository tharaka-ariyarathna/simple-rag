"""Writes synthetic test questions for every chunk that has none yet. Resumable: stop it any time.

Run from src:  python -m rag.evaluation.generate_questions [--limit N]
"""
import argparse
import logging
import time
from pathlib import Path
from rag.bootstrap import build_rag_components
from rag.constants import ACTIVE_EMBEDDING_MODEL, QUESTION_GENERATION_DELAY_SECONDS
from rag.evaluation.question_generator import QuestionGenerationError, QuestionGenerator
from rag.llm.base import LlmError
from rag.repository.chunk_repository import ChunkRepository
from rag.repository.question_repository import QuestionRepository

MAX_CONSECUTIVE_FAILURES = 5  # the LLM is probably down or out of quota: stop instead of burning through chunks


def generate_questions(limit: int | None) -> None:
    components = build_rag_components()
    try:
        chunk_repository = ChunkRepository(components.database)
        question_repository = QuestionRepository(components.database)
        generator = QuestionGenerator(components.llm_client)

        processed_keys = question_repository.processed_chunk_keys()
        pending_chunks = [
            chunk for chunk in chunk_repository.list_chunks(ACTIVE_EMBEDDING_MODEL)
            if (chunk["source"], chunk["chunk_index"]) not in processed_keys
        ]
        if limit is not None:
            pending_chunks = pending_chunks[:limit]
        print(f"{len(pending_chunks)} chunks to process ({len(processed_keys)} already done).")

        saved_questions = skipped_chunks = failed_chunks = consecutive_failures = 0
        for position, chunk in enumerate(pending_chunks, start=1):
            label = f"[{position}/{len(pending_chunks)}] {chunk['folder']} / {Path(chunk['source']).name} #{chunk['chunk_index']}"
            try:
                questions = generator.generate(chunk["content"])
            except (LlmError, QuestionGenerationError) as error:
                failed_chunks += 1
                consecutive_failures += 1
                print(f"{label}: failed ({error})")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"Stopping after {MAX_CONSECUTIVE_FAILURES} failures in a row. Run again to resume.")
                    break
                continue
            consecutive_failures = 0
            question_repository.save_questions(
                chunk["source"], chunk["chunk_index"], questions, components.llm_client.model_name
            )
            if questions:
                saved_questions += len(questions)
                print(f"{label}: {len(questions)} questions")
            else:
                skipped_chunks += 1
                print(f"{label}: nothing worth asking")
            time.sleep(QUESTION_GENERATION_DELAY_SECONDS)

        print(f"Done: {saved_questions} questions saved, {skipped_chunks} chunks skipped, {failed_chunks} failed.")
    finally:
        components.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argument_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argument_parser.add_argument("--limit", type=int, default=None, help="process at most this many chunks")
    generate_questions(argument_parser.parse_args().limit)

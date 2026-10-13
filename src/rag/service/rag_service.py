import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from rag.llm.base import LlmClient
from rag.llm.call_context import llm_call_context
from rag.repository.chunk_repository import ChunkRepository
from rag.repository.ledger_repository import LedgerRepository
from rag.constants import NO_RELEVANT_CONTEXT_ANSWER
from rag.embedding.base import EmbeddingModel
from rag.service.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

REFUSAL_REASON_RETRIEVAL = "retrieval"  # no chunk passed the similarity threshold, the LLM was never called
REFUSAL_REASON_LLM = "llm"  # chunks were found, but the LLM said the answer is not in them


@dataclass
class SourceReference:
    file_name: str
    folder: str
    similarity: float


@dataclass
class RagAnswer:
    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    refusal_reason: str | None = None  # retrieval / llm when the system said it doesn't know, else None


class RagService:
    """Question in, grounded answer out. Knows nothing about HTTP.

    Flow: embed question -> search chunks (with threshold) -> no chunks? fixed answer
    -> build prompt -> LLM -> answer + sources.
    All dependencies are injected (built once in api/main.py lifespan).
    """

    def __init__(
        self,
        repository: ChunkRepository,
        embedder: EmbeddingModel,
        prompt_builder: PromptBuilder,
        llm_client: LlmClient,
        ledger_repository: LedgerRepository | None = None,  # None = do not record requests
    ):
        self.repository = repository
        self.embedder = embedder
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.ledger_repository = ledger_repository

    def list_folders(self) -> dict[str, int]:
        """Folder name -> chunk count, for the active embedding model."""
        return self.repository.count_by_folder(self.embedder.model_name)

    def answer_question(
        self,
        question: str,
        folder: str | None,
        top_k: int,
        entry_point: str = "api",
        eval_run_id: int | None = None,
    ) -> RagAnswer:
        started_at = time.perf_counter()
        chunks = self.search_chunks(question, folder, top_k)
        rag_request_id = self.record_request_start(entry_point, question, folder, top_k, chunks, eval_run_id)
        answer_text = None
        refusal_reason = None
        try:
            if not chunks:
                refusal_reason = REFUSAL_REASON_RETRIEVAL
                answer_text = NO_RELEVANT_CONTEXT_ANSWER
                return RagAnswer(answer=answer_text, refusal_reason=refusal_reason)
            prompt = self.prompt_builder.build(question, chunks)
            with llm_call_context("answer", rag_request_id):
                answer_text = self.llm_client.generate(prompt.system, prompt.user).text
            if answer_text.strip() == NO_RELEVANT_CONTEXT_ANSWER:
                refusal_reason = REFUSAL_REASON_LLM
            sources_by_file: dict[str, SourceReference] = {}
            for chunk in chunks:
                file_name = Path(chunk["source"]).name
                if file_name not in sources_by_file:
                    sources_by_file[file_name] = SourceReference(file_name, chunk["folder"], chunk["similarity"])
            return RagAnswer(
                answer=answer_text, sources=list(sources_by_file.values()), refusal_reason=refusal_reason
            )
        finally:  # also runs when the LLM call fails: answer stays None and the failed call is in llm_calls
            self.record_request_end(rag_request_id, answer_text, refusal_reason, started_at)

    def search_chunks(
        self, question: str, folder: str | None, top_k: int, min_similarity: float | None = None
    ) -> list[dict]:
        """Retrieval only (no LLM): the most relevant chunks above the similarity threshold, best first.
        min_similarity overrides the default threshold (evaluation passes 0 to see what the threshold hides)."""
        query_vector = self.embedder.embed_query(question)
        threshold = {} if min_similarity is None else {"min_similarity": min_similarity}
        return self.repository.search_similar(
            query_vector,
            embedding_model=self.embedder.model_name,
            dimension=self.embedder.dimension,
            top_k=top_k,
            folder=folder,
            **threshold,
        )

    def record_request_start(
        self, entry_point: str, question: str, folder: str | None, top_k: int, chunks: list[dict],
        eval_run_id: int | None,
    ) -> int | None:
        if self.ledger_repository is None:
            return None
        try:
            return self.ledger_repository.insert_rag_request(
                entry_point=entry_point,
                question=question,
                folder=folder,
                top_k=top_k,
                chunks_found=len(chunks),
                top_similarity=float(chunks[0]["similarity"]) if chunks else None,
                eval_run_id=eval_run_id,
            )
        except Exception as error:  # a broken ledger must never break an answer
            logger.warning("Could not record the request: %s", error)
            return None

    def record_request_end(
        self, rag_request_id: int | None, answer_text: str | None, refusal_reason: str | None, started_at: float
    ) -> None:
        if self.ledger_repository is None or rag_request_id is None:
            return
        try:
            self.ledger_repository.finish_rag_request(
                rag_request_id, answer_text, refusal_reason, int((time.perf_counter() - started_at) * 1000)
            )
        except Exception as error:
            logger.warning("Could not finish the request record: %s", error)

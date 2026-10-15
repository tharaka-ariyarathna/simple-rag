import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from rag.llm.base import LlmClient
from rag.llm.call_context import llm_call_context
from rag.repository.chunk_repository import ChunkRepository
from rag.repository.ledger_repository import LedgerRepository
from rag.constants import NO_RELEVANT_CONTEXT_ANSWER, RERANK_CANDIDATES
from rag.reranking.base import Reranker
from rag.embedding.base import EmbeddingModel
from rag.service.keyword_query import KeywordQueryBuilder
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

    Flow: embed question -> hybrid search (vector + keywords, with threshold) -> rerank the candidates
    -> no chunks? fixed answer -> build prompt -> LLM -> answer + sources.
    All dependencies are injected (built once in api/main.py lifespan).
    """

    def __init__(
        self,
        repository: ChunkRepository,
        embedder: EmbeddingModel,
        prompt_builder: PromptBuilder,
        llm_client: LlmClient,
        ledger_repository: LedgerRepository | None = None,  # None = do not record requests
        keyword_query_builder: KeywordQueryBuilder | None = None,  # None = vector search only
        reranker: Reranker | None = None,  # None = keep the hybrid search order
        rerank_candidates: int = RERANK_CANDIDATES,
    ):
        self.repository = repository
        self.embedder = embedder
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.ledger_repository = ledger_repository
        self.keyword_query_builder = keyword_query_builder
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates

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
        conversation_id: str | None = None,
        standalone_question: str | None = None,  # a follow-up rewritten to make sense alone: searched instead
    ) -> RagAnswer:
        started_at = time.perf_counter()
        question_to_answer = standalone_question or question
        chunks = self.search_chunks(question_to_answer, folder, top_k)
        rag_request_id = self.record_request_start(
            entry_point, question, folder, top_k, chunks, eval_run_id, conversation_id,
            standalone_question if standalone_question and standalone_question != question else None,
        )
        answer_text = None
        refusal_reason = None
        try:
            if not chunks:
                refusal_reason = REFUSAL_REASON_RETRIEVAL
                answer_text = NO_RELEVANT_CONTEXT_ANSWER
                return RagAnswer(answer=answer_text, refusal_reason=refusal_reason)
            prompt = self.prompt_builder.build(question_to_answer, chunks)
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
        self, question: str, folder: str | None, top_k: int, min_similarity: float | None = None,
        use_keywords: bool = True, use_reranker: bool = True,
    ) -> list[dict]:
        """Retrieval only (no LLM): the most relevant chunks above the similarity threshold, best first.
        min_similarity overrides the default threshold (evaluation passes 0 to see what the threshold hides).
        use_keywords=False forces vector-only search and use_reranker=False keeps the search order
        (JD matching: a whole JD is neither a keyword query nor short enough for the reranker)."""
        rerank = use_reranker and self.reranker is not None
        query_vector = self.embedder.embed_query(question)
        threshold = {} if min_similarity is None else {"min_similarity": min_similarity}
        keyword_terms = (
            self.keyword_query_builder.terms_for(question, self.embedder.model_name)
            if use_keywords and self.keyword_query_builder is not None else []
        )
        chunks = self.repository.search_similar(
            query_vector,
            embedding_model=self.embedder.model_name,
            dimension=self.embedder.dimension,
            # the reranker gets a wider pool than the caller asked for; the threshold still applies to all of it
            top_k=max(top_k, self.rerank_candidates) if rerank else top_k,
            folder=folder,
            keyword_terms=keyword_terms,
            **threshold,
        )
        return self.rerank(question, chunks)[:top_k] if rerank else chunks

    def rerank(self, question: str, chunks: list[dict]) -> list[dict]:
        """Candidates reordered by the reranker, best first. Each gets a rerank_score; similarity is unchanged."""
        if not chunks:
            return chunks
        scores = self.reranker.score(question, [chunk["content"] for chunk in chunks])
        scored = [{**chunk, "rerank_score": score} for chunk, score in zip(chunks, scores, strict=True)]
        return sorted(scored, key=lambda chunk: chunk["rerank_score"], reverse=True)  # stable: ties keep search order

    def record_request_start(
        self, entry_point: str, question: str, folder: str | None, top_k: int, chunks: list[dict],
        eval_run_id: int | None, conversation_id: str | None = None, standalone_question: str | None = None,
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
                # max, not chunks[0]: with hybrid ordering the first chunk is not always the most similar
                top_similarity=max(float(chunk["similarity"]) for chunk in chunks) if chunks else None,
                eval_run_id=eval_run_id,
                conversation_id=conversation_id,
                standalone_question=standalone_question,
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

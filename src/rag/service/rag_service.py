from dataclasses import dataclass, field
from pathlib import Path
from rag.llm.base import LlmClient
from rag.repository.chunk_repository import ChunkRepository
from rag.constants import NO_RELEVANT_CONTEXT_ANSWER
from rag.embedding.base import EmbeddingModel
from rag.service.prompt_builder import PromptBuilder

@dataclass
class SourceReference:
    file_name: str
    folder: str
    similarity: float


@dataclass
class RagAnswer:
    answer: str
    sources: list[SourceReference] = field(default_factory=list)


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
    ):
        self.repository = repository
        self.embedder = embedder
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client

    def answer_question(self, question: str, folder: str | None, top_k: int) -> RagAnswer:
        query_vector = self.embedder.embed_query(question)
        chunks = self.repository.search_similar(
            query_vector,
            embedding_model=self.embedder.model_name,
            dimension=self.embedder.dimension,
            top_k=top_k,
            folder=folder,
        )
        if not chunks:
            return RagAnswer(answer=NO_RELEVANT_CONTEXT_ANSWER)
        prompt = self.prompt_builder.build(question, chunks)
        answer_text = self.llm_client.generate(prompt.system, prompt.user)
        sources_by_file: dict[str, SourceReference] = {}
        for chunk in chunks:
            file_name = Path(chunk["source"]).name
            if file_name not in sources_by_file:
                sources_by_file[file_name] = SourceReference(file_name, chunk["folder"], chunk["similarity"])
        return RagAnswer(answer=answer_text, sources=list(sources_by_file.values()))

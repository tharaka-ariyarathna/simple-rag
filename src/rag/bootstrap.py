import os
from dataclasses import dataclass
from dotenv import load_dotenv
from rag.constants import QWEN_TIMEOUT_SECONDS
from rag.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.llm.base import LlmClient
from rag.llm.fallback_client import FallbackLlmClient
from rag.llm.gemini_client import GeminiLlmClient
from rag.llm.logging_client import LoggingLlmClient
from rag.llm.ollama_client import OllamaLlmClient
from rag.repository.chunk_repository import ChunkRepository
from rag.repository.db import Database
from rag.repository.ledger_repository import LedgerRepository
from rag.service.prompt_builder import PromptBuilder
from rag.service.rag_service import RagService


@dataclass
class RagComponents:
    """Long-lived objects shared by every entry point (FastAPI, MCP server).
    The resources are kept alongside the service so the entry point can close them on shutdown."""

    rag_service: RagService
    database: Database
    llm_client: LlmClient

    def close(self) -> None:
        self.llm_client.close()
        self.database.close()


def build_rag_components() -> RagComponents:
    """Composition root: reads .env and wires DB pool, embedding model and LLM clients into RagService, once."""
    load_dotenv()
    database = Database(os.environ["DATABASE_URL"])
    ledger_repository = LedgerRepository(database)
    # Each real client is wrapped on its own, so a fallback shows as two ledger rows (failed primary, fallback)
    llm_client = FallbackLlmClient(
        primary_client=LoggingLlmClient(
            GeminiLlmClient(api_key=os.environ.get("GEMINI_API_KEY")), "gemini", "primary", ledger_repository
        ),
        fallback_client=LoggingLlmClient(
            OllamaLlmClient(timeout_seconds=QWEN_TIMEOUT_SECONDS), "ollama", "fallback", ledger_repository
        ),
    )
    rag_service = RagService(
        repository=ChunkRepository(database),
        embedder=SentenceTransformerEmbedder(),
        prompt_builder=PromptBuilder(),
        llm_client=llm_client,
        ledger_repository=ledger_repository,
    )
    return RagComponents(rag_service=rag_service, database=database, llm_client=llm_client)

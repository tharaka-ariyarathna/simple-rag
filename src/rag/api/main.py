from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from rag.api.routes import router
from rag.llm.fallback_client import FallbackLlmClient
from rag.llm.gemini_client import GeminiLlmClient
from rag.llm.ollama_client import OllamaLlmClient
from rag.repository.db import Database
from rag.repository.chunk_repository import ChunkRepository
from rag.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.service.prompt_builder import PromptBuilder
from rag.service.rag_service import RagService
from rag.constants import QWEN_TIMEOUT_SECONDS

load_dotenv()
database_url = os.environ["DATABASE_URL"]
gemini_api_key = os.environ.get("GEMINI_API_KEY")  # optional; if not set, Gemini LLM won't be used

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: build long-lived objects once (DB pool, embedding model, LLM client)
    # and inject them into RagService; every request reuses them.
    database = Database(database_url)
    chunk_repository = ChunkRepository(database)
    embedder = SentenceTransformerEmbedder()
    prompt_builder = PromptBuilder()
    llm_client = FallbackLlmClient(
        primary_client=GeminiLlmClient(api_key=gemini_api_key),
        fallback_client=OllamaLlmClient(timeout_seconds=QWEN_TIMEOUT_SECONDS)
    )
    app.state.rag_service = RagService(
        repository=chunk_repository, embedder=embedder, prompt_builder=prompt_builder, llm_client=llm_client
        )
    yield
    # Shutdown: release the HTTP client and the DB pool
    llm_client.close()
    database.close()


app = FastAPI(title="Simple RAG", lifespan=lifespan)
app.include_router(router)

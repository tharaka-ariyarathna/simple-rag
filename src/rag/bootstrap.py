import os
from collections.abc import Mapping
from dataclasses import dataclass
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from rag.constants import (
    CLOUD_QUESTION_GENERATION_DELAY_SECONDS, DEFAULT_ANSWER_LLM_CHAIN, DEFAULT_QUESTION_LLM_CHAIN, GEMINI_MODEL,
    GROQ_BASE_URL, GROQ_MODEL, OLLAMA_MODEL, OPENROUTER_BASE_URL, OPENROUTER_MODEL, QUESTION_GENERATION_DELAY_SECONDS,
    DB_POOL_MAX_SIZE, DB_SCHEMA, QUESTION_LLM_COOLDOWN_SECONDS, RERANKER_MODEL,
)
from rag.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.llm.base import LlmClient
from rag.llm.fallback_client import FallbackLlmClient
from rag.llm.gemini_client import GeminiLlmClient
from rag.llm.logging_client import LoggingLlmClient
from rag.llm.ollama_client import OllamaLlmClient
from rag.llm.openai_compatible_client import OpenAiCompatibleLlmClient
from rag.llm.retrying_client import RetryingLlmClient
from rag.memory.condenser import QuestionCondenser
from rag.memory.conversation_graph import ConversationGraph
from rag.reranking.base import Reranker
from rag.reranking.cross_encoder_reranker import CrossEncoderReranker
from rag.repository.chunk_repository import ChunkRepository
from rag.repository.db import Database
from rag.repository.ledger_repository import LedgerRepository
from rag.service.keyword_query import KeywordQueryBuilder
from rag.service.prompt_builder import PromptBuilder
from rag.service.rag_service import RagService

# Providers ANSWER_LLM_CHAIN / QUESTION_LLM_CHAIN may name, and the .env variable holding each one's API key (None = no key needed)
PROVIDER_API_KEY_VARIABLES = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "ollama": None,
}


class StartupConfigurationError(Exception):
    """.env is missing an API key the LLM chain needs, or names a provider that does not exist."""


@dataclass
class RagComponents:
    """Long-lived objects shared by every entry point (FastAPI, MCP server).
    The resources are kept alongside the service so the entry point can close them on shutdown."""

    rag_service: RagService
    database: Database
    llm_client: LlmClient
    conversation_graph: ConversationGraph
    checkpoint_pool: ConnectionPool

    def close(self) -> None:
        self.llm_client.close()
        self.checkpoint_pool.close()
        self.database.close()


@dataclass
class QuestionGenerationComponents:
    """What synthetic question generation needs: the database and the question LLM chain. No embedding model."""

    database: Database
    llm_client: LlmClient
    delay_seconds: float  # pause between chunks, to stay under a cloud provider's per-minute limit

    def close(self) -> None:
        self.llm_client.close()
        self.database.close()


def setting(environ: Mapping[str, str], name: str, default: str) -> str:
    """A .env value with surrounding spaces removed, or the default when it is missing or blank."""
    return (environ.get(name) or "").strip() or default


def answer_chain_providers(environ: Mapping[str, str]) -> list[str]:
    return chain_providers(environ, "ANSWER_LLM_CHAIN", DEFAULT_ANSWER_LLM_CHAIN)


def question_chain_providers(environ: Mapping[str, str]) -> list[str]:
    return chain_providers(environ, "QUESTION_LLM_CHAIN", DEFAULT_QUESTION_LLM_CHAIN)


def chain_providers(environ: Mapping[str, str], variable: str, default: str) -> list[str]:
    """Provider names from a chain variable, in fallback order. Fails when a name is unknown or a key is missing."""
    chain = setting(environ, variable, default)
    providers = [name.strip().lower() for name in chain.split(",") if name.strip()]
    unknown = [name for name in providers if name not in PROVIDER_API_KEY_VARIABLES]
    if not providers or unknown:
        raise StartupConfigurationError(
            f"{variable}={chain!r} must list providers from: {', '.join(PROVIDER_API_KEY_VARIABLES)}"
            + (f" (unknown: {', '.join(unknown)})" if unknown else "")
        )
    missing_keys = [
        PROVIDER_API_KEY_VARIABLES[name] for name in providers
        if PROVIDER_API_KEY_VARIABLES[name] and not setting(environ, PROVIDER_API_KEY_VARIABLES[name], "")
    ]
    if missing_keys:
        raise StartupConfigurationError(
            f"Missing API keys for {variable}={','.join(providers)}: {', '.join(missing_keys)}. Add them to .env."
        )
    return providers


def build_provider_client(provider: str, environ: Mapping[str, str]) -> LlmClient:
    """One real provider client. Keys were already checked by chain_providers."""
    if provider == "groq":
        return OpenAiCompatibleLlmClient("Groq", GROQ_BASE_URL, setting(environ, "GROQ_API_KEY", ""),
                                         setting(environ, "GROQ_MODEL", GROQ_MODEL))
    if provider == "openrouter":
        return OpenAiCompatibleLlmClient("OpenRouter", OPENROUTER_BASE_URL, setting(environ, "OPENROUTER_API_KEY", ""),
                                         setting(environ, "OPENROUTER_MODEL", OPENROUTER_MODEL))
    if provider == "gemini":
        return GeminiLlmClient(api_key=setting(environ, "GEMINI_API_KEY", ""),
                               model_name=setting(environ, "GEMINI_MODEL", GEMINI_MODEL))
    if provider == "ollama":
        return OllamaLlmClient(model_name=setting(environ, "OLLAMA_MODEL", OLLAMA_MODEL))
    raise StartupConfigurationError(f"Unknown LLM provider: {provider!r}")


def fallback_role(position: int) -> str:
    return "primary" if position == 0 else f"fallback_{position}"


def build_answer_llm_client(
    providers: list[str], ledger_repository: LedgerRepository, environ: Mapping[str, str]
) -> FallbackLlmClient:
    # Each real client is wrapped on its own, so a fallback shows as one ledger row per provider tried
    return FallbackLlmClient([
        LoggingLlmClient(build_provider_client(provider, environ), provider, fallback_role(position), ledger_repository)
        for position, provider in enumerate(providers)
    ])


def build_question_llm_client(
    providers: list[str], ledger_repository: LedgerRepository, environ: Mapping[str, str]
) -> FallbackLlmClient:
    """Each provider is retried with backoff before the next one is tried. A provider that still fails is skipped
    for QUESTION_LLM_COOLDOWN_SECONDS, so a capped OpenRouter is not retried for every remaining chunk."""
    return FallbackLlmClient([
        RetryingLlmClient(
            LoggingLlmClient(build_provider_client(provider, environ), provider, fallback_role(position),
                             ledger_repository)
        )
        for position, provider in enumerate(providers)
    ], cooldown_seconds=QUESTION_LLM_COOLDOWN_SECONDS)


def question_generation_delay(providers: list[str]) -> float:
    """Cloud providers (the ones that need an API key) get a pause between chunks; a local-only chain does not."""
    if any(PROVIDER_API_KEY_VARIABLES[provider] for provider in providers):
        return CLOUD_QUESTION_GENERATION_DELAY_SECONDS
    return QUESTION_GENERATION_DELAY_SECONDS


def build_reranker(environ: Mapping[str, str]) -> Reranker | None:
    """RERANKER_MODEL from .env (default BAAI/bge-reranker-base); "none" switches reranking off, e.g. to measure
    hybrid search alone with the retrieval eval."""
    model_name = setting(environ, "RERANKER_MODEL", RERANKER_MODEL)
    return None if model_name.lower() == "none" else CrossEncoderReranker(model_name)


def build_checkpointer(database_url: str) -> tuple[PostgresSaver, ConnectionPool]:
    """LangGraph's conversation memory, in the same database. It needs its own autocommit connections, which the
    app pool does not use. setup() creates its tables (checkpoints, checkpoint_blobs, ...) if they are missing."""
    pool = ConnectionPool(
        database_url, max_size=DB_POOL_MAX_SIZE, open=True,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                "options": f"-c search_path={DB_SCHEMA}"},
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    return checkpointer, pool


def build_rag_components() -> RagComponents:
    """Composition root: reads .env and wires DB pool, embedding model and LLM clients into RagService, once."""
    load_dotenv()
    providers = answer_chain_providers(os.environ)  # before anything slow starts: a missing key fails at once
    database = Database(os.environ["DATABASE_URL"])
    ledger_repository = LedgerRepository(database)
    llm_client = build_answer_llm_client(providers, ledger_repository, os.environ)
    chunk_repository = ChunkRepository(database)
    rag_service = RagService(
        repository=chunk_repository,
        embedder=SentenceTransformerEmbedder(),
        prompt_builder=PromptBuilder(),
        llm_client=llm_client,
        ledger_repository=ledger_repository,
        keyword_query_builder=KeywordQueryBuilder(chunk_repository),
        reranker=build_reranker(os.environ),
    )
    checkpointer, checkpoint_pool = build_checkpointer(os.environ["DATABASE_URL"])
    conversation_graph = ConversationGraph(rag_service, QuestionCondenser(llm_client), checkpointer)
    return RagComponents(rag_service=rag_service, database=database, llm_client=llm_client,
                         conversation_graph=conversation_graph, checkpoint_pool=checkpoint_pool)


def build_question_generation_components() -> QuestionGenerationComponents:
    load_dotenv()
    providers = question_chain_providers(os.environ)  # before the database: a missing key fails at once
    database = Database(os.environ["DATABASE_URL"])
    llm_client = build_question_llm_client(providers, LedgerRepository(database), os.environ)
    return QuestionGenerationComponents(database=database, llm_client=llm_client,
                                        delay_seconds=question_generation_delay(providers))

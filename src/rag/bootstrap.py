import os
from collections.abc import Mapping
from dataclasses import dataclass
from dotenv import load_dotenv
from rag.constants import (
    DEFAULT_ANSWER_LLM_CHAIN, GEMINI_MODEL, GROQ_BASE_URL, GROQ_MODEL, OLLAMA_MODEL, OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)
from rag.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.llm.base import LlmClient
from rag.llm.fallback_client import FallbackLlmClient
from rag.llm.gemini_client import GeminiLlmClient
from rag.llm.logging_client import LoggingLlmClient
from rag.llm.ollama_client import OllamaLlmClient
from rag.llm.openai_compatible_client import OpenAiCompatibleLlmClient
from rag.llm.retrying_client import RetryingLlmClient
from rag.repository.chunk_repository import ChunkRepository
from rag.repository.db import Database
from rag.repository.ledger_repository import LedgerRepository
from rag.service.keyword_query import KeywordQueryBuilder
from rag.service.prompt_builder import PromptBuilder
from rag.service.rag_service import RagService

# Providers ANSWER_LLM_CHAIN may name, and the .env variable holding each one's API key (None = no key needed)
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

    def close(self) -> None:
        self.llm_client.close()
        self.database.close()


@dataclass
class QuestionGenerationComponents:
    """What synthetic question generation needs: the database and the local LLM. No embedding model, no cloud keys."""

    database: Database
    llm_client: LlmClient

    def close(self) -> None:
        self.llm_client.close()
        self.database.close()


def setting(environ: Mapping[str, str], name: str, default: str) -> str:
    """A .env value with surrounding spaces removed, or the default when it is missing or blank."""
    return (environ.get(name) or "").strip() or default


def answer_chain_providers(environ: Mapping[str, str]) -> list[str]:
    """Provider names from ANSWER_LLM_CHAIN, in fallback order. Fails when a name is unknown or a key is missing."""
    chain = setting(environ, "ANSWER_LLM_CHAIN", DEFAULT_ANSWER_LLM_CHAIN)
    providers = [name.strip().lower() for name in chain.split(",") if name.strip()]
    unknown = [name for name in providers if name not in PROVIDER_API_KEY_VARIABLES]
    if not providers or unknown:
        raise StartupConfigurationError(
            f"ANSWER_LLM_CHAIN={chain!r} must list providers from: {', '.join(PROVIDER_API_KEY_VARIABLES)}"
            + (f" (unknown: {', '.join(unknown)})" if unknown else "")
        )
    missing_keys = [
        PROVIDER_API_KEY_VARIABLES[name] for name in providers
        if PROVIDER_API_KEY_VARIABLES[name] and not setting(environ, PROVIDER_API_KEY_VARIABLES[name], "")
    ]
    if missing_keys:
        raise StartupConfigurationError(
            f"Missing API keys for ANSWER_LLM_CHAIN={','.join(providers)}: {', '.join(missing_keys)}. Add them to .env."
        )
    return providers


def build_provider_client(provider: str, environ: Mapping[str, str]) -> LlmClient:
    """One real provider client. Keys were already checked by answer_chain_providers."""
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


def build_question_llm_client(ledger_repository: LedgerRepository, environ: Mapping[str, str]) -> RetryingLlmClient:
    """Question generation runs on the local model only: retried with backoff, never sent to a cloud provider."""
    return RetryingLlmClient(
        LoggingLlmClient(build_provider_client("ollama", environ), "ollama", "primary", ledger_repository)
    )


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
    )
    return RagComponents(rag_service=rag_service, database=database, llm_client=llm_client)


def build_question_generation_components() -> QuestionGenerationComponents:
    load_dotenv()
    database = Database(os.environ["DATABASE_URL"])
    llm_client = build_question_llm_client(LedgerRepository(database), os.environ)
    return QuestionGenerationComponents(database=database, llm_client=llm_client)

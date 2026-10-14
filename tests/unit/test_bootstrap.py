"""Startup configuration: which LLM chain .env asks for, and failing loudly when it is incomplete.
The environment is passed in as a plain dict, so these tests never read the real .env."""
import pytest
from rag.bootstrap import (
    StartupConfigurationError, answer_chain_providers, build_answer_llm_client, build_question_llm_client,
)
from rag.constants import GROQ_MODEL, OLLAMA_MODEL, OPENROUTER_MODEL
from rag.llm.fallback_client import FallbackLlmClient
from rag.llm.ollama_client import OllamaLlmClient
from rag.llm.openai_compatible_client import OpenAiCompatibleLlmClient
from rag.llm.retrying_client import RetryingLlmClient
from tests.fakes import FakeLedgerRepository

KEYS = {"GROQ_API_KEY": "groq-key", "OPENROUTER_API_KEY": "openrouter-key"}


class TestAnswerChainProviders:
    def test_default_chain_is_groq_then_openrouter_then_local(self):
        assert answer_chain_providers(KEYS) == ["groq", "openrouter", "ollama"]

    def test_chain_can_be_changed_in_env(self):
        assert answer_chain_providers({"ANSWER_LLM_CHAIN": " Ollama , groq ", **KEYS}) == ["ollama", "groq"]

    def test_missing_keys_fail_startup_and_name_every_missing_variable(self):
        with pytest.raises(StartupConfigurationError) as raised:
            answer_chain_providers({"GROQ_API_KEY": "   "})

        assert "GROQ_API_KEY" in str(raised.value) and "OPENROUTER_API_KEY" in str(raised.value)

    def test_only_keys_of_providers_in_the_chain_are_required(self):
        assert answer_chain_providers({"ANSWER_LLM_CHAIN": "ollama"}) == ["ollama"]

    def test_gemini_needs_its_key_when_it_is_in_the_chain(self):
        with pytest.raises(StartupConfigurationError, match="GEMINI_API_KEY"):
            answer_chain_providers({"ANSWER_LLM_CHAIN": "gemini,ollama"})

    @pytest.mark.parametrize("chain", ["groq,cerebras", " , "])
    def test_unknown_or_empty_chain_fails_startup(self, chain):
        with pytest.raises(StartupConfigurationError, match="ANSWER_LLM_CHAIN"):
            answer_chain_providers({"ANSWER_LLM_CHAIN": chain, **KEYS})


def test_answer_client_logs_each_provider_with_its_position_in_the_chain():
    client = build_answer_llm_client(["groq", "openrouter", "ollama"], FakeLedgerRepository(), KEYS)
    try:
        assert isinstance(client, FallbackLlmClient)
        assert [(logged.provider, logged.role) for logged in client.clients] == [
            ("groq", "primary"), ("openrouter", "fallback_1"), ("ollama", "fallback_2")]
        groq, openrouter, ollama = (logged.inner_client for logged in client.clients)
        assert isinstance(groq, OpenAiCompatibleLlmClient) and groq.model_name == GROQ_MODEL
        assert groq.http_client.headers["authorization"] == "Bearer groq-key"
        assert isinstance(openrouter, OpenAiCompatibleLlmClient) and openrouter.model_name == OPENROUTER_MODEL
        assert str(openrouter.http_client.base_url).startswith("https://openrouter.ai/api/v1")
        assert isinstance(ollama, OllamaLlmClient) and ollama.model_name == OLLAMA_MODEL
        assert groq.http_client.timeout.read == 60 and ollama.http_client.timeout.read == 240
    finally:
        client.close()


def test_model_ids_can_be_overridden_in_env():
    environment = {**KEYS, "GROQ_MODEL": "llama-3.3-70b", "OPENROUTER_MODEL": "google/gemma-4-31b-it:free",
                   "OLLAMA_MODEL": "qwen3:4b"}
    client = build_answer_llm_client(["groq", "openrouter", "ollama"], FakeLedgerRepository(), environment)
    try:
        assert client.model_name == "llama-3.3-70b -> google/gemma-4-31b-it:free -> qwen3:4b"
    finally:
        client.close()


def test_question_client_is_the_local_model_with_retries_and_needs_no_cloud_keys():
    client = build_question_llm_client(FakeLedgerRepository(), {})
    try:
        assert isinstance(client, RetryingLlmClient)
        assert client.retry_delays_seconds == [2, 4, 8]
        assert (client.inner_client.provider, client.inner_client.role) == ("ollama", "primary")
        assert isinstance(client.inner_client.inner_client, OllamaLlmClient)
    finally:
        client.close()

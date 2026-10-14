import pytest
from rag.llm.base import LlmTimeoutError, LlmUnavailableError
from rag.llm.call_context import current_purpose, current_rag_request_id, llm_call_context
from rag.llm.fallback_client import FallbackLlmClient
from rag.llm.logging_client import LoggingLlmClient
from tests.fakes import FakeLedgerRepository, FakeLlmClient


class TestFallback:
    def test_primary_answer_is_used_and_fallback_is_not_called(self):
        primary, fallback = FakeLlmClient(answer="from primary"), FakeLlmClient(answer="from fallback")

        result = FallbackLlmClient(primary, fallback).generate("s", "u")

        assert result.text == "from primary"
        assert fallback.calls == []

    @pytest.mark.parametrize("error", [LlmUnavailableError("down"), LlmTimeoutError("slow")])
    def test_llm_error_from_primary_switches_to_fallback(self, error):
        fallback = FakeLlmClient(answer="from fallback")

        result = FallbackLlmClient(FakeLlmClient(error=error), fallback).generate("s", "u")

        assert result.text == "from fallback"
        assert fallback.calls[0]["system"] == "s" and fallback.calls[0]["user"] == "u"

    def test_programming_errors_are_not_hidden_by_the_fallback(self):
        fallback = FakeLlmClient()

        with pytest.raises(KeyError):
            FallbackLlmClient(FakeLlmClient(error=KeyError("bug")), fallback).generate("s", "u")
        assert fallback.calls == []

    def test_when_both_fail_the_fallback_error_reaches_the_caller(self):
        client = FallbackLlmClient(FakeLlmClient(error=LlmUnavailableError("primary")),
                                   FakeLlmClient(error=LlmTimeoutError("fallback")))

        with pytest.raises(LlmTimeoutError, match="fallback"):
            client.generate("s", "u")

    def test_model_name_and_close_cover_both_clients(self):
        primary, fallback = FakeLlmClient(model_name="gemini"), FakeLlmClient(model_name="qwen")
        client = FallbackLlmClient(primary, fallback)

        client.close()

        assert client.model_name == "gemini (fallback: qwen)"
        assert primary.closed and fallback.closed


class TestLogging:
    def test_successful_call_is_recorded_with_tokens_and_context(self):
        ledger = FakeLedgerRepository()
        client = LoggingLlmClient(FakeLlmClient(model_name="qwen", input_tokens=30, output_tokens=4),
                                  "ollama", "fallback", ledger)

        with llm_call_context("question_generation", rag_request_id=12):
            client.generate("s", "u")

        call = ledger.llm_calls[0]
        assert call["purpose"] == "question_generation" and call["rag_request_id"] == 12
        assert (call["provider"], call["role"], call["model"]) == ("ollama", "fallback", "qwen")
        assert (call["input_tokens"], call["output_tokens"]) == (30, 4)
        assert call["success"] is True and call["error_type"] is None
        assert call["latency_ms"] >= 0

    @pytest.mark.parametrize("error, error_type", [
        (LlmTimeoutError("slow"), "timeout"),
        (LlmUnavailableError("down"), "unavailable"),
        (ValueError("bug"), "ValueError"),
    ])
    def test_failed_call_is_recorded_and_the_error_is_re_raised(self, error, error_type):
        ledger = FakeLedgerRepository()
        client = LoggingLlmClient(FakeLlmClient(error=error), "gemini", "primary", ledger)

        with pytest.raises(type(error)):
            client.generate("s", "u")

        assert ledger.llm_calls[0]["success"] is False
        assert ledger.llm_calls[0]["error_type"] == error_type
        assert ledger.llm_calls[0]["input_tokens"] is None

    def test_broken_ledger_never_breaks_the_call(self):
        client = LoggingLlmClient(FakeLlmClient(answer="still fine"), "gemini", "primary",
                                  FakeLedgerRepository(fail=True))

        assert client.generate("s", "u").text == "still fine"

    def test_close_closes_the_wrapped_client(self):
        inner = FakeLlmClient()

        LoggingLlmClient(inner, "gemini", "primary", FakeLedgerRepository()).close()

        assert inner.closed


def test_call_context_defaults_and_is_restored_after_the_block():
    assert (current_purpose.get(), current_rag_request_id.get()) == ("answer", None)

    with llm_call_context("question_generation", 5):
        with llm_call_context("answer", 6):
            assert (current_purpose.get(), current_rag_request_id.get()) == ("answer", 6)
        assert (current_purpose.get(), current_rag_request_id.get()) == ("question_generation", 5)

    assert (current_purpose.get(), current_rag_request_id.get()) == ("answer", None)

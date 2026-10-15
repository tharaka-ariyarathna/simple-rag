import pytest
from rag.llm.base import LlmConfigurationError, LlmTimeoutError, LlmUnavailableError
from rag.llm.call_context import current_purpose, current_rag_request_id, llm_call_context
from rag.llm.fallback_client import FallbackLlmClient
from rag.llm.logging_client import LoggingLlmClient
from rag.llm.retrying_client import RetryingLlmClient
from tests.fakes import FakeLedgerRepository, FakeLlmClient


class SequenceLlmClient(FakeLlmClient):
    """Raises the given errors on the first calls, then answers."""

    def __init__(self, errors: list[Exception], answer: str = "finally"):
        super().__init__(answer=answer)
        self.errors = list(errors)

    def generate(self, system_prompt, user_prompt):
        self.error = self.errors.pop(0) if self.errors else None
        return super().generate(system_prompt, user_prompt)


class TestFallback:
    def test_primary_answer_is_used_and_the_others_are_not_called(self):
        primary, second, third = FakeLlmClient(answer="from primary"), FakeLlmClient(), FakeLlmClient()

        result = FallbackLlmClient([primary, second, third]).generate("s", "u")

        assert result.text == "from primary"
        assert second.calls == [] and third.calls == []

    @pytest.mark.parametrize("error", [LlmUnavailableError("down"), LlmTimeoutError("slow")])
    def test_recoverable_error_moves_on_to_the_next_client(self, error):
        second = FakeLlmClient(answer="from second")

        result = FallbackLlmClient([FakeLlmClient(error=error), second]).generate("s", "u")

        assert result.text == "from second"
        assert second.calls[0]["system"] == "s" and second.calls[0]["user"] == "u"

    def test_third_client_answers_when_the_first_two_fail(self):
        third = FakeLlmClient(answer="from local")
        client = FallbackLlmClient([FakeLlmClient(error=LlmUnavailableError("rate limited")),
                                    FakeLlmClient(error=LlmTimeoutError("slow")), third])

        assert client.generate("s", "u").text == "from local"

    def test_configuration_error_stops_without_trying_the_others(self):
        second = FakeLlmClient()

        with pytest.raises(LlmConfigurationError):
            FallbackLlmClient([FakeLlmClient(error=LlmConfigurationError("401 bad key")), second]).generate("s", "u")
        assert second.calls == []

    def test_configuration_error_from_a_later_client_also_stops_the_chain(self):
        third = FakeLlmClient()
        client = FallbackLlmClient([FakeLlmClient(error=LlmUnavailableError("down")),
                                    FakeLlmClient(error=LlmConfigurationError("400")), third])

        with pytest.raises(LlmConfigurationError):
            client.generate("s", "u")
        assert third.calls == []

    def test_programming_errors_are_not_hidden_by_the_fallback(self):
        second = FakeLlmClient()

        with pytest.raises(KeyError):
            FallbackLlmClient([FakeLlmClient(error=KeyError("bug")), second]).generate("s", "u")
        assert second.calls == []

    def test_when_all_fail_the_last_error_reaches_the_caller(self):
        client = FallbackLlmClient([FakeLlmClient(error=LlmUnavailableError("first")),
                                    FakeLlmClient(error=LlmUnavailableError("second")),
                                    FakeLlmClient(error=LlmTimeoutError("last"))])

        with pytest.raises(LlmTimeoutError, match="last"):
            client.generate("s", "u")

    def test_model_name_and_close_cover_every_client(self):
        clients = [FakeLlmClient(model_name="gpt-oss"), FakeLlmClient(model_name="nemotron"),
                   FakeLlmClient(model_name="llama")]
        client = FallbackLlmClient(clients)

        client.close()

        assert client.model_name == "gpt-oss -> nemotron -> llama"
        assert all(inner.closed for inner in clients)

    def test_needs_at_least_one_client(self):
        with pytest.raises(ValueError):
            FallbackLlmClient([])

    def test_without_cooldown_a_failed_client_is_tried_again_on_the_next_call(self):
        primary = SequenceLlmClient([LlmUnavailableError("busy")], answer="from primary")
        client = FallbackLlmClient([primary, FakeLlmClient(answer="from second")])

        assert client.generate("s", "u").text == "from second"
        assert client.generate("s", "u").text == "from primary"


class TestFallbackCooldown:
    def setup_method(self):
        self.now = 1000.0

    def clock(self):
        return self.now

    def test_failed_client_is_skipped_until_its_cooldown_ends(self):
        capped = SequenceLlmClient([LlmUnavailableError("429 daily cap")], answer="from openrouter")
        local = FakeLlmClient(answer="from local")
        client = FallbackLlmClient([capped, local], cooldown_seconds=900, clock=self.clock)

        assert client.generate("s", "u").text == "from local"
        self.now += 899
        assert client.generate("s", "u").text == "from local"
        assert len(capped.calls) == 1  # not asked again during the cooldown
        self.now += 1
        assert client.generate("s", "u").text == "from openrouter"

    def test_when_every_client_is_cooling_down_all_are_tried_anyway(self):
        first = SequenceLlmClient([LlmUnavailableError("a"), LlmUnavailableError("a again")], answer="first ok")
        second = SequenceLlmClient([LlmTimeoutError("b")], answer="second ok")
        client = FallbackLlmClient([first, second], cooldown_seconds=900, clock=self.clock)

        with pytest.raises(LlmTimeoutError):
            client.generate("s", "u")  # both fail and start cooling down
        assert client.generate("s", "u").text == "second ok"  # first fails again, second recovered
        assert len(first.calls) == 2 and len(second.calls) == 2

    def test_configuration_error_does_not_start_a_cooldown_it_stops(self):
        broken = FakeLlmClient(error=LlmConfigurationError("401"))
        client = FallbackLlmClient([broken, FakeLlmClient()], cooldown_seconds=900, clock=self.clock)

        with pytest.raises(LlmConfigurationError):
            client.generate("s", "u")
        assert client.skipped_until == [0.0, 0.0]


class TestRetrying:
    def test_retries_recoverable_errors_with_growing_delays_until_it_answers(self):
        waits = []
        inner = SequenceLlmClient([LlmTimeoutError("slow"), LlmUnavailableError("busy")])

        result = RetryingLlmClient(inner, retry_delays_seconds=[2, 4, 8], sleep=waits.append).generate("s", "u")

        assert result.text == "finally"
        assert len(inner.calls) == 3
        assert waits == [2, 4]

    def test_gives_up_after_the_last_retry_with_the_last_error(self):
        waits = []
        inner = SequenceLlmClient([LlmUnavailableError(f"down {number}") for number in range(1, 5)])

        with pytest.raises(LlmUnavailableError, match="down 4"):
            RetryingLlmClient(inner, retry_delays_seconds=[2, 4, 8], sleep=waits.append).generate("s", "u")

        assert len(inner.calls) == 4  # the first try + 3 retries
        assert waits == [2, 4, 8]

    def test_configuration_error_is_not_retried(self):
        waits = []
        inner = SequenceLlmClient([LlmConfigurationError("404 model not pulled")])

        with pytest.raises(LlmConfigurationError):
            RetryingLlmClient(inner, retry_delays_seconds=[2, 4, 8], sleep=waits.append).generate("s", "u")

        assert len(inner.calls) == 1 and waits == []

    def test_every_attempt_is_a_ledger_row_when_it_wraps_the_logging_client(self):
        ledger = FakeLedgerRepository()
        inner = SequenceLlmClient([LlmTimeoutError("slow")])
        client = RetryingLlmClient(LoggingLlmClient(inner, "ollama", "primary", ledger), retry_delays_seconds=[1],
                                   sleep=lambda seconds: None)

        client.generate("s", "u")

        assert [call["success"] for call in ledger.llm_calls] == [False, True]

    def test_model_name_and_close_come_from_the_wrapped_client(self):
        inner = FakeLlmClient(model_name="llama3.2:3b")
        client = RetryingLlmClient(inner)

        client.close()

        assert client.model_name == "llama3.2:3b" and inner.closed


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
        (LlmConfigurationError("401"), "configuration"),
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

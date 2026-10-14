import logging
import time
from rag.llm.base import LlmClient, LlmConfigurationError, LlmResult, LlmTimeoutError, LlmUnavailableError
from rag.llm.call_context import current_purpose, current_rag_request_id
from rag.repository.ledger_repository import LedgerRepository

logger = logging.getLogger(__name__)


class LoggingLlmClient:
    """LlmClient decorator: records every call (latency, tokens, model, success) in the llm_calls table.

    Wrap each real client (Groq, OpenRouter, Ollama...), not the FallbackLlmClient, so a fallback shows as one
    row per provider tried: the failed ones and the one that answered."""

    def __init__(self, inner_client: LlmClient, provider: str, role: str, ledger_repository: LedgerRepository):
        self.inner_client = inner_client
        self.provider = provider
        self.role = role  # primary / fallback_1 / fallback_2 ...
        self.ledger_repository = ledger_repository
        self.model_name = inner_client.model_name

    def generate(self, system_prompt: str, user_prompt: str) -> LlmResult:
        started_at = time.perf_counter()
        try:
            result = self.inner_client.generate(system_prompt, user_prompt)
        except Exception as error:
            self.record_call(started_at, success=False, error_type=self.error_type_of(error))
            raise
        self.record_call(started_at, success=True, result=result)
        return result

    def record_call(
        self, started_at: float, success: bool, result: LlmResult | None = None, error_type: str | None = None
    ) -> None:
        try:
            self.ledger_repository.insert_llm_call(
                rag_request_id=current_rag_request_id.get(),
                purpose=current_purpose.get(),
                provider=self.provider,
                model=result.model if result else self.model_name,
                role=self.role,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                input_tokens=result.input_tokens if result else None,
                output_tokens=result.output_tokens if result else None,
                success=success,
                error_type=error_type,
            )
        except Exception as ledger_error:  # a broken ledger must never break an answer
            logger.warning("Could not write the LLM call ledger: %s", ledger_error)

    @staticmethod
    def error_type_of(error: Exception) -> str:
        if isinstance(error, LlmTimeoutError):
            return "timeout"
        if isinstance(error, LlmUnavailableError):
            return "unavailable"
        if isinstance(error, LlmConfigurationError):
            return "configuration"
        return type(error).__name__

    def close(self) -> None:
        self.inner_client.close()

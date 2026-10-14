import logging
import time
from collections.abc import Callable, Sequence
from rag.constants import LLM_RETRY_DELAYS_SECONDS
from rag.llm.base import RECOVERABLE_LLM_ERRORS, LlmClient, LlmResult

logger = logging.getLogger(__name__)


class RetryingLlmClient:
    """LlmClient that retries one client on LlmUnavailableError / LlmTimeoutError, waiting longer each time.

    LlmConfigurationError is raised at once (a retry cannot fix it). After the last retry the error reaches
    the caller. Wrap a LoggingLlmClient with it, so every attempt gets its own ledger row."""

    def __init__(
        self,
        inner_client: LlmClient,
        retry_delays_seconds: Sequence[float] = LLM_RETRY_DELAYS_SECONDS,  # one entry per retry
        sleep: Callable[[float], None] = time.sleep,  # injected so tests do not wait
    ):
        self.inner_client = inner_client
        self.retry_delays_seconds = list(retry_delays_seconds)
        self.sleep = sleep
        self.model_name = inner_client.model_name

    def generate(self, system_prompt: str, user_prompt: str) -> LlmResult:
        attempts = len(self.retry_delays_seconds) + 1
        for attempt in range(1, attempts + 1):
            try:
                return self.inner_client.generate(system_prompt, user_prompt)
            except RECOVERABLE_LLM_ERRORS as error:
                if attempt == attempts:
                    raise
                delay = self.retry_delays_seconds[attempt - 1]
                logger.warning("LLM '%s' failed (attempt %d of %d), retrying in %ss: %s",
                               self.model_name, attempt, attempts, delay, error)
                self.sleep(delay)
        raise AssertionError("unreachable: the loop returns or raises")

    def close(self) -> None:
        self.inner_client.close()

import logging
import time
from collections.abc import Callable
from rag.llm.base import RECOVERABLE_LLM_ERRORS, LlmClient, LlmConfigurationError, LlmResult

logger = logging.getLogger(__name__)


class FallbackLlmClient:
    """LlmClient that tries each client in order until one answers.

    Only LlmUnavailableError / LlmTimeoutError move on to the next client. LlmConfigurationError (bad key,
    bad request, unknown model) is our mistake, so no other provider can fix it: it is raised at once.

    With cooldown_seconds, a client that failed is skipped for that long, so a long batch (question generation)
    does not wait on a capped provider for every item. When every client is cooling down, all are tried anyway."""

    def __init__(
        self,
        clients: list[LlmClient],
        cooldown_seconds: float | None = None,  # None = try every client on every call
        clock: Callable[[], float] = time.monotonic,  # injected so tests do not wait
    ):
        if not clients:
            raise ValueError("FallbackLlmClient needs at least one client")
        self.clients = clients
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self.skipped_until = [0.0] * len(clients)
        self.model_name = " -> ".join(client.model_name for client in clients)

    def generate(self, system_prompt: str, user_prompt: str) -> LlmResult:
        now = self.clock()
        available = [client for client, until in zip(self.clients, self.skipped_until) if now >= until]
        candidates = available or self.clients
        for position, client in enumerate(candidates):
            try:
                return client.generate(system_prompt, user_prompt)
            except LlmConfigurationError as error:
                logger.error("LLM '%s' is misconfigured, not falling back: %s", client.model_name, error)
                raise
            except RECOVERABLE_LLM_ERRORS as error:
                self.start_cooldown(client)
                if position == len(candidates) - 1:
                    raise  # every client failed: the last error reaches the caller (route -> 503/504)
                logger.warning("LLM '%s' failed, trying '%s': %s",
                               client.model_name, candidates[position + 1].model_name, error)
        raise AssertionError("unreachable: the loop returns or raises")

    def start_cooldown(self, client: LlmClient) -> None:
        if self.cooldown_seconds:
            self.skipped_until[self.clients.index(client)] = self.clock() + self.cooldown_seconds
            logger.warning("Skipping LLM '%s' for %ss", client.model_name, self.cooldown_seconds)

    def close(self) -> None:
        for client in self.clients:
            client.close()

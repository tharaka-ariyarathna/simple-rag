import logging
from rag.llm.base import RECOVERABLE_LLM_ERRORS, LlmClient, LlmConfigurationError, LlmResult

logger = logging.getLogger(__name__)


class FallbackLlmClient:
    """LlmClient that tries each client in order until one answers.

    Only LlmUnavailableError / LlmTimeoutError move on to the next client. LlmConfigurationError (bad key,
    bad request, unknown model) is our mistake, so no other provider can fix it: it is raised at once."""

    def __init__(self, clients: list[LlmClient]):
        if not clients:
            raise ValueError("FallbackLlmClient needs at least one client")
        self.clients = clients
        self.model_name = " -> ".join(client.model_name for client in clients)

    def generate(self, system_prompt: str, user_prompt: str) -> LlmResult:
        last_position = len(self.clients) - 1
        for position, client in enumerate(self.clients):
            try:
                return client.generate(system_prompt, user_prompt)
            except LlmConfigurationError as error:
                logger.error("LLM '%s' is misconfigured, not falling back: %s", client.model_name, error)
                raise
            except RECOVERABLE_LLM_ERRORS as error:
                if position == last_position:
                    raise  # every client failed: the last error reaches the route -> 503/504
                logger.warning("LLM '%s' failed, trying '%s': %s",
                               client.model_name, self.clients[position + 1].model_name, error)
        raise AssertionError("unreachable: the loop returns or raises")

    def close(self) -> None:
        for client in self.clients:
            client.close()

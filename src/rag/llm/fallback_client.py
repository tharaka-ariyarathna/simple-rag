import logging
from rag.llm.base import LlmClient, LlmError

logger = logging.getLogger(__name__)


class FallbackLlmClient:
    """LlmClient that tries a primary client and, if it fails, answers with a fallback client."""

    def __init__(self, primary_client: LlmClient, fallback_client: LlmClient):
        self.primary_client = primary_client
        self.fallback_client = fallback_client
        self.model_name = f"{primary_client.model_name} (fallback: {fallback_client.model_name})"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            return self.primary_client.generate(system_prompt, user_prompt)
        except LlmError as error:
            logger.warning("Primary LLM '%s' failed, using fallback '%s': %s",
                           self.primary_client.model_name, self.fallback_client.model_name, error)
            return self.fallback_client.generate(system_prompt, user_prompt)  # its LlmError reaches the route -> 503/504

    def close(self) -> None:
        self.primary_client.close()
        self.fallback_client.close()
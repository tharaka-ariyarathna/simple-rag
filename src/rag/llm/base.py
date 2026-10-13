from dataclasses import dataclass
from typing import Protocol


class LlmError(Exception):
    """Base for all LLM failures. Callers catch these, never a provider's own errors (httpx, SDKs...)."""


class LlmUnavailableError(LlmError):
    """The LLM could not be reached or returned an error (server down, model not pulled, bad response)."""


class LlmTimeoutError(LlmError):
    """The LLM did not answer within the configured timeout."""


@dataclass
class LlmResult:
    """One LLM answer plus what the provider reported about the call. Token counts are None when not reported."""

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LlmClient(Protocol):
    """What the RAG service relies on. Any LLM provider (Ollama, Claude, ...) must offer this.
    Implementations must raise LlmUnavailableError / LlmTimeoutError instead of their own errors."""

    model_name: str

    def generate(self, system_prompt: str, user_prompt: str) -> LlmResult: ...

    def close(self) -> None: ...

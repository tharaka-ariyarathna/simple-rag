import httpx

from rag.constants import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LOCAL_LLM_TIMEOUT_SECONDS,
    LLM_TEMPERATURE,
    LLM_CONTEXT_TOKENS
)
from rag.llm.base import LlmResult, LlmUnavailableError
from rag.llm.http_call import post_json


class OllamaLlmClient:
    """LlmClient backed by a local Ollama server (POST /api/chat)."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model_name: str = OLLAMA_MODEL,
        timeout_seconds: float = LOCAL_LLM_TIMEOUT_SECONDS,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.http_client = httpx.Client(base_url=base_url, timeout=timeout_seconds)

    def generate(self, system_prompt: str, user_prompt: str) -> LlmResult:
        request_body = {
        "model": self.model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": LLM_TEMPERATURE, "num_ctx": LLM_CONTEXT_TOKENS},
        }
        # 404 means the model isn't pulled: post_json raises LlmConfigurationError for it
        response_body = post_json(self.http_client, "/api/chat", request_body, "Ollama", self.model_name)
        try:
            text = response_body["message"]["content"]
        except (KeyError, TypeError) as error:  # no message.content in the JSON
            raise LlmUnavailableError(f"Unexpected response from Ollama: {error}") from error
        return LlmResult(
            text=text,
            model=self.model_name,
            input_tokens=response_body.get("prompt_eval_count"),
            output_tokens=response_body.get("eval_count"),
        )

    def close(self) -> None:
        self.http_client.close()

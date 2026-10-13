import httpx

from rag.constants import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LLM_TIMEOUT_SECONDS,
    LLM_TEMPERATURE,
    LLM_CONTEXT_TOKENS
)
from rag.llm.base import LlmResult, LlmTimeoutError, LlmUnavailableError


class OllamaLlmClient:
    """LlmClient backed by a local Ollama server (POST /api/chat)."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model_name: str = OLLAMA_MODEL,
        timeout_seconds: float = LLM_TIMEOUT_SECONDS,
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
        # TimeoutException is a subclass of HTTPError, so it must be caught first
        try:
            response = self.http_client.post("/api/chat", json=request_body)
            response.raise_for_status()  # 4xx/5xx -> HTTPStatusError (e.g. 404 when the model isn't pulled)
            response_body = response.json()
            return LlmResult(
                text=response_body["message"]["content"],
                model=self.model_name,
                input_tokens=response_body.get("prompt_eval_count"),
                output_tokens=response_body.get("eval_count"),
            )
        except httpx.TimeoutException as error:
            raise LlmTimeoutError(
                f"Ollama did not answer within {self.timeout_seconds} seconds"
            ) from error
        except httpx.HTTPStatusError as error:
            raise LlmUnavailableError(
                f"Ollama returned {error.response.status_code} for model '{self.model_name}': {error.response.text}"
            ) from error
        except httpx.HTTPError as error:  # connection refused, DNS, ... -> Ollama not running
            raise LlmUnavailableError(f"Could not reach Ollama at {self.base_url}: {error}") from error
        except (KeyError, ValueError) as error:  # not JSON, or no message.content in it
            raise LlmUnavailableError(f"Unexpected response from Ollama: {error}") from error

    def close(self) -> None:
        self.http_client.close()

import httpx
from rag.constants import CLOUD_LLM_TIMEOUT_SECONDS, LLM_TEMPERATURE
from rag.llm.base import LlmResult, LlmUnavailableError
from rag.llm.http_call import post_json


class OpenAiCompatibleLlmClient:
    """LlmClient for any OpenAI-compatible chat completions API (POST {base_url}/chat/completions).
    Used for Groq and OpenRouter: only the base URL, key and model differ."""

    def __init__(
        self,
        provider: str,  # shown in error messages, e.g. "Groq"
        base_url: str,  # e.g. "https://api.groq.com/openai/v1"
        api_key: str,
        model_name: str,
        timeout_seconds: float = CLOUD_LLM_TIMEOUT_SECONDS,
    ):
        self.provider = provider
        self.model_name = model_name
        self.http_client = httpx.Client(
            base_url=base_url, timeout=timeout_seconds, headers={"Authorization": f"Bearer {api_key}"}
        )

    def generate(self, system_prompt: str, user_prompt: str) -> LlmResult:
        request_body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": LLM_TEMPERATURE,
        }
        response_body = post_json(self.http_client, "/chat/completions", request_body, self.provider,
                                  self.model_name)
        try:
            choice = response_body["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:  # e.g. an {"error": ...} body with status 200
            raise LlmUnavailableError(f"Unexpected response from {self.provider}: {error}") from error
        if not text or not text.strip():  # a reasoning model can spend its whole budget thinking
            raise LlmUnavailableError(
                f"{self.provider} returned an empty answer (finish_reason: {choice.get('finish_reason')})"
            )
        usage = response_body.get("usage") or {}
        return LlmResult(
            text=text,
            model=response_body.get("model") or self.model_name,  # the model that actually answered
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    def close(self) -> None:
        self.http_client.close()

import httpx
from rag.constants import CLOUD_LLM_TIMEOUT_SECONDS, GEMINI_BASE_URL, GEMINI_MODEL, LLM_TEMPERATURE
from rag.llm.base import LlmResult, LlmUnavailableError
from rag.llm.http_call import post_json


class GeminiLlmClient:
    """LlmClient backed by the Google Gemini API (generateContent).
    Not in the default answer chain; add "gemini" to ANSWER_LLM_CHAIN to use it."""

    def __init__(
        self,
        api_key: str,
        base_url: str = GEMINI_BASE_URL,  # "https://generativelanguage.googleapis.com"
        model_name: str = GEMINI_MODEL,
        timeout_seconds: float = CLOUD_LLM_TIMEOUT_SECONDS,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.http_client = httpx.Client(
            base_url=base_url, timeout=timeout_seconds, headers={"x-goog-api-key": api_key}
        )

    def generate(self, system_prompt: str, user_prompt: str) -> LlmResult:
        request_body = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": LLM_TEMPERATURE},
        }
        response_body = post_json(
            self.http_client, f"/v1beta/models/{self.model_name}:generateContent", request_body, "Gemini",
            self.model_name,
        )
        try:
            text = response_body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as error:  # no candidates (e.g. blocked by safety filter)
            raise LlmUnavailableError(f"Unexpected response from Gemini: {error}") from error
        usage = response_body.get("usageMetadata", {})
        return LlmResult(
            text=text,
            model=self.model_name,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
        )

    def close(self) -> None:
        self.http_client.close()

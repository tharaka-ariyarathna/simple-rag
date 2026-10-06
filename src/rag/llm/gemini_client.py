import httpx
from rag.constants import GEMINI_BASE_URL, GEMINI_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT_SECONDS
from rag.llm.base import LlmTimeoutError, LlmUnavailableError


class GeminiLlmClient:
    """LlmClient backed by the Google Gemini API (generateContent)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = GEMINI_BASE_URL,  # "https://generativelanguage.googleapis.com"
        model_name: str = GEMINI_MODEL,
        timeout_seconds: float = LLM_TIMEOUT_SECONDS,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.http_client = httpx.Client(
            base_url=base_url, timeout=timeout_seconds, headers={"x-goog-api-key": api_key}
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        request_body = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": LLM_TEMPERATURE},
        }
        # TimeoutException is a subclass of HTTPError, so it must be caught first
        try:
            response = self.http_client.post(f"/v1beta/models/{self.model_name}:generateContent", json=request_body)
            response.raise_for_status()  # 400 bad request, 403 bad key, 429 quota...
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except httpx.TimeoutException as error:
            raise LlmTimeoutError(f"Gemini did not answer within {self.timeout_seconds} seconds") from error
        except httpx.HTTPStatusError as error:
            raise LlmUnavailableError(
                f"Gemini returned {error.response.status_code} for model '{self.model_name}': {error.response.text}"
            ) from error
        except httpx.HTTPError as error:  # connection / DNS failure
            raise LlmUnavailableError(f"Could not reach Gemini at {self.base_url}: {error}") from error
        except (KeyError, IndexError, ValueError) as error:  # not JSON, or no candidates (e.g. blocked by safety filter)
            raise LlmUnavailableError(f"Unexpected response from Gemini: {error}") from error

    def close(self) -> None:
        self.http_client.close()
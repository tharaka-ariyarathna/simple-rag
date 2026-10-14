"""The HTTP clients are tested against httpx.MockTransport: real request building and response parsing,
but the "server" is a function in the test, so nothing goes over the network."""
import json
import httpx
import pytest
from rag.llm.base import LlmTimeoutError, LlmUnavailableError
from rag.llm.gemini_client import GeminiLlmClient
from rag.llm.ollama_client import OllamaLlmClient


@pytest.fixture
def mock_http(monkeypatch):
    """Makes every httpx.Client created in the test send its requests to `handler` instead of the network.
    Returns a setter: mock_http(handler). The clients build their httpx.Client in __init__, so call it first."""
    real_client = httpx.Client

    def install(handler):
        monkeypatch.setattr(httpx, "Client", lambda **options: real_client(transport=httpx.MockTransport(handler),
                                                                          **options))
    return install


def raise_error(error: Exception):
    def handler(request: httpx.Request):
        raise error
    return handler


class TestOllama:
    def test_sends_a_non_streaming_chat_request_and_parses_the_answer(self, mock_http):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"message": {"content": "Hi [1]."}, "prompt_eval_count": 42,
                                             "eval_count": 7})
        mock_http(handler)
        client = OllamaLlmClient(base_url="http://ollama.test", model_name="llama-test")

        result = client.generate("system rules", "user question")

        body = json.loads(requests[0].content)
        assert requests[0].url == "http://ollama.test/api/chat"
        assert body["model"] == "llama-test" and body["stream"] is False
        assert body["messages"] == [{"role": "system", "content": "system rules"},
                                    {"role": "user", "content": "user question"}]
        assert (result.text, result.model, result.input_tokens, result.output_tokens) == ("Hi [1].", "llama-test",
                                                                                          42, 7)

    def test_missing_token_counts_are_none(self, mock_http):
        mock_http(lambda request: httpx.Response(200, json={"message": {"content": "ok"}}))

        result = OllamaLlmClient().generate("s", "u")

        assert result.input_tokens is None and result.output_tokens is None

    def test_model_not_pulled_is_unavailable_and_names_the_model(self, mock_http):
        mock_http(lambda request: httpx.Response(404, text="model not found"))

        with pytest.raises(LlmUnavailableError, match="404.*llama-test"):
            OllamaLlmClient(model_name="llama-test").generate("s", "u")

    def test_timeout_becomes_llm_timeout(self, mock_http):
        mock_http(raise_error(httpx.ReadTimeout("slow")))

        with pytest.raises(LlmTimeoutError):
            OllamaLlmClient().generate("s", "u")

    def test_server_not_running_is_unavailable(self, mock_http):
        mock_http(raise_error(httpx.ConnectError("refused")))

        with pytest.raises(LlmUnavailableError, match="Could not reach Ollama"):
            OllamaLlmClient().generate("s", "u")

    @pytest.mark.parametrize("response", [httpx.Response(200, text="not json"), httpx.Response(200, json={"x": 1})])
    def test_unexpected_response_is_unavailable(self, mock_http, response):
        mock_http(lambda request: response)

        with pytest.raises(LlmUnavailableError, match="Unexpected response"):
            OllamaLlmClient().generate("s", "u")


class TestGemini:
    def test_sends_the_key_header_and_parses_the_answer(self, mock_http):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "Answer [2]."}]}}],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 9},
            })
        mock_http(handler)
        client = GeminiLlmClient(api_key="test-key", base_url="https://gemini.test", model_name="gemini-test")

        result = client.generate("system rules", "user question")

        request = requests[0]
        body = json.loads(request.content)
        assert request.url.path == "/v1beta/models/gemini-test:generateContent"
        assert request.headers["x-goog-api-key"] == "test-key"
        assert body["system_instruction"] == {"parts": [{"text": "system rules"}]}
        assert body["contents"] == [{"role": "user", "parts": [{"text": "user question"}]}]
        assert (result.text, result.input_tokens, result.output_tokens) == ("Answer [2].", 100, 9)

    @pytest.mark.parametrize("status_code", [400, 403, 429, 500])
    def test_error_status_is_unavailable(self, mock_http, status_code):
        mock_http(lambda request: httpx.Response(status_code, text="nope"))

        with pytest.raises(LlmUnavailableError, match=str(status_code)):
            GeminiLlmClient(api_key="k").generate("s", "u")

    def test_blocked_answer_without_candidates_is_unavailable(self, mock_http):
        mock_http(lambda request: httpx.Response(200, json={"candidates": []}))

        with pytest.raises(LlmUnavailableError, match="Unexpected response"):
            GeminiLlmClient(api_key="k").generate("s", "u")

    def test_timeout_becomes_llm_timeout(self, mock_http):
        mock_http(raise_error(httpx.ConnectTimeout("slow")))

        with pytest.raises(LlmTimeoutError):
            GeminiLlmClient(api_key="k").generate("s", "u")

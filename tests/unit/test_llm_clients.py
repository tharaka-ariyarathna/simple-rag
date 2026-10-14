"""The HTTP clients are tested against httpx.MockTransport: real request building and response parsing,
but the "server" is a function in the test, so nothing goes over the network."""
import json
import httpx
import pytest
from rag.llm.base import LlmConfigurationError, LlmTimeoutError, LlmUnavailableError
from rag.llm.gemini_client import GeminiLlmClient
from rag.llm.http_call import error_for_status
from rag.llm.ollama_client import OllamaLlmClient
from rag.llm.openai_compatible_client import OpenAiCompatibleLlmClient


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

    def test_model_not_pulled_is_a_configuration_error_naming_the_model(self, mock_http):
        mock_http(lambda request: httpx.Response(404, text="model not found"))

        with pytest.raises(LlmConfigurationError, match="404.*llama-test"):
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

    @pytest.mark.parametrize("status_code, error_type", [
        (400, LlmConfigurationError), (403, LlmConfigurationError),
        (429, LlmUnavailableError), (500, LlmUnavailableError),
    ])
    def test_error_status_maps_to_the_shared_error_types(self, mock_http, status_code, error_type):
        mock_http(lambda request: httpx.Response(status_code, text="nope"))

        with pytest.raises(error_type, match=str(status_code)):
            GeminiLlmClient(api_key="k").generate("s", "u")

    def test_blocked_answer_without_candidates_is_unavailable(self, mock_http):
        mock_http(lambda request: httpx.Response(200, json={"candidates": []}))

        with pytest.raises(LlmUnavailableError, match="Unexpected response"):
            GeminiLlmClient(api_key="k").generate("s", "u")

    def test_timeout_becomes_llm_timeout(self, mock_http):
        mock_http(raise_error(httpx.ConnectTimeout("slow")))

        with pytest.raises(LlmTimeoutError):
            GeminiLlmClient(api_key="k").generate("s", "u")


class TestOpenAiCompatible:
    def client(self, **options) -> OpenAiCompatibleLlmClient:
        return OpenAiCompatibleLlmClient(**({"provider": "Groq", "base_url": "https://api.groq.test/openai/v1",
                                             "api_key": "test-key", "model_name": "openai/gpt-oss-120b"} | options))

    def test_sends_a_bearer_chat_request_and_records_the_model_that_answered(self, mock_http):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={
                "model": "openai/gpt-oss-120b-2026-09",
                "choices": [{"message": {"role": "assistant", "content": "Alice [1]."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 8},
            })
        mock_http(handler)

        result = self.client().generate("system rules", "user question")

        request = requests[0]
        body = json.loads(request.content)
        assert str(request.url) == "https://api.groq.test/openai/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        assert body["model"] == "openai/gpt-oss-120b"
        assert body["messages"] == [{"role": "system", "content": "system rules"},
                                    {"role": "user", "content": "user question"}]
        assert (result.text, result.model) == ("Alice [1].", "openai/gpt-oss-120b-2026-09")
        assert (result.input_tokens, result.output_tokens) == (120, 8)

    def test_configured_model_is_used_when_the_response_names_none(self, mock_http):
        mock_http(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))

        result = self.client(model_name="nvidia/nemotron:free").generate("s", "u")

        assert result.model == "nvidia/nemotron:free"
        assert result.input_tokens is None

    @pytest.mark.parametrize("content", [None, "", "   "])
    def test_empty_answer_is_unavailable_so_the_next_provider_is_tried(self, mock_http, content):
        mock_http(lambda request: httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": "length"}]}))

        with pytest.raises(LlmUnavailableError, match="empty answer.*length"):
            self.client().generate("s", "u")

    @pytest.mark.parametrize("body", [{"error": {"message": "upstream failed"}}, {"choices": []}])
    def test_body_without_an_answer_is_unavailable(self, mock_http, body):
        mock_http(lambda request: httpx.Response(200, json=body))

        with pytest.raises(LlmUnavailableError, match="Unexpected response from OpenRouter"):
            self.client(provider="OpenRouter").generate("s", "u")

    def test_bad_key_is_a_configuration_error(self, mock_http):
        mock_http(lambda request: httpx.Response(401, text="invalid api key"))

        with pytest.raises(LlmConfigurationError, match="Groq returned 401"):
            self.client().generate("s", "u")

    def test_timeout_names_the_provider_and_its_timeout(self, mock_http):
        mock_http(raise_error(httpx.ReadTimeout("slow")))

        with pytest.raises(LlmTimeoutError, match="Groq did not answer within 60"):
            self.client().generate("s", "u")


@pytest.mark.parametrize("status_code, error_type", [
    (400, LlmConfigurationError),  # bad request: every provider would reject it
    (401, LlmConfigurationError),  # bad key
    (403, LlmConfigurationError),
    (404, LlmConfigurationError),  # unknown model / not pulled
    (422, LlmConfigurationError),
    (402, LlmUnavailableError),    # out of credits
    (408, LlmUnavailableError),
    (413, LlmUnavailableError),    # over a tokens-per-minute limit
    (429, LlmUnavailableError),    # rate limited
    (500, LlmUnavailableError),
    (502, LlmUnavailableError),
    (503, LlmUnavailableError),
])
def test_status_codes_decide_between_falling_back_and_stopping(status_code, error_type):
    error = error_for_status("Groq", "m", status_code, "body")

    assert type(error) is error_type
    assert str(error) == f"Groq returned {status_code} for model 'm': body"


def test_long_error_bodies_are_cut_in_the_message():
    assert len(str(error_for_status("Groq", "m", 500, "x" * 5000))) < 600

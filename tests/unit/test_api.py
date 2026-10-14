"""The FastAPI routes, with RagService replaced through dependency_overrides.

TestClient(app) is used without a `with` block, so the lifespan (DB pool, embedding model, LLM clients)
never runs: nothing real is started."""
import pytest
from fastapi.testclient import TestClient
from rag.api.main import app
from rag.api.routes import get_rag_service
from rag.constants import INTERNAL_ERROR_DETAIL, LLM_TIMEOUT_DETAIL, LLM_UNAVAILABLE_DETAIL, TOP_K_MAX
from rag.llm.base import LlmConfigurationError, LlmTimeoutError, LlmUnavailableError
from rag.service.rag_service import RagAnswer, SourceReference


class StubRagService:
    def __init__(self, rag_answer: RagAnswer | None = None, error: Exception | None = None):
        self.rag_answer = rag_answer or RagAnswer(answer="Alice [1].", sources=[SourceReference("a.pdf", "cvs", 0.8)])
        self.error = error
        self.calls = []

    def answer_question(self, question, folder, top_k, entry_point="api", eval_run_id=None):
        self.calls.append({"question": question, "folder": folder, "top_k": top_k, "entry_point": entry_point})
        if self.error:
            raise self.error
        return self.rag_answer


@pytest.fixture
def use_service():
    """use_service(stub) -> TestClient whose /chat calls that stub."""
    def install(stub: StubRagService) -> TestClient:
        app.dependency_overrides[get_rag_service] = lambda: stub
        return TestClient(app)
    yield install
    app.dependency_overrides.clear()


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_chat_returns_the_answer_and_its_sources(use_service):
    response = use_service(StubRagService()).post("/chat", json={"question": "Who?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Alice [1].",
                               "sources": [{"file_name": "a.pdf", "folder": "cvs", "similarity": 0.8}]}


def test_chat_strips_the_question_and_forwards_folder_and_top_k(use_service):
    stub = StubRagService()

    use_service(stub).post("/chat", json={"question": "  Who knows Go?  ", "folder": " cvs ", "top_k": 3})

    assert stub.calls == [{"question": "Who knows Go?", "folder": "cvs", "top_k": 3, "entry_point": "api"}]


def test_refusal_reason_is_not_part_of_the_response(use_service):
    stub = StubRagService(RagAnswer(answer="I couldn't find that in the documents.", refusal_reason="retrieval"))

    body = use_service(stub).post("/chat", json={"question": "q"}).json()

    assert body == {"answer": "I couldn't find that in the documents.", "sources": []}


@pytest.mark.parametrize("payload", [
    {},
    {"question": "   "},
    {"question": "x" * 1001},
    {"question": "q", "top_k": 0},
    {"question": "q", "top_k": TOP_K_MAX + 1},
    {"question": "q", "folder": "  "},
    {"question": "q", "unknown": 1},
])
def test_invalid_requests_are_rejected_before_the_service_runs(use_service, payload):
    stub = StubRagService()

    response = use_service(stub).post("/chat", json=payload)

    assert response.status_code == 422
    assert stub.calls == []


@pytest.mark.parametrize("error, status_code, detail", [
    (LlmTimeoutError("ollama took 240s on host x"), 504, LLM_TIMEOUT_DETAIL),
    (LlmUnavailableError("Groq returned 429: rate limit reached"), 503, LLM_UNAVAILABLE_DETAIL),
    (LlmConfigurationError("Groq returned 401: invalid api key gsk_..."), 500, INTERNAL_ERROR_DETAIL),
])
def test_llm_failures_map_to_a_status_with_a_generic_message(use_service, error, status_code, detail):
    response = use_service(StubRagService(error=error)).post("/chat", json={"question": "q"})

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}  # the provider's message stays in the server log

"""The FastAPI routes, with RagService replaced through dependency_overrides.

TestClient(app) is used without a `with` block, so the lifespan (DB pool, embedding model, LLM clients)
never runs: nothing real is started."""
import pytest
from fastapi.testclient import TestClient
from rag.api.main import app
from rag.api.routes import get_candidate_repository, get_conversation_graph, get_match_graph
from rag.constants import (
    INTERNAL_ERROR_DETAIL, LLM_TIMEOUT_DETAIL, LLM_UNAVAILABLE_DETAIL, REQUIREMENT_EXTRACTION_FAILED_DETAIL, TOP_K_MAX,
)
from rag.matching.requirements import RequirementExtractionError
from rag.llm.base import LlmConfigurationError, LlmTimeoutError, LlmUnavailableError
from rag.memory.conversation_graph import ConversationAnswer
from rag.service.rag_service import RagAnswer, SourceReference


class StubRagService:
    """Stands in for the conversation graph: records the call and returns a ready-made answer."""

    def __init__(self, rag_answer: RagAnswer | None = None, error: Exception | None = None):
        self.rag_answer = rag_answer or RagAnswer(answer="Alice [1].", sources=[SourceReference("a.pdf", "cvs", 0.8)])
        self.error = error
        self.calls = []

    def ask(self, question, folder, top_k, conversation_id=None, candidate_ids=None):
        self.calls.append({"question": question, "folder": folder, "top_k": top_k, "conversation_id": conversation_id})
        self.candidate_ids = candidate_ids
        if self.error:
            raise self.error
        return ConversationAnswer(conversation_id or "new-id", "Who is Alice?", self.rag_answer)


class StubNames:
    def __init__(self, names):
        self.names = names

    def names_by_id(self, candidate_ids):
        return {candidate_id: self.names.get(candidate_id) for candidate_id in candidate_ids}


@pytest.fixture
def use_service():
    """use_service(stub) -> TestClient whose /chat calls that stub."""
    def install(stub: StubRagService, names: dict | None = None) -> TestClient:
        app.dependency_overrides[get_conversation_graph] = lambda: stub
        app.dependency_overrides[get_candidate_repository] = lambda: StubNames(names or {})
        return TestClient(app)
    yield install
    app.dependency_overrides.clear()


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_chat_returns_the_answer_and_its_sources(use_service):
    response = use_service(StubRagService()).post("/chat", json={"question": "Who?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Alice [1].",
                               "sources": [{"file_name": "a.pdf", "folder": "cvs", "similarity": 0.8,
                                            "candidate_id": None, "candidate_name": None}],
                               "conversation_id": "new-id", "standalone_question": "Who is Alice?"}


def test_chat_strips_the_question_and_forwards_folder_and_top_k(use_service):
    stub = StubRagService()

    use_service(stub).post("/chat", json={"question": "  Who knows Go?  ", "folder": " cvs ", "top_k": 3})

    assert stub.calls == [{"question": "Who knows Go?", "folder": "cvs", "top_k": 3, "conversation_id": None}]


def test_chat_continues_the_conversation_it_is_given(use_service):
    stub = StubRagService()

    body = use_service(stub).post("/chat", json={"question": "And his Python?", "conversation_id": " abc "}).json()

    assert stub.calls[0]["conversation_id"] == "abc" and body["conversation_id"] == "abc"


def test_refusal_reason_is_not_part_of_the_response(use_service):
    stub = StubRagService(RagAnswer(answer="I couldn't find that in the documents.", refusal_reason="retrieval"))

    body = use_service(stub).post("/chat", json={"question": "q"}).json()

    assert "refusal_reason" not in body


@pytest.mark.parametrize("payload", [
    {},
    {"question": "   "},
    {"question": "x" * 1001},
    {"question": "q", "top_k": 0},
    {"question": "q", "top_k": TOP_K_MAX + 1},
    {"question": "q", "folder": "  "},
    {"question": "q", "unknown": 1},
    {"question": "q", "conversation_id": "  "},
    {"question": "q", "conversation_id": "x" * 101},
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


class StubCandidateRepository:
    def __init__(self, existing_ids):
        self.existing_ids = set(existing_ids)
        self.deleted = []

    def delete(self, candidate_id):
        self.deleted.append(candidate_id)
        return candidate_id in self.existing_ids


@pytest.fixture
def candidate_client():
    repository = StubCandidateRepository(existing_ids=[7])
    app.dependency_overrides[get_candidate_repository] = lambda: repository
    yield TestClient(app), repository
    app.dependency_overrides.clear()


def test_deleting_a_candidate_returns_no_content(candidate_client):
    client, repository = candidate_client

    response = client.delete("/candidates/7")

    assert response.status_code == 204 and repository.deleted == [7]


def test_deleting_an_unknown_candidate_is_a_404(candidate_client):
    client, _ = candidate_client

    assert client.delete("/candidates/99").status_code == 404
    assert client.delete("/candidates/abc").status_code == 422


def test_names_are_put_back_only_in_the_response(use_service):
    rag_answer = RagAnswer(answer="Candidate 17 knows Spark [1].",
                           sources=[SourceReference("cv_017.pdf", "cvs", 0.8, candidate_id=17)])
    stub = StubRagService(rag_answer)

    body = use_service(stub, names={17: "Ines Brooks"}).post(
        "/chat", json={"question": "Who knows Spark?", "candidate_ids": [17]}).json()

    assert body["answer"] == "Ines Brooks (Candidate 17) knows Spark [1]."
    assert body["sources"][0]["candidate_id"] == 17 and body["sources"][0]["candidate_name"] == "Ines Brooks"
    assert stub.candidate_ids == [17]


class StubMatchGraph:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def match(self, jd_text, top_n):
        self.calls.append((jd_text, top_n))
        if self.error:
            raise self.error
        return {"requirements": [{"skill": "Spark", "must_have": True, "min_years": None, "synonyms": []}],
                "min_total_years": 3.0, "shortlist": [17], "whole_jd_order": [17],
                "candidates": [{"candidate_id": 17, "score": 1.0, "gaps": [], "missing_must_haves": [],
                                "verdicts": [{"requirement": "Spark", "must_have": True, "verdict": "met",
                                              "evidence": "Spark pipelines"}]}]}


class StubConversations:
    def __init__(self):
        self.remembered = []

    def remember_match(self, conversation_id, request_text, match_result, summary):
        self.remembered.append((conversation_id, request_text, summary))


@pytest.fixture
def match_client():
    def install(match_graph):
        conversations = StubConversations()
        app.dependency_overrides[get_match_graph] = lambda: match_graph
        app.dependency_overrides[get_conversation_graph] = lambda: conversations
        app.dependency_overrides[get_candidate_repository] = lambda: StubNames({17: "Ines Brooks"})
        return TestClient(app), conversations
    yield install
    app.dependency_overrides.clear()


JD = "We need a data engineer with Spark and 3+ years of experience."


def test_match_returns_the_ranking_with_names_and_stores_it_for_follow_ups(match_client):
    client, conversations = match_client(StubMatchGraph())

    body = client.post("/match", json={"jd_text": JD, "top_n": 5}).json()

    assert body["candidates"][0]["candidate_id"] == 17 and body["candidates"][0]["candidate_name"] == "Ines Brooks"
    assert body["candidates"][0]["verdicts"][0]["evidence"] == "Spark pipelines"
    [(conversation_id, request_text, summary)] = conversations.remembered
    assert conversation_id == body["conversation_id"] and request_text.endswith(JD)
    assert summary == "Ranked 1 candidates: candidate 17 (1.00)." and "Ines" not in summary


def test_match_rejects_a_too_short_jd(match_client):
    client, _ = match_client(StubMatchGraph())

    assert client.post("/match", json={"jd_text": "Python dev"}).status_code == 422


def test_a_failed_requirement_extraction_is_a_503(match_client):
    client, _ = match_client(StubMatchGraph(error=RequirementExtractionError("bad json")))

    response = client.post("/match", json={"jd_text": JD})

    assert response.status_code == 503 and response.json() == {"detail": REQUIREMENT_EXTRACTION_FAILED_DETAIL}

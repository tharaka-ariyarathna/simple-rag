"""The MCP tools are plain functions (the @tool() decorator returns them unchanged), so they are called directly
with a fake Context whose lifespan_context holds a stub service. The SDK's own argument validation is not
retested here."""
from types import SimpleNamespace
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from rag.constants import LLM_TIMEOUT_DETAIL, LLM_UNAVAILABLE_DETAIL
from rag.llm.base import LlmTimeoutError, LlmUnavailableError
from rag.mcp_server import server
from rag.service.rag_service import RagAnswer, SourceReference
from tests.fakes import chunk_row


class StubRagService:
    def __init__(self, chunks=None, rag_answer=None, error=None):
        self.chunks = chunks or []
        self.rag_answer = rag_answer
        self.error = error
        self.calls = []

    def list_folders(self):
        return {"cvs": 4}

    def search_chunks(self, question, folder, top_k):
        self.calls.append(("search", question, folder, top_k))
        return self.chunks

    def answer_question(self, question, folder, top_k, entry_point="api"):
        self.calls.append(("answer", question, folder, top_k, entry_point))
        if self.error:
            raise self.error
        return self.rag_answer


def context_with(service: StubRagService):
    return SimpleNamespace(request_context=SimpleNamespace(lifespan_context=SimpleNamespace(rag_service=service)))


def test_list_folders():
    assert server.list_folders(context_with(StubRagService())) == {"cvs": 4}


def test_search_documents_returns_only_what_the_model_needs():
    service = StubRagService(chunks=[chunk_row("C:/Users/me/cvs/alice.pdf", "  Alice knows Python. \n",
                                               folder="cvs", similarity=0.87654, chunk_index=3)])

    passages = server.search_documents(context_with(service), "  python?  ", folder="cvs", top_k=2)

    assert passages == [{"file_name": "alice.pdf", "folder": "cvs", "similarity": 0.877,
                         "content": "Alice knows Python."}]
    assert service.calls == [("search", "python?", "cvs", 2)]


def test_answer_question_returns_the_answer_as_a_dict_and_marks_the_entry_point():
    rag_answer = RagAnswer(answer="Alice [1].", sources=[SourceReference("alice.pdf", "cvs", 0.9)])
    service = StubRagService(rag_answer=rag_answer)

    result = server.answer_question(context_with(service), " who? ", folder=None, top_k=5)

    assert result == {"answer": "Alice [1].", "sources": [{"file_name": "alice.pdf", "folder": "cvs",
                                                           "similarity": 0.9}], "refusal_reason": None}
    assert service.calls == [("answer", "who?", None, 5, "mcp")]


@pytest.mark.parametrize("error, message", [
    (LlmTimeoutError("internal detail"), LLM_TIMEOUT_DETAIL),
    (LlmUnavailableError("internal detail"), LLM_UNAVAILABLE_DETAIL),
])
def test_llm_failures_become_tool_errors_with_a_generic_message(error, message):
    with pytest.raises(ToolError) as raised:
        server.answer_question(context_with(StubRagService(error=error)), "q", folder=None, top_k=5)

    assert str(raised.value) == message
    assert "internal detail" not in str(raised.value)

from contextlib import contextmanager
from contextvars import ContextVar

# Like request-scoped globals: set by the caller, read by LoggingLlmClient, so generate() keeps its signature.
# Each thread (FastAPI / MCP worker threads) and each task gets its own copy.
current_purpose: ContextVar[str] = ContextVar("llm_call_purpose", default="answer")
current_rag_request_id: ContextVar[int | None] = ContextVar("llm_call_rag_request_id", default=None)


@contextmanager
def llm_call_context(purpose: str, rag_request_id: int | None = None):
    """Everything logged inside the with-block gets this purpose and request id."""
    purpose_token = current_purpose.set(purpose)
    request_token = current_rag_request_id.set(rag_request_id)
    try:
        yield
    finally:
        current_purpose.reset(purpose_token)
        current_rag_request_id.reset(request_token)

"""MCP adapter: exposes RagService as MCP tools over streamable HTTP.

Run from src:  python -m rag.mcp_server.server   ->  http://127.0.0.1:8001/mcp
"""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated
from pydantic import Field
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from rag.bootstrap import RagComponents, build_rag_components
from rag.constants import (
    FOLDER_MAX_LENGTH,
    LLM_TIMEOUT_DETAIL,
    LLM_UNAVAILABLE_DETAIL,
    MCP_HOST,
    MCP_PATH,
    MCP_PORT,
    MCP_SERVER_INSTRUCTIONS,
    MCP_SERVER_NAME,
    QUESTION_MAX_LENGTH,
    RETRIEVAL_TOP_K,
    TOP_K_MAX,
)
from rag.llm.base import LlmTimeoutError, LlmUnavailableError
from rag.service.rag_service import RagService

logger = logging.getLogger(__name__)

# Same limits as the REST API (api/schemas.py); the SDK turns them into the JSON schema the model sees
Question = Annotated[str, Field(min_length=1, max_length=QUESTION_MAX_LENGTH, description="The question to look up")]
Folder = Annotated[
    str | None, Field(max_length=FOLDER_MAX_LENGTH, description="Only search this folder (a name from list_folders)")
]
TopK = Annotated[int, Field(ge=1, le=TOP_K_MAX, description="How many chunks to retrieve")]


@asynccontextmanager
async def lifespan(server: MCPServer) -> AsyncIterator[RagComponents]:
    # Same as the FastAPI lifespan: build once at startup, close on shutdown
    rag_components = build_rag_components()
    try:
        yield rag_components
    finally:
        rag_components.close()


mcp_server = MCPServer(MCP_SERVER_NAME, instructions=MCP_SERVER_INSTRUCTIONS, lifespan=lifespan)


def get_rag_service(ctx: Context) -> RagService:
    return ctx.request_context.lifespan_context.rag_service  # what lifespan yielded (like app.state in FastAPI)


# The docstring is the tool description the model reads to decide when to call it.
# Plain def (not async): the SDK runs sync tools in a worker thread, so blocking DB work is fine.
@mcp_server.tool()
def list_folders(ctx: Context) -> dict[str, int]:
    """List the indexed document folders and how many chunks each holds.
    Use a folder name from here to narrow a search to that folder."""
    return get_rag_service(ctx).list_folders()



@mcp_server.tool()
def search_documents(
    ctx: Context, question: Question, folder: Folder = None, top_k: TopK = RETRIEVAL_TOP_K
) -> list[dict]:
    """Find the passages in the user's documents most relevant to a question, without generating an answer.
    Read the returned passages and answer yourself. Optionally limit the search to one folder.
    Returns file name, folder, similarity (0-1) and text per passage; an empty list means nothing relevant."""
    chunks = get_rag_service(ctx).search_chunks(question.strip(), folder, top_k)
    # Only what the model needs: no full local paths, no internal chunk indexes
    return [
        {
            "file_name": Path(chunk["source"]).name,
            "folder": chunk["folder"],
            "similarity": round(chunk["similarity"], 3),
            "content": chunk["content"].strip(),
        }
        for chunk in chunks
    ]


@mcp_server.tool()
def answer_question(ctx: Context, question: Question, folder: Folder = None, top_k: TopK = RETRIEVAL_TOP_K) -> dict:
    """Answer a question from the user's documents using this server's own LLM (slower; prefer search_documents
    when you can answer yourself). The answer cites its sources as [n]; if nothing relevant is found,
    it says so instead of guessing. Optionally limit the search to one folder."""
    try:
        rag_answer = get_rag_service(ctx).answer_question(question.strip(), folder, top_k, entry_point="mcp")
    except LlmTimeoutError as error:
        logger.warning("LLM timeout: %s", error)
        raise ToolError(LLM_TIMEOUT_DETAIL) from error  # generic message to the caller; details stay in the log
    except LlmUnavailableError as error:
        logger.error("LLM unavailable: %s", error)
        raise ToolError(LLM_UNAVAILABLE_DETAIL) from error
    return asdict(rag_answer)  # RagAnswer dataclass -> {"answer": ..., "sources": [...]}


if __name__ == "__main__":
    mcp_server.run("streamable-http", host=MCP_HOST, port=MCP_PORT, streamable_http_path=MCP_PATH)

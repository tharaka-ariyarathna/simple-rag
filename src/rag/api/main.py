from contextlib import asynccontextmanager
from fastapi import FastAPI
from rag.api.routes import router
from rag.bootstrap import build_rag_components


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: build long-lived objects once (DB pool, embedding model, LLM client); every request reuses them
    rag_components = build_rag_components()
    app.state.rag_service = rag_components.rag_service
    yield
    # Shutdown: release the HTTP clients and the DB pool
    rag_components.close()


app = FastAPI(title="Simple RAG", lifespan=lifespan)
app.include_router(router)

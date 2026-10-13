import subprocess
from rag.constants import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, MIN_SIMILARITY
from rag.bootstrap import RagComponents


def current_git_commit() -> str | None:
    """Short hash of the code being evaluated, so a run can be tied to a change. None outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def collect_settings(components: RagComponents, top_k: int) -> dict:
    """Everything that can change a score, saved with the run so runs can be compared fairly."""
    return {
        "embedding_model": components.rag_service.embedder.model_name,
        "chunk_size_tokens": CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "top_k": top_k,
        "min_similarity": MIN_SIMILARITY,
        "llm_model": components.llm_client.model_name,
    }

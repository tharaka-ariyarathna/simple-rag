# Embedding
# Changing this re-indexes into new rows next to the old ones; search only reads rows of the active model.
ACTIVE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 32
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Per-model text prefixes; models not listed here get none
EMBEDDING_MODEL_PREFIXES = {
    "BAAI/bge-small-en-v1.5": {"query": BGE_QUERY_PREFIX, "document": ""},
}

CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 50

# Tried in order: paragraph, line, sentence, word
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " "]

DB_SCHEMA = "main"
DB_POOL_MIN_SIZE = 1
DB_POOL_MAX_SIZE = 10

# Retrieval
RETRIEVAL_TOP_K = 5
MIN_SIMILARITY = 0.5  # chunks scoring below this are treated as not relevant
NO_RELEVANT_CONTEXT_ANSWER = "I couldn't find that in the documents."

# LLM
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
LLM_TIMEOUT_SECONDS = 120  # 3B model on CPU can be slow for long answers
LLM_CONTEXT_TOKENS=4096
LLM_TEMPERATURE=0.1  # low temperature = more deterministic, less creative
QWEN_TIMEOUT_SECONDS = 240
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"

# API request limits
QUESTION_MAX_LENGTH = 1000
FOLDER_MAX_LENGTH = 200
TOP_K_MAX = 20

# Prompt
CONTEXT_BLOCK_SEPARATOR = "\n\n"
SYSTEM_PROMPT = f"""You are an assistant that answers questions using ONLY the provided context.
Rules:
- Use only facts from the context. Do not use outside knowledge.
- If the context does not contain the answer, reply exactly: "{NO_RELEVANT_CONTEXT_ANSWER}"
- Cite the sources you used with their numbers, like [1] or [2][3].
- Be concise."""

# API error responses (details stay generic; full error goes to the server log)
LLM_UNAVAILABLE_DETAIL = "The language model is unavailable. Please try again later."
LLM_TIMEOUT_DETAIL = "The language model took too long to answer. Please try again."

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
# Answering tries these providers in order (override with ANSWER_LLM_CHAIN in .env)
DEFAULT_ANSWER_LLM_CHAIN = "groq,openrouter,ollama"
# Synthetic question generation (override with QUESTION_LLM_CHAIN, e.g. "openrouter,ollama")
DEFAULT_QUESTION_LLM_CHAIN = "ollama"
# Question generation: a provider that failed after its retries is skipped this long (e.g. OpenRouter's daily cap)
QUESTION_LLM_COOLDOWN_SECONDS = 900
# Model ids are defaults; GROQ_MODEL, OPENROUTER_MODEL, OLLAMA_MODEL and GEMINI_MODEL in .env override them
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
GEMINI_MODEL = "gemini-3.8-flash"  # not in the default chain; add "gemini" to ANSWER_LLM_CHAIN to use it
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
CLOUD_LLM_TIMEOUT_SECONDS = 60
LOCAL_LLM_TIMEOUT_SECONDS = 240  # 3B model on CPU can be slow for long answers
LLM_RETRY_DELAYS_SECONDS = (2, 4, 8)  # local question generation: 3 retries, waiting longer each time
LLM_CONTEXT_TOKENS=4096
LLM_TEMPERATURE=0.1  # low temperature = more deterministic, less creative

# API request limits
QUESTION_MAX_LENGTH = 1000
FOLDER_MAX_LENGTH = 200
TOP_K_MAX = 20

# API error responses (details stay generic; full error goes to the server log)
LLM_UNAVAILABLE_DETAIL = "The language model is unavailable. Please try again later."
LLM_TIMEOUT_DETAIL = "The language model took too long to answer. Please try again."
INTERNAL_ERROR_DETAIL = "Internal server error."  # LlmConfigurationError: our fault (bad key, bad request)

# MCP server (streamable HTTP; port differs from the FastAPI app's 8000)
MCP_SERVER_NAME = "simple-rag"
MCP_SERVER_INSTRUCTIONS = (
    "Searches the user's indexed personal documents. Use list_folders to see which folders exist, "
    "search_documents to read relevant passages, or answer_question for an answer written by this server's LLM."
)
MCP_HOST = "127.0.0.1"  # localhost only: no auth yet, so never expose this on the network
MCP_PORT = 8001
MCP_PATH = "/mcp"

# Evaluation
EVAL_MIN_HIT_RATE = 0.8  # retrieval eval passes when hit@k reaches this (starting guess, tune after the first run)
EVAL_MIN_MRR = 0.6
EVAL_MAX_REGRESSION = 0.03  # fails when a metric drops more than this against the previous run
EVAL_WORST_MISSES_SHOWN = 10
QUESTIONS_PER_CHUNK = 2
QUESTION_GENERATION_DELAY_SECONDS = 0.0  # pause between generator calls when the chain is local only
# ...and when it includes a cloud provider: OpenRouter's free models allow 20 requests a minute
CLOUD_QUESTION_GENERATION_DELAY_SECONDS = 3.5
EVAL_ANSWERABLE_SAMPLE = 20  # synthetic questions run through the full pipeline to measure false refusals
UNANSWERABLE_QUESTIONS_PATH = "evaluation/data/unanswerable.json"  # relative to the rag package
EVAL_MAX_HALLUCINATION_RATE = 0.1  # unanswerable questions that got an answer (starting guess)
EVAL_MAX_FALSE_REFUSAL_RATE = 0.3  # answerable questions that were refused (starting guess)
EVAL_RANDOM_SEED = 42  # same sample of answerable questions every run, so runs stay comparable

# JD -> CV matching evaluation
MATCHING_CHUNKS_RETRIEVED = 50  # chunks fetched per JD; they are then grouped by CV (about 2 chunks per CV)
MATCHING_CUTOFF = 10  # metrics are computed on the top 10 CVs
MATCHING_MIN_SIMILARITY = 0.0  # ranking, not refusing: the live Q&A path keeps MIN_SIMILARITY
MATCHING_STRATEGY_WHOLE_JD = "whole_jd"  # the whole JD text is used as one search query

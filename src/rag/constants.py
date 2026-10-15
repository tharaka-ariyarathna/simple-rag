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
RETRIEVAL_TOP_K = 10  # chunks given to the LLM
# Chunks scoring below this are treated as not relevant; with none left the question is refused without the LLM.
# Measured on the CV set: in-domain questions pass 120-150 of 160 chunks, off-topic ones top out at 0.44-0.49,
# so 0.4 would let off-topic questions through to the LLM.
MIN_SIMILARITY = 0.5
NO_RELEVANT_CONTEXT_ANSWER = "I couldn't find that in the documents."

# Hybrid search: vector ranking + keyword (Postgres full-text) ranking, merged with reciprocal rank fusion.
# Keywords only reorder chunks that already passed MIN_SIMILARITY; they never bring in a chunk below it.
RRF_K = 60  # standard RRF constant: score = sum of 1 / (RRF_K + rank) over the rankings a chunk appears in
HYBRID_VECTOR_CANDIDATES = 100  # nearest chunks taken from the vector ranking before fusion
KEYWORD_MAX_CHUNK_FRACTION = 0.5  # terms found in more chunks than this share say nothing about relevance
# Dropped from questions before the keyword search: they appear in questions but not as a reason to pick a chunk.
# "python" is in 24% of the CV chunks, so a frequency cut alone (without this list) would need to stay high.
KEYWORD_STOP_WORDS = frozenset("""
    a about above after all also am an and any anyone are as at be been best better can candidate candidates
    could did do does done for from good had has have having he her him his how i if in into is it its me
    more most my no not of on or our person people role roles she should so some someone than that the
    their them then there these they this those to us was we were what when where which who whom whose why
    will with would you your based experience experienced
    work works worked working know knows known knowledge use used uses using familiar skilled skill skills
    proficient strong year years find show list give tell please
""".split())

# Reranking: a cross-encoder reads the question and each candidate chunk together and reorders them.
# Hybrid search finds the right chunk in its top 20-30 far more often than in its top 10, so the candidates are
# reranked and only the best RETRIEVAL_TOP_K go on. It only reorders: the MIN_SIMILARITY refusal stays on cosine.
RERANKER_MODEL = "BAAI/bge-reranker-base"  # override with RERANKER_MODEL in .env; "none" switches reranking off
RERANK_CANDIDATES = 30  # chunks hybrid search hands to the reranker (an internal setting, not limited by TOP_K_MAX)
RERANK_BATCH_SIZE = 16

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
LLM_CONTEXT_TOKENS=8192  # Ollama only: room for ~10 chunks of ~400 tokens; Ollama silently cuts what does not fit
LLM_TEMPERATURE=0.1  # low temperature = more deterministic, less creative

# API request limits
QUESTION_MAX_LENGTH = 1000
FOLDER_MAX_LENGTH = 200
TOP_K_MAX = 20  # chunks an API / MCP caller may ask for
EVAL_TOP_K_MAX = 50  # the retrieval eval may look deeper, e.g. hit@30 to see what reranking could reach

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

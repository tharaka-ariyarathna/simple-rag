-- Ledger of every question asked and every LLM call made (latency, tokens, model, failures).
-- Run once against an existing database.
SET search_path TO main;

CREATE TABLE IF NOT EXISTS rag_requests (
    id               BIGSERIAL PRIMARY KEY,
    created_at       TIMESTAMPTZ      NOT NULL DEFAULT now(),
    entry_point      TEXT             NOT NULL,   -- api / mcp / eval
    eval_run_id      BIGINT,                      -- set when the question came from an evaluation run
    question         TEXT             NOT NULL,
    folder           TEXT,
    top_k            INT              NOT NULL,
    chunks_found     INT              NOT NULL,
    top_similarity   DOUBLE PRECISION,            -- NULL when nothing passed the threshold
    refusal_reason   TEXT,                        -- retrieval (no chunk found) / llm (LLM said it doesn't know) / NULL
    answer           TEXT,                        -- NULL when the LLM call failed
    total_latency_ms INT              NOT NULL,
    CONSTRAINT rag_requests_refusal_reason_check CHECK (refusal_reason IN ('retrieval', 'llm'))
);

CREATE INDEX IF NOT EXISTS rag_requests_created_at_idx ON rag_requests (created_at);
CREATE INDEX IF NOT EXISTS rag_requests_eval_run_idx ON rag_requests (eval_run_id);

CREATE TABLE IF NOT EXISTS llm_calls (
    id             BIGSERIAL PRIMARY KEY,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    rag_request_id BIGINT      REFERENCES rag_requests (id) ON DELETE SET NULL,
    purpose        TEXT        NOT NULL,          -- answer / question_generation
    provider       TEXT        NOT NULL,          -- groq / openrouter / ollama / gemini
    model          TEXT        NOT NULL,
    role           TEXT        NOT NULL,          -- primary / fallback_1 / fallback_2 (position in the chain)
    latency_ms     INT         NOT NULL,
    input_tokens   INT,                           -- NULL when the provider did not report it
    output_tokens  INT,
    success        BOOLEAN     NOT NULL,
    error_type     TEXT                           -- timeout / unavailable / configuration / NULL
);

CREATE INDEX IF NOT EXISTS llm_calls_created_at_idx ON llm_calls (created_at);
CREATE INDEX IF NOT EXISTS llm_calls_rag_request_idx ON llm_calls (rag_request_id);

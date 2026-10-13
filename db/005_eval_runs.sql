-- History of evaluation runs and their per-question results, so every run can be compared with the last one.
SET search_path TO main;

CREATE TABLE IF NOT EXISTS eval_runs (
    id         BIGSERIAL PRIMARY KEY,
    kind       TEXT        NOT NULL,              -- retrieval / unanswerable
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    git_commit TEXT,
    settings   JSONB       NOT NULL,              -- embedding model, chunk size, top_k, min_similarity, LLM model
    metrics    JSONB       NOT NULL,
    passed     BOOLEAN     NOT NULL
);

CREATE INDEX IF NOT EXISTS eval_runs_kind_created_idx ON eval_runs (kind, created_at DESC);

CREATE TABLE IF NOT EXISTS eval_results (
    id          BIGSERIAL PRIMARY KEY,
    eval_run_id BIGINT NOT NULL REFERENCES eval_runs (id) ON DELETE CASCADE,
    question    TEXT   NOT NULL,
    category    TEXT,                             -- out_of_domain / near_miss / answerable; NULL for retrieval
    rank        INT,                              -- retrieval: position of the evidence chunk, NULL = miss
    score       DOUBLE PRECISION,                 -- retrieval: reciprocal rank
    similarity  DOUBLE PRECISION,
    outcome     TEXT   NOT NULL,                  -- hit / miss / below_threshold / unmatchable / refused / answered
    details     JSONB
);

CREATE INDEX IF NOT EXISTS eval_results_run_idx ON eval_results (eval_run_id);

-- rag_requests.eval_run_id (003) now has a table to point at
ALTER TABLE rag_requests DROP CONSTRAINT IF EXISTS rag_requests_eval_run_fkey;
ALTER TABLE rag_requests ADD CONSTRAINT rag_requests_eval_run_fkey
    FOREIGN KEY (eval_run_id) REFERENCES eval_runs (id) ON DELETE SET NULL;

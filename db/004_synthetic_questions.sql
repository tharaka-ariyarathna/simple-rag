-- Questions an LLM wrote from chunks, used to test retrieval.
-- Not linked to chunks.id on purpose: re-indexing recreates chunk rows. A retrieval counts as a hit when a
-- retrieved chunk contains `evidence`, so the questions stay valid across re-indexing, chunk sizes and models.
SET search_path TO main;

CREATE TABLE IF NOT EXISTS synthetic_questions (
    id              BIGSERIAL PRIMARY KEY,
    source          TEXT        NOT NULL,         -- file the chunk came from (full path)
    chunk_index     INT         NOT NULL,         -- only used to remember which chunks were already processed
    question        TEXT        NOT NULL,
    expected_answer TEXT        NOT NULL,
    evidence        TEXT        NOT NULL,         -- text copied from the chunk that answers the question
    generator_model TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS synthetic_questions_chunk_idx ON synthetic_questions (source, chunk_index);

-- Chunks that were processed but produced no questions (headers, contact lines) must not be retried forever
CREATE TABLE IF NOT EXISTS synthetic_question_chunks (
    source          TEXT        NOT NULL,
    chunk_index     INT         NOT NULL,
    processed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source, chunk_index)
);

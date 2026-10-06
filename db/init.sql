SET search_path TO main;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    source          TEXT        NOT NULL,          -- full file path
    folder          TEXT        NOT NULL,          -- subfolder name (your metadata)
    chunk_index     INT         NOT NULL,
    content         TEXT        NOT NULL,
    embedding_model TEXT        NOT NULL,          -- which model made the vector
    embedding       VECTOR      NOT NULL,          -- no fixed size: models with different dimensions can coexist
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chunks_source_model_chunk_key UNIQUE (source, embedding_model, chunk_index)
);

-- HNSW indexes are per model, created by ChunkRepository.ensure_embedding_index
CREATE INDEX IF NOT EXISTS chunks_embedding_model_idx ON chunks (embedding_model);
CREATE INDEX IF NOT EXISTS chunks_folder_idx ON chunks (folder);

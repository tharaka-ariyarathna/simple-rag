-- Lets several embedding models live side by side in chunks.
-- Run once against an existing database created from the original init.sql.
SET search_path TO main;

-- Existing rows were all made by bge-small
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT NOT NULL DEFAULT 'BAAI/bge-small-en-v1.5';
ALTER TABLE chunks ALTER COLUMN embedding_model DROP DEFAULT;  -- new rows must state their model

-- The old index is tied to vector(384); per-model indexes replace it (ChunkRepository.ensure_embedding_index)
DROP INDEX IF EXISTS chunks_embedding_idx;
ALTER TABLE chunks ALTER COLUMN embedding TYPE vector;  -- no fixed size, so models with other dimensions fit

-- The same file can now have chunks from more than one model
ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_source_chunk_index_key;
ALTER TABLE chunks ADD CONSTRAINT chunks_source_model_chunk_key UNIQUE (source, embedding_model, chunk_index);

CREATE INDEX IF NOT EXISTS chunks_embedding_model_idx ON chunks (embedding_model);

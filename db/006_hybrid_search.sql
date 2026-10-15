-- Keyword (full-text) search next to vector search: a tsvector per chunk, kept up to date by Postgres.
-- Run once against an existing database; existing rows get their search_vector when the column is added.
SET search_path TO main;

-- Postgres' parser turns "C#" and "C++" both into "c" and splits ".NET" into "net", so technology names
-- are rewritten into single words first. Used for the stored chunks AND for questions (see
-- ChunkRepository.question_lexemes), so both sides are always normalised the same way.
CREATE OR REPLACE FUNCTION search_text(content TEXT) RETURNS TEXT
    LANGUAGE sql IMMUTABLE PARALLEL SAFE AS
$$
    SELECT regexp_replace(regexp_replace(regexp_replace(regexp_replace(content,
        '(^|[^[:alnum:]])c#',   '\1 csharp ', 'gi'),
        '(^|[^[:alnum:]])f#',   '\1 fsharp ', 'gi'),
        '(^|[^[:alnum:]])c\+\+', '\1 cpp ', 'gi'),
        '\.net([^[:alnum:]]|$)', ' dotnet \1', 'gi')
$$;

-- 'simple' keeps words as written (lower-cased): no stemming that would turn "AWS" into "aw"
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('simple', search_text(content))) STORED;

CREATE INDEX IF NOT EXISTS chunks_search_vector_idx ON chunks USING gin (search_vector);

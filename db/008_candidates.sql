-- Candidates as first-class rows: a stable id per CV (names are not unique), the profile extracted once at
-- ingestion, and the CV's content hash so unchanged CVs are not re-extracted or re-embedded.
-- Chunks point at their candidate and say which CV section they came from.
-- Run once against an existing database. Chunks indexed before this have no candidate_id until re-ingested.
SET search_path TO main;

CREATE TABLE IF NOT EXISTS candidates (
    id            BIGSERIAL PRIMARY KEY,
    source        TEXT        NOT NULL UNIQUE,      -- the CV file it came from (full path)
    content_hash  TEXT        NOT NULL,             -- sha256 of the CV text: unchanged hash = nothing to redo
    name          TEXT,                             -- NULL when extraction found no name that is in the CV
    total_years   NUMERIC(4, 1) NOT NULL DEFAULT 0, -- overlapping jobs counted once
    skills        JSONB       NOT NULL DEFAULT '[]',-- canonical skill names
    jobs          JSONB       NOT NULL DEFAULT '[]',-- validated jobs: title, company, start, end
    profile_model TEXT        NOT NULL,             -- the local model that extracted the profile
    extracted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS candidates_lower_name_idx ON candidates (lower(name));

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS candidate_id BIGINT REFERENCES candidates (id) ON DELETE CASCADE;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}';  -- {"section": "experience"}

CREATE INDEX IF NOT EXISTS chunks_candidate_idx ON chunks (candidate_id);

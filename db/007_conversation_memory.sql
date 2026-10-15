-- Conversation memory: which conversation a question belonged to, and what it was rewritten to before searching.
-- The conversation history itself lives in LangGraph's own checkpoint tables (created at startup by setup()).
-- Run once against an existing database.
SET search_path TO main;

ALTER TABLE rag_requests ADD COLUMN IF NOT EXISTS conversation_id     TEXT;  -- NULL for MCP / eval requests
ALTER TABLE rag_requests ADD COLUMN IF NOT EXISTS standalone_question TEXT;  -- NULL when the question was used as asked

CREATE INDEX IF NOT EXISTS rag_requests_conversation_idx ON rag_requests (conversation_id);

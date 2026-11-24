# Simple RAG: full project context

This file describes the whole project: what it does, how every part works, why each decision was made, which
alternatives were rejected, the evaluation numbers, and the known limitations. It is written so that an assistant
given only this file can answer detailed questions about the project (for example in interview preparation).

State as of 2026-10-18. Repository: `tharaka-ariyarathna/simple-rag` (local: `C:\Work\Private\Job Search\AI\Practice\Simple Rag`).
The full decision log, with every rejected alternative, is the "Simple RAG decision log" doc.

---

## 1. What the project is

A personal practice project: retrieval-augmented generation (RAG) over **80 synthetic CVs**, extended step by step
to learn production RAG patterns, LangGraph and LangChain. It has:

- **`POST /chat`**: grounded question answering over the CVs, with **conversation memory** (follow-up questions),
  **name resolution** (names map to candidate ids, because stored text is anonymised), and **follow-ups about a
  JD match**.
- **`POST /match`**: give a job description and get **ranked candidates**, each with a verdict per requirement,
  CV quotes as evidence, and gaps.
- **`DELETE /candidates/{id}`**: remove a candidate with all its chunks and embeddings.
- **An MCP server** (streamable HTTP, `127.0.0.1:8001/mcp`) with the tools `list_folders`, `search_documents` and
  `answer_question`. It is stateless and returns candidate ids, never names.
- **An evaluation suite**: retrieval, unanswerable/refusal, JD→CV matching (whole JD and the `/match` graph), and
  synthetic question generation.

**Data:** `C:\Work\Private\Job Search\AI\Practice Data\rag_v3\`
- `cvs/`: 80 PDFs, `cv_001`–`cv_080`. Half are single-column and half two-column. In two-column CVs the name is in
  the middle of the extracted text, not on line 1.
- `jds/`: 30 JD PDFs.
- `labels.json`: JD → CV grades (0 none, 1 partial, 2 strong).
- `manifest.csv`: per CV id, the name, format, area, level, years, number of roles and number of skills. It is the
  ground truth used to check profile extraction and redaction.

---

## 2. Stack

| Part | Choice |
|---|---|
| Language | Python 3.13, venv at `.venv` |
| API | FastAPI (sync route handlers run in a worker thread, because the pipeline is blocking) |
| MCP | the official MCP Python SDK (`MCPServer`), streamable HTTP |
| Database | Postgres 17 + pgvector 0.8.7 in Docker (container `rag-db`, database `rag`, schema `main`); psycopg3 with a pool and `dict_row` |
| Embeddings | sentence-transformers `BAAI/bge-small-en-v1.5` (384 dimensions, local; queries get the bge prefix) |
| Reranker | cross-encoder `BAAI/bge-reranker-base` (about 1 GB, local) |
| LLMs (answers, condensing, matching) | chain: Groq `openai/gpt-oss-120b` → OpenRouter `nvidia/nemotron-3-super-120b-a12b:free` → Ollama `llama3.2:3b` |
| LLM (profile extraction) | **local only**: Ollama `llama3.2:3b` (fallback option `qwen2.5:7b`) |
| Orchestration | LangGraph (`StateGraph`, conditional edges, `Send` fan-out, the Postgres checkpointer) |
| LangChain | only `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) and `langchain_core.messages` |
| Tests | pytest; hand-written fakes injected through constructors; integration tests on a `*_test` database |

Requirements: `requirements.txt`. It now includes `langgraph==1.2.15`, `langgraph-checkpoint-postgres==3.2.0` and
`langchain-text-splitters==1.1.3`. It is still missing some packages that are installed in `.venv` (`fastapi`,
`psycopg`, `pgvector`, `mcp`), a known gap.

---

## 3. Code layout (`src/rag/`)

```
api/            main.py (lifespan builds components once), routes.py, schemas.py (pydantic request/response)
bootstrap.py    composition root: reads .env, builds the DB pool, LLM chains, reranker, graphs; RagComponents
constants.py    every tunable number, with a comment saying why
prompt_constants.py  every prompt
embedding/      EmbeddingModel Protocol, SentenceTransformerEmbedder
indexing/       pdf_reader, chunker (TokenChunker on LangChain), section_chunker, redaction,
                profile_extractor (pydantic, Ollama), indexing_pipeline (CvIngestor)
llm/            base (LlmClient Protocol, errors), openai_compatible_client (Groq/OpenRouter), ollama_client,
                gemini_client, fallback_client, retrying_client, logging_client, http_call, call_context
reranking/      Reranker Protocol, CrossEncoderReranker
repository/     db (Database pool), chunk_repository, candidate_repository, ledger_repository,
                eval_repository, question_repository
service/        rag_service (RagService), prompt_builder, keyword_query, candidate_names (NameResolver, id↔name)
memory/         condenser (rewrite + checks), conversation_graph (chat LangGraph), follow_up (router)
matching/       requirements (extraction + pydantic), judge (verdicts + score), match_graph (match LangGraph)
mcp_server/     server.py
evaluation/     run_retrieval, run_unanswerable, run_matching, generate_questions + their *_eval modules
db/             init.sql, 002–008 migrations (applied by hand to `rag`; tests rebuild `rag_test` from them)
tests/          fakes.py, unit/, integration/ (conftest rebuilds the schema)
```

**Patterns used throughout:**
- **Protocols + constructor injection.** `LlmClient`, `EmbeddingModel` and `Reranker` are Protocols; repositories
  are duck-typed. Everything is wired in `bootstrap.py`. Tests inject fakes, with no patching.
- **Lifespan composition.** FastAPI's lifespan calls `build_rag_components()` once and puts the objects on
  `app.state`. Routes get them through `Depends(get_x)` getters. This avoids import-time database connections,
  allows a clean shutdown, and lets tests use `app.dependency_overrides`.
- **Constants in one file, each with a comment explaining why. Prompts in another file.**

---

## 4. Ingestion pipeline (`python -m rag.indexing.indexing_pipeline`)

For each PDF in `DOCS_DIR` (`CvIngestor.ingest`):

1. **Read** the text (`read_pdf_file`, pypdf).
2. **Hash** it (sha256). If the `candidates` row has the same `content_hash` and chunks exist for the active
   embedding model, the CV is **skipped as unchanged**. Re-runs are cheap, and nothing that didn't change is
   re-embedded.
3. **Extract the profile with a LOCAL model** (`ProfileExtractor`, Ollama `llama3.2:3b`). Raw CVs contain personal
   data, so they never go to a cloud LLM.
   - The pydantic `ExtractedProfile` JSON schema is sent as Ollama's `format`, which constrains the output.
   - The reply is validated again with pydantic.
   - Each `Job` is validated on its own; a bad job is dropped and logged, and the rest of the profile is kept:
     - dates may be `YYYY` or `YYYY-MM`, or "Present"
     - start ≤ end ≤ today
     - the company must appear in the CV, checked with `contains_evidence`, using pydantic validation context
   - **Total years** are computed **in code** by merging overlapping date ranges, so parallel jobs aren't counted
     twice.
   - **Skills** are kept only if they appear in the CV, and normalised with `SKILL_ALIASES` (k8s → Kubernetes).
   - **Name:** it must appear in the CV. If the model glued on a word ("Austin Pedro Patel", where the city comes
     from a two-column layout), the longest run of 2+ consecutive words that does appear is used. If no name is
     found, the CV is **not indexed**: better a missing CV than a leaking one.
4. **Section split** (`split_sections`). A line that **exactly** matches a known heading in `SECTION_HEADINGS` starts
   a section; "Work History" becomes `experience`, for example.
   - Sections: header (before the first heading) / summary / experience / education / skills / projects /
     certifications / languages / awards / contact / referees / other.
   - The match must be exact, because short body lines like "running in parallel" were otherwise detected as
     headings.
5. **Redact** (`redact_sections`). This text is the only thing ever chunked, embedded or sent to a cloud LLM.
   - The `contact` and `referees` sections are dropped whole. Referees are other people's personal data.
   - Regex replaces emails with `[EMAIL]`, URLs (http, www, LinkedIn/GitHub/GitLab) with `[URL]`, and phone numbers
     (9+ digits, so a year range like "2019 - 2021" isn't caught) with `[PHONE]`.
   - The candidate's name (the full name, then each part of 3+ letters, whole words, ignoring case) becomes
     "the candidate".
6. **Chunk each section on its own** (`SectionChunker`).
   - `TokenChunker` is LangChain's `RecursiveCharacterTextSplitter`, with the embedding tokenizer as the length
     function (300 tokens, overlap 50). Separators are paragraph, line, sentence, word, then `""` for a hard cut.
   - Each chunk is prefixed with its section name ("Skills\n…"), and `metadata = {"section": ...}`.
   - Chunks never cross sections.
7. **Leak check** (`find_leaks`). If any chunk still matches an email, URL or phone pattern, or contains the name,
   the CV is **not indexed**.
8. **Embed and save in ONE transaction** (`CandidateRepository.save`): upsert the candidate, delete its old chunks
   for this model (also matched by source, for chunks indexed before candidates existed), and insert the new chunks.
   A crash can never leave a new hash next to old chunks.

Each CV commits on its own, so the pipeline is resumable. The end-of-run summary counts new / updated / unchanged /
failed CVs. PDFs removed from the folder are left alone; delete them through the API.

**Results:** all 80 CVs new with 0 failed (about 45 s per CV, about 1 h in total); 609 chunks, all linked to a
candidate. The second run reported all 80 unchanged. A scan of every stored chunk found no email, phone, URL or name.
A dry run on 8 CVs against the manifest showed 0 JSON failures, every role found, exact skill counts, and years
within 1 (ours are about 0.8 higher because "2025 – Present" counts from January).

**Why the old chunker was replaced:** the hand-written `split_into_pieces` appended the separator to the last piece
too, adding text the document never had (a full stop). A strict `xfail` test documented this bug until it was fixed.
`from_huggingface_tokenizer` was not used because it rejects the test tokenizers; a `length_function` gives the same
result.

---

## 5. Database (schema `main`)

| Table | Purpose |
|---|---|
| `chunks` | `source, folder, chunk_index, content, embedding_model, embedding VECTOR (no fixed size), candidate_id → candidates ON DELETE CASCADE, metadata JSONB, search_vector TSVECTOR` (generated from `search_text(content)`) |
| `candidates` (008) | `id, source UNIQUE, content_hash, name, total_years NUMERIC(4,1), skills JSONB, jobs JSONB, profile_model, extracted_at`; index on `lower(name)` |
| `rag_requests` (003, 007) | one row per question: entry point, question, folder, top_k, chunks_found, top_similarity, refusal_reason (`retrieval`/`llm`), answer, latency, eval_run_id, **conversation_id, standalone_question** |
| `llm_calls` (003) | one row per LLM attempt: purpose (answer / condense / route / answer_from_match / requirement_extraction / judge / profile_extraction / question_generation), provider, model, role (primary / fallback_n), latency, tokens, success, error_type |
| `synthetic_questions`, `synthetic_question_chunks` (004) | generated eval questions with evidence quotes |
| `eval_runs`, `eval_results` (005) | eval runs with settings, metrics, pass/fail, git commit |
| LangGraph checkpoint tables | `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`, created by `PostgresSaver.setup()` at startup, not by our migrations |

- **One partial HNSW index per embedding model** (`ensure_embedding_index`), so several models can coexist. Queries
  cast `embedding::vector(dim)` to match the index expression.
- **006** adds `search_text()`, which rewrites C#, F#, C++ and .NET into `csharp`, `fsharp`, `cpp` and `dotnet`,
  because the Postgres parser otherwise turns C# and C++ into just "c". It also adds the generated `search_vector`
  column with the `simple` config (no stemming, so "AWS" doesn't become "aw") and a GIN index.
- **Migrations** are applied by hand: `psql` or a small Python call running each `db/00X_*.sql`. Integration tests
  rebuild `rag_test` from `init.sql` plus 003–008.

---

## 6. Retrieval (`RagService.search_chunks`)

1. **Embed** the question (with the bge query prefix).
2. **Hybrid search** in one SQL query (`ChunkRepository.search_similar`):
   - **Vector:** the nearest `HYBRID_VECTOR_CANDIDATES`=100 chunks, keeping those with similarity ≥ the threshold.
   - **Keywords:** Postgres full-text search with terms from `KeywordQueryBuilder`. It drops stop words (filler verbs
     like "worked" included) and drops terms that appear in more than 50% of chunks.
   - **Keywords only reorder chunks that already passed the similarity threshold.** They never bring in a chunk
     below it, so "nothing found" still means nothing relevant.
   - The two rankings are merged with **RRF** (k=60). `fused_score` is cast to `float8`, or psycopg returns it as a
     Decimal.
   - `hnsw.ef_search` is set per query, because by default HNSW returns at most 40 rows.
   - **Candidate filter** (`candidate_ids`): `candidate_id = ANY(...)` in both CTEs, plus
     `hnsw.iterative_scan = relaxed_order`. HNSW finds its nearest rows before the WHERE clause applies, so without
     iterative scan a strict filter could return too few. The outer query re-sorts the results.
3. **Thresholds:**
   - `MIN_SIMILARITY` = 0.5. On the old index, off-topic questions peaked at 0.44–0.49.
   - `MIN_SIMILARITY_FILTERED` = 0.35 when the search is limited to named candidates. At 0.5, "What is X's current
     job title?" lost the right chunk (0.44–0.49, rank 1–5). The answer LLM still refuses if the CV can't answer.
4. **Rerank:** the cross-encoder scores a pool of `max(top_k, 30)` candidates, and the top `top_k` (10) go to the
   LLM.
   - The order is stable, so ties keep the search order.
   - `rerank_score` is added to each chunk, and cosine `similarity` is kept.
   - Reranker scores are logits, comparable only within one query, so they are never used as a threshold.
5. **Timing logs:** hybrid search time, rerank time, answer LLM time.

**Answering** (`RagService.answer_question`):
- **No chunks** → the fixed answer "I couldn't find that in the documents.", `refusal_reason="retrieval"`, and no
  LLM call.
- **Otherwise** `PromptBuilder` numbers the chunks:
  `[n] (candidate 17, file: cv_017.pdf, folder: cvs)\n<content>`.
- The system prompt rules: use only the context, reply exactly with the refusal sentence if the answer isn't there,
  cite `[n]`, refer to people as "candidate N" and never guess names, be concise.
- An LLM refusal sets `refusal_reason="llm"`.
- Sources are de-duplicated by file and carry `candidate_id`.
- The ledger records the question, `standalone_question` (only if it differs), `conversation_id`, chunk count, the
  top similarity (the maximum, not the first), the answer and the latency. Ledger failures never break an answer.

**History of retrieval decisions (with eval evidence):**
- vector only at k=10: hit@k 0.74
- hybrid at k=10: 0.78
- hybrid at k=20: 0.92. The right chunk was found but ranked too low, which is why a reranker was added.
- hybrid + reranker at k=5: hit@1 0.843, hit@5 0.971, MRR 0.888 (70 questions). This was the old-index baseline.

---

## 7. LLM layer

- **`LlmClient` Protocol:** `generate(system, user) -> LlmResult(text, model, input_tokens, output_tokens)` and
  `close()`.
- **Errors:**
  - `LlmUnavailableError` and `LlmTimeoutError` are recoverable (`RECOVERABLE_LLM_ERRORS`).
  - `LlmConfigurationError` is not: bad key, bad request, unknown model. Retrying can't fix it.
- **`http_call.error_for_status`:** 402, 408, 413, 429 and 5xx are recoverable; any other 4xx is a configuration
  error.
- **Clients:** `OpenAiCompatibleLlmClient` (Groq and OpenRouter), `OllamaLlmClient` (optional `json_schema`, sent
  as Ollama's `format` for structured output), and `GeminiLlmClient` (kept, but not in the default chain).
- **Wrappers:**
  - `FallbackLlmClient(clients, cooldown_seconds)` tries the providers in order.
  - `RetryingLlmClient` retries with backoff of 2, 4 and 8 s; used for question generation.
  - `LoggingLlmClient` writes an `llm_calls` row per attempt. It reads purpose and request id from contextvars
    (`llm_call_context`).
- **Chains come from `.env`:**
  - `ANSWER_LLM_CHAIN` (default `groq,openrouter,ollama`) is used for answers, condensing, routing, requirement
    extraction and judging.
  - `QUESTION_LLM_CHAIN` (default `ollama`) is used for eval question generation, with retries and a 15-min cooldown.
  - **A missing API key for a listed provider fails startup** (`StartupConfigurationError`).
- **Timeouts:** 60 s for cloud providers and 240 s for local ones.
- **API errors:** timeout → 504, unavailable → 503, configuration → 500 with a generic message. Provider details stay
  in the logs.
- **Free-tier limits that affected design:** Groq allows 8,000 tokens a minute on `gpt-oss-120b`. OpenRouter free
  allows 20 requests a minute, and 50 a day under 10 credits.

---

## 8. Conversation memory, names and follow-ups: `ConversationGraph` (`memory/conversation_graph.py`)

### Graph
```
START ─► resolve ─┬─(a name fits 2+ candidates)──► clarify ─► END
                  ├─(first question)──────────────► answer ─► END
                  ├─(follow-up, conversation has a match_result) ► route ─┬─► answer_from_result ─► END
                  │                                                       └─► condense …
                  └─(normal follow-up)────────────► condense ─► check ─┬─► answer ─► END
                                                       ▲               │
                                                       └─(repair once)─┘
```

### State (`ChatState`, a TypedDict with `total=False`)
- **Kept across turns:**
  - `messages`, with the `add_messages` reducer, so it's appended to
  - `mentioned_candidates`: ids the conversation has referred to
  - `match_result`: the last `/match` result
- **Reset every turn:** `question, folder, top_k, candidate_ids (this turn's filter), clarification, route,
  standalone_question, rejection_reason, attempts, answer, sources, refusal_reason`.
- **Not in the state, on purpose:** retrieved chunks (so CV text isn't copied into checkpoints) and names (questions
  are stored after resolution).

### Nodes
- **`resolve`:** `NameResolver` (in `service/candidate_names.py`) finds names in the question.
  - Full names are matched first, then a single first or last name **only if written with a capital letter** and at
    least 3 letters long.
  - Each name is replaced with "candidate N", and the ids are added to `candidate_ids`.
  - A name that fits several candidates is narrowed by `mentioned_candidates` and the request's `candidate_ids`. If
    still ambiguous, the resolver sets `clarification`.
  - The `question` is **overwritten** with the resolved text, so no later step (cloud LLM or checkpoint) sees the
    name.
- **`clarify`:** answers "More than one candidate has that name: candidate 17 (8 years, Spark, Python); candidate 52
  (…). Which one do you mean?" There is no LLM call and no search, and the name itself is not repeated, because this
  text is saved in the history.
- **`route`** (only with a `match_result` and a router): `FollowUpRouter.classify` makes one LLM call and returns
  `from_result`, `search_candidate` or `new_question`. An unusable reply or an LLM failure leads to a search:
  routing only saves calls, and searching is never wrong.
- **`answer_from_result`:** the LLM answers from the compact ranking JSON only (ids, scores, verdicts, quotes), with
  no search.
- **`condense`:** `QuestionCondenser.rewrite` gives the LLM the last 6 messages and the question, and asks for
  `{"standalone_question": "..."}`. `check_standalone_question` then runs four checks:
  1. valid JSON with a non-empty question
  2. at most 300 characters
  3. no `[n]` citation and no refusal sentence
  4. every content word appears in the question or the history (stop words allowed, plus 5-character stem matches)

  A failed check stores the reason. An LLM failure falls back to the raw question.
- **`check`:** after 2 attempts without a valid rewrite, it uses the raw question. Otherwise it loops back to
  `condense` once, with the rejection reason in the prompt as a repair attempt.
- **`answer`:** builds the filter from `candidate_ids` plus any "candidate N" in the condensed question. That's how
  "what about her Java?", rewritten to "candidate 23's Java", keeps searching candidate 23.
  - It calls `RagService.answer_question(question, standalone_question, candidate_ids, conversation_id)`.
  - It appends **the question and its answer together** to `messages`, so a failed turn leaves nothing behind.
  - It updates `mentioned_candidates`.

### Persistence
- **LangGraph `PostgresSaver`**, keyed by `thread_id = conversation_id`.
  - It has its own psycopg pool (autocommit, `prepare_threshold=0`, dict rows, `search_path=main`).
  - `setup()` creates its tables at startup.
- **`durability="exit"`**: one checkpoint per turn instead of one per node.
- **`remember_match(conversation_id, request_text, match_result, summary)`:** `/match` uses `update_state(...,
  as_node="answer")` to write `match_result`, the ids and one message pair into a conversation, so `/chat` can
  continue it.
- **Why a checkpointer rather than our own `messages` table:** the app has no user-facing side to list or browse
  conversations, so automatic load and save wins. The cost is that the checkpointer stores the whole state, hence
  keeping chunks and names out of it.
- **Tests** use `InMemorySaver`. One integration test checks that the history survives a brand-new graph object
  through Postgres.

### `/chat` API (`routes.py`)
- **Request:** `question` (1–1000 chars, stripped), `folder?`, `top_k` (1–20), `conversation_id?` (≤100), and
  `candidate_ids?` (≤20, each ≥1). Extra fields are rejected.
- **Response:** `answer`, `sources[] {file_name, folder, similarity, candidate_id, candidate_name}`,
  `conversation_id`, `standalone_question`.
- **Names are put back only here:** `with_names()` turns "candidate 17" into "Ines Brooks (candidate 17)", using
  `CandidateRepository.names_by_id`.
- **MCP never maps names**, because its client is itself an LLM, often a cloud one.

---

## 9. JD matching: `MatchGraph` (`matching/match_graph.py`) and `POST /match`

### Graph
```
START ─► extract ─► validate ─(invalid, 1st time)─► extract   (repair: the error is quoted back to the LLM)
                       │ (invalid, 2nd time) → RequirementExtractionError → HTTP 503
                       ▼
                   prefilter ─► search ─► shortlist ──Send × N──► judge ──► rank ─► END
```

1. **extract:** a cloud LLM call (a JD has no personal data) returns
   `{"requirements": [{skill, must_have, min_years, synonyms}], "min_total_years"}`.
2. **validate:** the pydantic `JobRequirements` model checks 1–15 requirements (more are trimmed) and skills of 1–80
   characters; years are 0–50.
3. **prefilter:** if `min_total_years` is set, a SQL query keeps candidates with `total_years >=` it
   (`ids_with_min_years`). Hard filters only use reliable fields: total years are computed in code. Extracted skills
   are never used as a filter, because the LLM can miss one and wrongly drop a candidate.
4. **search:** one hybrid + reranker search per requirement (the skill plus its synonyms, 20 chunks), recording which
   candidates each requirement found. Plus one whole-JD vector search (50 chunks, no threshold) for recall.
5. **shortlist:** RRF of two rankings: requirement coverage (must-have = 2, nice-to-have = 1) and the whole-JD
   ranking. Top `MATCH_SHORTLIST_SIZE` = 10. Coverage alone gave nDCG 0.660, so the whole-JD ranking was fused in.
6. **judge:** `fan_out` returns `[Send("judge", {candidate_id, requirements}) …]`, LangGraph's map step.
   - Each judge reads the candidate's **whole redacted CV** (`cv_text`, all chunks in order) and returns one verdict
     per requirement: `met`, `partial` or `not_found`, plus a word-for-word quote.
   - **Code checks the reply:** a missing or unreadable verdict becomes `not_found`, and a quote not found in the CV
     turns met or partial into `not_found` with a note. A bad reply can only make a candidate weaker.
   - Results are collected through the `judgements: Annotated[list, operator.add]` reducer, the reduce step.
   - `max_concurrency` = 1, because 3 parallel judges hit Groq's 8,000 tokens a minute and fell back to weaker models.
7. **rank:** `score()` = the weighted share of requirements met (must = 2, nice = 1, partial counts 0.5), **capped at
   0.5 if any must-have is `not_found`**. Ties are broken by shortlist order. It returns the top `top_n` with
   `verdicts`, `gaps` and `missing_must_haves`.

There is no checkpointer: a match is one-shot. LLM calls per JD: about 1 + shortlist size, so about 11.

### `/match` API
- **Request:** `jd_text` (20–20,000 characters), `top_n` (1–20, default 10), `conversation_id?`.
- **Response:** `conversation_id`, `requirements[]`, `min_total_years`, and `candidates[] {candidate_id,
  candidate_name, score, verdicts[] {requirement, must_have, verdict, evidence, note?}, gaps, missing_must_haves}`.
- **After ranking**, it calls `remember_match` with "Match candidates to this job description: <first 300 chars>"
  and a summary like "Ranked N candidates: candidate 17 (0.92), …". The summary has no names.
- **Errors:** a failed extraction returns 503 with a generic message; LLM errors are mapped the same way as in
  `/chat`.

### Follow-ups after a match
Call `/chat` with the `conversation_id` from `/match`:
- "Why is candidate 17 above 23?" → `route` → `from_result` → answered from the stored ranking
- "Does candidate 17 know Terraform?" → `search_candidate` → condense → search filtered to 17
- An unrelated question → `new_question` → normal search

---

## 10. PII rules (the consistent policy)

1. **Raw CVs are only read by local code and the local model** (profile extraction).
2. **Only redacted text is stored as chunks.** The contact and referee sections are dropped; emails, phones, URLs and
   the name are masked.
3. **Cloud LLMs only see:** redacted chunks labelled "candidate N", questions with names already replaced, JDs, and
   match results with ids.
4. **The checkpoint history holds resolved questions and id-based answers.** Clarification texts don't repeat names.
5. **Names come back only in the REST response.** MCP returns ids only.
6. **Safety nets:**
   - a leak check before indexing
   - a real-CV leak test that uses the manifest's true names
   - a unit test asserting that a name never reaches the LLM prompt or the stored history
   - a scan of the live database (0 leaks)
7. **Deliberately not used (yet):** Microsoft Presidio (NER), because it's an extra dependency and flags tech names
   like "Jenkins" or "Kafka" as people. LLM-based redaction, because it may alter real text.

---

## 11. Evaluation

| Eval | Command (run from `src`) | What it measures |
|---|---|---|
| Retrieval | `python -m rag.evaluation.run_retrieval --top-k 5` | the synthetic questions; does the chunk containing the evidence quote rank in the top k? (hit@1, hit@k, file hit@k, MRR; misses due to the threshold; "unmatchable" when no chunk contains the evidence any more) |
| Unanswerable | `python -m rag.evaluation.run_unanswerable --top-k 10` | 6 out-of-domain + 9 near-miss questions (should refuse) + 20 answerable (seed 42): hallucination rate and false refusal rate |
| Matching, whole JD | `python -m rag.evaluation.run_matching` | 30 JDs: vector search with the whole JD, CVs ranked by best chunk; nDCG@10, recall@10, precision@10, MRR |
| Matching, graph | `python -m rag.evaluation.run_matching --strategy match_graph --limit 5` | `/match`'s workflow, plus the whole-JD baseline on the **same JDs in the same run**, plus shortlist recall |
| Question generation | `python -m rag.evaluation.generate_questions` | builds synthetic questions from chunks (`QUESTION_LLM_CHAIN`), with evidence quotes checked against the chunk |

- Runs are saved in `eval_runs` with their settings. A run is compared with the previous comparable run and fails on
  a regression of more than 0.03.
- **The retrieval and unanswerable evals resolve names exactly like `/chat`** (`resolved_search`). Without that they
  measured a path chat no longer uses: hit@5 0.64 and 60% false refusals.

### Results

| Eval | Before (old index) | After all phases |
|---|---|---|
| Retrieval k=5 | hit@1 0.843, hit@5 0.971, MRR 0.888 (70 questions, run 15) | hit@1 0.688, hit@5 0.922, MRR 0.773 (64 questions + 6 unmatchable, run 23) |
| Unanswerable k=10 | hallucination 0%, false refusals 5% (run 16) | hallucination 0%, false refusals 15% (run 24) |
| Matching whole JD (30 JDs) | nDCG@10 0.849, recall 0.850, precision 0.663, MRR 1.0 (run 17) | nDCG@10 0.875, recall 0.867, precision 0.680, MRR 0.978 (run 20) |
| Matching graph (5 JDs) | whole JD on the same 5: nDCG 0.819 | nDCG 0.732, MRR 0.900, shortlist recall 0.711 (run 25; it was 0.660 before the RRF shortlist) |

**How to read these:**
- **Retrieval fell** for two reasons. Five questions name nobody ("impact of introducing pair programming on the
  team"), and many CVs share that templated wording, while the smaller section chunks carry less context. And 6
  questions became unmatchable because their evidence contained redacted text.
- **Whole-JD matching improved** with section chunks.
- **The `/match` graph doesn't yet beat the whole-JD ranking on 5 JDs.** The main cause is shortlist recall (71% of
  relevant CVs reach the judge), plus free-tier fallbacks to weaker models. It's kept for its evidence and gaps, and
  wasn't tuned further, because tuning on 5 JDs would overfit.

---

## 12. Tests

- **327 tests pass** (unit plus integration).
- **Run:** `.venv/Scripts/python.exe -m pytest tests`. Integration tests need `TEST_DATABASE_URL`, and the database
  name must end in `_test`, because the schema is dropped and rebuilt.
- **Fakes** (`tests/fakes.py`): `WordTokenizer`, `CharTokenizer`, `FakeEmbedder`, `FakeLlmClient` (records purpose
  and request id), `FakeChunkRepository`, `FakeLedgerRepository`, `FakeKeywordRepository`, `FakeReranker`,
  `chunk_row`, `unit_vector`. Test files add their own `ScriptedLlm`, `FakeCandidateRepository`, `FakeDirectory`,
  `StubMatchGraph` and so on.
- **Notable tests:**
  - **Chunking:** chunking never invents text (formerly the xfail); no chunk crosses sections.
  - **Redaction:** leaks over all 80 real CVs.
  - **Profile extraction:** a glued-on name word is recovered.
  - **Memory:** the condense repair and fallback, the 6-message window, a failed turn not being remembered, and
    history surviving a new graph through Postgres.
  - **Names:** a name never reaches the LLM or the history; an ambiguous name asks for clarification; a pronoun
    follow-up keeps the candidate filter.
  - **Match follow-ups:** answered from the result without searching; a CV-detail question is searched with the
    filter.
  - **Matching:** the fan-out judges every shortlisted candidate; the years prefilter; the extraction repair, and
    failure after two bad replies; a fake quote becomes not_found; the must-have cap.
  - **API:** names appear only in responses; `/match` 503 handling; delete returns 204 or 404.

---

## 13. How to run

```bash
docker start rag-db                                   # Postgres + pgvector
cd src
../.venv/Scripts/python.exe -m rag.indexing.indexing_pipeline   # ingest (incremental)
../.venv/Scripts/python.exe -m uvicorn rag.api.main:app --port 8000
../.venv/Scripts/python.exe -m rag.mcp_server.server            # MCP on 127.0.0.1:8001/mcp
```

`.env` must contain `DOCS_DIR`, `DATABASE_URL`, `ANSWER_LLM_CHAIN` and the keys of the providers it lists,
`EVAL_DATASET_DIR`, and `TEST_DATABASE_URL`. Optional overrides: `GROQ_MODEL`, `OPENROUTER_MODEL`, `OLLAMA_MODEL`,
`PROFILE_LLM_MODEL`, `QUESTION_LLM_CHAIN`, and `RERANKER_MODEL` (`none` switches reranking off).

**Example flow:**
1. `POST /match {"jd_text": "...", "top_n": 5}` returns a ranking and a `conversation_id`.
2. `POST /chat {"question": "Why is candidate 17 first?", "conversation_id": "<id>"}` is answered from the ranking.
3. `POST /chat {"question": "Does Ines Brooks know Terraform?", "conversation_id": "<id>"}` resolves the name to an
   id and searches that CV.
4. `DELETE /candidates/17` removes the candidate and its chunks.

---

## 14. Where LangGraph and LangChain are used, and why

| Feature | Used | Problem it solves |
|---|---|---|
| `StateGraph` + conditional edges | chat and match graphs | explicit routing: first turn vs follow-up, clarification, follow-up routes |
| Loops | condense → check → condense; extract → validate → extract | repair attempts capped at one |
| `add_messages` reducer | chat memory | history is appended to automatically |
| `PostgresSaver` checkpointer + `thread_id` | chat memory | automatic load and save per conversation |
| `durability="exit"` | chat | one snapshot per turn; a failed turn saves nothing |
| `update_state(as_node=...)` | `/match` → chat | injects a match result into a conversation |
| `Send` + `operator.add` reducer | `/match` judge | map-reduce over candidates, with `max_concurrency` |
| LangChain `RecursiveCharacterTextSplitter` | chunking | replaced a buggy hand-written splitter |
| **Not used** | LangChain chains, agents, LLM wrappers | our own `LlmClient` chain already handles fallback, retries and the ledger |

**Interview points:**
- **Workflow before agent:** fixed graphs with capped loops. A tool-calling agent (Stage 5) is planned only as an
  experiment, kept only if it beats the baseline.
- **Why not more LangChain:** it would replace working, tested code without solving a problem.
- **Checkpointer trade-off:** it saves the whole state, hence no chunks or names in it.
- **Chunk-level retrieval vs candidate-level ranking:** grouping by candidate and the shortlist close the gap.
- **The PII design is one rule applied everywhere:** redacted storage, ids in LLM traffic, names only in the
  response.

---

## 15. Known limitations and open items

- **`/match` ranking below the whole-JD baseline** (5 JDs). Next steps: a larger or better-fused shortlist, the full
  30-JD eval (slow on the free tier), and checking the judge model's quality.
- **Retrieval misses on questions that name nobody**, plus the 6 unmatchable questions.
- **Two-column CVs:** the name and title line land inside the education section, which is harmless after redaction.
- **A name match needs a capital letter** for single names, and very short name parts (< 3 letters) aren't masked
  alone.
- **The years computation is slightly high** ("Present" counts from January of the start year). Per-skill years were
  deliberately skipped: complex, and not needed for v1.
- **Deferred:**
  - the conversation-memory eval (10–15 hand-written conversations)
  - a route-accuracy set for the router (about 20 labelled follow-ups)
  - a `condense_outcome` ledger column
  - the agent experiment
  - the synthetic 500-CV load test
  - Presidio
- **Housekeeping:** `requirements.txt` is incomplete; the old `Simple Rag-hybrid` worktree is obsolete; GitHub branch
  protection is suggested, because pushes went to `main` directly twice.
- **Git:** work is on `main`, with memory committed as `b78af79`. The phase 1–4 work is uncommitted and only
  partially staged. Suggest moving it to a feature branch before committing.
- **Working rules with the user:**
  - never commit or push unless asked
  - ask before evals or downloads that write to the user's database or are large
  - never echo API keys
  - integration tests only on a `*_test` database
  - plan before implementing
  - keep the decision log updated

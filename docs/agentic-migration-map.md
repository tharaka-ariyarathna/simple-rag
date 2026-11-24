# Skill Finder: agentic migration map

Milestone 0 of the agentic plan. This maps each part of today's code to its place in the agent, and lists what
does not fit. No code was changed in this milestone.

## Today

Two fixed LangGraph graphs:

- **`MatchGraph`** (`matching/match_graph.py`, `POST /match`, one-shot, no checkpointer):
  `extract → validate (repair once) → prefilter → search → shortlist → Send × N judge → rank`
- **`ConversationGraph`** (`memory/conversation_graph.py`, `POST /chat`, Postgres checkpointer keyed by
  `conversation_id`):
  `resolve → clarify | route → answer_from_result | condense → check → answer`

## After

```
API → NameResolver (names → "candidate N", outside the agent) → agent ⇄ tools → reply
                                                                 │
                                       ask_clarification ────────┴→ END
```

One agent node (tool-calling LLM) + one tool node, compiled with the same `PostgresSaver` and `thread_id`.

## Map: existing code → tool

| Existing code | Becomes | Notes |
|---|---|---|
| `RequirementExtractor.extract` + `parse_requirements` + `MatchGraph.validate` / `route_after_validate` (repair once) | **`extract_criteria`** | The extract → validate → repair loop moves inside the tool as a plain loop (max 2 attempts). Output `JobRequirements` is converted to `SearchCriteria`. Merges with criteria already in state. The extraction prompt is JD-oriented; it must also accept short requests like "best person for AWS deployments" (prompt tweak only, output schema unchanged). |
| *(none)* | **`update_criteria`** | New, deterministic. Edits `SearchCriteria` in state. |
| `RagService.search_chunks` (hybrid + reranker, `MATCHING_MIN_SIMILARITY`) | **`count_matches`** | New wrapper: one search per skill, count distinct `candidate_id`s. |
| `MatchGraph.prefilter` + `search` + `shortlist` | **`search_candidates`** | Called as plain methods, unchanged. Retrieves the fused order, keeps 30 in state, the top 10 go to ranking. Years prefilter (`min_total_years`) kept. Adds the D1 floor check before searching. Needs the JD text for the whole-JD search: for conversation-built criteria, a query text is built from role + requirements. |
| `MatchGraph.fan_out` + `judge` (`CandidateJudge.judge`, `checked_verdicts`) + `rank` (`score`) | **`rank_candidates`** | Unchanged logic, run as a loop over the shortlist (one at a time, as today's concurrency 1). Score shown as `score × 100`. Verdict labels stay `met / partial / not_found`. `strengths` = met (must-haves first), `gaps` = not met (must-haves first). |
| `CandidateRepository.cv_text` | **`get_candidate_detail`** | Whole redacted CV; only for ids in the current results. |
| `RagService.answer_question` / `search_chunks` (general Q&A, `ConversationGraph.answer`) | **`search_cv_chunks`** | General questions over CVs ("who knows Spark?"). Returns compact chunks with candidate ids; the agent writes the answer. Optional candidate filter. |
| *(none)* | **`ask_clarification`** | New. Limits enforced in code (≤3 questions, ≤2 rounds). Routes to `END`. |
| `ConversationGraph.resolve` (`NameResolver`) | **Stays outside the agent** | Runs on the user message before the agent, so names never reach the cloud LLM or the checkpoint. |
| `ConversationGraph.clarify` (ambiguous name) | **Stays outside the agent** | Same fixed reply, no LLM call. |
| `FollowUpRouter.classify` + `answer_from_result` (`memory/follow_up.py`) | **Removed (agent decides)** | The agent reads `ranking` from state and chooses `get_candidate_detail`, `search_cv_chunks` or a direct answer. |
| `QuestionCondenser` + `check_standalone_question` (`memory/condenser.py`) | **Removed (agent decides)** | The agent sees the history and writes standalone tool inputs itself. |
| `ChatState.messages`, `mentioned_candidates` | `AgentState.messages`, `mentioned_candidates` | `mentioned_candidates` is kept for name resolution. |
| `ChatState.match_result` | `AgentState.ranking` | |
| `ConversationGraph.remember_match` | **Removed** | `/match` routes through the agent, which writes its own state. |
| `MatchGraph` class | Kept behind a flag until approved for deletion (M4) | Its methods are reused by the tools. |
| `FallbackLlmClient` chain (Groq → OpenRouter → Ollama) | Unchanged for LLM calls **inside** tools | Extraction and judging keep `LlmClient`. |
| *(none)* | **Agent model**: LangChain chat models with `bind_tools` + `with_fallbacks` | Groq and OpenRouter via `ChatOpenAI` (OpenAI-compatible), Ollama via `ChatOllama` with `qwen2.5:7b`. |
| `POST /chat` (`ChatRequest.question`) | Agent turn | Response becomes `{reply, type, criteria, ranking, conversation_id}`, with names put back only here. |
| `POST /match` | Deprecated; sends the JD to the agent as the first message | |
| MCP server | **Unchanged** (out of scope for now) | Keeps using `RagService` directly. |

## What doesn't fit cleanly

1. **The plan's "LLM sufficiency check" and "retrieved enough?" loop don't exist.** Instead, the agent decides
   after each tool result whether it has enough to answer.
2. **No thin-CV rule.** `insufficient_info` is set only if the agent judges it from the evidence. No CV is ever
   dropped.
3. **Judging is done inside one tool.** Today the judges fan out with `Send`. Inside a tool there is no graph, so
   it is a plain loop. With concurrency 1 the behaviour is the same.
4. **Whole-JD search input.** `MatchGraph.search` searches the raw JD text. With criteria built in conversation
   there is no JD, so a query text is built from the criteria. With a full JD, the original text is kept in state
   so the ranking matches `MatchGraph`.
5. **Ranking regression.** The judge is an LLM, so "same scores" holds only for the same verdicts. The regression
   check fixes the judge's replies with a fake LLM and compares the scores.
6. **Existing checkpoints use `ChatState`.** Old conversations won't load into `AgentState`, so the agent uses a
   new checkpoint namespace and old conversations stay with the old graph.

## Tool-calling support

Checked on 2026-11-24 with one raw OpenAI-compatible request per provider: a `count_matches` tool definition and
"How many candidates know Kafka or RabbitMQ? Use the tool."

| Provider | Model | Result |
|---|---|---|
| Groq | `openai/gpt-oss-120b` | Tool call OK: `count_matches(skills=["Kafka", "RabbitMQ"])` |
| OpenRouter | `nvidia/nemotron-3-super-120b-a12b:free` | **HTTP 404: the free version of this model was withdrawn.** This also breaks the existing answer chain's second step. A different free OpenRouter model is needed. |
| Ollama (local) | `qwen2.5:7b` | Tool call OK: same call. Becomes the local fallback (replacing `llama3.2:3b`). |

One correct call on a simple prompt doesn't prove multi-step reliability. That shows up in M2.

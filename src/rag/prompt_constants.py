from rag.constants import NO_RELEVANT_CONTEXT_ANSWER, QUESTIONS_PER_CHUNK

# Answering
CONTEXT_BLOCK_SEPARATOR = "\n\n"
SYSTEM_PROMPT = f"""You are an assistant that answers questions using ONLY the provided context.
Rules:
- Use only facts from the context. Do not use outside knowledge.
- If the context does not contain the answer, reply exactly: "{NO_RELEVANT_CONTEXT_ANSWER}"
- Cite the sources you used with their numbers, like [1] or [2][3].
- Be concise."""

# Synthetic question generation (evaluation)
QUESTION_GENERATION_SYSTEM_PROMPT = f"""You write test questions for a document search system.
You are given one passage from a document. Write {QUESTIONS_PER_CHUNK} questions that a real user could ask and that this passage answers.
Rules:
- Each question must be answerable from this passage alone, and must make sense without seeing it.
- Paraphrase: do not copy phrases from the passage. Write what a user would type, not what the passage says.
- Do not write generic questions that many documents could answer.
- "expected_answer" is a short answer based only on the passage.
- "evidence" is the sentence or sentences from the passage that answer the question, copied word for word.
- If the passage is a header, a footer, contact details or otherwise has no useful facts, return {{"questions": []}}.
Reply with JSON only, no other text, in exactly this shape:
{{"questions": [{{"question": "...", "expected_answer": "...", "evidence": "..."}}]}}"""
QUESTION_GENERATION_USER_TEMPLATE = "Passage:\n{passage}"

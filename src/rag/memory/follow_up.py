import json
import logging
from rag.llm.base import LlmClient
from rag.llm.call_context import llm_call_context
from rag.prompt_constants import FROM_RESULT_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE

logger = logging.getLogger(__name__)

ROUTE_FROM_RESULT = "from_result"  # the stored match result answers it: no search
ROUTE_SEARCH_CANDIDATE = "search_candidate"  # needs a detail from a candidate's CV: filtered search
ROUTE_NEW_QUESTION = "new_question"  # unrelated to the match: normal search
ROUTES = (ROUTE_FROM_RESULT, ROUTE_SEARCH_CANDIDATE, ROUTE_NEW_QUESTION)


def compact_match_result(match_result: dict) -> str:
    """What the LLM needs from a match: ids, scores, verdicts and quotes. No names are ever in a match result."""
    return json.dumps({
        "requirements": [{"skill": r["skill"], "must_have": r["must_have"]} for r in match_result["requirements"]],
        "candidates": [{"candidate": c["candidate_id"], "rank": rank, "score": c["score"],
                        "verdicts": [{"requirement": v["requirement"], "verdict": v["verdict"], "evidence": v["evidence"]}
                                     for v in c["verdicts"]]}
                       for rank, c in enumerate(match_result["candidates"], start=1)],
    }, ensure_ascii=False)


def conversation_text(history: list[tuple[str, str]]) -> str:
    return "\n".join(f"{speaker}: {text}" for speaker, text in history)


class FollowUpRouter:
    """One LLM call classifies a follow-up into a fixed set of routes (a router node, not a free agent), so the
    number of calls per question is known and the routing can be measured on a labelled set."""

    def __init__(self, llm_client: LlmClient):
        self.llm_client = llm_client

    def classify(self, question: str, history: list[tuple[str, str]], match_result: dict) -> str:
        user_prompt = ROUTER_USER_TEMPLATE.format(match_result=compact_match_result(match_result),
                                                  history=conversation_text(history), question=question)
        with llm_call_context("route"):
            reply = self.llm_client.generate(ROUTER_SYSTEM_PROMPT, user_prompt).text
        route = parse_route(reply)
        logger.info("Follow-up routed to %s", route)
        return route

    def answer_from_result(self, question: str, history: list[tuple[str, str]], match_result: dict) -> str:
        user_prompt = ROUTER_USER_TEMPLATE.format(match_result=compact_match_result(match_result),
                                                  history=conversation_text(history), question=question)
        with llm_call_context("answer_from_match"):
            return self.llm_client.generate(FROM_RESULT_SYSTEM_PROMPT, user_prompt).text


def parse_route(reply: str) -> str:
    """The route in the reply; anything unusable becomes a search, which is slower but never wrong."""
    first_brace, last_brace = reply.find("{"), reply.rfind("}")
    try:
        route = json.loads(reply[first_brace:last_brace + 1])["route"]
    except (ValueError, KeyError, TypeError):
        return ROUTE_SEARCH_CANDIDATE
    return route if route in ROUTES else ROUTE_SEARCH_CANDIDATE

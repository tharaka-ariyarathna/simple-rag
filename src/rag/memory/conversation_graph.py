import logging
import uuid
from dataclasses import dataclass
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from rag.constants import CONDENSE_MAX_ATTEMPTS, CONVERSATION_HISTORY_MESSAGES
from rag.llm.base import RECOVERABLE_LLM_ERRORS
from rag.memory.condenser import QuestionCondenser, check_standalone_question
from rag.memory.follow_up import ROUTE_FROM_RESULT, ROUTE_SEARCH_CANDIDATE, FollowUpRouter
from rag.service.candidate_names import NameResolver, candidate_ids_in
from rag.service.rag_service import RagAnswer, RagService, SourceReference

logger = logging.getLogger(__name__)


class ChatState(TypedDict, total=False):
    """What the checkpointer saves for a conversation after each turn. Retrieved chunks and candidate names are
    deliberately not in it: questions are stored with names already replaced by "candidate N"."""

    messages: Annotated[list[AnyMessage], add_messages]  # appended to, never replaced: the conversation memory
    mentioned_candidates: list[int]  # kept across turns: who "Ines" or "she" can refer to
    match_result: dict | None  # kept across turns: the last /match result in this conversation (ids, verdicts)
    question: str  # the rest is overwritten every turn
    folder: str | None
    top_k: int
    candidate_ids: list[int]  # this turn's search filter: sent by the client or resolved from names
    clarification: str  # set when a name fits several candidates
    route: str  # follow-up route after a match
    standalone_question: str  # "" until a rewrite passes its checks (or the raw question is chosen)
    rejection_reason: str
    attempts: int
    answer: str
    sources: list[dict]
    refusal_reason: str | None


@dataclass
class ConversationAnswer:
    conversation_id: str
    standalone_question: str  # what retrieval actually searched for
    rag_answer: RagAnswer


class ConversationGraph:
    """The chat flow as a LangGraph graph with memory:

        START -> resolve -> clarify -> END                          (a name fits several candidates)
        START -> resolve -> route -> answer_from_result -> END      (follow-up a /match result can answer)
        START -> resolve [-> route] -> condense -> check -> answer -> END   (follow-ups; check loops back once)
        START -> resolve -> answer -> END                            (first question of a conversation)

    The checkpointer keeps the state per conversation_id (LangGraph's thread_id), so the next call sees the messages.
    resolver and router are optional: without them names are not resolved and there is no follow-up routing."""

    def __init__(self, rag_service: RagService, condenser: QuestionCondenser, checkpointer,
                 history_messages: int = CONVERSATION_HISTORY_MESSAGES, max_attempts: int = CONDENSE_MAX_ATTEMPTS,
                 resolver: NameResolver | None = None, router: FollowUpRouter | None = None):
        self.rag_service = rag_service
        self.condenser = condenser
        self.history_messages = history_messages
        self.max_attempts = max_attempts
        self.resolver = resolver
        self.router = router
        builder = StateGraph(ChatState)
        builder.add_node("resolve", self.resolve)
        builder.add_node("clarify", self.clarify)
        builder.add_node("route", self.route)
        builder.add_node("answer_from_result", self.answer_from_result)
        builder.add_node("condense", self.condense)
        builder.add_node("check", self.check)
        builder.add_node("answer", self.answer)
        builder.add_edge(START, "resolve")
        builder.add_conditional_edges("resolve", self.route_after_resolve, ["clarify", "route", "condense", "answer"])
        builder.add_edge("clarify", END)
        builder.add_conditional_edges("route", self.route_after_route, ["answer_from_result", "condense"])
        builder.add_edge("answer_from_result", END)
        builder.add_edge("condense", "check")
        builder.add_conditional_edges("check", self.route_after_check, ["condense", "answer"])
        builder.add_edge("answer", END)
        self.graph = builder.compile(checkpointer=checkpointer)

    def ask(self, question: str, folder: str | None, top_k: int, conversation_id: str | None = None,
            candidate_ids: list[int] | None = None) -> ConversationAnswer:
        conversation_id = conversation_id or str(uuid.uuid4())
        turn = {"question": question, "folder": folder, "top_k": top_k, "candidate_ids": list(candidate_ids or []),
                "clarification": "", "route": "", "standalone_question": "", "rejection_reason": "", "attempts": 0}
        # durability="exit": one checkpoint per turn instead of one per node. The question joins the messages only
        # together with its answer, so a turn that fails leaves no question without an answer behind.
        state = self.graph.invoke(turn, self.config(conversation_id), durability="exit")
        sources = [SourceReference(**source) for source in state["sources"]]
        rag_answer = RagAnswer(state["answer"], sources, state["refusal_reason"])
        return ConversationAnswer(conversation_id, state["standalone_question"], rag_answer)

    def remember_match(self, conversation_id: str, request_text: str, match_result: dict, summary: str) -> None:
        """Starts (or continues) a conversation with a /match result in it, so follow-ups can be routed to it."""
        candidate_ids = [candidate["candidate_id"] for candidate in match_result.get("candidates", [])]
        self.graph.update_state(self.config(conversation_id), {
            "match_result": match_result, "mentioned_candidates": candidate_ids,
            "messages": [HumanMessage(request_text), AIMessage(summary)],
        }, as_node="answer")

    @staticmethod
    def config(conversation_id: str) -> dict:
        return {"configurable": {"thread_id": conversation_id}}

    def history(self, state: ChatState) -> list[tuple[str, str]]:
        """The last messages of the conversation so far, as (speaker, text)."""
        earlier = state.get("messages", [])[-self.history_messages:]
        return [("User" if message.type == "human" else "Assistant", str(message.content)) for message in earlier]

    # --- nodes and routing -----------------------------------------------------------------------------------

    def resolve(self, state: ChatState) -> dict:
        """Names in the question become "candidate N" (the name goes no further) and join the search filter."""
        if self.resolver is None:
            return {}
        known = state.get("mentioned_candidates", []) + state.get("candidate_ids", [])
        resolution = self.resolver.resolve(state["question"], known)
        return {"question": resolution.question, "clarification": resolution.clarification or "",
                "candidate_ids": list(dict.fromkeys(state.get("candidate_ids", []) + resolution.candidate_ids))}

    def route_after_resolve(self, state: ChatState) -> str:
        if state.get("clarification"):
            return "clarify"
        if not state.get("messages"):
            return "answer"  # the first question has nothing to resolve against
        if self.router is not None and state.get("match_result"):
            return "route"
        return "condense"

    def clarify(self, state: ChatState) -> dict:
        return {"answer": state["clarification"], "sources": [], "refusal_reason": None,
                "standalone_question": state["question"],
                "messages": [HumanMessage(state["question"]), AIMessage(state["clarification"])]}

    def route(self, state: ChatState) -> dict:
        try:
            route = self.router.classify(state["question"], self.history(state), state["match_result"])
        except RECOVERABLE_LLM_ERRORS as error:  # routing is an optimisation: searching is always correct
            logger.warning("Follow-up routing failed, searching instead: %s", error)
            route = ROUTE_SEARCH_CANDIDATE
        return {"route": route}

    def route_after_route(self, state: ChatState) -> str:
        return "answer_from_result" if state["route"] == ROUTE_FROM_RESULT else "condense"

    def answer_from_result(self, state: ChatState) -> dict:
        answer = self.router.answer_from_result(state["question"], self.history(state), state["match_result"])
        return {"answer": answer, "sources": [], "refusal_reason": None, "standalone_question": state["question"],
                "mentioned_candidates": self.mentioned(state, candidate_ids_in(answer)),
                "messages": [HumanMessage(state["question"]), AIMessage(answer)]}

    def condense(self, state: ChatState) -> dict:
        attempts = state["attempts"] + 1
        history = self.history(state)
        try:
            reply = self.condenser.rewrite(history, state["question"], state["rejection_reason"])
        except RECOVERABLE_LLM_ERRORS as error:  # no rewrite is no reason to fail the whole question
            logger.warning("Condensing failed, using the question as asked: %s", error)
            return {"attempts": attempts, "standalone_question": state["question"], "rejection_reason": ""}
        verdict = check_standalone_question(reply, state["question"], history)
        if verdict.question:
            return {"attempts": attempts, "standalone_question": verdict.question, "rejection_reason": ""}
        logger.warning("Condensed question rejected (attempt %d of %d): %s", attempts, self.max_attempts,
                       verdict.reason)
        return {"attempts": attempts, "standalone_question": "", "rejection_reason": verdict.reason}

    def check(self, state: ChatState) -> dict:
        """Out of attempts without a usable rewrite: search with the question as asked."""
        if not state["standalone_question"] and state["attempts"] >= self.max_attempts:
            return {"standalone_question": state["question"]}
        return {}

    def route_after_check(self, state: ChatState) -> str:
        return "answer" if state["standalone_question"] else "condense"

    def answer(self, state: ChatState, config) -> dict:
        standalone_question = state.get("standalone_question") or state["question"]
        # "candidate 17" can come from the resolver, the client, or a rewrite of "what about her Java?"
        candidate_ids = list(dict.fromkeys(state.get("candidate_ids", []) + candidate_ids_in(standalone_question)))
        rag_answer = self.rag_service.answer_question(
            state["question"], state["folder"], state["top_k"], entry_point="api",
            conversation_id=config["configurable"]["thread_id"], standalone_question=standalone_question,
            candidate_ids=candidate_ids or None,
        )
        return {
            "standalone_question": standalone_question, "answer": rag_answer.answer,
            "sources": [{"file_name": s.file_name, "folder": s.folder, "similarity": s.similarity,
                         "candidate_id": s.candidate_id} for s in rag_answer.sources],
            "refusal_reason": rag_answer.refusal_reason,
            "mentioned_candidates": self.mentioned(state, candidate_ids + candidate_ids_in(rag_answer.answer)),
            "messages": [HumanMessage(state["question"]), AIMessage(rag_answer.answer)],
        }

    @staticmethod
    def mentioned(state: ChatState, new_ids: list[int]) -> list[int]:
        return list(dict.fromkeys(state.get("mentioned_candidates", []) + new_ids))

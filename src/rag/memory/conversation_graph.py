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
from rag.service.rag_service import RagAnswer, RagService, SourceReference

logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    """What the checkpointer saves for a conversation after each turn. Retrieved chunks are deliberately not in it."""

    messages: Annotated[list[AnyMessage], add_messages]  # appended to, never replaced: the conversation memory
    question: str  # the rest is overwritten every turn
    folder: str | None
    top_k: int
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

        START -> condense -> check -> answer -> END      (follow-ups; check loops back to condense once)
        START -> answer -> END                            (first question of a conversation)

    The checkpointer keeps the state per conversation_id (LangGraph's thread_id), so the next call sees the messages."""

    def __init__(self, rag_service: RagService, condenser: QuestionCondenser, checkpointer,
                 history_messages: int = CONVERSATION_HISTORY_MESSAGES, max_attempts: int = CONDENSE_MAX_ATTEMPTS):
        self.rag_service = rag_service
        self.condenser = condenser
        self.history_messages = history_messages
        self.max_attempts = max_attempts
        builder = StateGraph(ChatState)
        builder.add_node("condense", self.condense)
        builder.add_node("check", self.check)
        builder.add_node("answer", self.answer)
        builder.add_conditional_edges(START, self.route_start, ["condense", "answer"])
        builder.add_edge("condense", "check")
        builder.add_conditional_edges("check", self.route_after_check, ["condense", "answer"])
        builder.add_edge("answer", END)
        self.graph = builder.compile(checkpointer=checkpointer)

    def ask(self, question: str, folder: str | None, top_k: int,
            conversation_id: str | None = None) -> ConversationAnswer:
        conversation_id = conversation_id or str(uuid.uuid4())
        turn = {"question": question, "folder": folder, "top_k": top_k,
                "standalone_question": "", "rejection_reason": "", "attempts": 0}
        # durability="exit": one checkpoint per turn instead of one per node. The question joins the messages only
        # together with its answer (see answer), so a turn that fails leaves no question without an answer behind.
        state = self.graph.invoke(turn, {"configurable": {"thread_id": conversation_id}}, durability="exit")
        sources = [SourceReference(**source) for source in state["sources"]]
        rag_answer = RagAnswer(state["answer"], sources, state["refusal_reason"])
        return ConversationAnswer(conversation_id, state["standalone_question"], rag_answer)

    def history(self, state: ChatState) -> list[tuple[str, str]]:
        """The last messages of the conversation so far, as (speaker, text)."""
        earlier = state["messages"][-self.history_messages:]
        return [("User" if message.type == "human" else "Assistant", str(message.content)) for message in earlier]

    def route_start(self, state: ChatState) -> str:
        return "condense" if state["messages"] else "answer"  # the first question has nothing to resolve

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
        standalone_question = state["standalone_question"] or state["question"]
        rag_answer = self.rag_service.answer_question(
            state["question"], state["folder"], state["top_k"], entry_point="api",
            conversation_id=config["configurable"]["thread_id"], standalone_question=standalone_question,
        )
        return {
            "standalone_question": standalone_question, "answer": rag_answer.answer,
            "sources": [{"file_name": s.file_name, "folder": s.folder, "similarity": s.similarity}
                        for s in rag_answer.sources],
            "refusal_reason": rag_answer.refusal_reason, "messages": [HumanMessage(state["question"]), AIMessage(rag_answer.answer)],
        }

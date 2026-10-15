"""The memory graph, with LangGraph's in-memory checkpointer and a scripted rewriting LLM (no database, no network)."""
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from rag.llm.base import LlmResult, LlmUnavailableError
from rag.memory.condenser import QuestionCondenser
from rag.memory.conversation_graph import ConversationGraph
from rag.service.rag_service import RagService
from rag.service.prompt_builder import PromptBuilder
from tests.fakes import FakeChunkRepository, FakeEmbedder, FakeLedgerRepository, FakeLlmClient, chunk_row


class ScriptedLlm:
    """Gives the next scripted reply on each call; an Exception in the script is raised instead."""

    model_name = "scripted"

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append(user_prompt)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return LlmResult(text=step, model=self.model_name)

    def close(self):
        pass


def rewrite(question: str) -> str:
    return '{"standalone_question": "%s"}' % question


class Setup:
    def __init__(self, *script, answer="Nimal Perera knows Python [1]."):
        self.rewriter = ScriptedLlm(*script)
        self.answerer = FakeLlmClient(answer=answer)
        self.repository = FakeChunkRepository([chunk_row("nimal.pdf", "Nimal Perera, Python and Kubernetes")])
        self.ledger = FakeLedgerRepository()
        service = RagService(self.repository, FakeEmbedder(), PromptBuilder(), self.answerer, self.ledger)
        self.checkpointer = InMemorySaver()
        self.graph = ConversationGraph(service, QuestionCondenser(self.rewriter), self.checkpointer)

    def ask(self, question, conversation_id=None):
        return self.graph.ask(question, None, 5, conversation_id)


def test_the_first_question_is_searched_as_asked_and_never_rewritten():
    setup = Setup()

    result = setup.ask("Who has Kubernetes experience?")

    assert result.standalone_question == "Who has Kubernetes experience?"
    assert setup.rewriter.calls == [] and result.conversation_id
    assert setup.repository.search_calls and result.rag_answer.sources[0].file_name == "nimal.pdf"


def test_a_follow_up_is_rewritten_and_the_rewrite_is_what_gets_searched_and_answered():
    setup = Setup(rewrite("What is Nimal Perera Python experience?"))
    first = setup.ask("Who has Kubernetes experience?")

    second = setup.ask("What about his Python?", first.conversation_id)

    assert second.conversation_id == first.conversation_id
    assert second.standalone_question == "What is Nimal Perera Python experience?"
    assert "User: Who has Kubernetes experience?" in setup.rewriter.calls[0]
    assert "Assistant: Nimal Perera knows Python [1]." in setup.rewriter.calls[0]
    assert "What is Nimal Perera Python experience?" in setup.answerer.calls[1]["user"]


def test_the_ledger_records_the_conversation_and_the_rewrite_only_when_there_was_one():
    setup = Setup(rewrite("What is Nimal Perera Python experience?"))
    first = setup.ask("Who has Kubernetes experience?")
    setup.ask("What about his Python?", first.conversation_id)

    first_row, second_row = setup.ledger.requests.values()

    assert (first_row["conversation_id"], first_row["standalone_question"]) == (first.conversation_id, None)
    assert second_row["conversation_id"] == first.conversation_id
    assert second_row["question"] == "What about his Python?"
    assert second_row["standalone_question"] == "What is Nimal Perera Python experience?"


def test_conversations_do_not_see_each_other():
    setup = Setup()
    setup.ask("Who has Kubernetes experience?", "one")

    setup.ask("Who knows Go?", "two")

    assert setup.rewriter.calls == []  # "two" had no history, so nothing was rewritten


def test_a_rejected_rewrite_gets_one_repair_attempt():
    setup = Setup("Nimal Perera knows Python.", rewrite("What is Nimal Perera Python experience?"))
    first = setup.ask("Who has Kubernetes experience?")

    second = setup.ask("What about his Python?", first.conversation_id)

    assert second.standalone_question == "What is Nimal Perera Python experience?"
    assert len(setup.rewriter.calls) == 2 and "rejected" in setup.rewriter.calls[1]


def test_two_rejected_rewrites_fall_back_to_the_question_as_asked():
    setup = Setup("Nimal knows Python.", "Still not JSON.")
    first = setup.ask("Who has Kubernetes experience?")

    second = setup.ask("What about his Python?", first.conversation_id)

    assert second.standalone_question == "What about his Python?"
    assert len(setup.rewriter.calls) == 2  # no third try
    assert list(setup.ledger.requests.values())[1]["standalone_question"] is None


def test_an_unavailable_rewriting_llm_falls_back_without_failing_the_question():
    setup = Setup(LlmUnavailableError("429"))
    first = setup.ask("Who has Kubernetes experience?")

    second = setup.ask("What about his Python?", first.conversation_id)

    assert second.standalone_question == "What about his Python?" and second.rag_answer.answer


def test_only_the_last_messages_reach_the_rewriter():
    setup = Setup(*[rewrite(f"question {number}") for number in ("two", "three", "four", "five")])
    first = setup.ask("question one")
    for number in ("two", "three", "four", "five"):
        setup.ask(f"question {number}", first.conversation_id)

    # the 5th question follows 8 messages; the rewriter gets the last 6, so "question one" is gone
    assert "question one" not in setup.rewriter.calls[-1] and "question two" in setup.rewriter.calls[-1]


def test_a_failed_turn_is_not_remembered():
    setup = Setup(rewrite("Who knows Go?"))
    first = setup.ask("Who has Kubernetes experience?")
    setup.answerer.error = LlmUnavailableError("down")
    with pytest.raises(LlmUnavailableError):
        setup.ask("Who knows Go?", first.conversation_id)
    setup.answerer.error = None
    setup.rewriter.script = [rewrite("Who knows React?")]

    setup.ask("Who knows React?", first.conversation_id)

    assert "Who knows Go?" not in setup.rewriter.calls[-1]

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


# --- names, clarification and follow-ups after a match ---------------------------------------------------------

from rag.memory.follow_up import FollowUpRouter  # noqa: E402
from rag.service.candidate_names import NameResolver  # noqa: E402
from tests.unit.test_candidate_names import PEOPLE, FakeDirectory  # noqa: E402

MATCH = {"requirements": [{"skill": "Spark", "must_have": True}],
         "candidates": [{"candidate_id": 17, "score": 0.9,
                         "verdicts": [{"requirement": "Spark", "verdict": "met", "evidence": "Spark pipelines"}]},
                        {"candidate_id": 23, "score": 0.4,
                         "verdicts": [{"requirement": "Spark", "verdict": "not_found", "evidence": ""}]}]}


class NamedSetup(Setup):
    def __init__(self, *script, router_script=()):
        super().__init__(*script)
        self.router_llm = ScriptedLlm(*router_script)
        service = self.graph.rag_service
        self.graph = ConversationGraph(service, QuestionCondenser(self.rewriter), self.checkpointer,
                                       resolver=NameResolver(FakeDirectory(PEOPLE)),
                                       router=FollowUpRouter(self.router_llm))


def test_a_named_question_is_searched_only_in_that_candidates_cv_and_the_name_never_reaches_the_llm():
    setup = NamedSetup()

    result = setup.ask("What is Leah Walsh's Spark experience?")

    assert setup.repository.search_calls[0]["candidate_ids"] == [23]
    assert result.standalone_question == "What is candidate 23's Spark experience?"
    assert "Leah" not in setup.answerer.calls[0]["user"] and "Walsh" not in setup.answerer.calls[0]["user"]
    state = setup.checkpointer.get_tuple(setup.graph.config(result.conversation_id)).checkpoint["channel_values"]
    assert "Leah" not in str(state["messages"])  # not even in the stored history


def test_an_ambiguous_name_gets_a_clarifying_question_without_any_llm_call():
    setup = NamedSetup()

    result = setup.ask("What does Ines Brooks know?")

    assert "candidate 17" in result.rag_answer.answer and "candidate 52" in result.rag_answer.answer
    assert setup.answerer.calls == [] and setup.repository.search_calls == []


def test_a_pronoun_follow_up_keeps_searching_the_same_candidate():
    setup = NamedSetup(rewrite("What is candidate 23's Java experience?"))
    first = setup.ask("What is Leah Walsh's Spark experience?")

    setup.ask("What about her Java?", first.conversation_id)

    assert setup.repository.search_calls[1]["candidate_ids"] == [23]


def test_after_a_match_a_follow_up_the_result_answers_skips_the_search():
    setup = NamedSetup(router_script=['{"route": "from_result"}', "Candidate 17 has Spark pipelines; 23 lacks Spark."])
    setup.graph.remember_match("m1", "Match this JD", MATCH, "Ranked 2 candidates: candidate 17, candidate 23.")

    result = setup.ask("Why is candidate 17 above candidate 23?", "m1")

    assert result.rag_answer.answer.startswith("Candidate 17 has Spark")
    assert setup.repository.search_calls == [] and setup.rewriter.calls == []
    assert '"candidate": 17' in setup.router_llm.calls[0]  # the router saw the ranking, by id


def test_after_a_match_a_cv_detail_question_is_searched_in_that_candidates_cv():
    setup = NamedSetup(rewrite("Does candidate 17 know Terraform?"), router_script=['{"route": "search_candidate"}'])
    setup.graph.remember_match("m1", "Match this JD", MATCH, "Ranked 2 candidates: candidate 17, candidate 23.")

    setup.ask("Does candidate 17 know Terraform?", "m1")

    assert setup.repository.search_calls[0]["candidate_ids"] == [17]


def test_an_unusable_route_falls_back_to_searching():
    setup = NamedSetup(rewrite("Does candidate 17 know Terraform?"), router_script=["no idea"])
    setup.graph.remember_match("m1", "Match this JD", MATCH, "Ranked 2 candidates: candidate 17, candidate 23.")

    setup.ask("Does candidate 17 know Terraform?", "m1")

    assert len(setup.repository.search_calls) == 1

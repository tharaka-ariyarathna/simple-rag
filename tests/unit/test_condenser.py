"""The guards around the LLM that rewrites a follow-up into a standalone question."""
import pytest
from rag.constants import CONDENSED_QUESTION_MAX_LENGTH
from rag.memory.condenser import QuestionCondenser, check_standalone_question
from tests.fakes import FakeLlmClient

HISTORY = [("User", "Who has Kubernetes experience?"),
           ("Assistant", "Nimal Perera has 3 years of Kubernetes experience [1].")]


def reply(question: str) -> str:
    return '{"standalone_question": "%s"}' % question


def check(llm_reply: str, question: str = "What about his Python?"):
    return check_standalone_question(llm_reply, question, HISTORY)


def test_a_rewrite_that_resolves_the_reference_passes():
    assert check(reply("What is Nimal Perera Python experience?")).question == "What is Nimal Perera Python experience?"


def test_json_inside_a_code_fence_and_a_sentence_is_accepted():
    wrapped = "Sure!\n```json\n" + reply("What about Nimal Perera Python?") + "\n```"

    assert check(wrapped).question == "What about Nimal Perera Python?"


def test_an_unchanged_standalone_question_passes():
    assert check(reply("Who knows React?"), question="Who knows React?").question == "Who knows React?"


@pytest.mark.parametrize("bad_reply", ["Nimal Perera knows Python well.", "{}", '{"standalone_question": ""}',
                                       '{"standalone_question": 7}', "{not json}"])
def test_a_reply_that_is_not_the_expected_json_is_rejected(bad_reply):
    verdict = check(bad_reply)

    assert verdict.question is None and "JSON" in verdict.reason


def test_a_long_reply_is_rejected_as_an_answer():
    verdict = check(reply("Nimal " * CONDENSED_QUESTION_MAX_LENGTH))

    assert verdict.question is None and "longer than" in verdict.reason


@pytest.mark.parametrize("answer_like", ["Nimal Perera knows Python [1]?", "I couldn\'t find that in the documents."])
def test_citations_and_refusals_are_rejected(answer_like):
    assert check(reply(answer_like)).question is None


def test_words_that_are_nowhere_in_the_conversation_are_rejected():
    verdict = check(reply("What is Nimal Perera Python experience at Google?"))

    assert verdict.question is None and "google" in verdict.reason


def test_the_rewriter_sees_the_history_and_a_repair_attempt_names_the_rejection():
    llm = FakeLlmClient(answer=reply("Who knows React?"))
    condenser = QuestionCondenser(llm)

    condenser.rewrite(HISTORY, "Who knows React?")
    condenser.rewrite(HISTORY, "Who knows React?", rejection_reason="it was too long")

    first, repair = llm.calls
    assert "User: Who has Kubernetes experience?" in first["user"] and "Latest message: Who knows React?" in first["user"]
    assert "rejected" not in first["user"] and "it was too long" in repair["user"]
    assert first["purpose"] == "condense"

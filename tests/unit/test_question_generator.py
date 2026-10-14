import json
import pytest
from rag.evaluation.question_generator import QuestionGenerationError, QuestionGenerator
from tests.fakes import FakeLlmClient

CHUNK = "Tharaka built the billing service with AWS Lambda and SQS.\nIt processed 2 million events a day."


def reply_with(*questions: dict) -> str:
    return json.dumps({"questions": list(questions)})


def question(text="Which AWS services ran billing?", answer="Lambda and SQS",
             evidence="built the billing service with AWS Lambda and SQS"):
    return {"question": text, "expected_answer": answer, "evidence": evidence}


def test_generate_asks_the_llm_under_the_question_generation_purpose():
    llm = FakeLlmClient(answer=reply_with(question()))

    generated = QuestionGenerator(llm).generate(f"  {CHUNK}  ")

    assert generated.questions == [question()]
    assert generated.model == "fake-llm"  # LlmResult.model: what actually answered
    assert llm.calls[0]["purpose"] == "question_generation"
    assert llm.calls[0]["user"] == f"Passage:\n{CHUNK}"


def test_json_wrapped_in_code_fences_or_text_is_accepted():
    reply = f"Here you go:\n```json\n{reply_with(question())}\n```"

    assert QuestionGenerator(FakeLlmClient()).parse_questions(reply, CHUNK) == [question()]


def test_evidence_matches_despite_line_breaks_and_case():
    candidate = question(evidence="AWS LAMBDA and SQS.   It processed")

    assert QuestionGenerator(FakeLlmClient()).parse_questions(reply_with(candidate), CHUNK) == [candidate]


def test_questions_with_missing_fields_or_invented_evidence_are_dropped():
    reply = reply_with(
        question(),
        {"question": "No answer field?", "evidence": "AWS Lambda"},
        question(answer="   "),
        question(evidence="It used Kafka"),
        "not even a dict",
    )

    assert QuestionGenerator(FakeLlmClient()).parse_questions(reply, CHUNK) == [question()]


def test_fields_are_stripped():
    padded = {"question": " Q? ", "expected_answer": " A ", "evidence": " AWS Lambda "}

    assert QuestionGenerator(FakeLlmClient()).parse_questions(reply_with(padded), CHUNK) == [
        {"question": "Q?", "expected_answer": "A", "evidence": "AWS Lambda"}]


def test_empty_list_means_nothing_worth_asking():
    assert QuestionGenerator(FakeLlmClient()).parse_questions('{"questions": []}', CHUNK) == []


@pytest.mark.parametrize("reply", ["I cannot help with that.", '{"items": []}', '{"questions": "none"}', "[1, 2]"])
def test_unusable_reply_raises(reply):
    with pytest.raises(QuestionGenerationError):
        QuestionGenerator(FakeLlmClient()).parse_questions(reply, CHUNK)

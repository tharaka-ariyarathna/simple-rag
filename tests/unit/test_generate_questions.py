"""The question generation loop, with the generator and the repository replaced by stubs."""
from rag.evaluation.generate_questions import process_chunks
from rag.evaluation.question_generator import GeneratedQuestions, QuestionGenerationError
from rag.llm.base import LlmConfigurationError, LlmUnavailableError

QUESTION = {"question": "Q?", "expected_answer": "A", "evidence": "E"}


class StubGenerator:
    """Returns (or raises) one scripted outcome per chunk, in order."""

    def __init__(self, outcomes: list):
        self.outcomes = list(outcomes)
        self.chunks_seen = []

    def generate(self, chunk_content: str) -> GeneratedQuestions:
        self.chunks_seen.append(chunk_content)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RecordingQuestionRepository:
    def __init__(self):
        self.saved = []

    def save_questions(self, source, chunk_index, questions, generator_model):
        self.saved.append((source, chunk_index, questions, generator_model))


def chunks(count: int) -> list[dict]:
    return [{"source": f"C:/cvs/cv_{number}.pdf", "folder": "cvs", "chunk_index": 0, "content": f"chunk {number}"}
            for number in range(1, count + 1)]


def test_questions_are_saved_with_the_model_that_wrote_them():
    repository = RecordingQuestionRepository()
    generator = StubGenerator([GeneratedQuestions([QUESTION], "llama3.2:3b"), GeneratedQuestions([], "llama3.2:3b")])

    summary = process_chunks(chunks(2), generator, repository, delay_seconds=0)

    assert repository.saved == [("C:/cvs/cv_1.pdf", 0, [QUESTION], "llama3.2:3b"),
                                ("C:/cvs/cv_2.pdf", 0, [], "llama3.2:3b")]  # empty = processed, not retried
    assert (summary.saved_questions, summary.skipped_chunks, summary.stopped_by) == (1, 1, None)


def test_unusable_reply_skips_only_that_chunk():
    repository = RecordingQuestionRepository()
    generator = StubGenerator([QuestionGenerationError("not JSON"), GeneratedQuestions([QUESTION], "m")])

    summary = process_chunks(chunks(2), generator, repository, delay_seconds=0)

    assert [saved[0] for saved in repository.saved] == ["C:/cvs/cv_2.pdf"]  # cv_1 stays pending for next run
    assert summary.unusable_replies == 1 and summary.stopped_by is None


def test_llm_failure_after_retries_stops_the_run(caplog):
    repository = RecordingQuestionRepository()
    generator = StubGenerator([GeneratedQuestions([QUESTION], "m"), LlmUnavailableError("Ollama is not running")])

    summary = process_chunks(chunks(4), generator, repository, delay_seconds=0)

    assert generator.chunks_seen == ["chunk 1", "chunk 2"]  # chunks 3 and 4 were never sent
    assert len(repository.saved) == 1
    assert summary.stopped_by == "Ollama is not running"
    assert "Stopping at [2/4]" in caplog.text and "Ollama is not running" in caplog.text


def test_configuration_error_stops_the_run_too():
    summary = process_chunks(chunks(3), StubGenerator([LlmConfigurationError("404 model not pulled")]),
                             RecordingQuestionRepository(), delay_seconds=0)

    assert summary.stopped_by == "404 model not pulled"


def test_delay_between_chunks_only_when_configured():
    waits = []
    generator = StubGenerator([GeneratedQuestions([], "m"), GeneratedQuestions([], "m")])

    process_chunks(chunks(2), generator, RecordingQuestionRepository(), delay_seconds=1.5, sleep=waits.append)
    process_chunks([], StubGenerator([]), RecordingQuestionRepository(), delay_seconds=0, sleep=waits.append)

    assert waits == [1.5, 1.5]

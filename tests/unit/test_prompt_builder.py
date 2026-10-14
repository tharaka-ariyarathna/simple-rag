from rag.constants import NO_RELEVANT_CONTEXT_ANSWER
from rag.service.prompt_builder import PromptBuilder
from tests.fakes import chunk_row


def test_chunks_are_numbered_in_order_so_citations_match():
    chunks = [chunk_row("C:/docs/cvs/alice.pdf", "  Alice knows Python.  ", folder="cvs"),
              chunk_row("C:/docs/cvs/bob.pdf", "Bob knows Go.", folder="cvs")]

    prompt = PromptBuilder().build("Who knows Go?", chunks)

    assert "[1] (file: alice.pdf, folder: cvs)\nAlice knows Python." in prompt.user
    assert "[2] (file: bob.pdf, folder: cvs)\nBob knows Go." in prompt.user
    assert prompt.user.index("[1]") < prompt.user.index("[2]")


def test_only_the_file_name_is_shown_never_the_local_path():
    prompt = PromptBuilder().build("q", [chunk_row("C:/Users/someone/private/cv.pdf")])

    assert "C:/Users" not in prompt.user
    assert "file: cv.pdf" in prompt.user


def test_question_comes_last():
    prompt = PromptBuilder().build("What is E-417?", [chunk_row("a.pdf")])

    assert prompt.user.startswith("Context:\n")
    assert prompt.user.endswith("Question: What is E-417?")


def test_default_system_prompt_tells_the_model_the_exact_refusal_text():
    prompt = PromptBuilder().build("q", [chunk_row("a.pdf")])

    assert NO_RELEVANT_CONTEXT_ANSWER in prompt.system


def test_system_prompt_can_be_injected():
    assert PromptBuilder(system_prompt="custom rules").build("q", []).system == "custom rules"

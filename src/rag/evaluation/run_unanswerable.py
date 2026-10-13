"""Checks that the system says "I don't know" when the documents do not contain the answer,
and does not say it when they do. Calls the LLM, so it is slower and uses quota.

Run from src:  python -m rag.evaluation.run_unanswerable [--answerable-sample 20]
"""
import argparse
from rag.bootstrap import build_rag_components
from rag.constants import EVAL_ANSWERABLE_SAMPLE, RETRIEVAL_TOP_K
from rag.evaluation.run_settings import collect_settings, current_git_commit
from rag.evaluation.unanswerable_eval import (
    CATEGORY_ANSWERABLE, OUTCOME_ANSWERED, OUTCOME_REFUSED, ask, compute_metrics, judge,
    load_unanswerable_questions, sample_answerable_questions,
)
from rag.repository.eval_repository import EvalRepository
from rag.repository.question_repository import QuestionRepository

KIND = "unanswerable"


def print_scorecard(metrics: dict, passed: bool, problems: list[str]) -> None:
    print("\n=== Refusal scorecard ===")
    for category, stats in metrics["categories"].items():
        print(f"  {category:<14} refused {stats['refused']}/{stats['total']} ({stats['refusal_rate']:.0%})"
              f"  [no chunk found: {stats['refused_by_retrieval']}, LLM refused: {stats['refused_by_llm']}]")
    if "hallucination_rate" in metrics:
        print(f"  Hallucination rate (answered but should refuse): {metrics['hallucination_rate']:.0%}")
    if "false_refusal_rate" in metrics:
        print(f"  False refusal rate (refused but could answer):   {metrics['false_refusal_rate']:.0%}")
    if metrics["errors"]:
        print(f"  Questions skipped because the LLM failed: {metrics['errors']}")
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    for problem in problems:
        print(f"  - {problem}")


def print_mistakes(results: list[dict]) -> None:
    hallucinations = [r for r in results if r["category"] != CATEGORY_ANSWERABLE and r["outcome"] == OUTCOME_ANSWERED]
    false_refusals = [r for r in results if r["category"] == CATEGORY_ANSWERABLE and r["outcome"] == OUTCOME_REFUSED]
    if hallucinations:
        print("\nAnswered, but the documents do not contain the answer:")
        for result in hallucinations:
            print(f"  [{result['category']}] {result['question']}\n      -> {result['details']['answer'][:200]!r}")
    if false_refusals:
        print("\nRefused, but the documents contain the answer:")
        for result in false_refusals:
            print(f"  [{result['details']['refusal_reason']}] {result['question']}")


def run_unanswerable_eval(answerable_sample: int, top_k: int) -> bool:
    components = build_rag_components()
    try:
        eval_repository = EvalRepository(components.database)
        cases = load_unanswerable_questions()
        if answerable_sample:
            synthetic_questions = QuestionRepository(components.database).list_questions()
            if not synthetic_questions:
                print("No synthetic questions yet, so false refusals are not measured this run.")
            cases += sample_answerable_questions(synthetic_questions, answerable_sample)

        settings = collect_settings(components, top_k) | {"answerable_sample": answerable_sample}
        run_id = eval_repository.start_run(KIND, settings, current_git_commit())
        results = []
        for position, case in enumerate(cases, start=1):
            results.append(ask(components.rag_service, case, top_k, run_id))
            print(f"  {position}/{len(cases)} {results[-1]['outcome']:<9} [{case['category']}] {case['question'][:70]}")

        metrics = compute_metrics(results)
        passed, problems = judge(metrics)
        eval_repository.save_results(run_id, results)
        eval_repository.finish_run(run_id, metrics, passed)

        print_scorecard(metrics, passed, problems)
        print_mistakes(results)
        print(f"\nSaved as eval run {run_id}. Each question is in rag_requests with eval_run_id = {run_id}.")
        return passed
    finally:
        components.close()


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argument_parser.add_argument("--answerable-sample", type=int, default=EVAL_ANSWERABLE_SAMPLE,
                                 help="synthetic questions to run too, to measure false refusals (0 = skip)")
    argument_parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    arguments = argument_parser.parse_args()
    raise SystemExit(0 if run_unanswerable_eval(arguments.answerable_sample, arguments.top_k) else 1)

"""Scores retrieval: for each synthetic question, is the chunk it was written from found, and at which rank?
No LLM is called here, so it is fast and free: run it after every change to chunking, top_k, threshold or model.

Run from src:  python -m rag.evaluation.run_retrieval [--top-k 5]
"""
import argparse
from rag.bootstrap import build_rag_components
from rag.constants import ACTIVE_EMBEDDING_MODEL, EVAL_WORST_MISSES_SHOWN, RETRIEVAL_TOP_K, TOP_K_MAX
from rag.evaluation.retrieval_eval import (
    OUTCOME_HIT, OUTCOME_UNMATCHABLE, RetrievalCase, compute_metrics, evaluate_case, judge,
)
from rag.evaluation.run_settings import collect_settings, current_git_commit
from rag.repository.chunk_repository import ChunkRepository
from rag.repository.eval_repository import EvalRepository
from rag.repository.question_repository import QuestionRepository

KIND = "retrieval"


def print_scorecard(metrics: dict, previous_metrics: dict | None, passed: bool, problems: list[str]) -> None:
    print("\n=== Retrieval scorecard ===")
    print(f"Questions evaluated: {metrics['evaluated']}  (unmatchable, skipped: {metrics['unmatchable']})")
    for name, label in (("hit_at_1", "hit@1"), ("hit_at_k", "hit@k"), ("file_hit_at_k", "file hit@k"), ("mrr", "MRR")):
        change = ""
        if previous_metrics and name in previous_metrics:
            change = f"  (previous {previous_metrics[name]:.3f}, {metrics[name] - previous_metrics[name]:+.3f})"
        print(f"  {label:<11} {metrics[name]:.3f}{change}")
    print(f"  Misses caused by the similarity threshold: {metrics['below_threshold_misses']}")
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    for problem in problems:
        print(f"  - {problem}")


def print_worst_misses(results: list[dict]) -> None:
    misses = [result for result in results if result["outcome"] not in (OUTCOME_HIT, OUTCOME_UNMATCHABLE)]
    if not misses:
        return
    print(f"\nMisses (first {EVAL_WORST_MISSES_SHOWN} of {len(misses)}):")
    for result in misses[:EVAL_WORST_MISSES_SHOWN]:
        extra = ""
        if result["outcome"] == "below_threshold":
            extra = f" (similarity {result['similarity']:.3f}, rank {result['details']['rank_without_threshold']} without threshold)"
        print(f"  [{result['outcome']}]{extra} {result['question']}")


def run_retrieval_eval(top_k: int) -> bool:
    components = build_rag_components()
    try:
        questions = QuestionRepository(components.database).list_questions()
        if not questions:
            print("No synthetic questions yet. Run: python -m rag.evaluation.generate_questions")
            return False
        eval_repository = EvalRepository(components.database)
        all_chunks = ChunkRepository(components.database).list_chunks(ACTIVE_EMBEDDING_MODEL)

        run_id = eval_repository.start_run(KIND, collect_settings(components, top_k), current_git_commit())
        previous_run = eval_repository.latest_finished_run(KIND, before_run_id=run_id)

        results = []
        for position, row in enumerate(questions, start=1):
            case = RetrievalCase(row["question"], row["source"], row["evidence"])
            results.append(evaluate_case(components.rag_service, case, all_chunks, top_k))
            if position % 25 == 0:
                print(f"  {position}/{len(questions)} questions done")

        metrics = compute_metrics(results)
        previous_metrics = previous_run["metrics"] if previous_run else None
        passed, problems = judge(metrics, previous_metrics)
        eval_repository.save_results(run_id, results)
        eval_repository.finish_run(run_id, metrics, passed)

        print_scorecard(metrics, previous_metrics, passed, problems)
        print_worst_misses(results)
        print(f"\nSaved as eval run {run_id}.")
        return passed
    finally:
        components.close()


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argument_parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K, choices=range(1, TOP_K_MAX + 1),
                                 metavar=f"1-{TOP_K_MAX}")
    raise SystemExit(0 if run_retrieval_eval(argument_parser.parse_args().top_k) else 1)

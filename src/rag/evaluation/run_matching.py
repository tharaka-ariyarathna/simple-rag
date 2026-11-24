"""Scores JD -> CV matching: for each job description, rank the CVs and compare with the labelled matches.
No LLM is called, so it is fast and free: run it after every change to chunking, the model or the search.

Needs EVAL_DATASET_DIR in .env: a folder with jds/ (PDFs) and labels.json.
Run from src:  python -m rag.evaluation.run_matching [--chunks 50] [--cutoff 10]
               python -m rag.evaluation.run_matching --strategy match_graph --limit 5
The match_graph strategy runs /match's workflow (about 1 + shortlist-size LLM calls per JD, so --limit keeps it
within free-tier quotas) and scores the whole-JD baseline on the same JDs in the same run, for a fair comparison.
"""
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from rag.bootstrap import build_rag_components
from rag.constants import (
    EVAL_WORST_MISSES_SHOWN, MATCHING_CHUNKS_RETRIEVED, MATCHING_CUTOFF, MATCHING_MIN_SIMILARITY,
    MATCHING_STRATEGY_GRAPH, MATCHING_STRATEGY_WHOLE_JD, MATCH_SHORTLIST_SIZE,
)
from rag.evaluation.matching_dataset import load_matching_dataset
from rag.evaluation.matching_eval import (
    METRIC_NAMES, candidate_id_of, changed_settings, compute_metrics, evaluate_jd, evaluate_match, is_comparable,
    judge,
)
from rag.evaluation.run_settings import collect_settings, current_git_commit
from rag.repository.eval_repository import EvalRepository

KIND = "matching"
METRIC_LABELS = {"ndcg_at_k": "nDCG@k", "recall_at_k": "recall@k", "precision_at_k": "precision@k", "mrr": "MRR"}


def print_scorecard(metrics: dict, previous_run: dict | None, settings: dict, passed: bool, problems: list[str]):
    previous_metrics = previous_run["metrics"] if previous_run else None
    print(f"\n=== JD -> CV matching scorecard (k = {metrics['cutoff']}) ===")
    print(f"JDs evaluated: {metrics['jd_count']}   CVs ranked per JD (average): {metrics['avg_candidates_ranked']:.1f}")
    for name in METRIC_NAMES:
        change = ""
        if previous_metrics:
            change = f"  (previous {previous_metrics[name]:.3f}, {metrics[name] - previous_metrics[name]:+.3f})"
        print(f"  {METRIC_LABELS[name]:<15} {metrics[name]:.3f}{change}")
    if previous_run:
        differences = changed_settings(previous_run["settings"], settings)
        print("Changed since the previous run:", differences if differences else "nothing (same settings)")
    else:
        print("No comparable previous run: this is the baseline.")
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    for problem in problems:
        print(f"  - {problem}")


def print_weakest_jds(results: list[dict], cutoff: int) -> None:
    weakest = sorted(results, key=lambda result: result["score"])[:EVAL_WORST_MISSES_SHOWN]
    print(f"\nWeakest JDs (lowest nDCG@{cutoff}):")
    for result in weakest:
        details = result["details"]
        top = ", ".join(f"{entry['cv']}[{entry['grade']}]" for entry in details["top"][:5])
        print(f"  {result['question']}: nDCG {result['score']:.2f}, recall {details['recall_at_k']:.2f}"
              f" ({details['relevant_count']} relevant). Top 5 (grade in brackets): {top}")


def evaluate_with_match_graph(components, dataset, cutoff: int) -> list[dict]:
    results = []
    for position, (jd_id, jd_text) in enumerate(dataset.jd_texts.items(), start=1):
        match_result = components.match_graph.match(jd_text, top_n=MATCH_SHORTLIST_SIZE)
        candidate_ids = set(match_result["whole_jd_order"]) | set(match_result["shortlist"])
        sources = components.candidate_repository.sources_by_id(list(candidate_ids))
        cv_id_by_candidate = {cid: candidate_id_of(source) for cid, source in sources.items()}
        results.append(evaluate_match(jd_id, match_result, cv_id_by_candidate, dataset.labels[jd_id], cutoff))
        print(f"  {position}/{len(dataset.jd_texts)} {jd_id}: nDCG {results[-1]['score']:.2f}")
    return results


def run_matching_eval(chunks_retrieved: int, cutoff: int, strategy: str = MATCHING_STRATEGY_WHOLE_JD,
                      limit: int | None = None) -> bool:
    load_dotenv()
    dataset = load_matching_dataset(Path(os.environ["EVAL_DATASET_DIR"]))
    if limit:
        dataset.jd_texts = dict(list(dataset.jd_texts.items())[:limit])
    if dataset.skipped_jd_ids:
        print(f"Skipped {len(dataset.skipped_jd_ids)} labelled JDs without a PDF: {', '.join(dataset.skipped_jd_ids)}")

    components = build_rag_components()
    try:
        eval_repository = EvalRepository(components.database)
        settings = collect_settings(components, chunks_retrieved) | {
            "strategy": strategy,
            "aggregation": "max_chunk_similarity" if strategy == MATCHING_STRATEGY_WHOLE_JD else "llm_judge_score",
            **({"shortlist_size": MATCH_SHORTLIST_SIZE} if strategy == MATCHING_STRATEGY_GRAPH else {}),
            "min_similarity": MATCHING_MIN_SIMILARITY,
            "chunks_retrieved": chunks_retrieved,
            "cutoff": cutoff,
            "jd_count": len(dataset.jd_texts),
        }
        run_id = eval_repository.start_run(KIND, settings, current_git_commit())
        previous_run = eval_repository.latest_finished_run(KIND, before_run_id=run_id)
        if previous_run and not is_comparable(previous_run["settings"], settings):
            print("The previous run used a different k, chunk count or JD set, so it is not compared.")
            previous_run = None

        if strategy == MATCHING_STRATEGY_GRAPH:
            results = evaluate_with_match_graph(components, dataset, cutoff)
            baseline = compute_metrics([
                evaluate_jd(components.rag_service, jd_id, jd_text, dataset.labels[jd_id], chunks_retrieved, cutoff)
                for jd_id, jd_text in dataset.jd_texts.items()], cutoff)
        else:
            results = [
                evaluate_jd(components.rag_service, jd_id, jd_text, dataset.labels[jd_id], chunks_retrieved, cutoff)
                for jd_id, jd_text in dataset.jd_texts.items()
            ]
            baseline = None
        metrics = compute_metrics(results, cutoff)
        passed, problems = judge(metrics, previous_run["metrics"] if previous_run else None)
        eval_repository.save_results(run_id, results)
        eval_repository.finish_run(run_id, metrics, passed)

        print_scorecard(metrics, previous_run, settings, passed, problems)
        if baseline:
            print("Whole-JD baseline on the same JDs: " + ", ".join(
                f"{METRIC_LABELS[name]} {baseline[name]:.3f}" for name in METRIC_NAMES))
            print(f"Shortlist recall (relevant CVs that reached the judge): {metrics['shortlist_recall']:.3f}")
        print_weakest_jds(results, cutoff)
        print(f"\nSaved as eval run {run_id}.")
        return passed
    finally:
        components.close()


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argument_parser.add_argument("--chunks", type=int, default=MATCHING_CHUNKS_RETRIEVED,
                                 help="chunks retrieved per JD before grouping by CV")
    argument_parser.add_argument("--cutoff", type=int, default=MATCHING_CUTOFF, help="k for the metrics")
    argument_parser.add_argument("--strategy", choices=[MATCHING_STRATEGY_WHOLE_JD, MATCHING_STRATEGY_GRAPH],
                                 default=MATCHING_STRATEGY_WHOLE_JD)
    argument_parser.add_argument("--limit", type=int, default=None, help="only the first N JDs (LLM quota)")
    arguments = argument_parser.parse_args()
    raise SystemExit(0 if run_matching_eval(arguments.chunks, arguments.cutoff, arguments.strategy,
                                            arguments.limit) else 1)

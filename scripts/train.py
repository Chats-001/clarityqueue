"""Train ClarityQueue and generate every reported benchmark artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from clarityqueue.config import PRIORITIES, RANDOM_SEED, ROUTES, settings
from clarityqueue.evaluation import classification_summary, retrieval_summary
from clarityqueue.model import TriageModel, combine_text
from clarityqueue.persistence import content_hash, save_bundle, write_json
from clarityqueue.retrieval import EvidenceIndex
from scripts.generate_data import generate_tickets


def train(
    knowledge_path: Path,
    ticket_path: Path,
    artifact_dir: Path,
    answer_threshold: float,
) -> dict:
    if not ticket_path.exists():
        ticket_path.parent.mkdir(parents=True, exist_ok=True)
        generate_tickets().to_csv(ticket_path, index=False)
    tickets = pd.read_csv(ticket_path)
    articles = pd.read_csv(knowledge_path)
    # Hold out one complete issue family per route. A row-random split would leak near-identical
    # generated templates into both sets and overstate generalization.
    held_out_articles = {f"KB-{prefix}-03" for prefix in ["ACCESS", "BILL", "SYNC", "INT", "PERF"]}
    test_rows = tickets[tickets["article_id"].isin(held_out_articles)].copy()
    train_rows = tickets[~tickets["article_id"].isin(held_out_articles)].copy()
    train_texts = [
        combine_text(row.title, row.description) for row in train_rows.itertuples(index=False)
    ]
    test_texts = [
        combine_text(row.title, row.description) for row in test_rows.itertuples(index=False)
    ]
    model = TriageModel().fit(train_texts, train_rows["route"], train_rows["priority"])
    index = EvidenceIndex().fit(articles)
    routes, route_confidence, priorities, priority_confidence = model.predict_many(test_texts)
    route_metrics = classification_summary(test_rows["route"], routes, list(ROUTES))
    priority_metrics = classification_summary(test_rows["priority"], priorities, list(PRIORITIES))
    retrieval = retrieval_summary(index, test_rows, answer_threshold)
    out_of_domain_queries = [
        "What will the weather be tomorrow?",
        "Please approve my employee vacation request",
        "How should I calculate quarterly payroll tax?",
        "Book a hotel near the airport",
        "Our office coffee machine is leaking",
        "Write a marketing slogan for a new shoe",
        "What stocks should I buy this week?",
        "Translate this contract into French",
    ]
    out_of_domain_answers = [
        index.answer(query, answer_threshold) for query in out_of_domain_queries
    ]
    out_of_domain_abstention = sum(
        not answer["answered"] for answer in out_of_domain_answers
    ) / len(out_of_domain_answers)
    details = pd.DataFrame(retrieval.pop("details"))
    predictions = test_rows.reset_index(drop=True).copy()
    predictions["predicted_route"] = routes
    predictions["route_confidence"] = route_confidence
    predictions["predicted_priority"] = priorities
    predictions["priority_confidence"] = priority_confidence
    predictions = predictions.merge(details, on="ticket_id", validate="one_to_one")
    predictions["route_correct"] = predictions["route"] == predictions["predicted_route"]
    predictions["automation_candidate"] = (predictions["route_confidence"] >= 0.45) & predictions[
        "answered"
    ]
    candidates = predictions[predictions["automation_candidate"]]
    end_to_end_correct = candidates["route_correct"] & candidates["correct"]
    metrics = {
        "dataset": {
            "generated_tickets": int(len(tickets)),
            "knowledge_articles": int(len(articles)),
            "held_out_tickets": int(len(test_rows)),
            "held_out_issue_families": int(len(held_out_articles)),
            "random_seed": RANDOM_SEED,
        },
        "route_model": route_metrics,
        "priority_model": priority_metrics,
        "retrieval": retrieval,
        "operational": {
            "automation_candidate_rate": float(predictions["automation_candidate"].mean()),
            "end_to_end_accuracy_when_automated": float(end_to_end_correct.mean())
            if len(candidates)
            else 0.0,
            "abstention_rate": float(1 - retrieval["answer_coverage"]),
            "out_of_domain_abstention_rate": float(out_of_domain_abstention),
            "critical_priority_recall": priority_metrics["per_class"]["critical"]["recall"],
        },
        "evaluation_note": (
            "All metrics use a seeded held-out split of an original synthetic benchmark. "
            "They demonstrate methodology, not production performance."
        ),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(artifact_dir / "test_predictions.csv", index=False)
    pd.DataFrame(route_metrics["confusion_matrix"], index=ROUTES, columns=ROUTES).to_csv(
        artifact_dir / "route_confusion.csv"
    )
    per_route = pd.DataFrame(route_metrics["per_class"]).T.reset_index(names="route")
    per_route.to_csv(artifact_dir / "route_quality.csv", index=False)
    source_hash = content_hash(knowledge_path, ticket_path)
    metadata = save_bundle(model, index, metrics, artifact_dir, source_hash)
    write_json(
        artifact_dir / "run.json",
        {"model_version": metadata["model_version"], "metrics": metrics},
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge", type=Path, default=Path("data/source/knowledge_base.csv"))
    parser.add_argument("--tickets", type=Path, default=Path("data/generated/tickets.csv"))
    parser.add_argument("--artifacts", type=Path, default=settings.artifact_dir)
    parser.add_argument("--answer-threshold", type=float, default=settings.answer_threshold)
    args = parser.parse_args()
    train(args.knowledge, args.tickets, args.artifacts, args.answer_threshold)


if __name__ == "__main__":
    main()

r"""Measure retrieval alone: evidence recall at several depths, no answers.

Builds the retriever named in a config the way ``load_service`` does,
retrieves the top 20 paragraphs for every question within its own paper,
and only then reads gold. Recall@k is the best recall over references at
each depth, the definition ``tools/score.py`` uses at depth 5. For reading
failures, each question also gets its top-20 ranking and the paragraphs of
its best-matching reference (chosen at recall@5) that the top 5 missed,
with their rank and section. Dense retrievers only call the API to embed
the questions.

Example::

    python tools/retrieval_eval.py --config config/composed-dense.json \
      --index-dir .index-full --questions data/eval-120/dev_questions.jsonl \
      --gold data/eval-120/dev_gold.jsonl \
      --output experiments/runs/002-dense.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qasper_rag.config import load_class, load_config
from qasper_rag.pipeline.retrieval import EmbeddingRetriever
from qasper_rag.storage.jsonl import read_jsonl

Record = dict[str, Any]
DEPTHS = (5, 10, 20)
ANSWER_DEPTH = 5  # the most paragraphs the output contract allows


def setup_args() -> argparse.ArgumentParser:
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    return parser


def retrieve_all(
    config_path: str | Path, index_dir: str | Path, questions: list[Record]
) -> dict[str, list[str]]:
    """Return each question's top-20 paragraph IDs, best first.

    Each question searches only its own ``paper_id``, as in evaluation runs.
    """
    config = load_config(config_path)
    paragraphs = read_jsonl(Path(index_dir) / "paragraphs.jsonl")
    retriever_class = load_class(config.retriever, EmbeddingRetriever)
    retriever = retriever_class(paragraphs, config.retriever, index_dir)
    return {
        q["question_id"]: [
            p["paragraph_id"]
            for p in retriever.retrieve(
                {"question": q["question"], "paper_ids": [q["paper_id"]]},
                DEPTHS[-1],
            )
        ]
        for q in questions
    }


def _recall(ranked: list[str], evidence: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(evidence)) / len(evidence)


def recall_at(
    ranked: list[str], evidence_sets: list[list[str]], k: int
) -> float:
    """Return the best recall of the top ``k`` over the references."""
    return max(_recall(ranked, evidence, k) for evidence in evidence_sets)


def best_reference(ranked: list[str], evidence_sets: list[list[str]]) -> int:
    """Return the index of the reference that best matches the ranking.

    Highest recall@5 wins; ties go to higher recall@20, then to the earliest
    reference.
    """
    return max(
        range(len(evidence_sets)),
        key=lambda i: (
            _recall(ranked, evidence_sets[i], ANSWER_DEPTH),
            _recall(ranked, evidence_sets[i], DEPTHS[-1]),
            -i,
        ),
    )


def missed_paragraphs(
    ranked: list[str], evidence: list[str], paragraphs: dict[str, Record]
) -> list[Record]:
    """Return the evidence missing from the top 5, with rank and section.

    ``rank`` is the 1-based position in ``ranked``, or ``None`` when the
    paragraph is not in it at all.
    """
    top = set(ranked[:ANSWER_DEPTH])
    return [
        {
            "paragraph_id": pid,
            "rank": ranked.index(pid) + 1 if pid in ranked else None,
            "section": paragraphs[pid]["section"],
        }
        for pid in evidence
        if pid not in top
    ]


def evaluate(
    rankings: dict[str, list[str]],
    gold: list[Record],
    paragraphs: dict[str, Record],
) -> dict[str, Any]:
    """Score rankings against gold for the answerable questions.

    Parameters
    ----------
    rankings : dict[str, list[str]]
        Retrieved paragraph IDs per question ID, best first.
    gold : list[dict]
        Gold records with ``question_id`` and ``references``.
    paragraphs : dict[str, dict]
        Indexed paragraphs by ID, for the sections of missed paragraphs.

    Returns
    -------
    dict
        ``answerable_questions``, mean ``recall_at_{5,10,20}``, and per
        question its ``recall_at_5``, ``best_reference``, ``missed`` and
        ``ranking``.
    """
    totals = dict.fromkeys(DEPTHS, 0.0)
    rows = []
    for record in gold:
        evidence_sets = [
            ref["evidence_paragraph_ids"]
            for ref in record["references"]
            if not ref["abstain"]
        ]
        if not evidence_sets:
            continue
        ranked = rankings[record["question_id"]]
        for k in DEPTHS:
            totals[k] += recall_at(ranked, evidence_sets, k)
        best = best_reference(ranked, evidence_sets)
        rows.append(
            {
                "question_id": record["question_id"],
                "recall_at_5": recall_at(ranked, evidence_sets, ANSWER_DEPTH),
                "best_reference": best,
                "missed": missed_paragraphs(
                    ranked, evidence_sets[best], paragraphs
                ),
                "ranking": ranked,
            }
        )
    count = len(rows)
    means = {
        f"recall_at_{k}": totals[k] / count if count else 0.0 for k in DEPTHS
    }
    return {"answerable_questions": count, **means, "questions": rows}


def main(args: argparse.Namespace) -> None:
    """Retrieve for every question, then score against gold."""
    questions = read_jsonl(args.questions)
    rankings = retrieve_all(args.config, args.index_dir, questions)
    # Gold is read only after every question has been retrieved.
    gold = read_jsonl(args.gold)
    paragraphs = {
        p["paragraph_id"]: p
        for p in read_jsonl(Path(args.index_dir) / "paragraphs.jsonl")
    }
    paper_of = {q["question_id"]: q["paper_id"] for q in questions}
    result = {"config": args.config, **evaluate(rankings, gold, paragraphs)}
    for row in result["questions"]:
        row["paper_id"] = paper_of[row["question_id"]]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k != "questions"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(setup_args().parse_args())

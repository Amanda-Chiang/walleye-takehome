"""Score a predictions file against the gold labels.

Every prediction must first pass the output rules in
``qasper_rag.validation.contracts``. The scorer does not depend on the
model provider or on how the predictions were made.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qasper_rag.storage.jsonl import read_jsonl, unique_by
from qasper_rag.validation.contracts import validate_prediction

Record = dict[str, Any]
MAX_EVIDENCE = 5


def tokens(text: str) -> list[str]:
    """Lowercase, drop punctuation and articles, and split on spaces."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def token_f1(prediction: str, reference: str) -> float:
    """Return the F1 of the shared tokens of two answers."""
    predicted, gold = tokens(prediction), tokens(reference)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    return 2 * overlap / (len(predicted) + len(gold)) if overlap else 0.0


def set_f1(predicted: Any, gold: Any) -> float:
    """Return the F1 of two ID sets. Two empty sets score 1."""
    predicted, gold = set(predicted), set(gold)
    if not predicted and not gold:
        return 1.0
    return 2 * len(predicted & gold) / (len(predicted) + len(gold))


@dataclass
class Tally:
    """Running totals for one scoring pass."""

    answers: list[float] = field(default_factory=list)
    recalls: list[float] = field(default_factory=list)
    citations: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    missing: int = 0
    covered: int = 0
    answerable: int = 0
    true_abstain: int = 0
    false_abstain: int = 0
    missed_abstain: int = 0


def check_references(
    qid: str, question: Record, refs: list[Record], paragraphs: dict
) -> bool:
    """Check the gold references and return whether they are abstentions."""
    if not refs or len({ref["abstain"] for ref in refs}) != 1:
        raise ValueError(f"{qid}: references must agree on answerability")
    abstain = refs[0]["abstain"]
    for ref in refs:
        evidence = ref["evidence_paragraph_ids"]
        if abstain and (ref["answer"] != "Unanswerable" or evidence):
            raise ValueError(f"{qid}: invalid abstention reference")
        if not abstain and not (
            evidence
            and len(evidence) <= MAX_EVIDENCE
            and len(evidence) == len(set(evidence))
        ):
            raise ValueError(f"{qid}: invalid evidence reference")
        for pid in evidence:
            paper = paragraphs.get(pid, {}).get("paper_id")
            if paper != question["paper_id"]:
                raise ValueError(f"{qid}: gold evidence outside target paper")
    return abstain


def add_missing(tally: Tally, abstain_gold: bool) -> None:
    """Score a question that has no prediction as zero everywhere."""
    tally.missing += 1
    tally.answers.append(0.0)
    if abstain_gold:
        tally.missed_abstain += 1
    else:
        tally.recalls.append(0.0)
        tally.citations.append(0.0)


def add_prediction(
    tally: Tally, prediction: Record, refs: list[Record], abstain_gold: bool
) -> None:
    """Add one prediction's scores, using the best reference per metric."""
    tally.latencies.append(prediction["latency_ms"])
    tally.answers.append(
        max(token_f1(prediction["answer"], ref["answer"]) for ref in refs)
    )
    abstain = prediction["abstain"]
    tally.true_abstain += abstain and abstain_gold
    tally.false_abstain += abstain and not abstain_gold
    tally.missed_abstain += not abstain and abstain_gold
    if abstain_gold:
        return
    tally.covered += not abstain
    retrieved = set(prediction["retrieved_paragraph_ids"])
    cited = {item["paragraph_id"] for item in prediction["citations"]}
    evidence = [ref["evidence_paragraph_ids"] for ref in refs]
    tally.recalls.append(
        max(len(retrieved & set(e)) / len(e) for e in evidence)
    )
    tally.citations.append(max(set_f1(cited, e) for e in evidence))


def ratio(top: float, bottom: float) -> float:
    """Return ``top / bottom``, or 0 when ``bottom`` is 0."""
    return top / bottom if bottom else 0.0


def summarize(tally: Tally, questions: int) -> dict[str, Any]:
    """Turn the totals into the metrics dictionary."""
    precision = ratio(
        tally.true_abstain, tally.true_abstain + tally.false_abstain
    )
    recall = ratio(
        tally.true_abstain, tally.true_abstain + tally.missed_abstain
    )
    return {
        "questions": questions,
        "answer_token_f1": mean(tally.answers),
        "paragraph_evidence_recall_at_5": mean(tally.recalls or [0.0]),
        "citation_evidence_f1": mean(tally.citations or [0.0]),
        "abstention_precision": precision,
        "abstention_recall": recall,
        "abstention_f1": ratio(2 * precision * recall, precision + recall),
        "answerable_question_coverage": ratio(tally.covered, tally.answerable),
        "missing_outputs": tally.missing,
        "mean_query_latency_ms": mean(tally.latencies)
        if tally.latencies
        else None,
        "latency_observations": len(tally.latencies),
    }


def score(
    corpus: list[Record],
    questions: list[Record],
    gold: list[Record],
    predictions: list[Record],
) -> dict[str, Any]:
    """Check every prediction, then score them against the gold labels.

    Parameters
    ----------
    corpus, questions, gold, predictions : list[dict]
        The records of each JSON Lines file.

    Returns
    -------
    dict
        Each metric by name. A missing prediction scores 0.

    Raises
    ------
    ValueError
        If the files disagree, a prediction breaks the output rules, or a
        gold reference is malformed.
    """
    paragraphs = unique_by(corpus, "paragraph_id")
    question_map = unique_by(questions, "question_id")
    gold_map = unique_by(gold, "question_id")
    prediction_map = unique_by(predictions, "question_id")
    if not questions or set(gold_map) != set(question_map):
        raise ValueError("gold IDs must exactly match a nonempty question set")
    if set(prediction_map) - set(question_map):
        raise ValueError("predictions contain unknown question IDs")
    for qid, record in prediction_map.items():
        validate_prediction(record, question_map[qid], paragraphs)
    tally = Tally()
    for qid, question in question_map.items():
        refs = gold_map[qid]["references"]
        abstain_gold = check_references(qid, question, refs, paragraphs)
        tally.answerable += not abstain_gold
        prediction = prediction_map.get(qid)
        if prediction is None:
            add_missing(tally, abstain_gold)
        else:
            add_prediction(tally, prediction, refs, abstain_gold)
    return summarize(tally, len(questions))


def setup_args() -> argparse.ArgumentParser:
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "questions", "gold", "predictions", "output"):
        parser.add_argument(f"--{name}", required=True)
    return parser


def main(args: argparse.Namespace) -> None:
    """Score the files, print the metrics, and write them to ``--output``."""
    names = ("corpus", "questions", "gold", "predictions")
    result = score(*(read_jsonl(getattr(args, name)) for name in names))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(setup_args().parse_args())

"""The output rules every prediction must follow.

The batch runner and ``tools/score.py`` both use these checks. A prediction
passes only if each citation quotes a retrieved paragraph exactly, every
paragraph comes from the requested papers, and an abstention has no citations.
"""

from __future__ import annotations

import math
from typing import Any

# The scorer reads at most five retrieved paragraphs and five citations.
MAX_RETRIEVED_PASSAGES = 5
MAX_CITATIONS = 5
ABSTAIN_ANSWER = "Unanswerable"

REQUIRED_PREDICTION_FIELDS = frozenset(
    {
        "question_id",
        "answer",
        "abstain",
        "citations",
        "retrieved_paragraph_ids",
        "latency_ms",
    }
)
CITATION_FIELDS = frozenset({"paragraph_id", "quote"})

Paragraphs = dict[str, dict[str, Any]]


def target_papers(question: dict[str, Any]) -> list[str]:
    """Return the papers a question may use: ``paper_ids`` or ``paper_id``.

    Parameters
    ----------
    question : dict
        A question with ``paper_ids`` (a list) or ``paper_id`` (one ID).

    Returns
    -------
    list[str]
        The allowed paper IDs.

    Raises
    ------
    ValueError
        If neither field holds valid, unique, nonempty IDs.
    """
    papers = question.get("paper_ids", [question.get("paper_id")])
    is_valid = (
        isinstance(papers, list)
        and bool(papers)
        and all(isinstance(paper, str) and paper for paper in papers)
        and len(papers) == len(set(papers))
    )
    if not is_valid:
        raise ValueError("question requires paper_id or unique paper_ids")
    return papers


def validate_prediction(
    record: Any, question: dict[str, Any], paragraphs: Paragraphs
) -> None:
    """Raise ``ValueError`` unless ``record`` follows every output rule.

    Parameters
    ----------
    record : Any
        The prediction to check.
    question : dict
        The question it answers.
    paragraphs : dict
        Every known paragraph, keyed by ``paragraph_id``.

    Raises
    ------
    ValueError
        Naming the question and the first rule the record breaks.
    """
    try:
        _check_fields(record, question)
        retrieved = _check_retrieval(record, question, paragraphs)
        _check_citations(record["citations"], retrieved, paragraphs)
        _check_abstention(record)
    except ValueError as exc:
        raise ValueError(f"{question['question_id']}: {exc}") from None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_fields(record: Any, question: dict[str, Any]) -> None:
    _require(isinstance(record, dict), "prediction must be an object")
    _require(
        record.keys() >= REQUIRED_PREDICTION_FIELDS,
        f"required fields: {sorted(REQUIRED_PREDICTION_FIELDS)}",
    )
    _require(
        record["question_id"] == question["question_id"], "wrong question ID"
    )
    _require(type(record["abstain"]) is bool, "abstain must be boolean")
    answer = record["answer"]
    _require(
        isinstance(answer, str) and bool(answer.strip()),
        "answer must be a nonempty string",
    )
    latency = record["latency_ms"]
    _require(
        type(latency) in (int, float)
        and math.isfinite(latency)
        and latency >= 0,
        "latency_ms must be finite and non-negative",
    )


def _check_retrieval(
    record: dict[str, Any], question: dict[str, Any], paragraphs: Paragraphs
) -> list[str]:
    ids = record["retrieved_paragraph_ids"]
    _require(
        isinstance(ids, list)
        and len(ids) <= MAX_RETRIEVED_PASSAGES
        and all(isinstance(pid, str) for pid in ids),
        f"retrieval must contain up to {MAX_RETRIEVED_PASSAGES} string IDs",
    )
    _require(len(ids) == len(set(ids)), "duplicate retrieval ID")
    papers = set(target_papers(question))
    for pid in ids:
        _require(pid in paragraphs, f"unknown paragraph: {pid}")
        _require(
            paragraphs[pid]["paper_id"] in papers,
            "retrieval outside requested paper scope",
        )
    return ids


def _check_citations(
    citations: Any, retrieved: list[str], paragraphs: Paragraphs
) -> None:
    _require(
        isinstance(citations, list) and len(citations) <= MAX_CITATIONS,
        f"up to {MAX_CITATIONS} citations required",
    )
    seen: set[str] = set()
    for citation in citations:
        _require(
            isinstance(citation, dict) and set(citation) == CITATION_FIELDS,
            "citation fields must be paragraph_id and quote only",
        )
        pid, quote = citation["paragraph_id"], citation["quote"]
        _require(pid in retrieved, "citation must be in retrieved set")
        _require(pid not in seen, "duplicate citation ID")
        seen.add(pid)
        _require(
            isinstance(quote, str)
            and bool(quote.strip())
            and quote in paragraphs[pid]["text"],
            "quote must be a nonempty exact substring of the cited paragraph",
        )


def _check_abstention(record: dict[str, Any]) -> None:
    if record["abstain"]:
        _require(
            record["answer"] == ABSTAIN_ANSWER and not record["citations"],
            f"abstention requires answer {ABSTAIN_ANSWER} and empty citations",
        )
        return
    _require(
        record["answer"].strip().lower() != ABSTAIN_ANSWER.lower()
        and bool(record["citations"]),
        "an answer requires citations and cannot be Unanswerable",
    )

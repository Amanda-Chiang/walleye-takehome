"""Build the text-only data files from the QASPER dev JSON.

Writes ``corpus.jsonl``, ``dev_questions.jsonl``, ``dev_gold.jsonl`` and
``manifest.json``. Questions are chosen by a hash of their IDs, so the output
is the same on every run. This is a maintainer tool; inference never uses it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qasper_rag.storage.jsonl import write_jsonl

Record = dict[str, Any]
Selected = tuple[Record, list[Record]]
MAX_EVIDENCE = 5
SELECTION = (
    "SHA256(question_id), fixed answerability quotas; no performance selection"
)
CURATION = (
    "Unanimous answerability; all answerable annotations fully map to <=5 text "
    "paragraphs; no FLOAT SELECTED evidence. Duplicate text maps to first "
    "occurrence."
)


def canonical_paragraphs(paper_id: str, paper: Record) -> list[Record]:
    """Return a paper's title, abstract, paragraphs and captions as records.

    Parameters
    ----------
    paper_id : str
        The arXiv ID that keys the paper in the QASPER JSON.
    paper : dict
        The paper's QASPER record.

    Returns
    -------
    list[dict]
        One record per nonblank piece, numbered in reading order.
    """
    pieces = [("Title", paper["title"]), ("Abstract", paper["abstract"])]
    for section in paper["full_text"]:
        name = section.get("section_name") or ""
        pieces.extend((name, text) for text in section["paragraphs"])
    # Raw captions only; the figures themselves are out of scope.
    pieces.extend(
        ("Caption", item.get("caption") or "")
        for item in paper.get("figures_and_tables", [])
    )
    return [
        {
            "paper_id": paper_id,
            "paragraph_id": f"{paper_id}:p{i:05d}",
            "section": section,
            "text": text.strip(),
        }
        for i, (section, text) in enumerate(pieces)
        if text.strip()
    ]


def answer_text(answer: Record) -> str | None:
    """Return the reference answer text, or None if the annotation has none."""
    if answer["extractive_spans"]:
        return ", ".join(answer["extractive_spans"])
    if answer["free_form_answer"]:
        return answer["free_form_answer"]
    if answer["yes_no"] is not None:
        return "Yes" if answer["yes_no"] else "No"
    return None


def convert_annotation(
    answer: Record, text_ids: dict[str, list[str]]
) -> Record | None:
    """Convert one annotation, or return None if it cannot be used."""
    if answer["unanswerable"]:
        return {
            "answer": "Unanswerable",
            "abstain": True,
            "evidence_paragraph_ids": [],
        }
    evidence = answer["evidence"]
    # Skip the annotation if any evidence is a table, a figure or missing.
    if not evidence or any(
        "FLOAT SELECTED" in text or text.strip() not in text_ids
        for text in evidence
    ):
        return None
    # Repeated paragraph text maps to its first occurrence.
    ids = list(dict.fromkeys(text_ids[text.strip()][0] for text in evidence))
    value = answer_text(answer)
    if len(ids) > MAX_EVIDENCE or value is None:
        return None
    return {"answer": value, "abstain": False, "evidence_paragraph_ids": ids}


def convert_references(
    qa: Record, text_ids: dict[str, list[str]]
) -> list[Record] | None:
    """Return every annotation of a question, or None to exclude it.

    A question is excluded if any annotation cannot be used, or if the
    annotators disagree on whether it can be answered.
    """
    references = []
    for annotation in qa["answers"]:
        reference = convert_annotation(annotation["answer"], text_ids)
        if reference is None:
            return None
        references.append(reference)
    if not references or len({ref["abstain"] for ref in references}) != 1:
        return None
    return references


def question_order(item: Selected) -> str:
    """Sort key: the SHA-256 hash of the question ID."""
    return hashlib.sha256(item[0]["question_id"].encode()).hexdigest()


def collect(papers: dict[str, Record]) -> tuple[list[Record], dict, int]:
    """Return the corpus, usable questions by answerability, and exclusions."""
    corpus: list[Record] = []
    pools: dict[bool, list[Selected]] = {True: [], False: []}
    excluded = 0
    for paper_id, paper in sorted(papers.items()):
        paragraphs = canonical_paragraphs(paper_id, paper)
        corpus.extend(paragraphs)
        text_ids: dict[str, list[str]] = defaultdict(list)
        for paragraph in paragraphs:
            text_ids[paragraph["text"]].append(paragraph["paragraph_id"])
        for qa in paper["qas"]:
            references = convert_references(qa, text_ids)
            if references is None:
                excluded += 1
                continue
            question = {
                "question_id": qa["question_id"],
                "paper_id": paper_id,
                "question": qa["question"],
            }
            pools[references[0]["abstain"]].append((question, references))
    return corpus, pools, excluded


def select(pools: dict, count: int, abstentions: int) -> list[Selected]:
    """Take the first questions by hash, with a fixed number of abstentions."""
    selected: list[Selected] = []
    for abstain, needed in ((False, count - abstentions), (True, abstentions)):
        pool = sorted(pools[abstain], key=question_order)
        if len(pool) < needed:
            raise ValueError(
                f"only {len(pool)} eligible abstain={abstain} questions; "
                f"requested {needed}"
            )
        selected.extend(pool[:needed])
    return sorted(selected, key=question_order)


def build_manifest(
    source: Path,
    papers: dict[str, Record],
    corpus: list[Record],
    selected: list[Selected],
    excluded: int,
) -> dict[str, Any]:
    """Describe the source file and what was selected from it."""
    return {
        "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "questions": len(selected),
        "unanswerable": sum(refs[0]["abstain"] for _, refs in selected),
        "paragraphs": len(corpus),
        "papers": len(papers),
        "excluded_questions": excluded,
        "selection": SELECTION,
        "curation": CURATION,
    }


def prepare(
    source: str | Path, output_dir: str | Path, count: int, abstentions: int
) -> None:
    """Write the four data files from the QASPER dev JSON.

    Parameters
    ----------
    source : str or Path
        The QASPER JSON file.
    output_dir : str or Path
        The folder to write.
    count : int
        How many questions to select.
    abstentions : int
        How many of them are unanswerable.
    """
    if not 0 <= abstentions <= count:
        raise ValueError("require 0 <= abstentions <= count")
    source = Path(source)
    papers = json.loads(source.read_text(encoding="utf-8"))
    corpus, pools, excluded = collect(papers)
    selected = select(pools, count, abstentions)
    output = Path(output_dir)
    write_jsonl(output / "corpus.jsonl", corpus)
    write_jsonl(output / "dev_questions.jsonl", (q for q, _ in selected))
    write_jsonl(
        output / "dev_gold.jsonl",
        (
            {"question_id": q["question_id"], "references": refs}
            for q, refs in selected
        ),
    )
    manifest = build_manifest(source, papers, corpus, selected, excluded)
    text = json.dumps(manifest, indent=2) + "\n"
    (output / "manifest.json").write_text(text)
    print(text, end="")


def setup_args() -> argparse.ArgumentParser:
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/qasper-dev-v0.3.json")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--abstentions", type=int, default=5)
    return parser


if __name__ == "__main__":
    args = setup_args().parse_args()
    prepare(args.source, args.output_dir, args.count, args.abstentions)

"""Copy the first N questions, their gold labels and their papers' paragraphs.

Each question searches only its own paper, so answers on the slice match
answers on the full corpus, and indexing the slice costs much less.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qasper_rag.storage.jsonl import read_jsonl, write_jsonl


def setup_args() -> argparse.ArgumentParser:
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--questions", default="data/dev_questions.jsonl")
    parser.add_argument("--gold", default="data/dev_gold.jsonl")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(args: argparse.Namespace) -> None:
    """Write ``questions.jsonl``, ``gold.jsonl`` and ``corpus.jsonl``."""
    questions = read_jsonl(args.questions)[: args.count]
    ids = {question["question_id"] for question in questions}
    papers = {question["paper_id"] for question in questions}
    gold = [row for row in read_jsonl(args.gold) if row["question_id"] in ids]
    corpus = [
        row for row in read_jsonl(args.corpus) if row["paper_id"] in papers
    ]
    target = Path(args.output_dir)
    write_jsonl(target / "questions.jsonl", questions)
    write_jsonl(target / "gold.jsonl", gold)
    write_jsonl(target / "corpus.jsonl", corpus)
    unanswerable = sum(row["references"][0]["abstain"] for row in gold)
    print(
        f"{len(questions)} questions ({unanswerable} unanswerable), "
        f"{len(papers)} papers, {len(corpus)} paragraphs -> {target}"
    )


if __name__ == "__main__":
    main(setup_args().parse_args())

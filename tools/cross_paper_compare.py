r"""Compare plain top-5 retrieval with a per-paper cap, across all papers.

For each question (no paper ID), the retriever named in the config searches
every paper and returns its ranking 50 deep. Two selections are made from
that one ranking:

- ``plain``: the first 5, which is what ``qasper-rag answer`` uses.
- ``capped``: walk the same ranking and skip a paragraph once its paper
  already has ``--per-paper`` passages, until 5 are chosen.

Both keep the retriever's scores, fusion and paragraph IDs; only the
selection differs. There is no gold for cross-paper questions, so the
output lists each selected passage (paper, section, text) to be judged by
reading. Retrieval only: dense scorers embed each question once.

Example::

    python tools/cross_paper_compare.py --config config/composed-hybrid.json \
      --index-dir .index-full --questions data/cross-paper/questions.jsonl \
      --output experiments/runs/008-cross-paper-cap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qasper_rag.config import load_class, load_config
from qasper_rag.pipeline.retrieval import EmbeddingRetriever
from qasper_rag.storage.jsonl import read_jsonl

Record = dict[str, Any]
TOP_K = 5  # the most passages the output contract allows
DEPTH = 50  # how deep the capped selection may look


def setup_args() -> argparse.ArgumentParser:
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-paper", type=int, default=2)
    return parser


def cap_per_paper(
    ranking: list[Record], per_paper: int, k: int = TOP_K
) -> list[Record]:
    """Return the first ``k`` passages, at most ``per_paper`` from each paper.

    Keeps the ranking's order. If the ranking runs out first, fewer than
    ``k`` passages are returned.
    """
    counts: Counter[str] = Counter()
    chosen = []
    for passage in ranking:
        if counts[passage["paper_id"]] < per_paper:
            counts[passage["paper_id"]] += 1
            chosen.append(passage)
        if len(chosen) == k:
            break
    return chosen


def describe(passages: list[Record]) -> list[Record]:
    """Return the fields needed to judge each passage by reading."""
    return [
        {
            "paragraph_id": p["paragraph_id"],
            "section": p.get("section", ""),
            "text": p["text"],
        }
        for p in passages
    ]


def main(args: argparse.Namespace) -> None:
    """Retrieve across all papers and write both selections."""
    config = load_config(args.config)
    paragraphs = read_jsonl(Path(args.index_dir) / "paragraphs.jsonl")
    retriever_class = load_class(config.retriever, EmbeddingRetriever)
    retriever = retriever_class(paragraphs, config.retriever, args.index_dir)
    all_papers = sorted({p["paper_id"] for p in paragraphs})
    rows = []
    for question in read_jsonl(args.questions):
        query = {"question": question["question"], "paper_ids": all_papers}
        ranking = retriever.retrieve(query, DEPTH)
        plain = ranking[:TOP_K]
        capped = cap_per_paper(ranking, args.per_paper)
        rows.append(
            {
                "question_id": question["question_id"],
                "question": question["question"],
                "plain": describe(plain),
                "capped": describe(capped),
                "plain_papers": len({p["paper_id"] for p in plain}),
                "capped_papers": len({p["paper_id"] for p in capped}),
            }
        )
    result = {
        "config": args.config,
        "per_paper": args.per_paper,
        "questions": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        papers = f"{row['plain_papers']} -> {row['capped_papers']} papers"
        print(f"{papers}: {row['question']}")


if __name__ == "__main__":
    main(setup_args().parse_args())

r"""Write a per-question error report for one scored run.

Reads the questions, gold, predictions and corpus of a run folder (the layout
``tools/run_eval.py`` writes) and tags each question with a mechanical
outcome. Gold is read only here, after prediction, for analysis.

Example::

    python tools/error_analysis.py --run-dir results/baseline-20 \\
      --output experiments/runs/001-baseline-20.md
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qasper_rag.storage.jsonl import read_jsonl

Row = dict[str, Any]


def setup_args() -> argparse.ArgumentParser:
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--predictions", default="predictions.jsonl")
    parser.add_argument("--output", required=True)
    return parser


def gold_evidence(references: list[Row]) -> set[str]:
    """Return the union of evidence paragraph IDs over all references."""
    return {pid for ref in references for pid in ref["evidence_paragraph_ids"]}


def outcome(prediction: Row, references: list[Row]) -> str:
    """Tag a prediction with one mechanical outcome.

    Parameters
    ----------
    prediction : dict
        One prediction record.
    references : list[dict]
        The gold references for the question.

    Returns
    -------
    str
        One of ``correct_abstain``, ``missed_abstain``, ``retrieval_miss``,
        ``retrieval_partial``, ``over_abstain`` or ``retrieved_answered``.
    """
    if all(ref["abstain"] for ref in references):
        return "correct_abstain" if prediction["abstain"] else "missed_abstain"
    evidence = gold_evidence(references)
    hits = evidence & set(prediction["retrieved_paragraph_ids"])
    if not hits:
        return "retrieval_miss"
    if prediction["abstain"]:
        return "over_abstain"
    return "retrieved_answered" if hits == evidence else "retrieval_partial"


def short(paragraph_id: str) -> str:
    """Drop the paper prefix from a paragraph ID."""
    return paragraph_id.split(":", 1)[-1]


def question_section(
    index: int, question: Row, prediction: Row, gold: Row, corpus: dict
) -> list[str]:
    """Render the markdown block for one question."""
    refs = gold["references"]
    evidence = gold_evidence(refs)
    retrieved = prediction["retrieved_paragraph_ids"]
    lines = [
        f"### Q{index} `{question['paper_id']}`: {outcome(prediction, refs)}",
        "",
        f"- **Question:** {question['question']}",
        f"- **Prediction:** {prediction['answer']}",
    ]
    lines += [f"- **Gold:** {ref['answer']}" for ref in refs]
    cited = [short(c["paragraph_id"]) for c in prediction["citations"]]
    lines.append(
        f"- **Retrieved:** {[short(p) for p in retrieved]}; "
        f"**cited:** {cited}; "
        f"**gold evidence:** {sorted(short(p) for p in evidence)}"
    )
    for pid in sorted(evidence - set(retrieved)):
        para = corpus.get(pid, {})
        text = para.get("text", "")[:240]
        lines.append(
            f"- **Missed** `{short(pid)}` [{para.get('section', '')}]: {text}"
        )
    return [*lines, ""]


def main(args: argparse.Namespace) -> None:
    """Write the markdown report."""
    run = Path(args.run_dir)
    questions = read_jsonl(run / "questions.jsonl")
    gold = {row["question_id"]: row for row in read_jsonl(run / "gold.jsonl")}
    preds = {
        row["question_id"]: row for row in read_jsonl(run / args.predictions)
    }
    corpus = {
        row["paragraph_id"]: row for row in read_jsonl(run / "corpus.jsonl")
    }
    counts = Counter(
        outcome(preds[q["question_id"]], gold[q["question_id"]]["references"])
        for q in questions
    )
    lines = [
        f"# Error report: `{run}`",
        "",
        "| Outcome | Count |",
        "| --- | ---: |",
    ]
    lines += [f"| {name} | {n} |" for name, n in counts.most_common()]
    lines += ["", "## Questions", ""]
    for i, question in enumerate(questions):
        qid = question["question_id"]
        lines += question_section(i, question, preds[qid], gold[qid], corpus)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}: {dict(counts)}")


if __name__ == "__main__":
    main(setup_args().parse_args())

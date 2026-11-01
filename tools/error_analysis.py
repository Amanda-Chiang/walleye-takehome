r"""Write a per-question error report for one scored run.

Reads the questions, gold, predictions and corpus of a run folder (the layout
``tools/run_eval.py`` writes) and tags each question with a mechanical
outcome. Gold is read only here, after prediction, for analysis.

When a question has several references, outcomes and missed evidence use
the one whose evidence best matches the retrieved paragraphs, as
``tools/score.py`` takes the best reference for evidence recall.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qasper_rag.storage.jsonl import read_jsonl
from tools.retrieval_eval import best_reference

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


def best_evidence(retrieved: list[str], references: list[Row]) -> set[str]:
    """Return the evidence of the reference that best matches ``retrieved``.

    Uses ``best_reference`` from ``tools/retrieval_eval.py``, so a question
    whose references cite different paragraphs is judged against the one it
    matched. Abstention references have no evidence, so they give an empty
    set.
    """
    evidence_sets = [ref["evidence_paragraph_ids"] for ref in references]
    if not all(evidence_sets):
        return set()
    return set(evidence_sets[best_reference(retrieved, evidence_sets)])


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
    retrieved = prediction["retrieved_paragraph_ids"]
    evidence = best_evidence(retrieved, references)
    hits = evidence & set(retrieved)
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
    retrieved = prediction["retrieved_paragraph_ids"]
    evidence = best_evidence(retrieved, refs)
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
        f"**gold evidence (best of {len(refs)} references):** "
        f"{sorted(short(p) for p in evidence)}"
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

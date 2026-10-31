"""Score a predictions file with RAGAS, using the gold labels as references.

RAGAS is an optional extra. Install it into the project environment with:

    python -m pip install --editable ".[ragas]"
    python tools/ragas_eval.py --help

Only answerable questions are scored. RAGAS has no reference answer for an
unanswerable question, so use the abstention metrics from tools/score.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qasper_rag.storage.jsonl import read_jsonl

warnings.filterwarnings("ignore", category=DeprecationWarning)

Record = dict[str, Any]
INPUT_FILES = ("corpus", "questions", "gold", "predictions")


def build_rows(
    corpus: list[Record],
    questions: list[Record],
    gold: list[Record],
    predictions: list[Record],
) -> list[Record]:
    """Join each answerable question with its prediction and passage text.

    Parameters
    ----------
    corpus, questions, gold, predictions : list[dict]
        The records of each JSON Lines file.

    Returns
    -------
    list[dict]
        One RAGAS row per answerable question that has a prediction.
    """
    text = {row["paragraph_id"]: row["text"] for row in corpus}
    gold_by_id = {row["question_id"]: row["references"] for row in gold}
    predicted = {row["question_id"]: row for row in predictions}
    rows = []
    for question in questions:
        reference = gold_by_id[question["question_id"]][0]
        prediction = predicted.get(question["question_id"])
        if reference["abstain"] or prediction is None:
            continue
        rows.append(
            {
                "user_input": question["question"],
                "response": prediction["answer"],
                "retrieved_contexts": [
                    text[pid] for pid in prediction["retrieved_paragraph_ids"]
                ],
                "reference": reference["answer"],
                "reference_contexts": [
                    text[pid] for pid in reference["evidence_paragraph_ids"]
                ],
            }
        )
    return rows


def make_models(judge_model: str, embedding_model: str) -> tuple[Any, Any]:
    """Return the RAGAS judge model and embedding model, via OpenRouter."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    key = os.environ.get("RAG_API_KEY") or os.environ["OPENROUTER_API_KEY"]
    base_url = os.environ.get("RAG_BASE_URL", "https://openrouter.ai/api/v1")
    llm = ChatOpenAI(
        model=judge_model, api_key=key, base_url=base_url, temperature=0
    )
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=key,
        base_url=base_url,
        check_embedding_ctx_length=False,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def run_ragas(rows: list[Record], llm: Any, embeddings: Any) -> Any:
    """Run the five RAGAS metrics and return the results as a DataFrame."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    result = evaluate(
        EvaluationDataset.from_list(rows),
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=4, timeout=120),
    )
    return result.to_pandas()


def clean(value: float) -> float | None:
    """Return None for NaN, which marks a metric RAGAS failed to compute."""
    return None if math.isnan(value) else value


def summarize(frame: Any, rows: list[Record], judge: str) -> dict[str, Any]:
    """Return means, failure counts and per-question scores."""
    names = [column for column in frame.columns if column not in rows[0]]
    per_question = [
        {
            "question": row["user_input"],
            **{name: clean(float(frame[name][i])) for name in names},
        }
        for i, row in enumerate(rows)
    ]
    return {
        "questions_scored": len(rows),
        "judge_model": judge,
        "means": {name: clean(float(frame[name].mean())) for name in names},
        "failed_cells": {name: int(frame[name].isna().sum()) for name in names},
        "per_question": per_question,
    }


def setup_args() -> argparse.ArgumentParser:
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for name in (*INPUT_FILES, "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--judge-model", default="openai/gpt-6-luna")
    parser.add_argument(
        "--embedding-model", default="openai/text-embedding-3-small"
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """Score the predictions, print the means, and write the full result."""
    rows = build_rows(*(read_jsonl(getattr(args, n)) for n in INPUT_FILES))
    if not rows:
        raise SystemExit("No answerable questions with predictions to score.")
    llm, embeddings = make_models(args.judge_model, args.embedding_model)
    summary = summarize(
        run_ragas(rows, llm, embeddings), rows, args.judge_model
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    brief = ("questions_scored", "means", "failed_cells")
    print(json.dumps({key: summary[key] for key in brief}, indent=2))


if __name__ == "__main__":
    main(setup_args().parse_args())

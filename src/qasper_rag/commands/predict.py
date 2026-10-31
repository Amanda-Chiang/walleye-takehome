"""Write one prediction per question, in the format ``tools/score.py`` reads.

Run as ``qasper-rag predict`` or ``python -m qasper_rag.commands.predict``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from qasper_rag.cli import run_cli
from qasper_rag.pipeline.service import load_service
from qasper_rag.storage.jsonl import read_jsonl, unique_by

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add this command's options to ``parser``."""
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--config", default="config/baseline.json")
    parser.add_argument("--output", required=True)


def execute(args: argparse.Namespace) -> int:
    """Run the command and return the exit code."""
    run_batch(args.index_dir, args.questions, args.config, args.output)
    return 0


def run_batch(
    index_dir: str | Path,
    questions_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> int:
    """Answer every question in a file and write the predictions.

    Each prediction is written as soon as it is ready. The run stops at the
    first failed question, naming it.

    Parameters
    ----------
    index_dir : str or Path
        The folder written by ``qasper-rag index``.
    questions_path : str or Path
        A JSON Lines file of questions.
    config_path : str or Path
        The settings file the index was built with.
    output_path : str or Path
        The predictions file to write.

    Returns
    -------
    int
        The number of predictions written.
    """
    service = load_service(index_dir, config_path)
    questions = read_jsonl(questions_path)
    unique_by(questions, "question_id")
    if not questions:
        raise ValueError("question file must contain at least one question")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for question in questions:
            try:
                result = service.answer(question)
            except Exception as exc:
                qid = question.get("question_id", "<unknown>")
                raise RuntimeError(f"question {qid} failed: {exc}") from exc
            record = json.dumps(
                result.prediction, ensure_ascii=False, allow_nan=False
            )
            stream.write(record + "\n")
            stream.flush()
    logger.info("Wrote %d predictions to %s", len(questions), output)
    return len(questions)


def main(argv: list[str] | None = None) -> int:
    """Run ``python -m qasper_rag.commands.predict``."""
    return run_cli(__doc__, add_arguments, execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

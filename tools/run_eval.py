"""Run the whole evaluation in one command: subset, index, predict, score.

Example, using the first 10 development questions and the baseline config:

    python tools/run_eval.py --count 10 --config config/baseline.json

Each step is the same command shown in docs/evaluation.md, so you can also
run the steps by hand. Pass --ragas to also score with RAGAS, which needs the
optional packages: python -m pip install --editable ".[ragas]".
Only the standard library is used by this script.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvalPaths:
    """Locations of the files one evaluation run reads and writes."""

    out_dir: str
    index_dir: str

    def file(self, name: str) -> str:
        """Return the path of ``name`` inside the output directory."""
        return f"{self.out_dir}/{name}"


def setup_args() -> argparse.ArgumentParser:
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/baseline.json")
    parser.add_argument(
        "--count", type=int, default=10, help="dev questions to evaluate"
    )
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--questions", default="data/dev_questions.jsonl")
    parser.add_argument("--gold", default="data/dev_gold.jsonl")
    parser.add_argument(
        "--output-dir", help="default: results/<config name>-<count>"
    )
    parser.add_argument(
        "--index-dir", help="default: .index-<config name>-<count>"
    )
    parser.add_argument(
        "--existing-index",
        action="store_true",
        help="use the index already in --index-dir and the full corpus",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="also score with RAGAS (needs the ragas extra)",
    )
    return parser


def resolve_paths(args: argparse.Namespace) -> EvalPaths:
    """Build the output and index directories from the arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with ``config``, ``count``, ``output_dir`` and
        ``index_dir``.

    Returns
    -------
    EvalPaths
        The explicit directories, or defaults named after the config.
    """
    name = f"{Path(args.config).stem}-{args.count}"
    return EvalPaths(
        out_dir=args.output_dir or f"results/{name}",
        index_dir=args.index_dir or f".index-{name}",
    )


def run(command: list[str]) -> None:
    """Run one step from the repository root and stop if it fails.

    Parameters
    ----------
    command : list[str]
        The program and its arguments.
    """
    logging.info("+ %s", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def command(program: list[str], **options: str) -> list[str]:
    """Return ``program`` followed by ``--name value`` for each option."""
    flags = [
        [f"--{name.replace('_', '-')}", value]
        for name, value in options.items()
    ]
    return [
        sys.executable,
        *program,
        *(part for flag in flags for part in flag),
    ]


def eval_files(args: argparse.Namespace, paths: EvalPaths) -> dict[str, str]:
    """Return the four files that predict and score read or write."""
    corpus = args.corpus if args.existing_index else paths.file("corpus.jsonl")
    return {
        "corpus": corpus,
        "questions": paths.file("questions.jsonl"),
        "gold": paths.file("gold.jsonl"),
        "predictions": paths.file("predictions.jsonl"),
    }


def data_steps(args: argparse.Namespace, paths: EvalPaths) -> list[list[str]]:
    """Return the subset step, plus the index step unless one exists."""
    subset = command(
        ["tools/subset.py"],
        corpus=args.corpus,
        questions=args.questions,
        gold=args.gold,
        count=str(args.count),
        output_dir=paths.out_dir,
    )
    if args.existing_index:
        return [subset]
    index = command(
        ["-m", "qasper_rag.commands.index"],
        corpus=paths.file("corpus.jsonl"),
        config=args.config,
        index_dir=paths.index_dir,
    )
    return [subset, index]


def build_steps(args: argparse.Namespace, paths: EvalPaths) -> list[list[str]]:
    """Return the commands of one evaluation run, in order.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    paths : EvalPaths
        Output and index directories.

    Returns
    -------
    list[list[str]]
        Subset, index (unless ``--existing-index``), predict and score, plus
        RAGAS when ``--ragas`` is given.
    """
    files = eval_files(args, paths)
    predict = command(
        ["-m", "qasper_rag.commands.predict"],
        index_dir=paths.index_dir,
        questions=files["questions"],
        config=args.config,
        output=files["predictions"],
    )
    steps = [*data_steps(args, paths), predict]
    steps.append(
        command(["tools/score.py"], **files, output=paths.file("metrics.json"))
    )
    if args.ragas:
        steps.append(
            command(
                ["tools/ragas_eval.py"],
                **files,
                output=paths.file("ragas.json"),
            )
        )
    return steps


def main(args: argparse.Namespace) -> None:
    """Run every step of the evaluation."""
    paths = resolve_paths(args)
    if args.existing_index and not Path(ROOT, paths.index_dir).is_dir():
        raise SystemExit(f"No index in {paths.index_dir}. Build it first.")
    for step in build_steps(args, paths):
        run(step)
    logging.info("Results are in %s", paths.out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(setup_args().parse_args())

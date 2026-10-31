"""Answer one question over one paper, several papers, or every paper.

Run as ``qasper-rag answer`` or ``python -m qasper_rag.commands.answer``.
"""

from __future__ import annotations

import argparse
import json

from qasper_rag.cli import run_cli
from qasper_rag.pipeline.service import load_service


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add this command's options to ``parser``."""
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--config", default="config/baseline.json")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--paper-id",
        action="append",
        dest="paper_ids",
        help="search only this paper; repeat for several; omit for all",
    )


def execute(args: argparse.Namespace) -> int:
    """Print the answer as JSON and return the exit code."""
    service = load_service(args.index_dir, args.config)
    response = service.ask(args.question, args.paper_ids)
    print(json.dumps(response, ensure_ascii=False, allow_nan=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run ``python -m qasper_rag.commands.answer``."""
    return run_cli(__doc__, add_arguments, execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

"""The ``qasper-rag`` command: index, answer, predict and serve.

Each command module has ``add_arguments`` and ``execute``, so
``qasper-rag <command>`` and ``python -m qasper_rag.<module>`` behave the same.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Callable, Sequence

# Command name: (module, help text). Modules load when the command runs.
COMMANDS = {
    "index": ("qasper_rag.commands.index", "build the paper index"),
    "answer": ("qasper_rag.commands.answer", "answer one question"),
    "predict": (
        "qasper_rag.commands.predict",
        "write predictions for a question file",
    ),
    "serve": ("qasper_rag.web.server", "start the local chat page"),
}

AddArguments = Callable[[argparse.ArgumentParser], None]
Execute = Callable[[argparse.Namespace], int]


def run_cli(
    description: str | None,
    add_arguments: AddArguments,
    execute: Execute,
    argv: Sequence[str] | None = None,
) -> int:
    """Parse arguments, run a command, and turn expected errors into exit 2.

    Parameters
    ----------
    description : str or None
        Help text for the parser.
    add_arguments : callable
        Adds the command's options to the parser.
    execute : callable
        Runs the command and returns its exit code.
    argv : sequence of str, optional
        The arguments. Defaults to ``sys.argv``.

    Returns
    -------
    int
        The exit code.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    parser = argparse.ArgumentParser(description=description)
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        return execute(args)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        logging.getLogger("qasper_rag").error("%s", exc)
        return 2


def add_commands(parser: argparse.ArgumentParser) -> None:
    """Add one subcommand per entry in ``COMMANDS``."""
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (module_name, help_text) in COMMANDS.items():
        module = importlib.import_module(module_name)
        module.add_arguments(commands.add_parser(name, help=help_text))


def execute(args: argparse.Namespace) -> int:
    """Run the chosen subcommand."""
    module = importlib.import_module(COMMANDS[args.command][0])
    return module.execute(args)


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``qasper-rag``."""
    return run_cli(__doc__, add_commands, execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

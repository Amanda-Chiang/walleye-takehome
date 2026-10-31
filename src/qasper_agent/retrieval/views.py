"""Text views: decide what text represents a piece of a paragraph."""

from __future__ import annotations

from typing import Any

from qasper_agent.retrieval.base import Passage


class RawText:
    """Index the piece as it is, as the baseline does.

    Parameters
    ----------
    settings : dict
        Unused.
    """

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings

    def render(self, paragraph: Passage, piece: str) -> str:
        """Return ``piece`` unchanged."""
        return piece


class SectionPrefixed:
    """Put the section path before the piece, e.g. "Results > Parsing".

    Some evidence paragraphs never use the question's words, but their
    section title does (EXPERIMENTS.md 002: "Baseline Approaches",
    "Features."). Only the indexed text changes; answers and quotes still
    use the original paragraph.

    Parameters
    ----------
    settings : dict
        Unused.
    """

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings

    def render(self, paragraph: Passage, piece: str) -> str:
        """Return the section path, a newline, then ``piece``.

        ``piece`` is returned unchanged when the paragraph has no section.
        """
        section = paragraph.get("section", "").replace(" ::: ", " > ")
        return f"{section}\n{piece}" if section else piece

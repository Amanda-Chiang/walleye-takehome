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

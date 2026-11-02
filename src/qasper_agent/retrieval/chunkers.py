"""Chunkers: split a paragraph into the pieces that get indexed."""

from __future__ import annotations

from typing import Any

from qasper_agent.retrieval.base import Passage


class WholeParagraph:
    """Index each paragraph as one piece, as the baseline does.

    Parameters
    ----------
    settings : dict
        Unused.
    """

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings

    def split(self, paragraph: Passage) -> list[str]:
        """Return the paragraph text as the only piece."""
        return [paragraph["text"]]

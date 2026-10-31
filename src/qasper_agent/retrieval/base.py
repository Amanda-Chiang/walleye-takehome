"""The parts a composed retriever is built from, and the unit they share.

A paragraph goes through the parts in this order::

    paragraph ─► Chunker.split ─► pieces ─► TextView.render ─► Unit.text
    question  ─► each Scorer.score(units) ─► Fusion.fuse ─► best paragraphs
              ─► Reranker.rerank (optional) ─► top_k paragraphs

Constructors, as ``ComposedRetriever`` calls them:

- ``TextView(settings)``, ``Chunker(settings)``, ``Fusion(settings)`` and
  ``Reranker(settings)``.
- ``Scorer(settings, index_dir)``. ``index_dir`` is where a scorer may cache
  what it builds, e.g. embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

Passage = dict[str, Any]
Question = dict[str, Any]


@dataclass(frozen=True)
class Unit:
    """One indexed piece of a paragraph.

    Attributes
    ----------
    paragraph_id : str
        The paragraph this unit came from. Scores roll up to it.
    paper_id : str
        The paper of that paragraph, used to limit the search.
    text : str
        The text that scorers see. It may differ from the paragraph text,
        e.g. with the section title in front. Answers and quotes always use
        the original paragraph.
    """

    paragraph_id: str
    paper_id: str
    text: str


class Chunker(Protocol):
    """Split a paragraph into the pieces that get indexed."""

    def split(self, paragraph: Passage) -> list[str]:
        """Return one or more nonempty pieces of ``paragraph["text"]``."""
        ...


class TextView(Protocol):
    """Decide what text represents one piece of a paragraph."""

    def render(self, paragraph: Passage, piece: str) -> str:
        """Return the text to index for ``piece`` of ``paragraph``."""
        ...


class Scorer(Protocol):
    """Score units against a question. Higher means more relevant."""

    def fit(self, units: list[Unit]) -> None:
        """Prepare to score ``units``, e.g. embed them or count terms."""
        ...

    def score(self, query: str, candidates: list[int]) -> list[float]:
        """Return one score per candidate.

        ``candidates`` are positions in the ``units`` list given to ``fit``.
        A score of exactly 0 means "no match": fusion gives it no credit.
        """
        ...


class Fusion(Protocol):
    """Combine the scores of several scorers into one score per unit."""

    def fuse(self, score_lists: list[list[float]]) -> list[float]:
        """Return one fused score per unit. Higher means more relevant.

        Each inner list holds one scorer's scores, in the same unit order.
        """
        ...


class Reranker(Protocol):
    """Reorder the best paragraphs, e.g. with a cross-encoder or a model."""

    def rerank(
        self, question: Question, passages: list[Passage]
    ) -> list[Passage]:
        """Return ``passages`` reordered, best first, unchanged."""
        ...

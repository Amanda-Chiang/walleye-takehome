"""A retriever assembled from a chunker, index text, scorers and fusion.

Each part is named in the ``retriever`` config section, so one part can be
changed without touching the others::

    "retriever": {
      "class": "qasper_agent.retrieval.composed:ComposedRetriever",
      "chunker": {"class": "qasper_agent.retrieval.chunkers:WholeParagraph"},
      "index_text": {"class": "qasper_agent.retrieval.index_text:RawText"},
      "scorers": [{"class": "qasper_agent.retrieval.dense:DenseScorer",
                   "embedding": {...}}],
      "fusion": {"class": "qasper_agent.retrieval.fusion:ReciprocalRankFusion"},
      "candidate_k": 20,
      "reranker": {"class": "..."}
    }

Only ``scorers`` is required. ``reranker`` is optional, and ``candidate_k``
sets how many paragraphs it sees.

``qasper-rag index`` calls ``prepare_index``, which fits every scorer once so
their caches (e.g. embeddings) are built at index time, not at first load.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from qasper_agent.components import build_component
from qasper_agent.retrieval.base import Passage, Question, Unit
from qasper_agent.retrieval.chunkers import WholeParagraph
from qasper_agent.retrieval.fusion import ReciprocalRankFusion
from qasper_agent.retrieval.index_text import RawText
from qasper_rag.config import ConfigurationError


class ComposedRetriever:
    """Score indexed units, then return their best original paragraphs.

    Parameters
    ----------
    paragraphs : list[dict]
        The indexed paragraphs.
    settings : dict
        The ``retriever`` settings; see the module docstring.
    index_dir : str or Path
        The index folder. Scorers may cache what they build here.

    Raises
    ------
    ConfigurationError
        If no scorer is configured or a part cannot be loaded.
    """

    def __init__(
        self,
        paragraphs: list[Passage],
        settings: dict[str, Any],
        index_dir: str | Path,
    ) -> None:
        self.settings = settings
        self.chunker = build_component(
            "chunker", settings.get("chunker"), WholeParagraph
        )
        self.index_text = build_component(
            "index_text", settings.get("index_text"), RawText
        )
        self.fusion = build_component(
            "fusion", settings.get("fusion"), ReciprocalRankFusion
        )
        self.reranker = (
            build_component("reranker", settings["reranker"], None)
            if settings.get("reranker")
            else None
        )
        self.candidate_k: int = settings.get("candidate_k", 0)
        self.paragraphs = {p["paragraph_id"]: p for p in paragraphs}
        self.units = self._make_units()
        self.units_by_paper: dict[str, list[int]] = defaultdict(list)
        for position, unit in enumerate(self.units):
            self.units_by_paper[unit.paper_id].append(position)
        self.scorers = self._make_scorers(index_dir)

    @classmethod
    def prepare_index(
        cls,
        paragraphs: list[Passage],
        settings: dict[str, Any],
        index_dir: str | Path,
    ) -> None:
        """Build every part once so scorer caches are filled.

        ``qasper-rag index`` calls this after writing the index files. Later
        loads read the caches instead of embedding at startup.

        Parameters
        ----------
        paragraphs : list[dict]
            The corpus paragraphs.
        settings : dict
            The ``retriever`` settings.
        index_dir : str or Path
            The index folder.
        """
        cls(paragraphs, settings, index_dir)

    def _make_units(self) -> list[Unit]:
        return [
            Unit(
                p["paragraph_id"],
                p["paper_id"],
                self.index_text.render(p, piece),
            )
            for p in self.paragraphs.values()
            for piece in self.chunker.split(p)
        ]

    def _make_scorers(self, index_dir: str | Path) -> list[Any]:
        sections = self.settings.get("scorers")
        if not isinstance(sections, list) or not sections:
            raise ConfigurationError(
                "retriever.scorers must be a nonempty list"
            )
        scorers = [
            build_component("scorer", section, None, index_dir)
            for section in sections
        ]
        for scorer in scorers:
            scorer.fit(self.units)
        return scorers

    def retrieve(self, question: Question, top_k: int) -> list[Passage]:
        """Return up to ``top_k`` paragraphs from ``question["paper_ids"]``.

        Parameters
        ----------
        question : dict
            Holds ``question`` (the text) and ``paper_ids``.
        top_k : int
            How many paragraphs to return.

        Returns
        -------
        list[dict]
            The best original paragraphs, best first.
        """
        candidates = [
            position
            for paper_id in question["paper_ids"]
            for position in self.units_by_paper[paper_id]
        ]
        if not candidates:
            return []
        score_lists = [
            scorer.score(question["question"], candidates)
            for scorer in self.scorers
        ]
        fused = self.fusion.fuse(score_lists)
        ranked = best_paragraphs(
            [self.units[i].paragraph_id for i in candidates], fused
        )
        depth = max(top_k, self.candidate_k)
        passages = [self.paragraphs[pid] for pid in ranked[:depth]]
        if self.reranker is not None:
            passages = self.reranker.rerank(question, passages)
        return passages[:top_k]


def best_paragraphs(paragraph_ids: list[str], scores: list[float]) -> list[str]:
    """Rank paragraphs by the best score among their units.

    Parameters
    ----------
    paragraph_ids : list[str]
        The paragraph of each scored unit.
    scores : list[float]
        The score of each unit, in the same order.

    Returns
    -------
    list[str]
        Paragraph IDs, best first. Ties keep the earlier paragraph first.
    """
    best: dict[str, float] = {}
    for paragraph_id, score in zip(paragraph_ids, scores, strict=True):
        if paragraph_id not in best or score > best[paragraph_id]:
            best[paragraph_id] = score
    first_seen = {pid: i for i, pid in reversed(list(enumerate(paragraph_ids)))}
    return sorted(best, key=lambda pid: (-best[pid], first_seen[pid]))

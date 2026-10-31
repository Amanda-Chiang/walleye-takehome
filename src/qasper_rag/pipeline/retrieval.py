"""Find the passages most similar to a question, using stored embeddings."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from qasper_rag.providers.embeddings import (
    EmbeddingClient,
    Vector,
    corpus_fingerprint,
    normalize,
)
from qasper_rag.storage.jsonl import read_jsonl, unique_by

Passage = dict[str, Any]


class EmbeddingRetriever:
    """Rank passages by cosine similarity to the question's embedding.

    Parameters
    ----------
    paragraphs : list[dict]
        The indexed paragraphs.
    settings : dict
        The ``retriever`` settings. ``embedding`` must match the settings
        the index was built with.
    index_dir : str or Path
        The folder written by ``qasper-rag index``.

    Raises
    ------
    ValueError
        If the index is missing, or was built with other settings or another
        corpus.
    """

    def __init__(
        self,
        paragraphs: list[Passage],
        settings: dict[str, Any],
        index_dir: str | Path,
    ) -> None:
        self.embedder = EmbeddingClient(settings["embedding"])
        metadata = self._read_metadata(Path(index_dir), paragraphs)
        self.dimensions: int = metadata["dimensions"]
        vectors = unique_by(
            read_jsonl(Path(index_dir) / "embeddings.jsonl"), "paragraph_id"
        )
        if set(vectors) != {p["paragraph_id"] for p in paragraphs}:
            raise ValueError("embedding index IDs do not match the corpus")
        self.by_paper: dict[str, list[tuple[Passage, Vector]]] = defaultdict(
            list
        )
        for paragraph in paragraphs:
            vector = vectors[paragraph["paragraph_id"]]["vector"]
            self.by_paper[paragraph["paper_id"]].append(
                (paragraph, normalize(vector, self.dimensions))
            )

    def _read_metadata(
        self, index_dir: Path, paragraphs: list[Passage]
    ) -> dict[str, Any]:
        path = index_dir / "embedding_metadata.json"
        if not path.exists():
            raise ValueError("missing embedding index; run `qasper-rag index`")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        is_current = metadata[
            "embedding"
        ] == self.embedder.signature and metadata[
            "corpus_sha256"
        ] == corpus_fingerprint(paragraphs)
        if not is_current:
            raise ValueError(
                "embedding config or corpus changed; rebuild the index"
            )
        return metadata

    def retrieve(self, question: dict[str, Any], top_k: int) -> list[Passage]:
        """Return the ``top_k`` passages closest to the question.

        Parameters
        ----------
        question : dict
            Holds ``question`` (the text) and ``paper_ids`` (the papers to
            search). The service checks ``paper_ids`` before this call.
        top_k : int
            How many passages to return.

        Returns
        -------
        list[dict]
            The closest passages, best first.
        """
        query = normalize(
            self.embedder.embed([question["question"]])[0], self.dimensions
        )
        candidates = [
            item
            for paper_id in question["paper_ids"]
            for item in self.by_paper[paper_id]
        ]
        scored = [
            (
                sum(a * b for a, b in zip(query, vector, strict=True)),
                position,
                passage,
            )
            for position, (passage, vector) in enumerate(candidates)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [passage for _, _, passage in scored[:top_k]]

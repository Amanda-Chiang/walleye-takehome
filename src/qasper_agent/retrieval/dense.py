"""Score units by cosine similarity of embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qasper_agent.retrieval.base import Unit
from qasper_rag.providers.embeddings import EmbeddingClient, Vector, normalize

# The starter's cache file. Its keys include the embedding settings and the
# text, so sharing it reuses vectors already paid for and cannot mix models.
CACHE_FILE = "embedding-cache.sqlite"


class DenseScorer:
    """Rank units by cosine similarity to the question embedding.

    Uses the starter's ``EmbeddingClient``, so the model and API settings
    work the same way as in ``config/baseline.json``.

    Parameters
    ----------
    settings : dict
        Holds ``embedding``, the same settings as the starter's
        ``retriever.embedding``.
    index_dir : str or Path
        Where the embedding cache is kept.
    """

    def __init__(self, settings: dict[str, Any], index_dir: str | Path) -> None:
        self.embedder = EmbeddingClient(settings["embedding"])
        self.batch_size: int = settings["embedding"].get("batch_size", 32)
        self.cache_path = Path(index_dir) / CACHE_FILE
        self.vectors: list[Vector] = []

    def fit(self, units: list[Unit]) -> None:
        """Embed every unit, reusing cached vectors."""
        self.vectors = self.embedder.embed_cached(
            [unit.text for unit in units], self.cache_path, self.batch_size
        )

    def score(self, query: str, candidates: list[int]) -> list[float]:
        """Return the cosine similarity of each candidate to ``query``."""
        query_vector = normalize(
            self.embedder.embed([query])[0], self.embedder.dimensions
        )
        return [
            sum(
                a * b
                for a, b in zip(query_vector, self.vectors[i], strict=True)
            )
            for i in candidates
        ]

"""Embed text through the API and cache the vectors in SQLite."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from qasper_rag.providers.api_client import APIClient

Vector = list[float]


def normalize(vector: Any, dimensions: int | None = None) -> Vector:
    """Check a vector and scale it to length 1.

    Parameters
    ----------
    vector : Any
        The vector to check, as returned by the API or read from disk.
    dimensions : int, optional
        The required length, if known.

    Returns
    -------
    list[float]
        The vector divided by its length.

    Raises
    ------
    ValueError
        If the vector is empty, has a non-numeric or non-finite value, has
        the wrong length, or has zero length.
    """
    is_numeric = (
        isinstance(vector, list)
        and bool(vector)
        and all(type(x) in (int, float) and math.isfinite(x) for x in vector)
    )
    if not is_numeric:
        raise ValueError("embedding must be a nonempty finite numeric vector")
    if dimensions is not None and len(vector) != dimensions:
        raise ValueError(
            "embedding dimension mismatch; rebuild the index with this config"
        )
    length = math.sqrt(sum(x * x for x in vector))
    if not math.isfinite(length) or length == 0:
        raise ValueError("embedding has an invalid norm")
    return [x / length for x in vector]


def corpus_fingerprint(paragraphs: list[dict[str, Any]]) -> str:
    """Return a SHA-256 hash that changes whenever any paragraph changes."""
    canonical = json.dumps(
        paragraphs, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class EmbeddingClient:
    """Embed text with one embedding model.

    Parameters
    ----------
    config : dict
        The ``retriever.embedding`` settings. Uses ``model``, ``dimensions``,
        ``max_input_chars`` and the API settings read by ``APIClient``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.client = APIClient(config, "RAG_EMBEDDING_MODEL")
        self.dimensions: int | None = config.get("dimensions")
        self.max_input_chars: int = config.get("max_input_chars", 12_000)
        if self.dimensions is not None and (
            type(self.dimensions) is not int or self.dimensions < 1
        ):
            raise ValueError("embedding dimensions must be a positive integer")
        if type(self.max_input_chars) is not int or self.max_input_chars < 1:
            raise ValueError("max_input_chars must be a positive integer")

    @property
    def signature(self) -> dict[str, Any]:
        """Settings that change the vectors. A change needs a new index."""
        return {
            "model": self.client.model,
            "base_url": self.client.base_url,
            "dimensions": self.dimensions,
            "max_input_chars": self.max_input_chars,
        }

    def input_text(self, text: str) -> str:
        """Shorten text for embedding only. Quotes still use the full text."""
        return text[: self.max_input_chars]

    def embed(self, texts: list[str]) -> list[Vector]:
        """Embed texts in one request and return vectors in the same order.

        Parameters
        ----------
        texts : list[str]
            The texts to embed.

        Returns
        -------
        list[list[float]]
            One normalized vector per text.

        Raises
        ------
        ValueError
            If the response has the wrong number of vectors, a bad or repeated
            index, or an invalid vector.
        """
        if not texts:
            return []
        payload: dict[str, Any] = {
            "model": self.client.model,
            "input": [self.input_text(text) for text in texts],
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        data = self.client.post("/embeddings", payload)["data"]
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError("embedding response count mismatch")
        by_index = {item["index"]: item["embedding"] for item in data}
        if sorted(by_index, key=str) != sorted(range(len(texts)), key=str):
            raise ValueError("invalid or duplicate embedding response index")
        vectors = [
            normalize(by_index[i], self.dimensions) for i in range(len(texts))
        ]
        if len({len(vector) for vector in vectors}) != 1:
            raise ValueError("inconsistent embedding dimensions")
        return vectors

    def embed_cached(
        self, texts: list[str], cache_path: str | Path, batch_size: int = 32
    ) -> list[Vector]:
        """Embed texts, reusing vectors cached by an earlier run.

        Each finished batch is saved at once, so a failed run can be resumed.

        Parameters
        ----------
        texts : list[str]
            The texts to embed.
        cache_path : str or Path
            The SQLite cache file. It is created if missing.
        batch_size : int
            How many texts to send per request.

        Returns
        -------
        list[list[float]]
            One normalized vector per text.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("embedding batch_size must be a positive integer")
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        keys = [self._cache_key(text) for text in texts]
        with closing(sqlite3.connect(cache_path)) as cache:
            cache.execute(
                "CREATE TABLE IF NOT EXISTS embeddings "
                "(key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
            )
            found = self._read_cache(cache, keys)
            missing = list(
                {
                    key: text
                    for key, text in zip(keys, texts, strict=True)
                    if key not in found
                }.items()
            )
            for offset in range(0, len(missing), batch_size):
                batch = missing[offset : offset + batch_size]
                found.update(self._embed_and_store(cache, batch))
        return [found[key] for key in keys]

    def _cache_key(self, text: str) -> str:
        signature = json.dumps(self.signature, sort_keys=True)
        source = f"{signature}\n{self.input_text(text)}"
        return hashlib.sha256(source.encode()).hexdigest()

    def _read_cache(
        self, cache: sqlite3.Connection, keys: list[str]
    ) -> dict[str, Vector]:
        found = {}
        for key in dict.fromkeys(keys):
            row = cache.execute(
                "SELECT vector FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
            if row:
                found[key] = normalize(json.loads(row[0]), self.dimensions)
        return found

    def _embed_and_store(
        self, cache: sqlite3.Connection, batch: list[tuple[str, str]]
    ) -> dict[str, Vector]:
        vectors = self.embed([text for _, text in batch])
        stored = dict(zip((key for key, _ in batch), vectors, strict=True))
        cache.executemany(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
            [(key, json.dumps(vector)) for key, vector in stored.items()],
        )
        cache.commit()
        return stored

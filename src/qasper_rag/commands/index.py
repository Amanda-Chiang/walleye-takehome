"""Build the paper index: a corpus snapshot plus one embedding per paragraph.

Run as ``qasper-rag index`` or ``python -m qasper_rag.commands.index``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from qasper_rag.cli import run_cli
from qasper_rag.config import AppConfig, load_class, load_config
from qasper_rag.pipeline.retrieval import EmbeddingRetriever
from qasper_rag.providers.embeddings import EmbeddingClient, corpus_fingerprint
from qasper_rag.storage.jsonl import read_jsonl, unique_by, write_jsonl

logger = logging.getLogger(__name__)

REQUIRED_PARAGRAPH_FIELDS = ("paragraph_id", "paper_id", "text")
INDEX_FILES = (
    "embeddings.jsonl",
    "paragraphs.jsonl",
    "embedding_metadata.json",
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add this command's options to ``parser``."""
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--config", default="config/baseline.json")


def execute(args: argparse.Namespace) -> int:
    """Run the command and return the exit code."""
    build_index(args.corpus, args.index_dir, args.config)
    return 0


def load_corpus(corpus_path: str | Path) -> list[dict[str, Any]]:
    """Read the corpus and check every paragraph.

    Parameters
    ----------
    corpus_path : str or Path
        A JSON Lines file of paragraphs.

    Returns
    -------
    list[dict]
        The paragraphs, in file order.

    Raises
    ------
    ValueError
        If the corpus is empty, an ID repeats, ``paragraph_id``, ``paper_id``
        or ``text`` is empty, or ``section`` is not a string. Some QASPER
        sections have no name, so an empty ``section`` is allowed.
    """
    paragraphs = read_jsonl(corpus_path)
    if not paragraphs:
        raise ValueError("corpus must not be empty")
    for paragraph_id, paragraph in unique_by(
        paragraphs, "paragraph_id"
    ).items():
        for key in REQUIRED_PARAGRAPH_FIELDS:
            value = paragraph.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"paragraph {paragraph_id!r} requires a nonempty "
                    f"string {key}"
                )
        if not isinstance(paragraph.get("section"), str):
            raise ValueError(
                f"paragraph {paragraph_id!r} requires a string section"
            )
    return paragraphs


def build_index(
    corpus_path: str | Path, index_dir: str | Path, config_path: str | Path
) -> int:
    """Embed every paragraph and write the index files.

    Files are written under temporary names and renamed at the end, so a
    failed run never leaves a half-built index. Embeddings are cached in
    ``embedding-cache.sqlite``, so a rerun only embeds new text.

    Parameters
    ----------
    corpus_path : str or Path
        A JSON Lines file of paragraphs.
    index_dir : str or Path
        The folder to write.
    config_path : str or Path
        The settings file.

    Returns
    -------
    int
        The number of paragraphs indexed.
    """
    paragraphs = load_corpus(corpus_path)
    config = load_config(config_path)
    embedding_config = config.embedding
    target = Path(index_dir)
    target.mkdir(parents=True, exist_ok=True)
    if embedding_config is None:
        # A custom retriever without embeddings only needs the corpus.
        write_jsonl(target / "paragraphs.jsonl", paragraphs)
        prepare_retriever(config, paragraphs, target)
        return len(paragraphs)
    embedder = EmbeddingClient(embedding_config)
    logger.info(
        "Embedding %d paragraphs with %s; finished batches are cached",
        len(paragraphs),
        embedder.client.model,
    )
    vectors = embedder.embed_cached(
        [paragraph["text"] for paragraph in paragraphs],
        target / "embedding-cache.sqlite",
        embedding_config.get("batch_size", 32),
    )
    if len({len(vector) for vector in vectors}) != 1:
        raise ValueError("embedding provider returned inconsistent dimensions")
    _write_pending(target, paragraphs, vectors, embedder.signature)
    for name in INDEX_FILES:
        (target / f"{name}.pending").replace(target / name)
    logger.info("Indexed %d paragraphs in %s", len(paragraphs), target)
    prepare_retriever(config, paragraphs, target)
    return len(paragraphs)


def prepare_retriever(
    config: AppConfig, paragraphs: list[dict[str, Any]], index_dir: Path
) -> None:
    """Let a custom retriever build what it needs once, at index time.

    A retriever class may define ``prepare_index(paragraphs, settings,
    index_dir)``, e.g. to embed text the default index does not hold, such
    as chunks or section titles. It runs after the index files are written,
    so later loads by ``predict``, ``answer`` and ``serve`` can read its
    cache instead of building at startup. Classes without it are unaffected.

    Parameters
    ----------
    config : AppConfig
        The checked settings.
    paragraphs : list[dict]
        The corpus paragraphs.
    index_dir : Path
        The index folder.
    """
    retriever_class = load_class(config.retriever, EmbeddingRetriever)
    prepare_index = getattr(retriever_class, "prepare_index", None)
    if prepare_index is None:
        return
    logger.info("Preparing %s for the index", retriever_class.__name__)
    prepare_index(paragraphs, config.retriever, index_dir)


def _write_pending(
    target: Path,
    paragraphs: list[dict[str, Any]],
    vectors: list[list[float]],
    signature: dict[str, Any],
) -> None:
    write_jsonl(
        target / "embeddings.jsonl.pending",
        (
            {"paragraph_id": paragraph["paragraph_id"], "vector": vector}
            for paragraph, vector in zip(paragraphs, vectors, strict=True)
        ),
    )
    write_jsonl(target / "paragraphs.jsonl.pending", paragraphs)
    metadata = {
        "embedding": signature,
        "dimensions": len(vectors[0]),
        "corpus_sha256": corpus_fingerprint(paragraphs),
    }
    (target / "embedding_metadata.json.pending").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    """Run ``python -m qasper_rag.commands.index``."""
    return run_cli(__doc__, add_arguments, execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

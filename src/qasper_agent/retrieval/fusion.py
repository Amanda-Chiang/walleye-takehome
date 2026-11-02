"""Fusion: combine several scorers' scores into one score per unit."""

from __future__ import annotations

from typing import Any


class ReciprocalRankFusion:
    """Sum ``1 / (k + rank)`` over scorers.

    Uses ranks, not raw scores, so scorers on different scales (BM25 and
    cosine similarity) combine without tuning. With one scorer it keeps that
    scorer's order.

    A score of exactly 0 means the scorer found no match (e.g. BM25 with no
    query term), so it earns no credit. Otherwise every unmatched paragraph
    would gain credit, more of it the earlier it appears in the paper.
    Equal nonzero scores share the best rank of their group.

    Parameters
    ----------
    settings : dict
        ``k`` (default 60) damps the weight of the top ranks.
    """

    def __init__(self, settings: dict[str, Any]) -> None:
        self.k: int = settings.get("k", 60)

    def fuse(self, score_lists: list[list[float]]) -> list[float]:
        """Return the fused score of each unit."""
        fused = [0.0] * len(score_lists[0])
        for scores in score_lists:
            for position, rank in match_ranks(scores).items():
                fused[position] += 1 / (self.k + rank)
        return fused


def match_ranks(scores: list[float]) -> dict[int, int]:
    """Return the 1-based rank of each unit with a nonzero score.

    Equal scores share the best rank of their group. Units scoring exactly
    0 are left out.
    """
    matched = sorted(
        (i for i, score in enumerate(scores) if score != 0),
        key=lambda i: -scores[i],
    )
    ranks: dict[int, int] = {}
    for rank, position in enumerate(matched, start=1):
        previous = matched[rank - 2] if rank > 1 else None
        if previous is not None and scores[previous] == scores[position]:
            ranks[position] = ranks[previous]
        else:
            ranks[position] = rank
    return ranks

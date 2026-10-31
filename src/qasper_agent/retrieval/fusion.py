"""Fusion: combine several scorers' scores into one score per unit."""

from __future__ import annotations

from typing import Any


class ReciprocalRankFusion:
    """Sum ``1 / (k + rank)`` over scorers.

    Uses ranks, not raw scores, so scorers on different scales (BM25 and
    cosine similarity) combine without tuning. With one scorer it keeps that
    scorer's order. Ties keep the earlier unit first.

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
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            for rank, position in enumerate(order, start=1):
                fused[position] += 1 / (self.k + rank)
        return fused

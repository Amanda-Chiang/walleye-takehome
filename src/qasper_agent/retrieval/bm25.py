"""Score units by BM25 keyword relevance, with no API calls."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from qasper_agent.retrieval.base import Unit

TOKEN = re.compile(r"[a-z0-9]+")

# Function words and question words. Questions are short and mostly made of
# these, and a paragraph matching only them earned a top BM25 rank
# (EXPERIMENTS.md 003).
STOPWORD_TEXT = (
    "a an and are as at be been but by can could did do does for from had "
    "has have how i if in into is it its no not of on or our such than "
    "that the their them then there these they this those to was we were "
    "what when where which while who whom why will with would"
)
STOPWORDS = frozenset(STOPWORD_TEXT.split())


def tokenize(text: str, stopwords: frozenset[str] = frozenset()) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords and a plural "s".

    Stopwords are matched before the plural rule. The plural rule lets
    "baselines" match "baseline". It skips words of 3 letters or fewer and
    words ending in "ss" (e.g. "class").
    """
    return [
        t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t
        for t in TOKEN.findall(text.lower())
        if t not in stopwords
    ]


class BM25Scorer:
    """Rank units by Okapi BM25 over their tokens.

    Term rarity (IDF) is computed over every indexed unit, all papers
    together. A unit with no query term scores exactly 0, which fusion
    treats as "no match".

    Parameters
    ----------
    settings : dict
        ``k1`` (default 1.5) limits how much repeated terms add. ``b``
        (default 0.75) sets how much long units are penalized.
        ``stopwords`` (default false) drops ``STOPWORDS`` from the query.
        Units keep them, so their lengths and term rarity do not change.
    index_dir : str or Path
        Unused; BM25 needs no cache.
    """

    def __init__(self, settings: dict[str, Any], index_dir: str | Path) -> None:
        self.k1: float = settings.get("k1", 1.5)
        self.b: float = settings.get("b", 0.75)
        self.stopwords = STOPWORDS if settings.get("stopwords") else frozenset()
        self.terms: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.average_length = 1.0
        self.idf: dict[str, float] = {}

    def fit(self, units: list[Unit]) -> None:
        """Count the terms of every unit and how rare each term is."""
        self.terms = [Counter(tokenize(unit.text)) for unit in units]
        self.lengths = [sum(terms.values()) for terms in self.terms]
        self.average_length = sum(self.lengths) / len(self.lengths) or 1.0
        frequency = Counter(term for terms in self.terms for term in terms)
        count = len(self.terms)
        # Always positive, so any matching term gives a positive score.
        self.idf = {
            term: math.log(1 + (count - df + 0.5) / (df + 0.5))
            for term, df in frequency.items()
        }

    def score(self, query: str, candidates: list[int]) -> list[float]:
        """Return the BM25 score of each candidate for ``query``."""
        query_terms = list(dict.fromkeys(tokenize(query, self.stopwords)))
        return [self._score(query_terms, i) for i in candidates]

    def _score(self, query_terms: list[str], position: int) -> float:
        terms = self.terms[position]
        length = self.lengths[position] / self.average_length
        norm = self.k1 * (1 - self.b + self.b * length)
        return sum(
            self.idf[t] * terms[t] * (self.k1 + 1) / (terms[t] + norm)
            for t in query_terms
            if terms[t]
        )

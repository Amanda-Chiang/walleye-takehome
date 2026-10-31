"""Repair citation quotes that are almost, but not exactly, paragraph text.

Models often copy a quote nearly perfectly but drop or change one character,
e.g. the closing ``$`` of inline math. The output rules require exact
substrings, so such a quote would fail the whole run. A repaired quote is
always text taken from the paragraph itself. A quote that does not closely
match anything is left unchanged, so the output check still rejects it.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

MIN_MATCH_RATIO = 0.9


def snap_quote(quote: str, paragraph: str) -> str:
    """Return ``quote`` if it is exact, else the closest span of ``paragraph``.

    Parameters
    ----------
    quote : str
        The quote the model wrote.
    paragraph : str
        The text of the paragraph it cites.

    Returns
    -------
    str
        An exact substring of ``paragraph`` that matches at least 90% of
        ``quote``, or ``quote`` unchanged if no span matches that closely.
    """
    if not quote.strip() or quote in paragraph:
        return quote
    matcher = SequenceMatcher(None, quote, paragraph, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    if not blocks:
        return quote
    matched = sum(block.size for block in blocks)
    first, last = blocks[0], blocks[-1]
    start = max(first.b - first.a, 0)
    end = last.b + last.size + (len(quote) - last.a - last.size)
    span = paragraph[start : min(end, len(paragraph))].strip()
    if matched / len(quote) < MIN_MATCH_RATIO or not span:
        return quote
    return span


def snap_citations(citations: Any, paragraphs: dict[str, str]) -> Any:
    """Return citations with each quote repaired against its paragraph.

    Parameters
    ----------
    citations : Any
        The model's citations. Malformed values pass through unchanged, so
        the output check can report them.
    paragraphs : dict[str, str]
        Paragraph text keyed by ``paragraph_id``.

    Returns
    -------
    Any
        The citations, with repaired quotes where a repair was possible.
    """
    if not isinstance(citations, list):
        return citations
    return [_snap_one(citation, paragraphs) for citation in citations]


def _snap_one(citation: Any, paragraphs: dict[str, str]) -> Any:
    is_well_formed = (
        isinstance(citation, dict)
        and isinstance(citation.get("quote"), str)
        and citation.get("paragraph_id") in paragraphs
    )
    if not is_well_formed:
        return citation
    text = paragraphs[citation["paragraph_id"]]
    return {**citation, "quote": snap_quote(citation["quote"], text)}

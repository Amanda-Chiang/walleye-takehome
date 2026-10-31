"""The RAG pipeline: retrieve passages, generate an answer, check it.

A question can search one paper, several papers, or every paper. Evaluation
questions name one ``paper_id``, so they search one paper. The chat page and
``qasper-rag answer`` can search the whole collection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from qasper_rag.config import load_class, load_config
from qasper_rag.pipeline.generation import ChatGenerator
from qasper_rag.pipeline.retrieval import EmbeddingRetriever
from qasper_rag.storage.jsonl import read_jsonl, unique_by
from qasper_rag.validation.contracts import target_papers, validate_prediction
from qasper_rag.validation.quotes import snap_citations

Question = dict[str, Any]
Passage = dict[str, Any]

# One regeneration is allowed because model output varies between calls.
MAX_GENERATION_ATTEMPTS = 2
MAX_QUESTION_CHARS = 2_000


class Retriever(Protocol):
    """Anything that returns passages for a question, best first."""

    def retrieve(self, question: Question, top_k: int) -> list[Passage]:
        """Return up to ``top_k`` passages from ``question["paper_ids"]``."""
        ...


class Generator(Protocol):
    """Anything that writes an answer from passages."""

    def generate(
        self, question: Question, passages: list[Passage]
    ) -> dict[str, Any]:
        """Return ``answer``, ``abstain`` and ``citations``."""
        ...


class InvalidQuestionError(ValueError):
    """The question or its paper scope is invalid, so the caller can fix it."""


@dataclass
class Draft:
    """The passages and answer from one run, before the final check.

    Attributes
    ----------
    passages : list[dict]
        The passages the answer was written from, at most ``top_k``.
    answer : dict
        ``answer``, ``abstain`` and ``citations``.
    notes : list[str]
        Extra trace lines, e.g. "Retried retrieval with a new query."
    """

    passages: list[Passage]
    answer: dict[str, Any]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnswerResult:
    """A checked prediction and the steps taken to produce it."""

    prediction: dict[str, Any]
    trace: tuple[str, ...]


class RAGService:
    """Answer questions over the indexed papers.

    Parameters
    ----------
    paragraphs : dict
        Every indexed paragraph, keyed by ``paragraph_id``.
    retriever : Retriever
        Finds passages for a question.
    generator : Generator
        Writes the answer from the passages.
    top_k : int
        How many passages to give the generator.
    settings : dict, optional
        The ``service`` section of the settings file, for subclasses.

    Notes
    -----
    To change how the stages work together, e.g. to retry retrieval when the
    evidence is weak, subclass this class and override ``run``. Call
    ``self.retrieve`` and ``self.generate`` from it. ``answer`` still checks
    the scope and the output rules, so a subclass cannot skip them.
    """

    def __init__(
        self,
        *,
        paragraphs: dict[str, Passage],
        retriever: Retriever,
        generator: Generator,
        top_k: int,
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.paragraphs = paragraphs
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.settings = settings or {}
        self.paper_titles = paper_titles(paragraphs.values())
        self.paper_ids = list(self.paper_titles)

    def answer(self, question: Question) -> AnswerResult:
        """Answer one question and check the result against the output rules.

        Parameters
        ----------
        question : dict
            Holds ``question_id`` and ``question``, plus ``paper_ids`` or
            ``paper_id`` to limit the search. Without either, every paper is
            searched.

        Returns
        -------
        AnswerResult
            The prediction and a short trace of the steps.

        Raises
        ------
        InvalidQuestionError
            If the question or its paper scope is invalid.
        ValueError
            If a stage breaks the rules, e.g. the model twice writes an
            answer that fails the output check.
        """
        validate_question(question)
        started_at = time.perf_counter()
        question = {**question, "paper_ids": self.scope(question)}
        draft = self.run(question)
        # Check again here, in case a subclass built its own passage list.
        self._check_passages(draft.passages, question["paper_ids"])
        record = self.to_record(question, draft)
        record["latency_ms"] = (time.perf_counter() - started_at) * 1000
        validate_prediction(record, question, self.paragraphs)
        return AnswerResult(
            record, describe_steps(question, record, draft.notes)
        )

    def run(self, question: Question) -> Draft:
        """Retrieve passages and write an answer. Override to change the flow.

        Parameters
        ----------
        question : dict
            The question, with ``paper_ids`` already checked.

        Returns
        -------
        Draft
            The passages and the answer. ``answer`` checks it afterwards.
        """
        passages = self.retrieve(question)
        return self.generate(question, passages)

    def retrieve(self, question: Question) -> list[Passage]:
        """Return checked passages for ``question`` from the retriever.

        Pass a copy of the question with a new ``question`` text to search
        with a rewritten query.
        """
        passages = self.retriever.retrieve(question, self.top_k)
        self._check_passages(passages, question["paper_ids"])
        return passages

    def generate(self, question: Question, passages: list[Passage]) -> Draft:
        """Ask the generator for an answer that passes the output rules.

        The answer is regenerated once if it breaks the rules, because model
        output varies between calls.

        Raises
        ------
        ValueError
            If both answers break the rules.
        """
        self._check_passages(passages, question["paper_ids"])
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            answer = self.generator.generate(question, passages)
            if not isinstance(answer, dict):
                raise ValueError("generator must return an object")
            draft = Draft(passages, answer)
            try:
                record = self.to_record(question, draft)
                validate_prediction(record, question, self.paragraphs)
            except ValueError:
                if attempt == MAX_GENERATION_ATTEMPTS:
                    raise
                continue
            if attempt > 1:
                draft.notes.append(
                    "Regenerated the answer after an invalid model output "
                    f"({attempt} attempts)."
                )
            return Draft(passages, record_answer(record), draft.notes)
        raise AssertionError("unreachable")

    def to_record(self, question: Question, draft: Draft) -> dict[str, Any]:
        """Build the prediction record, repairing near-miss quotes."""
        ids = [p["paragraph_id"] for p in draft.passages]
        # Repair quotes against the stored text, never a changed copy.
        texts = {pid: self.paragraphs[pid]["text"] for pid in ids}
        return {
            "question_id": question["question_id"],
            "answer": draft.answer.get("answer"),
            "abstain": draft.answer.get("abstain"),
            "citations": snap_citations(draft.answer.get("citations"), texts),
            "retrieved_paragraph_ids": ids,
            "latency_ms": 0.0,
        }

    def ask(
        self,
        question_text: str,
        paper_ids: list[str] | None = None,
        question_id: str | None = None,
    ) -> dict[str, Any]:
        """Answer free text for the chat page and the ``answer`` command.

        Parameters
        ----------
        question_text : str
            The question, at most 2,000 characters.
        paper_ids : list[str], optional
            The papers to search. ``None`` searches every paper.
        question_id : str, optional
            An ID for the trace. One is made up if missing.

        Returns
        -------
        dict
            The prediction, plus paper titles and sections for each source.
        """
        if not isinstance(question_text, str) or not question_text.strip():
            raise InvalidQuestionError("question must not be empty")
        if len(question_text) > MAX_QUESTION_CHARS:
            raise InvalidQuestionError(
                f"question must be at most {MAX_QUESTION_CHARS:,} characters"
            )
        question: Question = {
            "question_id": question_id or f"ask-{time.time_ns()}",
            "question": question_text.strip(),
        }
        if paper_ids is not None:
            question["paper_ids"] = paper_ids
        return self._response(question, self.answer(question))

    def scope(self, question: Question) -> list[str]:
        """Return the papers to search, checking that each one is indexed."""
        if "paper_ids" not in question and "paper_id" not in question:
            return list(self.paper_ids)
        try:
            papers = target_papers(question)
        except ValueError as exc:
            raise InvalidQuestionError(str(exc)) from None
        unknown = [paper for paper in papers if paper not in self.paper_titles]
        if unknown:
            raise InvalidQuestionError(f"paper is not indexed: {unknown[0]}")
        return papers

    def _check_passages(self, passages: Any, scope: list[str]) -> None:
        if not isinstance(passages, list) or len(passages) > self.top_k:
            raise ValueError(
                f"retriever must return at most {self.top_k} passages"
            )
        if not all(isinstance(p, dict) for p in passages):
            raise ValueError("retriever returned a non-object passage")
        ids = [p.get("paragraph_id") for p in passages]
        if len(ids) != len(set(ids)):
            raise ValueError("retriever returned a duplicate paragraph")
        for passage, pid in zip(passages, ids, strict=True):
            canonical = self.paragraphs.get(pid)
            is_canonical = canonical is not None and all(
                passage.get(key) == canonical[key]
                for key in ("paper_id", "text")
            )
            if not is_canonical:
                raise ValueError(f"retriever changed or invented {pid}")
            if canonical["paper_id"] not in scope:
                raise ValueError("retriever left the requested papers")

    def _response(
        self, question: Question, result: AnswerResult
    ) -> dict[str, Any]:
        record = result.prediction
        scope = self.scope(question)
        response: dict[str, Any] = {
            **record,
            "question": question["question"],
            "answerable": not record["abstain"],
            "citations": [self._describe(c) for c in record["citations"]],
            "retrieved_sources": [
                self._describe({"paragraph_id": pid})
                for pid in record["retrieved_paragraph_ids"]
            ],
            "trace": list(result.trace),
            "paper_scope": "all papers"
            if set(scope) == set(self.paper_ids)
            else "selected papers",
            "papers_searched": len(scope),
        }
        if len(scope) == 1:
            response["paper_id"] = scope[0]
        return response

    def _describe(self, item: dict[str, Any]) -> dict[str, Any]:
        passage = self.paragraphs[item["paragraph_id"]]
        paper_id = passage["paper_id"]
        return {
            **item,
            "paper_id": paper_id,
            "paper_title": self.paper_titles.get(paper_id, paper_id),
            "section": passage["section"],
        }


def validate_question(question: Any) -> None:
    """Raise ``InvalidQuestionError`` unless the question has an ID and text."""
    if not isinstance(question, dict):
        raise InvalidQuestionError("question must be an object")
    for name in ("question_id", "question"):
        value = question.get(name)
        if not isinstance(value, str) or not value.strip():
            raise InvalidQuestionError(
                f"question requires a nonempty string {name}"
            )


def paper_titles(paragraphs: Any) -> dict[str, str]:
    """Map each paper ID to its title, or to the ID when there is no title."""
    titles: dict[str, str] = {}
    for passage in paragraphs:
        if passage.get("section") == "Title":
            titles[passage["paper_id"]] = passage["text"]
        else:
            titles.setdefault(passage["paper_id"], passage["paper_id"])
    return titles


def record_answer(record: dict[str, Any]) -> dict[str, Any]:
    """Return the ``answer``, ``abstain`` and ``citations`` of a record."""
    return {key: record[key] for key in ("answer", "abstain", "citations")}


def describe_steps(
    question: Question, record: dict[str, Any], notes: list[str]
) -> tuple[str, ...]:
    """Return the trace lines shown on the chat page."""
    papers = len(question["paper_ids"])
    count = len(record["retrieved_paragraph_ids"])
    steps = [f"Retrieved {count} passages from {papers} paper(s).", *notes]
    if record["abstain"]:
        steps.append(
            "Abstained because the supplied evidence was insufficient."
        )
    else:
        steps.append(
            f"Answered with {len(record['citations'])} source citation(s)."
        )
    return tuple(steps)


def load_service(index_dir: str | Path, config_path: str | Path) -> RAGService:
    """Build the service from an index folder and a settings file.

    The retriever is built as ``Class(paragraphs, retriever_settings,
    index_dir)`` and the generator as ``Class(generator_settings)``. The
    optional ``service`` section can name a ``RAGService`` subclass. Set
    ``class`` in any of these sections to use your own class.

    Parameters
    ----------
    index_dir : str or Path
        The folder written by ``qasper-rag index``.
    config_path : str or Path
        The settings file the index was built with.

    Returns
    -------
    RAGService
        The ready service.
    """
    config = load_config(config_path)
    paragraphs_path = Path(index_dir) / "paragraphs.jsonl"
    paragraphs = read_jsonl(paragraphs_path)
    if not paragraphs:
        raise ValueError(f"index contains no passages: {paragraphs_path}")
    retriever_class = load_class(config.retriever, EmbeddingRetriever)
    generator_class = load_class(config.generator, ChatGenerator)
    service_class = load_class(config.service, RAGService)
    return service_class(
        paragraphs=unique_by(paragraphs, "paragraph_id"),
        retriever=retriever_class(paragraphs, config.retriever, index_dir),
        generator=generator_class(config.generator),
        top_k=config.top_k,
        settings=config.service,
    )

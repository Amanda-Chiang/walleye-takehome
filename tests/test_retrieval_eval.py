import io
import json
import os
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from qasper_agent.retrieval.composed import ComposedRetriever
from qasper_rag.commands.index import build_index
from qasper_rag.pipeline.retrieval import EmbeddingRetriever
from qasper_rag.providers.embeddings import EmbeddingClient
from qasper_rag.storage.jsonl import read_jsonl, write_jsonl
from test_composed_retriever import EMBEDDING, ENV, fake_embed
from tools.retrieval_eval import (
    best_reference,
    evaluate,
    main,
    missed_paragraphs,
    recall_at,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SECTIONS = {"p10": "Results", "p20": "Method", "p30": "Results"}
PARAGRAPHS = {pid: {"paragraph_id": pid, "section": s} for pid, s in SECTIONS.items()}
WORDS = ["model accuracy", "data set", "baseline model", "accuracy data",
         "training model", "test set", "baseline accuracy", "model data"]


def gold_row(question_id, *evidence_sets):
    refs = [{"answer": "a", "abstain": False, "evidence_paragraph_ids": list(e)}
            for e in evidence_sets]
    return {"question_id": question_id, "references": refs or
            [{"answer": "Unanswerable", "abstain": True, "evidence_paragraph_ids": []}]}


class RecallTests(unittest.TestCase):
    def test_matches_score_py_best_reference_recall(self):
        self.assertEqual(recall_at(["a", "b", "c"], [["a", "x"], ["c"]], 1), 0.5)
        self.assertEqual(recall_at(["a", "b", "c"], [["a", "x"], ["c"]], 3), 1.0)


class BestReferenceTests(unittest.TestCase):
    def test_fully_retrieved_reference_leaves_nothing_missed(self):
        # One accepted reference is fully retrieved, so nothing counts as missed.
        ranked = ["p10", "p30"]
        result = evaluate({"q": ranked}, [gold_row("q", ["p10"], ["p20"])], PARAGRAPHS)
        self.assertEqual(result["recall_at_5"], 1.0)
        self.assertEqual(result["questions"][0]["best_reference"], 0)
        self.assertEqual(result["questions"][0]["missed"], [])

    def test_ties_at_5_go_to_higher_recall_at_20(self):
        ranked = ["x1", "x2", "x3", "x4", "x5", "p20"]
        self.assertEqual(best_reference(ranked, [["p10"], ["p20"]]), 1)

    def test_full_ties_go_to_the_first_reference(self):
        self.assertEqual(best_reference(["x"], [["p10"], ["p20"]]), 0)


class MissedTests(unittest.TestCase):
    def test_reports_rank_within_20_and_section(self):
        ranked = ["x1", "x2", "x3", "x4", "x5", "p30"]
        self.assertEqual(missed_paragraphs(ranked, ["p30", "p20"], PARAGRAPHS), [
            {"paragraph_id": "p30", "rank": 6, "section": "Results"},
            {"paragraph_id": "p20", "rank": None, "section": "Method"},
        ])


class EvaluateTests(unittest.TestCase):
    def test_skips_unanswerable_and_maximizes_each_depth_separately(self):
        ranked = ["p10", "x1", "x2", "x3", "x4", "p20", "p30"]
        gold = [gold_row("q", ["p10", "p30"], ["p20"]), gold_row("u")]
        result = evaluate({"q": ranked, "u": ["p10"]}, gold, PARAGRAPHS)
        self.assertEqual(result["answerable_questions"], 1)
        self.assertEqual(result["recall_at_5"], 0.5)
        self.assertEqual(result["recall_at_10"], 1.0)
        question = result["questions"][0]
        self.assertEqual(question["best_reference"], 0)
        self.assertEqual(question["missed"], [{"paragraph_id": "p30", "rank": 7, "section": "Results"}])
        self.assertEqual(question["ranking"], ranked)


class CommandTests(unittest.TestCase):
    def test_runs_on_fixtures_without_an_api(self):
        config = {"retriever": {"class": "qasper_agent.retrieval.composed:ComposedRetriever",
                                "scorers": [{"class": "test_composed_retriever:KeywordScorer"}]},
                  "generator": {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            build_index(FIXTURES / "corpus.jsonl", tmp, path)
            output = Path(tmp) / "out.json"
            with redirect_stdout(io.StringIO()):
                main(Namespace(config=str(path), index_dir=tmp,
                               questions=str(FIXTURES / "questions.jsonl"),
                               gold=str(FIXTURES / "gold.jsonl"), output=str(output)))
            result = json.loads(output.read_text())
        self.assertEqual(result["config"], str(path))
        self.assertEqual(result["answerable_questions"], 1)
        self.assertEqual(result["recall_at_5"], 1.0)


class PrefixConsistencyTests(unittest.TestCase):
    """recall@5 read from a depth-20 ranking must match a top_k=5 call."""

    paragraphs = [{"paragraph_id": f"x:p{i:05d}", "paper_id": "x", "section": "S",
                   "text": WORDS[i]} for i in range(len(WORDS))]
    questions = ["model accuracy", "baseline data set", "test", "nothing matches"]

    def assert_prefix_consistent(self, retriever):
        for text in self.questions:
            question = {"question": text, "paper_ids": ["x"]}
            with self.subTest(text=text):
                self.assertEqual(retriever.retrieve(question, 20)[:5],
                                 retriever.retrieve(question, 5))

    def test_composed_retriever(self):
        settings = {"scorers": [{"class": "test_composed_retriever:KeywordScorer"}]}
        self.assert_prefix_consistent(ComposedRetriever(self.paragraphs, settings, "unused"))

    def test_starter_embedding_retriever(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV), \
                patch.object(EmbeddingClient, "embed", fake_embed):
            corpus = Path(tmp) / "corpus.jsonl"
            write_jsonl(corpus, self.paragraphs)
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps({"retriever": {"embedding": EMBEDDING}, "generator": {}}))
            build_index(corpus, tmp, config)
            retriever = EmbeddingRetriever(read_jsonl(corpus), {"embedding": EMBEDDING}, tmp)
            self.assert_prefix_consistent(retriever)


if __name__ == "__main__":
    unittest.main()

import copy
import json
import io
import os
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from qasper_rag.validation.contracts import validate_prediction
from qasper_rag.pipeline.generation import ChatGenerator
from qasper_rag.storage.jsonl import read_jsonl, unique_by
from qasper_rag.pipeline.service import RAGService
from tools.prepare_data import convert_references
from tools.score import score

ROOT = Path(__file__).resolve().parents[1]


class StarterTests(unittest.TestCase):
    def setUp(self):
        self.corpus = read_jsonl(ROOT / "tests/fixtures/corpus.jsonl")
        self.questions = read_jsonl(ROOT / "tests/fixtures/questions.jsonl")
        self.gold = read_jsonl(ROOT / "tests/fixtures/gold.jsonl")
        self.predictions = read_jsonl(ROOT / "tests/fixtures/predictions.jsonl")
        self.paragraphs = unique_by(self.corpus, "paragraph_id")

    def test_perfect_example_and_missing_penalties(self):
        metrics = score(self.corpus, self.questions, self.gold, self.predictions)
        for key in ("answer_token_f1", "paragraph_evidence_recall_at_5", "citation_evidence_f1",
                    "abstention_precision", "abstention_recall", "abstention_f1",
                    "answerable_question_coverage"):
            self.assertEqual(metrics[key], 1.0, key)
        metrics = score(self.corpus, self.questions, self.gold, [])
        self.assertEqual(metrics["missing_outputs"], 2)
        self.assertEqual(metrics["answer_token_f1"], 0)
        self.assertEqual(metrics["citation_evidence_f1"], 0)
        self.assertEqual(metrics["abstention_recall"], 0)
        self.assertIsNone(metrics["mean_query_latency_ms"])

    def test_invalid_predictions_rejected(self):
        mutations = [
            lambda p: p.update(latency_ms=float("nan")),
            lambda p: p.update(latency_ms=True),
            lambda p: p.update(abstain="false"),
            lambda p: p.update(retrieved_paragraph_ids=["other:p00000"]),
            lambda p: p.update(retrieved_paragraph_ids=["demo:p00000"] * 2),
            lambda p: p.update(retrieved_paragraph_ids=[]),
            lambda p: p["citations"][0].update(quote="99% accuracy"),
            lambda p: p["citations"][0].update(extra="invalid"),
            lambda p: p.update(citations=p["citations"] * 2),
            lambda p: p.update(abstain=True, answer="Unanswerable"),
            lambda p: p.update(citations=[]),
        ]
        for mutate in mutations:
            prediction = copy.deepcopy(self.predictions[0])
            mutate(prediction)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_prediction(prediction, self.questions[0], self.paragraphs)
        with self.assertRaises(ValueError):
            score(self.corpus, self.questions, self.gold, self.predictions * 2)
        unknown = dict(self.predictions[0], question_id="unknown")
        with self.assertRaises(ValueError):
            score(self.corpus, self.questions, self.gold, [unknown])

    def test_best_reference_is_independent_per_metric(self):
        gold = copy.deepcopy(self.gold)
        gold[0]["references"].append({"answer": "eighty two percent", "abstain": False,
                                      "evidence_paragraph_ids": ["demo:p00001"]})
        predictions = copy.deepcopy(self.predictions)
        predictions[0]["citations"] = [{"paragraph_id": "demo:p00001", "quote": "English documents"}]
        metrics = score(self.corpus, self.questions, gold, predictions)
        self.assertEqual(metrics["answer_token_f1"], 1)
        self.assertEqual(metrics["citation_evidence_f1"], 1)

    def test_extension_interfaces_and_abstention_preserve_retrieval(self):
        class CustomRetriever:
            def retrieve(inner, question, top_k):
                return self.corpus[:1]
        class CustomGenerator:
            def generate(inner, question, passages):
                return {"answer": "Unanswerable", "abstain": True, "citations": []}
        service = RAGService(paragraphs=self.paragraphs, retriever=CustomRetriever(),
                             generator=CustomGenerator(), top_k=5)
        result = service.answer(self.questions[1]).prediction
        self.assertEqual(result["retrieved_paragraph_ids"], ["demo:p00000"])
        self.assertTrue(result["abstain"])
        self.assertGreaterEqual(result["latency_ms"], 0)

    def test_curation_excludes_visual_and_conflicting_annotations(self):
        answer = {"unanswerable": False, "evidence": ["text"], "extractive_spans": ["text"],
                  "free_form_answer": "", "yes_no": None}
        qa = {"answers": [{"answer": answer}]}
        self.assertIsNotNone(convert_references(qa, {"text": ["p"]}))
        answer["evidence"] = ["FLOAT SELECTED: table"]
        self.assertIsNone(convert_references(qa, {"text": ["p"]}))
        answer["evidence"] = ["text"]
        qa["answers"].append({"answer": {"unanswerable": True}})
        self.assertIsNone(convert_references(qa, {"text": ["p"]}))

    def test_provider_retries_and_failure_is_visible(self):
        env = {"RAG_BASE_URL": "http://localhost/v1", "RAG_MODEL": "fake", "RAG_API_KEY": "test"}
        with patch.dict(os.environ, env):
            generator = ChatGenerator({"attempts": 2})
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("url", 429, "retry", {}, None)) as request, patch("time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "retries exhausted"):
                generator.generate(self.questions[0], self.corpus[:1])
            self.assertEqual(request.call_count, 2)

    def test_provider_error_explains_access_failure_without_key(self):
        env = {"RAG_BASE_URL": "http://localhost/v1", "RAG_MODEL": "fake", "RAG_API_KEY": "secret-test-key"}
        with patch.dict(os.environ, env):
            generator = ChatGenerator({"attempts": 1})
        body = io.BytesIO(json.dumps({"error": {"message": "privacy policy blocked secret-test-key"}}).encode())
        error = urllib.error.HTTPError("url", 404, "blocked", {}, body)
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                generator.generate(self.questions[0], self.corpus[:1])
        self.assertIn("privacy policy blocked", str(caught.exception))
        self.assertNotIn("secret-test-key", str(caught.exception))

    def test_distributed_data_is_consistent(self):
        corpus = read_jsonl(ROOT / "data/corpus.jsonl")
        questions = read_jsonl(ROOT / "data/dev_questions.jsonl")
        gold = read_jsonl(ROOT / "data/dev_gold.jsonl")
        manifest = json.loads((ROOT / "data/manifest.json").read_text())
        paragraphs = unique_by(corpus, "paragraph_id")
        unique_by(questions, "question_id")
        self.assertEqual(len(paragraphs), manifest["paragraphs"])
        self.assertEqual(len(questions), manifest["questions"])
        self.assertTrue(all(set(q) == {"question_id", "paper_id", "question"} for q in questions))
        self.assertTrue(all(set(p) == {"paragraph_id", "paper_id", "section", "text"} for p in corpus))
        self.assertEqual(sum(g["references"][0]["abstain"] for g in gold), manifest["unanswerable"])
        result = score(corpus, questions, gold, [])
        self.assertEqual(result["missing_outputs"], len(questions))


if __name__ == "__main__":
    unittest.main()

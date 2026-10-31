import unittest

from qasper_rag.pipeline.service import RAGService

PASSAGES = {
    "a:p0": {"paragraph_id": "a:p0", "paper_id": "a", "section": "Title", "text": "Paper A"},
    "a:p1": {"paragraph_id": "a:p1", "paper_id": "a", "section": "Results", "text": "Model A reached 82% accuracy."},
    "b:p0": {"paragraph_id": "b:p0", "paper_id": "b", "section": "Title", "text": "Paper B"},
    "b:p1": {"paragraph_id": "b:p1", "paper_id": "b", "section": "Results", "text": "Model B reached 99% accuracy."},
}


class ScopedRetriever:
    def __init__(self):
        self.scopes = []

    def retrieve(self, question, top_k):
        self.scopes.append(question["paper_ids"])
        return [p for p in PASSAGES.values()
                if p["paper_id"] in question["paper_ids"] and p["section"] != "Title"][:top_k]


class CitingGenerator:
    def generate(self, question, passages):
        first = passages[0]
        quote = first["text"].split(" reached ")[1]
        return {"answer": quote, "abstain": False,
                "citations": [{"paragraph_id": first["paragraph_id"], "quote": quote}]}


class ServiceScopeTests(unittest.TestCase):
    def setUp(self):
        self.retriever = ScopedRetriever()
        self.service = RAGService(paragraphs=PASSAGES, retriever=self.retriever,
                                  generator=CitingGenerator(), top_k=5)

    def test_no_scope_searches_every_paper(self):
        response = self.service.ask("What accuracy was reached?")
        self.assertEqual(self.retriever.scopes, [["a", "b"]])
        self.assertEqual(response["papers_searched"], 2)
        self.assertEqual(response["paper_scope"], "all papers")
        self.assertEqual({s["paper_id"] for s in response["retrieved_sources"]}, {"a", "b"})
        self.assertEqual(response["citations"][0]["paper_title"], "Paper A")

    def test_one_paper_scope_stays_in_that_paper(self):
        response = self.service.ask("What accuracy was reached?", ["b"])
        self.assertEqual(self.retriever.scopes, [["b"]])
        self.assertEqual(response["paper_id"], "b")
        self.assertEqual(response["paper_scope"], "selected papers")
        self.assertEqual(response["retrieved_paragraph_ids"], ["b:p1"])

    def test_batch_question_with_one_paper_id_searches_one_paper(self):
        question = {"question_id": "q1", "paper_id": "a", "question": "What accuracy?"}
        record = self.service.answer(question).prediction
        self.assertEqual(self.retriever.scopes, [["a"]])
        self.assertEqual(record["retrieved_paragraph_ids"], ["a:p1"])

    def test_unknown_paper_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not indexed"):
            self.service.ask("What accuracy?", ["missing"])


class FlakyGenerator:
    """Breaks the abstention contract on the first calls, then answers correctly."""

    def __init__(self, bad_calls):
        self.bad_calls = bad_calls
        self.calls = 0

    def generate(self, question, passages):
        self.calls += 1
        if self.calls <= self.bad_calls:
            return {"answer": "No evidence", "abstain": True, "citations": []}
        return CitingGenerator().generate(question, passages)


class RegenerationTests(unittest.TestCase):
    def service(self, bad_calls):
        generator = FlakyGenerator(bad_calls)
        service = RAGService(paragraphs=PASSAGES, retriever=ScopedRetriever(),
                             generator=generator, top_k=5)
        return service, generator

    def test_one_invalid_output_is_regenerated(self):
        service, generator = self.service(bad_calls=1)
        response = service.ask("What accuracy?", ["a"])
        self.assertEqual(generator.calls, 2)
        self.assertFalse(response["abstain"])
        self.assertTrue(any("Regenerated" in line for line in response["trace"]))

    def test_repeated_invalid_output_still_fails_loudly(self):
        service, generator = self.service(bad_calls=2)
        with self.assertRaisesRegex(ValueError, "abstention requires"):
            service.ask("What accuracy?", ["a"])
        self.assertEqual(generator.calls, 2)


if __name__ == "__main__":
    unittest.main()

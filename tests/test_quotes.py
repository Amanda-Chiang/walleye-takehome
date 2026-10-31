import unittest

from qasper_rag.validation.quotes import snap_citations, snap_quote
from qasper_rag.pipeline.service import RAGService

PARAGRAPH = (
    "All experiments ran on Windows 7. We learn a LDA model with 100 topics; "
    "$\\alpha =0.01$, $\\beta = 0.01$ and using Gibbs sampling. Related words follow."
)
DROPPED_DOLLAR = (
    "We learn a LDA model with 100 topics; $\\alpha =0.01$, $\\beta = 0.01 "
    "and using Gibbs sampling."
)


class SnapQuoteTests(unittest.TestCase):
    def test_exact_quote_is_unchanged(self):
        self.assertEqual(snap_quote("Windows 7", PARAGRAPH), "Windows 7")

    def test_near_miss_becomes_exact_paragraph_text(self):
        snapped = snap_quote(DROPPED_DOLLAR, PARAGRAPH)
        self.assertIn(snapped, PARAGRAPH)
        self.assertIn("$\\beta = 0.01$", snapped)

    def test_unrelated_quote_is_left_for_validation_to_reject(self):
        quote = "We trained on eight GPUs for two days."
        self.assertEqual(snap_quote(quote, PARAGRAPH), quote)

    def test_malformed_citations_pass_through(self):
        self.assertIsNone(snap_citations(None, {}))
        self.assertEqual(snap_citations([{"paragraph_id": "x"}], {}), [{"paragraph_id": "x"}])


class ServiceRepairTests(unittest.TestCase):
    def setUp(self):
        self.passage = {"paragraph_id": "p:1", "paper_id": "p", "section": "Methods", "text": PARAGRAPH}
        self.question = {"question_id": "q1", "paper_id": "p", "question": "How is the LDA model trained?"}

    def service(self, quote):
        passage = self.passage

        class Retriever:
            def retrieve(inner, question, top_k):
                return [passage]

        class Generator:
            def generate(inner, question, passages):
                return {"answer": "Gibbs sampling", "abstain": False,
                        "citations": [{"paragraph_id": "p:1", "quote": quote}]}

        return RAGService(paragraphs={"p:1": passage}, retriever=Retriever(),
                          generator=Generator(), top_k=5)

    def test_near_miss_quote_no_longer_fails_the_run(self):
        record = self.service(DROPPED_DOLLAR).answer(self.question).prediction
        self.assertIn(record["citations"][0]["quote"], PARAGRAPH)

    def test_invented_quote_still_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "exact substring"):
            self.service("We trained on eight GPUs for two days.").answer(self.question)


if __name__ == "__main__":
    unittest.main()

import unittest

from tools.error_analysis import best_evidence, outcome, question_section


def ref(*evidence, abstain=False):
    answer = "Unanswerable" if abstain else "an answer"
    return {"answer": answer, "abstain": abstain, "evidence_paragraph_ids": list(evidence)}


def prediction(*retrieved, abstain=False):
    return {"answer": "x", "abstain": abstain, "citations": [],
            "retrieved_paragraph_ids": list(retrieved)}


class BestReferenceTests(unittest.TestCase):
    def test_alternative_reference_fully_retrieved_is_not_partial(self):
        # Reference A cites p10, reference B cites p20; retrieving p10 gives
        # score.py recall 1.0, so it must not be tagged retrieval_partial.
        refs = [ref("x:p10"), ref("x:p20")]
        self.assertEqual(outcome(prediction("x:p10"), refs), "retrieved_answered")
        self.assertEqual(best_evidence(["x:p10"], refs), {"x:p10"})

    def test_best_reference_partly_retrieved_is_partial(self):
        refs = [ref("x:p10", "x:p11"), ref("x:p20")]
        self.assertEqual(outcome(prediction("x:p10"), refs), "retrieval_partial")

    def test_no_reference_retrieved_is_a_miss(self):
        refs = [ref("x:p10"), ref("x:p20")]
        self.assertEqual(outcome(prediction("x:p30"), refs), "retrieval_miss")
        self.assertEqual(outcome(prediction("x:p10", abstain=True), refs), "over_abstain")

    def test_abstention_references_have_no_evidence(self):
        refs = [ref(abstain=True), ref(abstain=True)]
        self.assertEqual(best_evidence(["x:p10"], refs), set())
        self.assertEqual(outcome(prediction("x:p10"), refs), "missed_abstain")

    def test_report_lists_only_the_best_references_missed_evidence(self):
        question = {"paper_id": "x", "question": "q"}
        refs = [ref("x:p10", "x:p11"), ref("x:p20")]
        corpus = {pid: {"section": "S", "text": pid} for pid in ("x:p11", "x:p20")}
        lines = question_section(0, question, prediction("x:p10"), {"references": refs}, corpus)
        missed = [line for line in lines if line.startswith("- **Missed**")]
        self.assertEqual(len(missed), 1)
        self.assertIn("p11", missed[0])


if __name__ == "__main__":
    unittest.main()

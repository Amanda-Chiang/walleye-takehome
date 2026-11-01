import unittest

from qasper_agent.retrieval.base import Unit
from qasper_agent.retrieval.bm25 import STOPWORDS, BM25Scorer, tokenize
from qasper_agent.retrieval.composed import ComposedRetriever
from qasper_agent.retrieval.fusion import ReciprocalRankFusion

BM25 = {"class": "qasper_agent.retrieval.bm25:BM25Scorer"}


def fitted(*texts, **settings):
    scorer = BM25Scorer(settings, "unused")
    scorer.fit([Unit(f"p:p{i:05d}", "p", t) for i, t in enumerate(texts)])
    return scorer


def paragraphs(paper, *texts):
    return [{"paragraph_id": f"{paper}:p{i:05d}", "paper_id": paper, "section": "S", "text": t}
            for i, t in enumerate(texts)]


def ids(passages):
    return [p["paragraph_id"] for p in passages]


class FixedDense:
    """A stand-in for DenseScorer: fixed scores per unit text, no API."""

    def __init__(self, settings, index_dir):
        self.table = settings["scores"]

    def fit(self, units):
        self.units = units

    def score(self, query, candidates):
        return [self.table[self.units[i].text] for i in candidates]


class TokenizeTests(unittest.TestCase):
    def test_lowercases_splits_and_drops_plural_s(self):
        self.assertEqual(tokenize("Our Baselines: TransE, 82% F1."),
                         ["our", "baseline", "transe", "82", "f1"])

    def test_keeps_short_words_and_double_s(self):
        self.assertEqual(tokenize("is class gas"), ["is", "class", "gas"])

    def test_drops_stopwords_before_the_plural_rule(self):
        # "this" and "does" are stopwords, not "thi" and "doe".
        self.assertEqual(tokenize("Does this model use baselines?", STOPWORDS),
                         ["model", "use", "baseline"])


class BM25ScoreTests(unittest.TestCase):
    def test_matching_unit_beats_non_matching(self):
        scores = fitted("our baseline is TransE", "we use embeddings").score(
            "What baselines are used?", [0, 1])
        self.assertGreater(scores[0], scores[1])

    def test_no_overlap_scores_exactly_zero(self):
        self.assertEqual(fitted("alpha beta", "beta gamma").score("delta", [0, 1]), [0, 0])

    def test_rare_term_outweighs_common_term(self):
        scorer = fitted("model data", "model data", "model transe", "model")
        self.assertGreater(scorer.score("transe", [2])[0], scorer.score("model", [2])[0])

    def test_shorter_unit_wins_on_equal_term_count(self):
        short, long = fitted("baseline", "baseline " + "filler " * 30).score("baseline", [0, 1])
        self.assertGreater(short, long)

    def test_stopwords_setting_ignores_function_word_matches(self):
        # EXPERIMENTS.md 003: a unit matching only "the" and "they" ranked first.
        texts = ("the results they report", "our baseline is TransE")
        question = "What is the baseline they use?"
        self.assertGreater(fitted(*texts).score(question, [0])[0], 0)
        self.assertEqual(fitted(*texts, stopwords=True).score(question, [0, 1])[0], 0)
        self.assertGreater(fitted(*texts, stopwords=True).score(question, [1])[0], 0)

    def test_stopwords_are_off_by_default(self):
        self.assertEqual(BM25Scorer({}, "unused").stopwords, frozenset())

    def test_scores_exactly_the_requested_candidates(self):
        scorer = fitted("a b", "b c", "c d")
        self.assertEqual(scorer.score("c", [2, 0]), [scorer.score("c", [2])[0], 0])


class FusionTests(unittest.TestCase):
    def test_zero_scores_get_no_credit(self):
        fused = ReciprocalRankFusion({"k": 60}).fuse([[0, 0, 0, 5]])
        self.assertEqual(fused[:3], [0, 0, 0])
        self.assertEqual(fused[3], 1 / 61)

    def test_zero_score_only_loses_that_scorers_credit(self):
        fused = ReciprocalRankFusion({"k": 60}).fuse([[0, 2], [0.9, 0.1]])
        self.assertEqual(fused, [1 / 61, 1 / 61 + 1 / 62])

    def test_equal_scores_share_the_best_rank(self):
        fused = ReciprocalRankFusion({"k": 60}).fuse([[3, 3, 1]])
        self.assertEqual(fused, [1 / 61, 1 / 61, 1 / 63])


class HybridTests(unittest.TestCase):
    texts = ["We describe the data.", "Results are shown.", "Training details follow.",
             "Our baseline is TransE.", "Embedding models encode entities."]
    dense = {"class": "test_bm25:FixedDense",
             "scores": dict(zip(texts, [0.4, 0.3, 0.2, 0.5, 0.9]))}

    def hybrid(self, corpus):
        return ComposedRetriever(corpus, {"scorers": [BM25, self.dense]}, "unused")

    def test_unmatched_early_paragraphs_do_not_push_down_the_dense_favourite(self):
        # Only p3 matches BM25. Under position-based ranks, p0-p2 would earn
        # BM25 credit for coming early and overtake p4, dense's first choice.
        retriever = self.hybrid(paragraphs("a", *self.texts))
        ranked = ids(retriever.retrieve({"question": "Which TransE variant?", "paper_ids": ["a"]}, 5))
        self.assertEqual(ranked, ["a:p00003", "a:p00004", "a:p00000", "a:p00001", "a:p00002"])

    def test_searches_only_the_requested_paper(self):
        corpus = paragraphs("a", *self.texts[:3]) + paragraphs("b", *self.texts[3:])
        retriever = self.hybrid(corpus)
        passages = retriever.retrieve({"question": "Which TransE variant?", "paper_ids": ["a"]}, 5)
        self.assertEqual({p["paper_id"] for p in passages}, {"a"})


class PrefixConsistencyTests(unittest.TestCase):
    texts = [f"text {w}" for w in ("model accuracy", "data set", "baseline model", "accuracy data",
                                    "training model", "test set", "baseline accuracy", "model data")]
    questions = ["model accuracy", "baseline data set", "test", "nothing matches"]

    def assert_prefix_consistent(self, scorers):
        retriever = ComposedRetriever(paragraphs("x", *self.texts), {"scorers": scorers}, "unused")
        for text in self.questions:
            question = {"question": text, "paper_ids": ["x"]}
            with self.subTest(text=text):
                self.assertEqual(retriever.retrieve(question, 20)[:5], retriever.retrieve(question, 5))

    def test_bm25(self):
        self.assert_prefix_consistent([BM25])

    def test_hybrid(self):
        dense = {"class": "test_bm25:FixedDense",
                 "scores": {t: 1 / (i + 2) for i, t in enumerate(self.texts)}}
        self.assert_prefix_consistent([BM25, dense])


if __name__ == "__main__":
    unittest.main()

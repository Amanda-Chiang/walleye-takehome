import unittest

from tools.cross_paper_compare import cap_per_paper


def ranking(*papers):
    return [{"paragraph_id": f"{paper}:p{i:05d}", "paper_id": paper}
            for i, paper in enumerate(papers)]


def ids(passages):
    return [p["paragraph_id"] for p in passages]


class CapPerPaperTests(unittest.TestCase):
    def test_skips_a_paper_after_its_limit_and_keeps_order(self):
        passages = ranking("a", "a", "a", "b", "a", "c", "d")
        self.assertEqual(ids(cap_per_paper(passages, 2)),
                         ["a:p00000", "a:p00001", "b:p00003", "c:p00005", "d:p00006"])

    def test_diverse_ranking_is_unchanged(self):
        passages = ranking("a", "b", "c", "d", "e", "f")
        self.assertEqual(cap_per_paper(passages, 2), passages[:5])

    def test_returns_fewer_when_the_ranking_runs_out(self):
        self.assertEqual(len(cap_per_paper(ranking("a", "a", "a", "b"), 2)), 3)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from qasper_rag.commands.index import build_index

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests/fixtures/corpus.jsonl"


class PreparingRetriever:
    """Records the prepare_index call in a file in the index folder."""

    def __init__(self, paragraphs, settings, index_dir):
        pass

    @classmethod
    def prepare_index(cls, paragraphs, settings, index_dir):
        (Path(index_dir) / "prepared.json").write_text(json.dumps(
            {"count": len(paragraphs), "note": settings["note"],
             "corpus_written": (Path(index_dir) / "paragraphs.jsonl").exists()}))

    def retrieve(self, question, top_k):
        return []


class PlainRetriever:
    """A custom retriever without the hook."""

    def __init__(self, paragraphs, settings, index_dir):
        pass

    def retrieve(self, question, top_k):
        return []


class IndexHookTests(unittest.TestCase):
    def build(self, tmp, retriever):
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps({"retriever": retriever, "generator": {}}))
        return build_index(CORPUS, tmp, path)

    def test_hook_runs_once_after_the_corpus_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            count = self.build(tmp, {"class": "test_index_hook:PreparingRetriever", "note": "x"})
            prepared = json.loads((Path(tmp) / "prepared.json").read_text())
        self.assertEqual(count, 3)
        self.assertEqual(prepared, {"count": 3, "note": "x", "corpus_written": True})

    def test_retriever_without_hook_only_saves_the_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.build(tmp, {"class": "test_index_hook:PlainRetriever"})
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["config.json", "paragraphs.jsonl"])


if __name__ == "__main__":
    unittest.main()

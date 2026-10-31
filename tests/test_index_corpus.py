import json
import tempfile
import unittest
from pathlib import Path

from qasper_rag.commands.index import load_corpus


def paragraph(section):
    return {"paragraph_id": "p:1", "paper_id": "p", "section": section, "text": "Some text."}


class LoadCorpusTests(unittest.TestCase):
    def load(self, record):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.jsonl"
            path.write_text(json.dumps(record) + "\n")
            return load_corpus(path)

    def test_unnamed_section_is_allowed(self):
        self.assertEqual(self.load(paragraph(""))[0]["section"], "")

    def test_missing_section_is_rejected(self):
        record = paragraph("")
        del record["section"]
        with self.assertRaisesRegex(ValueError, "string section"):
            self.load(record)

    def test_empty_text_is_rejected(self):
        record = {**paragraph("Intro"), "text": " "}
        with self.assertRaisesRegex(ValueError, "nonempty string text"):
            self.load(record)


if __name__ == "__main__":
    unittest.main()

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from qasper_rag.config import ConfigurationError, load_class, load_config
from qasper_rag.commands.index import build_index
from qasper_rag.pipeline.service import Draft, RAGService, load_service

ROOT = Path(__file__).resolve().parents[1]


class FirstPassageRetriever:
    """A custom retriever: the first passage of each requested paper."""

    def __init__(self, paragraphs, settings, index_dir):
        self.paragraphs = paragraphs
        self.settings = settings

    def retrieve(self, question, top_k):
        return [p for p in self.paragraphs if p["paper_id"] in question["paper_ids"]][:1]


class QuoteGenerator:
    """A custom generator: quotes the first passage, with no API call."""

    def __init__(self, settings):
        self.settings = settings

    def generate(self, question, passages):
        first = passages[0]
        return {"answer": first["text"], "abstain": False,
                "citations": [{"paragraph_id": first["paragraph_id"], "quote": first["text"]}]}


class ExtensionTests(unittest.TestCase):
    def write_config(self, tmp, retriever_class, generator_class):
        config = {"retriever": {"class": retriever_class, "note": "custom"},
                  "generator": {"class": generator_class}}
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(config))
        return path

    def test_custom_classes_load_from_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.write_config(tmp, "test_extension:FirstPassageRetriever",
                                       "test_extension:QuoteGenerator")
            build_index(ROOT / "tests/fixtures/corpus.jsonl", tmp, config)
            service = load_service(tmp, config)
            self.assertEqual(service.retriever.settings["note"], "custom")
            response = service.ask("Anything?")
        self.assertEqual(response["retrieved_paragraph_ids"], ["demo:p00000"])
        self.assertFalse(response["abstain"])

    def test_bad_class_name_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("no_colon", "missing_module:Thing", "test_extension:Missing"):
                config = self.write_config(tmp, name, "test_extension:QuoteGenerator")
                with self.subTest(name=name), self.assertRaises(ConfigurationError):
                    build_index(ROOT / "tests/fixtures/corpus.jsonl", tmp, config)
                    load_service(tmp, config)

    def test_builtin_retriever_needs_embedding_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"retriever": {}, "generator": {}}))
            with self.assertRaisesRegex(ConfigurationError, "retriever.embedding"):
                load_config(path)


class AbstainOnceGenerator:
    """Abstains on the first call, then quotes the first passage."""

    def __init__(self, settings):
        self.calls = 0

    def generate(self, question, passages):
        self.calls += 1
        if self.calls == 1:
            return {"answer": "Unanswerable", "abstain": True, "citations": []}
        return QuoteGenerator({}).generate(question, passages)


class RetryOnAbstainService(RAGService):
    """A custom service: search again once if the first answer abstains."""

    def run(self, question):
        draft = self.generate(question, self.retrieve(question))
        if not draft.answer["abstain"]:
            return draft
        suffix = self.settings.get("retry_suffix", "")
        broader = {**question, "question": f"{question['question']} {suffix}"}
        retry = self.generate(question, self.retrieve(broader))
        retry.notes.insert(0, "Retried retrieval with a broader query.")
        return retry


class ServiceHookTests(unittest.TestCase):
    def test_service_class_loads_from_settings_and_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "retriever": {"class": "test_extension:FirstPassageRetriever"},
                "generator": {"class": "test_extension:AbstainOnceGenerator"},
                "service": {"class": "test_extension:RetryOnAbstainService",
                            "retry_suffix": "more words"},
            }
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            build_index(ROOT / "tests/fixtures/corpus.jsonl", tmp, path)
            service = load_service(tmp, path)
            self.assertIsInstance(service, RetryOnAbstainService)
            self.assertEqual(service.settings["retry_suffix"], "more words")
            response = service.ask("Anything?")
        self.assertFalse(response["abstain"])
        self.assertEqual(service.generator.calls, 2)
        self.assertIn("Retried retrieval with a broader query.", response["trace"])

    def test_subclass_output_is_still_checked(self):
        class BrokenService(RetryOnAbstainService):
            def run(self, question):
                draft = super().run(question)
                draft.answer = {"answer": "made up", "abstain": False, "citations": []}
                return draft

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "retriever": {"class": "test_extension:FirstPassageRetriever"},
                "generator": {"class": "test_extension:QuoteGenerator"}}))
            build_index(ROOT / "tests/fixtures/corpus.jsonl", tmp, path)
            base = load_service(tmp, path)
        service = BrokenService(paragraphs=base.paragraphs, retriever=base.retriever,
                                generator=base.generator, top_k=5)
        with self.assertRaisesRegex(ValueError, "requires citations"):
            service.ask("Anything?")


    def test_subclass_passages_are_checked(self):
        class InventingService(RetryOnAbstainService):
            def run(self, question):
                passage = dict(self.retrieve(question)[0], text="changed text")
                return self.generate(question, [passage])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "top_k": 1,
                "retriever": {"class": "test_extension:FirstPassageRetriever"},
                "generator": {"class": "test_extension:QuoteGenerator"}}))
            build_index(ROOT / "tests/fixtures/corpus.jsonl", tmp, path)
            base = load_service(tmp, path)
        service = InventingService(paragraphs=base.paragraphs, retriever=base.retriever,
                                   generator=base.generator, top_k=1)
        with self.assertRaisesRegex(ValueError, "changed or invented"):
            service.ask("Anything?")


class ImportPathTests(unittest.TestCase):
    def test_class_in_current_folder_loads_without_python_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "my_cwd_rag"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "stage.py").write_text("class Stage:\n    pass\n")
            old_cwd, old_path = os.getcwd(), list(sys.path)
            try:
                os.chdir(tmp)
                sys.path[:] = [p for p in sys.path if p not in ("", tmp, os.path.realpath(tmp))]
                loaded = load_class({"class": "my_cwd_rag.stage:Stage"}, object)
            finally:
                os.chdir(old_cwd)
                sys.path[:] = old_path
                sys.modules.pop("my_cwd_rag.stage", None)
                sys.modules.pop("my_cwd_rag", None)
        self.assertEqual(loaded.__name__, "Stage")


if __name__ == "__main__":
    unittest.main()

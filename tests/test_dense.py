import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from qasper_rag.providers.embeddings import EmbeddingClient, normalize
from qasper_rag.storage.jsonl import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
ENV = {"RAG_API_KEY": "fixture", "RAG_MODEL": "fixture-chat",
       "RAG_EMBEDDING_MODEL": "fixture-embedding"}


class FakeProvider(BaseHTTPRequestHandler):
    """Answers embedding and chat requests the way the real API would."""

    calls = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append((self.path, body))
        if self.path.endswith("/embeddings"):
            size = body.get("dimensions", 256)
            result = {"data": [{"index": i, "embedding": [1.0] + [0.0] * (size - 1)}
                               for i in range(len(body["input"]))]}
        else:
            prompt = json.loads(body["messages"][0]["content"].split("\n", 1)[1])
            if "accuracy" in prompt["question"]:
                answer = {"answer": "82%", "abstain": False,
                          "citations": [{"paragraph_id": "demo:p00000", "quote": "82% accuracy"}]}
            else:
                answer = {"answer": "Unanswerable", "abstain": True, "citations": []}
            result = {"choices": [{"message": {"content": json.dumps(answer)}}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def log_message(self, *args):
        pass


class EmbeddingTests(unittest.TestCase):
    def test_vectors_validate_and_response_order_is_respected(self):
        with patch.dict(os.environ, ENV):
            embedder = EmbeddingClient({"dimensions": 2})
        with patch.object(embedder.client, "post", return_value={"data": [
                {"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 0]}]}):
            self.assertEqual(embedder.embed(["a", "b"]), [[1, 0], [0, 1]])
        for vector in ([0, 0], [float("nan"), 1], [1], [True, 1]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                normalize(vector, 2)
        with patch.object(embedder.client, "post", return_value={"data": [
                {"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [1, 0]}]}):
            with self.assertRaises(ValueError):
                embedder.embed(["a", "b"])

    def test_cache_reuses_embeddings_and_separates_models_and_text(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV):
            embedder = EmbeddingClient({"dimensions": 2, "max_input_chars": 10})
            cache = Path(tmp) / "cache.sqlite"
            with patch.object(embedder, "embed", return_value=[[1, 0]]) as call:
                self.assertEqual(embedder.embed_cached(["text", "text"], cache), [[1, 0], [1, 0]])
                self.assertEqual(embedder.embed_cached(["text"], cache), [[1, 0]])
                self.assertEqual(call.call_count, 1)
                embedder.client.model = "new-model"
                embedder.embed_cached(["text"], cache)
                self.assertEqual(call.call_count, 2)
                embedder.embed_cached(["changed"], cache)
                self.assertEqual(call.call_count, 3)


class EndToEndTests(unittest.TestCase):
    """Index, predict and score through the real commands and a fake provider."""

    def setUp(self):
        FakeProvider.calls = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeProvider)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.env = dict(os.environ, **ENV,
                        RAG_BASE_URL=f"http://127.0.0.1:{self.server.server_port}/v1")

    def run_command(self, *args, check=True):
        return subprocess.run([sys.executable, *args], cwd=ROOT, env=self.env,
                              capture_output=True, text=True, check=check)

    def count(self, endpoint):
        return sum(path.endswith(endpoint) for path, _ in FakeProvider.calls)

    def test_index_predict_and_score(self):
        index = ("-m", "qasper_rag.commands.index", "--corpus", "tests/fixtures/corpus.jsonl",
                 "--index-dir", self.tmp, "--config", "config/baseline.json")
        self.run_command(*index)
        self.run_command(*index)
        self.assertEqual(self.count("/embeddings"), 1, "the second build reuses the cache")

        output = str(Path(self.tmp) / "predictions.jsonl")
        predict = ("-m", "qasper_rag.commands.predict", "--index-dir", self.tmp,
                   "--questions", "tests/fixtures/questions.jsonl",
                   "--config", "config/baseline.json", "--output", output)
        self.assertEqual(self.run_command(*predict).stdout, "")
        predictions = read_jsonl(output)
        self.assertEqual(predictions[0]["answer"], "82%")
        self.assertTrue(predictions[1]["abstain"])
        self.assertEqual(self.count("/chat/completions"), 2)
        self.assertNotIn("references", json.dumps(FakeProvider.calls))

        metrics = str(Path(self.tmp) / "metrics.json")
        self.run_command("tools/score.py", "--corpus", "tests/fixtures/corpus.jsonl",
                         "--questions", "tests/fixtures/questions.jsonl",
                         "--gold", "tests/fixtures/gold.jsonl",
                         "--predictions", output, "--output", metrics)
        self.assertEqual(json.loads(Path(metrics).read_text())["answer_token_f1"], 1)

    def test_changed_embedding_settings_need_a_new_index(self):
        self.run_command("-m", "qasper_rag.commands.index", "--corpus", "tests/fixtures/corpus.jsonl",
                         "--index-dir", self.tmp, "--config", "config/baseline.json")
        metadata = Path(self.tmp) / "embedding_metadata.json"
        data = json.loads(metadata.read_text())
        data["embedding"]["model"] = "incompatible-model"
        metadata.write_text(json.dumps(data))
        result = self.run_command(
            "-m", "qasper_rag.commands.predict", "--index-dir", self.tmp,
            "--questions", "tests/fixtures/questions.jsonl", "--config", "config/baseline.json",
            "--output", str(Path(self.tmp) / "p.jsonl"), check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rebuild the index", result.stderr)


if __name__ == "__main__":
    unittest.main()

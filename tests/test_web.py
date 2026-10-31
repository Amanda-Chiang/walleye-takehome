import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from qasper_rag.pipeline.service import RAGService
from qasper_rag.web.server import RAGRequestHandler
from test_service import PASSAGES, CitingGenerator, ScopedRetriever


class ChatEndpointTests(unittest.TestCase):
    def setUp(self):
        service = RAGService(paragraphs=PASSAGES, retriever=ScopedRetriever(),
                             generator=CitingGenerator(), top_k=5)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RAGRequestHandler)
        self.server.rag_service = service
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, body):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.server.server_port}/api/chat",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            with error:
                return error.code, json.load(error)

    def test_answer_for_one_paper(self):
        status, body = self.post({"question": "What accuracy?", "paper_id": "b"})
        self.assertEqual(status, 200)
        self.assertEqual(body["paper_id"], "b")

    def test_service_rules_become_client_errors(self):
        for request, message in [
            ({"question": "What accuracy?", "paper_id": "missing"}, "paper is not indexed"),
            ({"question": "   "}, "must not be empty"),
            ({"question": "x" * 2_001}, "at most 2,000 characters"),
        ]:
            with self.subTest(request=request):
                status, body = self.post(request)
                self.assertEqual(status, 400)
                self.assertIn(message, body["error"])


if __name__ == "__main__":
    unittest.main()

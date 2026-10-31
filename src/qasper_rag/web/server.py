"""Serve the local chat page.

Run as ``qasper-rag serve`` or ``python -m qasper_rag.web.server``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from qasper_rag.cli import run_cli
from qasper_rag.pipeline.service import (
    InvalidQuestionError,
    RAGService,
    load_service,
)

WEB_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 16_384
logger = logging.getLogger(__name__)


class ClientError(Exception):
    """A bad request, reported to the browser as HTTP 400 or 413."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add this command's options to ``parser``."""
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--config", default="config/baseline.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)


def execute(args: argparse.Namespace) -> int:
    """Run the server until interrupted."""
    run_server(args.index_dir, args.config, args.host, args.port)
    return 0


def parse_chat_request(request: Any) -> tuple[Any, list[str] | None]:
    """Return the question text and the papers to search.

    Parameters
    ----------
    request : Any
        The decoded JSON body, with ``question`` and optionally ``scope``
        (``"all"`` or ``"paper"``) and ``paper_id``.

    Returns
    -------
    tuple
        The question text, and ``[paper_id]`` or ``None`` for every paper.

    Raises
    ------
    ClientError
        If the body is not an object or the scope is invalid. The service
        checks the question text and the paper ID.
    """
    if not isinstance(request, dict):
        raise ClientError("Request body must be a JSON object.")
    paper_id = request.get("paper_id")
    scope = request.get("scope", "paper" if paper_id is not None else "all")
    if scope not in ("all", "paper"):
        raise ClientError("scope must be 'all' or 'paper'.")
    if scope == "paper" and not isinstance(paper_id, str):
        raise ClientError("Choose a paper from the list.")
    return request.get("question"), [paper_id] if scope == "paper" else None


class RAGRequestHandler(BaseHTTPRequestHandler):
    """Serve the page, the paper list and the chat endpoint."""

    server_version = "QASPER-RAG"

    def log_message(self, format: str, *args: Any) -> None:
        """Log the request line only, never questions or keys."""
        stamp = self.log_date_time_string()
        print(f"{stamp} {self.client_address[0]} {args[0]}", file=sys.stderr)

    @property
    def service(self) -> RAGService:
        """The service shared by every request thread."""
        return self.server.rag_service  # type: ignore[attr-defined]

    def send_body(self, status: int, payload: bytes, content_type: str) -> None:
        """Send a complete response with headers that stop caching."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, status: int, body: Any) -> None:
        """Send ``body`` as JSON."""
        payload = json.dumps(body, ensure_ascii=False, allow_nan=False)
        content_type = "application/json; charset=utf-8"
        self.send_body(status, payload.encode("utf-8"), content_type)

    def do_GET(self) -> None:
        """Serve the chat page and the paper list."""
        if self.path in ("/", "/index.html"):
            page = (WEB_DIR / "index.html").read_bytes()
            self.send_body(200, page, "text/html; charset=utf-8")
        elif self.path == "/api/papers":
            papers = [
                {"paper_id": paper_id, "title": title}
                for paper_id, title in self.service.paper_titles.items()
            ]
            papers.sort(key=lambda p: (p["title"].casefold(), p["paper_id"]))
            self.send_json(200, {"papers": papers})
        else:
            self.send_json(404, {"error": "Route not found."})

    def do_POST(self) -> None:
        """Answer one chat question."""
        if self.path != "/api/chat":
            self.send_json(404, {"error": "Route not found."})
            return
        try:
            question, paper_ids = parse_chat_request(self.read_json())
            question_id = f"chat-{uuid.uuid4().hex}"
            result = self.service.ask(question, paper_ids, question_id)
        except ClientError as exc:
            self.send_json(exc.status, {"error": str(exc)})
        except InvalidQuestionError as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            logger.error("RAG request failed: %s", exc)
            message = "The RAG request failed. Check the server log."
            self.send_json(502, {"error": message})
        else:
            self.send_json(200, result)

    def read_json(self) -> Any:
        """Read and decode the request body."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ClientError("Content-Length must be a number.") from None
        if not 1 <= length <= MAX_BODY_BYTES:
            raise ClientError(
                f"Request must be 1 to {MAX_BODY_BYTES} bytes.", status=413
            )
        try:
            return json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            raise ClientError("Request body must be valid JSON.") from None


def run_server(
    index_dir: str | Path,
    config_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Load the index and serve the chat page until interrupted.

    Parameters
    ----------
    index_dir : str or Path
        The folder written by ``qasper-rag index``.
    config_path : str or Path
        The settings file the index was built with.
    host : str
        The address to listen on.
    port : int
        The port to listen on. ``0`` picks a free port.
    """
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    service = load_service(index_dir, config_path)
    server = ThreadingHTTPServer((host, port), RAGRequestHandler)
    server.rag_service = service  # type: ignore[attr-defined]
    logger.info("Chat page at http://%s:%d/", host, server.server_port)
    logger.info(
        "Loaded %d paragraphs across %d papers",
        len(service.paragraphs),
        len(service.paper_titles),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping the chat server")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    """Run ``python -m qasper_rag.web.server``."""
    return run_cli(__doc__, add_arguments, execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())

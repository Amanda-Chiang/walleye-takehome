"""Send JSON requests to an OpenRouter-compatible API, with retries."""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Some Python installs ship without CA roots, so these are tried in order.
CA_BUNDLES = (
    "/etc/ssl/cert.pem",
    "/opt/homebrew/etc/ca-certificates/cert.pem",
    "/usr/local/etc/ca-certificates/cert.pem",
    "/etc/ssl/certs/ca-certificates.crt",
)
MAX_ERROR_CHARS = 600


def tls_context() -> ssl.SSLContext:
    """Return a verifying TLS context, loading a system CA bundle if needed.

    Returns
    -------
    ssl.SSLContext
        The default context, with a CA bundle added when Python has none.
    """
    context = ssl.create_default_context()
    has_roots = context.cert_store_stats()["x509_ca"] > 0
    has_override = os.environ.get("SSL_CERT_FILE") or os.environ.get(
        "SSL_CERT_DIR"
    )
    if has_roots or has_override:
        return context
    bundle = next((path for path in CA_BUNDLES if Path(path).is_file()), None)
    if bundle is not None:
        context.load_verify_locations(cafile=bundle)
    return context


def is_retryable(status: int) -> bool:
    """Return True for rate limits and server errors."""
    return status == 429 or 500 <= status < 600


class APIClient:
    """POST JSON to one API, reading the key and model from the environment.

    Parameters
    ----------
    config : dict
        Settings for one stage. Uses ``model``, ``base_url``, ``attempts``
        and ``timeout_s``.
    model_env : str
        Environment variable that overrides ``config["model"]``.
    """

    def __init__(
        self, config: dict[str, Any], model_env: str = "RAG_MODEL"
    ) -> None:
        base_url = config.get("base_url", DEFAULT_BASE_URL)
        self.base_url = os.environ.get("RAG_BASE_URL", base_url).rstrip("/")
        self.model = os.environ.get(model_env) or config.get("model")
        self.api_key = os.environ.get("RAG_API_KEY") or os.environ.get(
            "OPENROUTER_API_KEY"
        )
        if not (self.base_url and self.model and self.api_key):
            raise ValueError(
                "set OPENROUTER_API_KEY (or RAG_API_KEY) and a model"
            )
        self.attempts = int(config.get("attempts", 3))
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")
        self.timeout_s = config.get("timeout_s", 60)
        self.tls = tls_context()

    def post(self, endpoint: str, payload: dict[str, Any]) -> Any:
        """POST ``payload`` and return the decoded JSON response.

        Rate limits, server errors and connection failures are retried with
        a growing pause. Other HTTP errors fail at once.

        Parameters
        ----------
        endpoint : str
            Path after the base URL, e.g. ``/chat/completions``.
        payload : dict
            The JSON body.

        Returns
        -------
        Any
            The decoded response body.

        Raises
        ------
        RuntimeError
            If the request fails. The API key never appears in the message.
        """
        request = self._request(endpoint, payload)
        for attempt in range(1, self.attempts + 1):
            try:
                return self._send(request)
            except urllib.error.HTTPError as exc:
                if not is_retryable(exc.code) or attempt == self.attempts:
                    raise self._http_error(endpoint, exc) from None
            except (urllib.error.URLError, TimeoutError) as exc:
                reason = getattr(exc, "reason", None)
                if isinstance(reason, ssl.SSLCertVerificationError):
                    raise RuntimeError(
                        "TLS verification failed; set SSL_CERT_FILE"
                    ) from None
                if attempt == self.attempts:
                    raise RuntimeError(
                        f"{endpoint}: connection failed; retries exhausted"
                    ) from None
            time.sleep(min(2 ** (attempt - 1), 8))
        raise AssertionError("unreachable")

    def _request(
        self, endpoint: str, payload: dict[str, Any]
    ) -> urllib.request.Request:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = json.dumps(payload).encode()
        return urllib.request.Request(
            self.base_url + endpoint, data=body, headers=headers
        )

    def _send(self, request: urllib.request.Request) -> Any:
        with urllib.request.urlopen(
            request, timeout=self.timeout_s, context=self.tls
        ) as response:
            return json.load(response)

    def _http_error(
        self, endpoint: str, exc: urllib.error.HTTPError
    ) -> RuntimeError:
        detail = self._provider_message(exc)
        retried = "; retries exhausted" if is_retryable(exc.code) else ""
        return RuntimeError(
            f"{endpoint}: provider HTTP {exc.code}{retried}{detail}"
        )

    def _provider_message(self, exc: urllib.error.HTTPError) -> str:
        try:
            message = json.loads(exc.read())["error"]["message"]
        except (ValueError, KeyError, TypeError, AttributeError):
            return ""
        if not isinstance(message, str):
            return ""
        redacted = message.replace(self.api_key, "[redacted]")
        return f": {redacted[:MAX_ERROR_CHARS]}"

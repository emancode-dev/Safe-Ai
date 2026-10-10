"""Local Ollama connectivity and model invocation.

This client talks ONLY to a local Ollama HTTP endpoint. It never calls a
hosted or remote inference provider. It is responsible for:

- Reading ``OLLAMA_BASE_URL`` and ``OLLAMA_MODEL`` from the environment.
- Checking that the local service is reachable.
- Checking that the configured model is installed.
- Warning when the configured endpoint is not a loopback address.
- Invoking the model with a bounded timeout and returning the raw text.

It deliberately does not log user input and never puts raw user text into
exception messages.
"""

from __future__ import annotations

import os
import json
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import requests

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:1.5b"
DEFAULT_TIMEOUT = 120  # seconds (a cold 1.5B model load can take a while)

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class OllamaStatus(str, Enum):
    """High-level outcome categories, surfaced to the UI."""

    OK = "ok"
    SERVICE_UNAVAILABLE = "service_unavailable"
    MODEL_MISSING = "model_missing"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    INVALID_RESPONSE = "invalid_response"


class OllamaError(Exception):
    """Error raised for Ollama interaction failures.

    Carries a machine-readable :class:`OllamaStatus` and a human-safe message
    that never contains the user's submitted text.
    """

    def __init__(self, status: OllamaStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class OllamaConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: int = DEFAULT_TIMEOUT

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        timeout_raw = os.getenv("OLLAMA_TIMEOUT", str(DEFAULT_TIMEOUT)).strip()
        try:
            timeout = max(1, int(timeout_raw))
        except ValueError:
            timeout = DEFAULT_TIMEOUT
        return cls(base_url=base_url.rstrip("/"), model=model, timeout=timeout)

    def is_loopback(self) -> bool:
        """True if the configured endpoint targets the local machine."""
        try:
            host = urllib.parse.urlparse(self.base_url).hostname or ""
        except ValueError:
            return False
        return host.lower() in _LOOPBACK_HOSTS


class OllamaClient:
    """Thin wrapper over the local Ollama HTTP API."""

    def __init__(self, config: Optional[OllamaConfig] = None, session: Optional[requests.Session] = None):
        self.config = config or OllamaConfig.from_env()
        self._session = session or requests.Session()

    # -- connectivity -------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the local Ollama service responds to /api/tags."""
        try:
            resp = self._session.get(f"{self.config.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list:
        """Return the list of installed model tags, or raise OllamaError."""
        try:
            resp = self._session.get(f"{self.config.base_url}/api/tags", timeout=5)
        except requests.Timeout as exc:
            raise OllamaError(OllamaStatus.TIMEOUT, "Timed out contacting the local Ollama service.") from exc
        except requests.RequestException as exc:
            raise OllamaError(
                OllamaStatus.SERVICE_UNAVAILABLE,
                "Could not reach the local Ollama service.",
            ) from exc
        if resp.status_code != 200:
            raise OllamaError(OllamaStatus.SERVER_ERROR, f"Ollama returned HTTP {resp.status_code}.")
        try:
            data = resp.json()
        except ValueError as exc:
            raise OllamaError(OllamaStatus.INVALID_RESPONSE, "Ollama returned malformed JSON for model list.") from exc
        return [m.get("name", "") for m in data.get("models", [])]

    def model_installed(self) -> bool:
        """Return True if the configured model tag is installed locally."""
        try:
            tags = self.list_models()
        except OllamaError:
            return False
        if self.config.model in tags:
            return True
        # Accept a match on the base name (e.g. "qwen2.5:1.5b" vs
        # "qwen2.5:1.5b-instruct" installed, or ":latest" variants).
        base = self.config.model.split(":")[0]
        return any(t.split(":")[0] == base for t in tags)

    # -- generation ---------------------------------------------------------

    def generate(self, system_prompt: str, user_content: str, *, as_json: bool = True) -> str:
        """Invoke the model and return the raw response text.

        Uses /api/chat with a low temperature for deterministic output. Raises
        :class:`OllamaError` with an appropriate status on any failure.
        """
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 1024},
        }
        if as_json:
            payload["format"] = "json"

        try:
            resp = self._session.post(
                f"{self.config.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout,
            )
        except requests.Timeout as exc:
            raise OllamaError(OllamaStatus.TIMEOUT, "The model request timed out.") from exc
        except requests.RequestException as exc:
            raise OllamaError(
                OllamaStatus.SERVICE_UNAVAILABLE,
                "Could not reach the local Ollama service.",
            ) from exc

        if resp.status_code == 404:
            raise OllamaError(
                OllamaStatus.MODEL_MISSING,
                "The configured model is not installed in Ollama.",
            )
        if resp.status_code != 200:
            raise OllamaError(OllamaStatus.SERVER_ERROR, f"Ollama returned HTTP {resp.status_code}.")

        try:
            data = resp.json()
        except ValueError as exc:
            raise OllamaError(OllamaStatus.INVALID_RESPONSE, "Ollama returned malformed JSON.") from exc

        content = data.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise OllamaError(OllamaStatus.INVALID_RESPONSE, "Ollama returned an empty model message.")
        return content

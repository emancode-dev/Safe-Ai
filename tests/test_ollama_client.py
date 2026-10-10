"""Tests for the Ollama client. No live server required; HTTP is mocked."""

import json

import pytest
import requests

from src.ollama_client import (
    OllamaClient,
    OllamaConfig,
    OllamaError,
    OllamaStatus,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, raise_json=False):
        self.status_code = status_code
        self._payload = payload or {}
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("bad json")
        return self._payload


class FakeSession:
    """Configurable fake requests.Session."""

    def __init__(self, get_handler=None, post_handler=None):
        self._get = get_handler
        self._post = post_handler

    def get(self, url, **kwargs):
        if self._get is None:
            raise requests.ConnectionError("no server")
        return self._get(url, **kwargs)

    def post(self, url, **kwargs):
        if self._post is None:
            raise requests.ConnectionError("no server")
        return self._post(url, **kwargs)


def _client(session):
    return OllamaClient(OllamaConfig(base_url="http://localhost:11434", model="qwen2.5:1.5b"), session=session)


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:1234/")
    monkeypatch.setenv("OLLAMA_MODEL", "mymodel:tag")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "30")
    cfg = OllamaConfig.from_env()
    assert cfg.base_url == "http://localhost:1234"
    assert cfg.model == "mymodel:tag"
    assert cfg.timeout == 30


def test_is_loopback_true():
    assert OllamaConfig(base_url="http://localhost:11434").is_loopback()
    assert OllamaConfig(base_url="http://127.0.0.1:11434").is_loopback()


def test_is_loopback_false():
    assert not OllamaConfig(base_url="http://10.1.2.3:11434").is_loopback()


def test_is_available_true():
    session = FakeSession(get_handler=lambda url, **k: FakeResponse(200, {"models": []}))
    assert _client(session).is_available() is True


def test_is_available_false_on_connection_error():
    assert _client(FakeSession()).is_available() is False


def test_model_installed_exact_match():
    payload = {"models": [{"name": "qwen2.5:1.5b"}]}
    session = FakeSession(get_handler=lambda url, **k: FakeResponse(200, payload))
    assert _client(session).model_installed() is True


def test_model_installed_base_name_match():
    payload = {"models": [{"name": "qwen2.5:1.5b-instruct"}]}
    session = FakeSession(get_handler=lambda url, **k: FakeResponse(200, payload))
    assert _client(session).model_installed() is True


def test_model_installed_false_when_absent():
    payload = {"models": [{"name": "llama3:8b"}]}
    session = FakeSession(get_handler=lambda url, **k: FakeResponse(200, payload))
    assert _client(session).model_installed() is False


def test_generate_success():
    content = json.dumps({"findings": [], "summary": "ok"})
    payload = {"message": {"content": content}}
    session = FakeSession(post_handler=lambda url, **k: FakeResponse(200, payload))
    out = _client(session).generate("sys", "user")
    assert "summary" in out


def test_generate_service_unavailable():
    session = FakeSession(post_handler=None)  # raises ConnectionError
    with pytest.raises(OllamaError) as exc:
        _client(session).generate("sys", "user")
    assert exc.value.status == OllamaStatus.SERVICE_UNAVAILABLE


def test_generate_model_missing():
    session = FakeSession(post_handler=lambda url, **k: FakeResponse(404))
    with pytest.raises(OllamaError) as exc:
        _client(session).generate("sys", "user")
    assert exc.value.status == OllamaStatus.MODEL_MISSING


def test_generate_timeout():
    def handler(url, **k):
        raise requests.Timeout("slow")
    session = FakeSession(post_handler=handler)
    with pytest.raises(OllamaError) as exc:
        _client(session).generate("sys", "user")
    assert exc.value.status == OllamaStatus.TIMEOUT


def test_generate_server_error():
    session = FakeSession(post_handler=lambda url, **k: FakeResponse(500))
    with pytest.raises(OllamaError) as exc:
        _client(session).generate("sys", "user")
    assert exc.value.status == OllamaStatus.SERVER_ERROR


def test_generate_malformed_json():
    session = FakeSession(post_handler=lambda url, **k: FakeResponse(200, raise_json=True))
    with pytest.raises(OllamaError) as exc:
        _client(session).generate("sys", "user")
    assert exc.value.status == OllamaStatus.INVALID_RESPONSE


def test_generate_empty_content():
    payload = {"message": {"content": "   "}}
    session = FakeSession(post_handler=lambda url, **k: FakeResponse(200, payload))
    with pytest.raises(OllamaError) as exc:
        _client(session).generate("sys", "user")
    assert exc.value.status == OllamaStatus.INVALID_RESPONSE


def test_error_message_contains_no_user_text():
    """Regression: error messages must never echo submitted content."""
    session = FakeSession(post_handler=lambda url, **k: FakeResponse(500))
    secret = "SECRET_USER_PAYLOAD_12345"
    try:
        _client(session).generate("sys", secret)
    except OllamaError as exc:
        assert secret not in exc.message

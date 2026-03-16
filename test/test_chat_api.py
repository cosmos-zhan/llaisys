import sys


def _has_optional_deps() -> bool:
    try:
        import fastapi  # noqa: F401
        import httpx  # noqa: F401
    except ImportError:
        return False
    return True


if not _has_optional_deps():
    print("Skipped: optional server dependencies are not installed.")
    sys.exit(0)

from fastapi.testclient import TestClient

from chat_test_utils import FakeModel, FakeTokenizer
from llaisys.chat_server import create_app


def test_non_stream_response() -> None:
    app = create_app(tokenizer=FakeTokenizer(), model=FakeModel(), model_name="fake-qwen")
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "top_k": 10,
            "top_p": 0.9,
            "temperature": 0.8,
        },
    )
    response.raise_for_status()
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "fake-qwen"
    assert payload["session_id"].startswith("session-")
    assert payload["cache_reused_tokens"] == 0
    assert payload["choices"][0]["message"]["content"] == "Hello!"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"]["completion_tokens"] == 6


def test_stream_response() -> None:
    app = create_app(tokenizer=FakeTokenizer(), model=FakeModel(), model_name="fake-qwen")
    client = TestClient(app)
    chunks = []
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "top_k": 10,
            "top_p": 0.9,
            "temperature": 0.8,
            "stream": True,
        },
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                chunks.append(line)

    assert chunks[-1] == "data: [DONE]"
    content_parts = []
    for line in chunks[:-1]:
        if not line.startswith("data: "):
            continue
        event = line[6:]
        if event == "[DONE]":
            continue
        payload = __import__("json").loads(event)
        delta = payload["choices"][0]["delta"]
        if "content" in delta:
            content_parts.append(delta["content"])
    assert "".join(content_parts) == "Hello!"


def test_session_reuse_response() -> None:
    model = FakeModel()
    app = create_app(tokenizer=FakeTokenizer(), model=model, model_name="fake-qwen")
    client = TestClient(app)
    session_id = "session-reuse"

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "session_id": session_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
        },
    )
    first.raise_for_status()
    assistant_content = first.json()["choices"][0]["message"]["content"]

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": "again"},
            ],
            "max_tokens": 8,
        },
    )
    second.raise_for_status()
    payload = second.json()
    assert payload["cache_reused_tokens"] > 0
    assert model.calls[0]["reset_state"] is True
    assert model.calls[1]["reset_state"] is False
    assert len(model.calls[1]["inputs"]) < len(model.calls[0]["inputs"]) + len(assistant_content)


def test_session_management_crud() -> None:
    app = create_app(tokenizer=FakeTokenizer(), model=FakeModel(), model_name="fake-qwen")
    client = TestClient(app)

    created = client.post(
        "/v1/sessions",
        json={
            "session_id": "session-crud",
            "messages": [{"role": "system", "content": "You are terse."}],
        },
    )
    created.raise_for_status()
    assert created.json()["id"] == "session-crud"

    listed = client.get("/v1/sessions")
    listed.raise_for_status()
    session_ids = [item["id"] for item in listed.json()["data"]]
    assert "session-crud" in session_ids

    fetched = client.get("/v1/sessions/session-crud")
    fetched.raise_for_status()
    assert fetched.json()["messages"][0]["content"] == "You are terse."

    updated = client.put(
        "/v1/sessions/session-crud",
        json={"messages": [{"role": "user", "content": "edited"}]},
    )
    updated.raise_for_status()
    assert updated.json()["messages"][0]["content"] == "edited"

    deleted = client.delete("/v1/sessions/session-crud")
    deleted.raise_for_status()
    assert deleted.json()["deleted"] is True


def test_regenerate_uses_truncate_for_active_session() -> None:
    model = FakeModel()
    app = create_app(tokenizer=FakeTokenizer(), model=model, model_name="fake-qwen")
    client = TestClient(app)
    session_id = "session-regen"

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "session_id": session_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
        },
    )
    first.raise_for_status()

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-qwen",
            "session_id": session_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
        },
    )
    second.raise_for_status()

    truncate_calls = [call for call in model.calls if call["mode"] == "truncate"]
    assert truncate_calls, model.calls
    assert truncate_calls[-1]["position"] > 0
    generate_calls = [call for call in model.calls if call["mode"] == "generate"]
    assert generate_calls[0]["reset_state"] is True
    assert generate_calls[1]["reset_state"] is False


if __name__ == "__main__":
    test_non_stream_response()
    test_stream_response()
    test_session_reuse_response()
    test_session_management_crud()
    test_regenerate_uses_truncate_for_active_session()
    print("\033[92mTest passed!\033[0m\n")

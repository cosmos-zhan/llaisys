import argparse
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import llaisys
from transformers import AutoTokenizer

try:
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "FastAPI support is optional. Install with `pip install ./python[server]`."
    ) from exc


def _device_from_name(device_name: str) -> llaisys.DeviceType:
    if device_name == "cpu":
        return llaisys.DeviceType.CPU
    if device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    raise ValueError(f"Unsupported device: {device_name}")


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object")
        role = str(message.get("role", "")).strip()
        content = message.get("content", "")
        if not role:
            raise ValueError("Each message must include a role")
        if isinstance(content, list):
            text_parts = [str(part.get("text", "")) for part in content if isinstance(part, dict)]
            content = "".join(text_parts)
        normalized.append({"role": role, "content": str(content)})
    if not normalized:
        raise ValueError("messages must not be empty")
    return normalized


def _clone_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [{"role": message["role"], "content": message["content"]} for message in messages]


def _build_input_ids(tokenizer, messages: List[Dict[str, Any]]) -> List[int]:
    prompt = tokenizer.apply_chat_template(
        conversation=_normalize_messages(messages),
        add_generation_prompt=True,
        tokenize=False,
    )
    return list(tokenizer.encode(prompt))


def _finish_reason(model, generated_ids: List[int], max_tokens: int) -> str:
    if generated_ids and generated_ids[-1] == getattr(model.meta, "end_token", None):
        return "stop"
    if len(generated_ids) >= max_tokens:
        return "length"
    return "stop"


def _usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _sse_payload(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _resolve_session_id(payload: Dict[str, Any]) -> str:
    session_id = payload.get("session_id")
    return str(session_id) if session_id else f"session-{uuid.uuid4().hex}"


def _session_title(messages: List[Dict[str, str]]) -> str:
    for message in messages:
        if message["role"] == "user" and message["content"]:
            title = message["content"].strip().replace("\n", " ")
            return title[:64] if title else "Untitled Session"
    return "Untitled Session"


def _longest_common_prefix_len(lhs: List[int], rhs: List[int]) -> int:
    limit = min(len(lhs), len(rhs))
    idx = 0
    while idx < limit and lhs[idx] == rhs[idx]:
        idx += 1
    return idx


@dataclass
class SessionState:
    session_id: str
    messages: List[Dict[str, str]]
    token_ids: List[int]
    created_at: float
    updated_at: float


@dataclass
class ReusePlan:
    reset_state: bool
    generation_inputs: List[int]
    reused_tokens: int
    truncate_to: Optional[int]


def _session_payload(state: SessionState, include_messages: bool = False) -> Dict[str, Any]:
    payload = {
        "id": state.session_id,
        "object": "session",
        "title": _session_title(state.messages),
        "created": int(state.created_at),
        "updated": int(state.updated_at),
        "message_count": len(state.messages),
        "token_count": len(state.token_ids),
    }
    if include_messages:
        payload["messages"] = _clone_messages(state.messages)
    return payload


def _clear_active_session(app: FastAPI) -> None:
    app.state.model.reset()
    app.state.active_session_id = None
    app.state.active_token_ids = []


def _prepare_session_reuse(app: FastAPI, session_id: str, input_ids: List[int]) -> ReusePlan:
    active_session_id = app.state.active_session_id
    active_token_ids = app.state.active_token_ids

    if active_session_id != session_id or not active_token_ids:
        return ReusePlan(reset_state=True, generation_inputs=input_ids, reused_tokens=0, truncate_to=None)

    prefix_len = _longest_common_prefix_len(active_token_ids, input_ids)
    if prefix_len <= 0:
        return ReusePlan(reset_state=True, generation_inputs=input_ids, reused_tokens=0, truncate_to=None)

    reusable_prefix = prefix_len
    if reusable_prefix >= len(input_ids):
        reusable_prefix = len(input_ids) - 1

    if reusable_prefix <= 0:
        return ReusePlan(reset_state=True, generation_inputs=input_ids, reused_tokens=0, truncate_to=None)

    truncate_to = reusable_prefix if reusable_prefix < len(active_token_ids) else None
    return ReusePlan(
        reset_state=False,
        generation_inputs=input_ids[reusable_prefix:],
        reused_tokens=reusable_prefix,
        truncate_to=truncate_to,
    )


def _commit_session(
    app: FastAPI,
    session_id: str,
    messages: List[Dict[str, str]],
    cached_tokens: List[int],
) -> None:
    now = time.time()
    previous = app.state.sessions.get(session_id)
    created_at = previous.created_at if previous is not None else now
    app.state.sessions[session_id] = SessionState(
        session_id=session_id,
        messages=_clone_messages(messages),
        token_ids=list(cached_tokens),
        created_at=created_at,
        updated_at=now,
    )
    app.state.active_session_id = session_id
    app.state.active_token_ids = list(cached_tokens)


def create_app(
    model_path: Optional[str] = None,
    device: str = "cpu",
    device_id: int = 0,
    *,
    tokenizer=None,
    model=None,
    model_name: Optional[str] = None,
) -> FastAPI:
    if tokenizer is None or model is None:
        if model_path is None:
            raise ValueError("model_path is required when tokenizer/model are not injected")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = llaisys.models.create_model(model_path, _device_from_name(device), device_id)
        resolved_model_name = model_name or llaisys.models.default_model_name(model_path)
    else:
        resolved_model_name = model_name or "llaisys-chat"

    app = FastAPI(title="LLAISYS Chat API", version="0.2.0")
    app.state.lock = threading.Lock()
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.model_name = resolved_model_name
    app.state.sessions = {}
    app.state.active_session_id = None
    app.state.active_token_ids = []

    @app.get("/health")
    def health() -> Dict[str, Any]:
        with app.state.lock:
            session_count = len(app.state.sessions)
        return {"status": "ok", "model": app.state.model_name, "sessions": session_count}

    @app.get("/v1/sessions")
    def list_sessions() -> Dict[str, Any]:
        with app.state.lock:
            sessions = sorted(app.state.sessions.values(), key=lambda item: item.updated_at, reverse=True)
            return {"object": "list", "data": [_session_payload(session) for session in sessions]}

    @app.post("/v1/sessions")
    def create_session(payload: Optional[Dict[str, Any]] = Body(default=None)) -> JSONResponse:
        payload = payload or {}
        session_id = _resolve_session_id(payload)
        messages = payload.get("messages")
        normalized_messages = _normalize_messages(messages) if messages else []

        with app.state.lock:
            if session_id in app.state.sessions:
                raise HTTPException(status_code=409, detail=f"Session `{session_id}` already exists")

            now = time.time()
            app.state.sessions[session_id] = SessionState(
                session_id=session_id,
                messages=normalized_messages,
                token_ids=[],
                created_at=now,
                updated_at=now,
            )
            return JSONResponse(_session_payload(app.state.sessions[session_id], include_messages=True))

    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: str) -> Dict[str, Any]:
        with app.state.lock:
            state = app.state.sessions.get(session_id)
            if state is None:
                raise HTTPException(status_code=404, detail=f"Unknown session `{session_id}`")
            payload = _session_payload(state, include_messages=True)
            payload["active"] = app.state.active_session_id == session_id
            return payload

    @app.put("/v1/sessions/{session_id}")
    def replace_session(session_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if "messages" not in payload:
            raise HTTPException(status_code=400, detail="`messages` is required")

        try:
            normalized_messages = _normalize_messages(payload["messages"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        with app.state.lock:
            state = app.state.sessions.get(session_id)
            if state is None:
                raise HTTPException(status_code=404, detail=f"Unknown session `{session_id}`")

            state.messages = normalized_messages
            state.token_ids = []
            state.updated_at = time.time()
            if app.state.active_session_id == session_id:
                _clear_active_session(app)
            return _session_payload(state, include_messages=True)

    @app.delete("/v1/sessions/{session_id}")
    def delete_session(session_id: str) -> Dict[str, Any]:
        with app.state.lock:
            state = app.state.sessions.pop(session_id, None)
            if state is None:
                raise HTTPException(status_code=404, detail=f"Unknown session `{session_id}`")
            if app.state.active_session_id == session_id:
                _clear_active_session(app)
            return {"id": session_id, "object": "session.deleted", "deleted": True}

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: Dict[str, Any] = Body(...)):
        session_id = _resolve_session_id(payload)
        with app.state.lock:
            stored_session = app.state.sessions.get(session_id)

        raw_messages = payload.get("messages")
        if raw_messages is None:
            if stored_session is None:
                raise HTTPException(status_code=400, detail="`messages` is required for a new session")
            raw_messages = stored_session.messages

        try:
            messages = _normalize_messages(raw_messages)
            input_ids = _build_input_ids(app.state.tokenizer, messages)
            max_tokens = int(payload.get("max_tokens", payload.get("max_completion_tokens", 128)))
            top_k = int(payload.get("top_k", 1))
            top_p = float(payload.get("top_p", 1.0))
            temperature = float(payload.get("temperature", 1.0))
            stream = bool(payload.get("stream", False))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if max_tokens <= 0:
            raise HTTPException(status_code=400, detail="`max_tokens` must be positive")

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        request_model = str(payload.get("model", app.state.model_name))

        if stream:
            def event_stream() -> Iterable[str]:
                generated_ids: List[int] = []
                with app.state.lock:
                    plan = _prepare_session_reuse(app, session_id, input_ids)
                    if plan.truncate_to is not None:
                        app.state.model.truncate(plan.truncate_to)

                    yield _sse_payload(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request_model,
                            "session_id": session_id,
                            "cache_reused_tokens": plan.reused_tokens,
                            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                        }
                    )

                    for token_id, text_chunk in app.state.model.stream_generate(
                        plan.generation_inputs,
                        tokenizer=app.state.tokenizer,
                        max_new_tokens=max_tokens,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        reset_state=plan.reset_state,
                    ):
                        generated_ids.append(token_id)
                        if text_chunk:
                            yield _sse_payload(
                                {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": request_model,
                                    "session_id": session_id,
                                    "cache_reused_tokens": plan.reused_tokens,
                                    "choices": [{"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}],
                                }
                            )

                    assistant_content = app.state.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    _commit_session(
                        app,
                        session_id,
                        messages + [{"role": "assistant", "content": assistant_content}],
                        input_ids + generated_ids,
                    )

                finish_reason = _finish_reason(app.state.model, generated_ids, max_tokens)
                yield _sse_payload(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request_model,
                        "session_id": session_id,
                        "cache_reused_tokens": plan.reused_tokens,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    }
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        with app.state.lock:
            plan = _prepare_session_reuse(app, session_id, input_ids)
            if plan.truncate_to is not None:
                app.state.model.truncate(plan.truncate_to)

            output_ids = app.state.model.generate(
                plan.generation_inputs,
                max_new_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                reset_state=plan.reset_state,
            )
            generated_ids = output_ids[len(plan.generation_inputs):]
            assistant_content = app.state.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            _commit_session(
                app,
                session_id,
                messages + [{"role": "assistant", "content": assistant_content}],
                input_ids + generated_ids,
            )

        finish_reason = _finish_reason(app.state.model, generated_ids, max_tokens)
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": request_model,
                "session_id": session_id,
                "cache_reused_tokens": plan.reused_tokens,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": assistant_content},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": _usage(len(input_ids), len(generated_ids)),
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--device-id", default=0, type=int)
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model-name", default=None, type=str)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Uvicorn is optional. Install with `pip install ./python[server]`."
        ) from exc

    app = create_app(
        model_path=args.model_path,
        device=args.device,
        device_id=args.device_id,
        model_name=args.model_name,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

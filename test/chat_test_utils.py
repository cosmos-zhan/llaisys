import socket
import time
from typing import Iterator, Tuple


class FakeTokenizer:
    def apply_chat_template(self, conversation, add_generation_prompt=True, tokenize=False):
        rendered = []
        for message in conversation:
            role = message["role"]
            content = message["content"]
            rendered.append(f"{role}:{content}" if content else f"{role}:")
        prompt = "\n".join(rendered)
        if add_generation_prompt:
            prompt += "\nassistant:"
        return prompt

    def encode(self, text, return_tensors=None):
        tokens = [ord(ch) for ch in text]
        if return_tensors == "pt":
            raise NotImplementedError("FakeTokenizer does not support tensor outputs")
        return tokens or [0]

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(token_id) for token_id in token_ids if 0 <= token_id < 256)


class _Meta:
    end_token = 33


class FakeModel:
    def __init__(self):
        self.meta = _Meta()
        self._reply = [72, 101, 108, 108, 111, 33]
        self.calls = []

    def generate(self, inputs, max_new_tokens=20, top_k=1, top_p=1.0, temperature=1.0, reset_state=True):
        self.calls.append({"mode": "generate", "inputs": list(inputs), "reset_state": reset_state})
        return list(inputs) + self._reply[:max_new_tokens]

    def stream_generate(
        self,
        inputs,
        *,
        tokenizer=None,
        max_new_tokens=20,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        reset_state=True,
    ) -> Iterator[Tuple[int, str]]:
        self.calls.append({"mode": "stream_generate", "inputs": list(inputs), "reset_state": reset_state})
        produced = []
        for token in self._reply[:max_new_tokens]:
            produced.append(token)
            text = tokenizer.decode(produced, skip_special_tokens=True, clean_up_tokenization_spaces=False) if tokenizer else ""
            prev = tokenizer.decode(produced[:-1], skip_special_tokens=True, clean_up_tokenization_spaces=False) if tokenizer else ""
            yield token, text[len(prev):] if text.startswith(prev) else text
            if token == self.meta.end_token:
                break

    def truncate(self, position: int) -> None:
        self.calls.append({"mode": "truncate", "position": int(position)})

    def reset(self) -> None:
        self.calls.append({"mode": "reset"})


def serve_fake_app(port: int) -> None:
    import uvicorn
    from llaisys.chat_server import create_app

    app = create_app(tokenizer=FakeTokenizer(), model=FakeModel(), model_name="fake-qwen")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


def wait_for_server(port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"Server on port {port} did not start in time")

"""Chat client for the fine-tuned model served on the RunPod GPU via vLLM.

The chat model runs entirely on the pod now (no local Mac inference, no GGUF
conversion) — reach it from the Mac through an SSH tunnel:

    ssh -f -N -L 8000:localhost:8000 -p <port> -i <key> root@<pod-ip>

Shared by the Gradio app and the eval runner so both use identical generation
settings.
"""
from __future__ import annotations

import json

import requests


class PodChatModel:
    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", timeout: int = 300) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self, system_prompt: str, user_message: str, history: list[dict] | None = None,
        max_tokens: int = 512, temperature: float = 0.2,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}, *(history or []), {"role": "user", "content": user_message}]
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def chat_stream(
        self, system_prompt: str, user_message: str, history: list[dict] | None = None,
        max_tokens: int = 512, temperature: float = 0.2,
    ):
        messages = [{"role": "system", "content": system_prompt}, *(history or []), {"role": "user", "content": user_message}]
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "messages": messages,
            },
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[len(b"data: "):]
            if payload == b"[DONE]":
                break
            delta = json.loads(payload)["choices"][0]["delta"].get("content")
            if delta:
                yield delta

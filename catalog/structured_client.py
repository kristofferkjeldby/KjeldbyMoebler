"""Unified structured-generation client: Claude Sonnet or a locally-served
open model (via vLLM's OpenAI-compatible API), behind one interface.

Lets catalog/generate_catalog.py (and, if useful later, the training-data /
test-question generators) swap backends with a single `--backend` flag
without duplicating the prompt-building logic per backend.

The local backend expects a vLLM (or any OpenAI-compatible) server reachable
at `base_url`, started with structured-output support, e.g.:

    vllm serve <model> --quantization bitsandbytes --load-format bitsandbytes \
        --port 8000 --enforce-eager

and reached from the Mac via an SSH tunnel:

    ssh -f -N -L 8000:localhost:8000 -p <port> -i <key> root@<pod-ip>
"""
from __future__ import annotations

import json
import time
from typing import Protocol

import anthropic
import requests

RETRY_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 3


def with_retries(fn, *args, **kwargs):
    """Retry transient failures (dropped connections, timeouts) a few times
    with linear backoff before giving up — a single flaky network blip or a
    slow generation under concurrent load shouldn't permanently lose a job."""
    last_exc: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_exc


class StructuredClient(Protocol):
    def generate(self, prompt: str, schema: dict, max_tokens: int, temperature: float = 0.4) -> dict: ...


class ClaudeStructuredClient:
    def __init__(self, model: str) -> None:
        self.model = model
        self.client = anthropic.Anthropic()

    def generate(self, prompt: str, schema: dict, max_tokens: int, temperature: float = 0.4) -> dict:
        # `temperature` is rejected (400 invalid_request_error, "deprecated
        # for this model") by claude-sonnet-5 — the param is kept on this
        # method for interface parity with LocalStructuredClient, just not
        # forwarded to the API call for this backend.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)


class LocalStructuredClient:
    """Talks to an OpenAI-compatible server (vLLM) using guided JSON decoding."""

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", timeout: int = 600) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, schema: dict, max_tokens: int, temperature: float = 0.4) -> dict:
        # vLLM defaults to temperature=1.0 when unset, which drifts badly on
        # long structured-JSON arrays in Danish (nonsense colors/materials,
        # literal "nan" tokens creeping in by the last few array items) —
        # a lower, explicit temperature keeps the whole batch coherent.
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": schema, "strict": True},
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return json.loads(text)


def build_client(backend: str, model: str, local_base_url: str = "http://localhost:8000/v1") -> StructuredClient:
    if backend == "claude":
        return ClaudeStructuredClient(model)
    if backend == "local":
        return LocalStructuredClient(model, base_url=local_base_url)
    raise ValueError(f"Unknown backend: {backend!r} (expected 'claude' or 'local')")

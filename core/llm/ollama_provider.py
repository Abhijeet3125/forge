# core/llm/ollama_provider.py

import json
from typing import Any, Iterator, Sequence
import httpx

from core.llm.provider import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    normalize_messages,
)

import os
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("FORGE_MODEL", "qwen2.5-coder:14b-instruct-q5_K_M")
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("FORGE_OLLAMA_TIMEOUT", "300.0"))


class OllamaProvider(LLMProvider):
    """
    Local LLM provider using Ollama's native HTTP REST API.
    Interacts with Ollama via /api/chat.
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Checks if the local Ollama instance is running and reachable."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def list_installed_models(self) -> list[str]:
        """Returns a list of model names currently installed in Ollama."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def generate(
        self,
        messages: Sequence[LLMMessage | dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Sends a synchronous chat completion request to Ollama's /api/chat endpoint.
        """
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        # Merge any additional options passed
        if "options" in kwargs:
            options.update(kwargs.pop("options"))

        payload = {
            "model": self.model,
            "messages": normalize_messages(messages),
            "stream": False,
            "options": options,
            **kwargs,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()

            message_data = data.get("message", {})
            content = message_data.get("content", "")
            role = message_data.get("role", "assistant")

            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            return LLMResponse(
                content=content,
                model=data.get("model", self.model),
                role=role,
                finish_reason=data.get("done_reason", "stop" if data.get("done") else None),
                usage=usage,
                raw=data,
            )

        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Please ensure Ollama is running (`ollama serve`) and model '{self.model}' is pulled "
                f"(`ollama pull {self.model}`). Details: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s for model '{self.model}'."
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama returned HTTP error {e.response.status_code}: {e.response.text}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Ollama: {e}") from e

    def generate_stream(
        self,
        messages: Sequence[LLMMessage | dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Streams response tokens chunk-by-chunk from Ollama.
        """
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload = {
            "model": self.model,
            "messages": normalize_messages(messages),
            "stream": True,
            "options": options,
            **kwargs,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield token
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. Ensure Ollama is running. Error: {e}"
            ) from e

# core/llm/provider.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Sequence

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class LLMMessage:
    role: Role
    content: str
    name: str | None = None

    def to_dict(self) -> dict[str, str]:
        data = {"role": self.role, "content": self.content}
        if self.name:
            data["name"] = self.name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMMessage":
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            name=data.get("name"),
        )


@dataclass
class LLMResponse:
    content: str
    model: str
    role: str = "assistant"
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


def normalize_messages(messages: Sequence[LLMMessage | dict[str, Any]]) -> list[dict[str, str]]:
    """Converts a mix of LLMMessage and dicts into standardized message dictionaries."""
    normalized: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, LLMMessage):
            normalized.append(msg.to_dict())
        elif isinstance(msg, dict):
            normalized.append({"role": str(msg.get("role", "user")), "content": str(msg.get("content", ""))})
    return normalized


class LLMProvider(ABC):
    """
    Abstract Base Class for all LLM providers (Ollama, Claude, OpenAI, Router).
    Allows switching between local models and cloud providers without changing agent code.
    """

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.extra_kwargs = kwargs

    @abstractmethod
    def generate(
        self,
        messages: Sequence[LLMMessage | dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Synchronously generate a response from the model."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        messages: Sequence[LLMMessage | dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """Stream response tokens as they arrive."""
        pass

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        """Convenience method for generating a completion from a single prompt."""
        messages: list[LLMMessage] = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content=prompt))

        response = self.generate(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
        return response.content

    def is_available(self) -> bool:
        """Check if the provider/service is reachable."""
        return True

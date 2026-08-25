"""LLM 抽象层。上层所有模块只认这里的类型, 不认任何厂商 SDK。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from agentp.util import new_id


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    @staticmethod
    def create(name: str, arguments: dict[str, Any]) -> "ToolCall":
        return ToolCall(id=new_id("call"), name=name, arguments=arguments)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str = ""
    name: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    # 上下文组装器写进来的元数据: 属于哪个 zone、来源、是否被压缩过
    meta: dict[str, Any] = field(default_factory=dict)

    def to_api(self) -> dict[str, Any]:
        """转成 OpenAI Chat Completions 的 wire 格式。"""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": __import__("json").dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "meta": self.meta,
        }


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"  # stop | tool_calls | length | error
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    latency_ms: float = 0.0

    def to_message(self) -> Message:
        return Message(role="assistant", content=self.content, tool_calls=self.tool_calls)


# 每 1K token 的价格 (USD)。真实定价会变, 这里的作用是让成本护栏有数可算。
PRICING: dict[str, tuple[float, float]] = {
    "mock-medium": (0.0, 0.0),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "deepseek-chat": (0.00014, 0.00028),
    "qwen-plus": (0.0008, 0.002),
}
_DEFAULT_PRICE = (0.001, 0.003)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = PRICING.get(model, _DEFAULT_PRICE)
    return prompt_tokens / 1000 * p_in + completion_tokens / 1000 * p_out


@runtime_checkable
class LLMProvider(Protocol):
    """所有 Provider 的契约。

    `hint` 是给离线 Mock 用的旁路信号(planner/critic/reflect/summarize/react),
    真实 Provider 直接忽略即可 —— 这样同一份业务代码能同时跑通两种后端。
    """

    name: str
    model: str

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        hint: str = "react",
    ) -> LLMResponse: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

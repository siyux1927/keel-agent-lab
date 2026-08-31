"""Provider 注册表 + 带 Trace/缓存的装饰器。"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from keel.config import settings
from keel.llm.base import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    Usage,
    estimate_cost,
)
from keel.llm.mock import MockProvider
from keel.llm.openai_compat import OpenAICompatProvider
from keel.observability.trace import current_tracer
from keel.util import stable_hash


class TracedProvider:
    """装饰器: 给任意 Provider 加上 Trace 埋点 + embedding 缓存。

    用组合而不是继承, 这样 Mock 和真实 Provider 都不需要知道 Trace 的存在。
    embedding 缓存很关键 —— 检索链路上同一段文本会被反复 embed。
    """

    def __init__(self, inner: LLMProvider) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self._embed_cache: dict[str, list[float]] = {}
        self.stats = {"chat_calls": 0, "embed_calls": 0, "cache_hits": 0, "total_cost": 0.0}

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        hint: str = "react",
    ) -> LLMResponse:
        tracer = current_tracer()
        with tracer.span(f"llm.{hint}", kind="llm", model=self.model, messages=len(messages)) as span:
            resp = await self.inner.chat(
                messages, tools=tools, temperature=temperature, max_tokens=max_tokens, hint=hint
            )
            self.stats["chat_calls"] += 1
            self.stats["total_cost"] += resp.usage.cost_usd
            span.attributes.update(
                {
                    "finish_reason": resp.finish_reason,
                    "tool_calls": [tc.name for tc in resp.tool_calls],
                    "latency_ms": round(resp.latency_ms, 1),
                    "preview": resp.content[:160],
                }
            )
            tracer.record_usage(
                resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.cost_usd
            )
            return resp

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        missing = [t for t in texts if stable_hash(t) not in self._embed_cache]
        if missing:
            tracer = current_tracer()
            with tracer.span("llm.embed", kind="llm", count=len(missing)) as span:
                vectors = await self.inner.embed(missing)
                # 把降级状态带进 trace: 语义检索悄悄退化成词法匹配是最难发现的一类故障,
                # 至少要让它在调用链上留下痕迹
                if getattr(self.inner, "embed_degraded", False):
                    span.attributes["degraded"] = True
                    span.attributes["degraded_reason"] = self.inner.embed_degraded_reason[:160]
                if vectors:
                    span.attributes["dim"] = len(vectors[0])
            for text, vec in zip(missing, vectors):
                self._embed_cache[stable_hash(text)] = vec
            self.stats["embed_calls"] += 1
        self.stats["cache_hits"] += len(texts) - len(missing)
        return [self._embed_cache[stable_hash(t)] for t in texts]

    async def aclose(self) -> None:
        closer = getattr(self.inner, "aclose", None)
        if closer:
            await closer()


def build_provider(name: Optional[str] = None, model: Optional[str] = None) -> TracedProvider:
    name = (name or settings.llm.provider).lower()
    if name == "mock":
        return TracedProvider(MockProvider(model=model or "mock-medium"))
    if name in ("openai", "deepseek", "qwen", "compat"):
        return TracedProvider(OpenAICompatProvider(model=model))
    raise ValueError(f"未知的 provider: {name}")


_default: Optional[TracedProvider] = None


def get_provider() -> TracedProvider:
    """进程级单例 —— embedding 缓存要跨请求复用才有意义。"""
    global _default
    if _default is None:
        # 选了真实 provider 却没配 Key 时自动退回 mock, 避免演示时一片 401
        if settings.llm.provider != "mock" and not settings.llm.api_key:
            _default = build_provider("mock")
        else:
            _default = build_provider()
    return _default


def reset_provider() -> None:
    global _default
    _default = None


def set_provider(provider: Optional[TracedProvider]) -> None:
    """显式指定全局 Provider。

    存在的理由是 kb_search 这类工具走的是 get_provider()/get_memory() 单例, 而不是
    调用方注入的实例。做 A/B 实验时若只注入引擎, 工具侧仍会用旧 Provider, 两组数据
    就不可比了。
    """
    global _default
    _default = provider


__all__ = [
    "LLMProvider", "LLMResponse", "Message", "ToolCall", "Usage", "estimate_cost",
    "MockProvider", "OpenAICompatProvider", "TracedProvider",
    "build_provider", "get_provider", "reset_provider", "set_provider",
]

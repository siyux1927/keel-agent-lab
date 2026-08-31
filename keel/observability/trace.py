"""树状 Trace + Token/成本核算。

Agent 是非确定性系统, 出问题时"复现"比"日志"重要得多。
这里做的是 OpenTelemetry 的极简同构版: Span 树 + contextvar 隐式父子关系,
额外把 token/成本沿树向上汇总 —— 这样能一眼看出钱花在哪个环节。
"""

from __future__ import annotations

import contextvars
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from keel.util import new_id, now_ms, now_ts


@dataclass
class Span:
    id: str
    name: str
    kind: str = "internal"  # llm | tool | memory | context | loop | agent | internal
    parent_id: Optional[str] = None
    start_ms: float = field(default_factory=now_ms)
    end_ms: Optional[float] = None
    status: str = "running"  # running | ok | error
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        return (self.end_ms or now_ms()) - self.start_ms

    def add_event(self, name: str, **attrs: Any) -> None:
        self.events.append({"name": name, "ts": now_ts(), "attrs": attrs})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "parent_id": self.parent_id,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "error": self.error,
        }


# span 栈存在 contextvar 里而不是 Tracer 实例上。
# 原因: 并发编排时多个 asyncio.Task 共用同一个 Tracer, 如果栈是共享列表,
# 各任务的 push/pop 会交错, 父子关系全乱。contextvar 在每个 Task 创建时
# 自动复制一份, 天然做到"同一条 trace, 每个分支各自嵌套"。
_span_stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "keel_span_stack", default=()
)


class Tracer:
    def __init__(self, trace_id: Optional[str] = None, name: str = "root") -> None:
        self.trace_id = trace_id or new_id("trace")
        self.name = name
        self.spans: list[Span] = []
        self._by_id: dict[str, Span] = {}
        self.created_at = now_ts()

    # -- 采集 ---------------------------------------------------------------

    @contextmanager
    def span(self, name: str, kind: str = "internal", **attrs: Any) -> Iterator[Span]:
        stack = _span_stack.get()
        span = Span(
            id=new_id("span"),
            name=name,
            kind=kind,
            parent_id=stack[-1] if stack else None,
            attributes=dict(attrs),
        )
        self.spans.append(span)
        self._by_id[span.id] = span
        token = _span_stack.set(stack + (span.id,))
        try:
            yield span
            span.status = "ok" if span.status == "running" else span.status
        except Exception as exc:  # noqa: BLE001 - 兜住一切, 记录后原样抛出
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end_ms = now_ms()
            _span_stack.reset(token)

    def record_usage(self, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
        """记到当前 span 上; 汇总时再沿树向上滚。"""
        stack = _span_stack.get()
        span = self._by_id.get(stack[-1]) if stack else None
        if span is not None:
            span.prompt_tokens += prompt_tokens
            span.completion_tokens += completion_tokens
            span.cost_usd += cost_usd

    # -- 汇总 ---------------------------------------------------------------

    def totals(self) -> dict[str, Any]:
        by_kind: dict[str, dict[str, Any]] = {}
        for span in self.spans:
            bucket = by_kind.setdefault(
                span.kind, {"count": 0, "duration_ms": 0.0, "tokens": 0, "cost_usd": 0.0}
            )
            bucket["count"] += 1
            bucket["duration_ms"] += span.duration_ms
            bucket["tokens"] += span.prompt_tokens + span.completion_tokens
            bucket["cost_usd"] += span.cost_usd
        for bucket in by_kind.values():
            bucket["duration_ms"] = round(bucket["duration_ms"], 2)
            bucket["cost_usd"] = round(bucket["cost_usd"], 6)
        return {
            "trace_id": self.trace_id,
            "spans": len(self.spans),
            "prompt_tokens": sum(s.prompt_tokens for s in self.spans),
            "completion_tokens": sum(s.completion_tokens for s in self.spans),
            "total_tokens": sum(s.prompt_tokens + s.completion_tokens for s in self.spans),
            "cost_usd": round(sum(s.cost_usd for s in self.spans), 6),
            "llm_calls": sum(1 for s in self.spans if s.kind == "llm"),
            "tool_calls": sum(1 for s in self.spans if s.kind == "tool"),
            "errors": sum(1 for s in self.spans if s.status == "error"),
            "wall_ms": round(now_ms() - self.created_at * 1000, 2),
            "by_kind": by_kind,
        }

    def tree(self) -> list[dict[str, Any]]:
        """扁平 span 列表还原成嵌套树, 前端直接渲染。"""
        nodes = {s.id: {**s.to_dict(), "children": []} for s in self.spans}
        roots: list[dict[str, Any]] = []
        for span in self.spans:
            node = nodes[span.id]
            parent = nodes.get(span.parent_id) if span.parent_id else None
            (parent["children"] if parent else roots).append(node)
        return roots

    def to_dict(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, "name": self.name,
                "totals": self.totals(), "tree": self.tree()}

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# 用 contextvar 传 tracer, 避免每个函数签名都拖一个 tracer 参数;
# asyncio 下每个 Task 自带 context 副本, 并发编排时天然隔离。
_current: contextvars.ContextVar[Optional[Tracer]] = contextvars.ContextVar(
    "keel_tracer", default=None
)

_NOOP = Tracer(trace_id="noop", name="noop")


def current_tracer() -> Tracer:
    return _current.get() or _NOOP


def get_tracer() -> Tracer:
    return current_tracer()


@contextmanager
def use_tracer(tracer: Tracer) -> Iterator[Tracer]:
    token = _current.set(tracer)
    try:
        yield tracer
    finally:
        _current.reset(token)


class TraceStore:
    """内存里的 trace 环形缓冲, 支撑 UI 的"回放"页。"""

    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self._order: list[str] = []
        self._items: dict[str, dict[str, Any]] = {}

    def put(self, tracer: Tracer) -> None:
        self._items[tracer.trace_id] = tracer.to_dict()
        # 同一条 trace 会被 put 多次(编排共享 tracer, 每个 worker 收尾都会写一遍),
        # 不去重的话 _order 里堆满重复 id, 淘汰时会把还在用的条目误删。
        if tracer.trace_id in self._order:
            self._order.remove(tracer.trace_id)
        self._order.append(tracer.trace_id)
        while len(self._order) > self.capacity:
            self._items.pop(self._order.pop(0), None)

    def get(self, trace_id: str) -> Optional[dict[str, Any]]:
        return self._items.get(trace_id)

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        out = []
        for tid in reversed(self._order[-limit:]):
            item = self._items[tid]
            out.append({"trace_id": tid, "name": item["name"], "totals": item["totals"]})
        return out


trace_store = TraceStore()

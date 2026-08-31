"""进程内事件总线, 给 SSE 提供逐步推送的能力。

Agent 单轮可能跑十几秒, 全程静默的产品体验很糟。
引擎每走一步就往总线丢事件, HTTP 层订阅后转成 SSE 推给前端。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from keel.util import now_ts


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=now_ts)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "ts": self.ts, **self.data}


class EventBus:
    """单生产者 / 多消费者。消费者慢了就丢事件, 绝不阻塞 Agent 主循环。"""

    def __init__(self, maxsize: int = 512) -> None:
        self.maxsize = maxsize
        self._subscribers: list[asyncio.Queue[Optional[Event]]] = []
        self.history: list[Event] = []

    def emit(self, type_: str, **data: Any) -> Event:
        event = Event(type=type_, data=data)
        self.history.append(event)
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # 背压策略: 宁可丢展示事件, 也不拖慢推理
        return event

    def close(self) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(None)  # None 是流结束哨兵
            except asyncio.QueueFull:
                pass

    def register(self) -> "asyncio.Queue[Optional[Event]]":
        """同步注册一个订阅队列。

        单独暴露这一步是为了消除 SSE 的竞态: 必须先注册再启动 Agent 任务,
        否则最开始那几个事件会在订阅生效前就发完了。
        """
        queue: asyncio.Queue[Optional[Event]] = asyncio.Queue(maxsize=self.maxsize)
        self._subscribers.append(queue)
        return queue

    async def stream(self, queue: "asyncio.Queue[Optional[Event]]") -> AsyncIterator[Event]:
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    async def subscribe(self) -> AsyncIterator[Event]:
        async for event in self.stream(self.register()):
            yield event


class NullBus(EventBus):
    """非流式调用时用它, 免得上层到处判空。"""

    def emit(self, type_: str, **data: Any) -> Event:  # noqa: D102
        return Event(type=type_, data=data)

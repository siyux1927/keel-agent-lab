"""工作记忆: 容量固定的近期对话缓冲。

关键约束是 **token 而不是轮数**。按轮数截断(常见做法)会在用户贴了一大段日志时瞬间爆窗。

溢出策略是"滚动摘要 + 尾部保真":
  最近 N 轮原样保留(细节对当前推理最有用),
  更早的轮次压成一段不断迭代更新的摘要,
  同时原文下沉到情景记忆 —— 摘要有损, 但原文还能被检索回来。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agentp.config import settings
from agentp.llm.base import Message
from agentp.util import count_tokens, new_id, now_ts


@dataclass
class Turn:
    role: str
    content: str
    id: str = field(default_factory=lambda: new_id("turn"))
    ts: float = field(default_factory=now_ts)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return count_tokens(self.content) + 4

    def to_message(self) -> Message:
        return Message(role=self.role, content=self.content, meta=self.meta)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "role": self.role, "content": self.content,
                "ts": self.ts, "tokens": self.tokens, "meta": self.meta}


class WorkingMemory:
    def __init__(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
        keep_recent: Optional[int] = None,
    ) -> None:
        cfg = settings.memory
        self.session_id = session_id
        self.max_tokens = max_tokens or cfg.working_max_tokens
        self.keep_recent = keep_recent or cfg.working_keep_recent
        self.turns: deque[Turn] = deque()
        self.rolling_summary: str = ""
        self.evicted_count = 0

    # ------------------------------------------------------------------

    def add(self, role: str, content: str, **meta: Any) -> Turn:
        turn = Turn(role=role, content=content, meta=meta)
        self.turns.append(turn)
        return turn

    @property
    def token_count(self) -> int:
        return sum(t.tokens for t in self.turns) + count_tokens(self.rolling_summary)

    def is_overflowing(self) -> bool:
        return self.token_count > self.max_tokens

    async def maybe_compact(
        self,
        summarizer: Optional[Callable] = None,
        on_evict: Optional[Callable[[list[Turn]], Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """溢出时压缩。返回本次压缩的报告, 没触发就返回 None。"""
        if not self.is_overflowing() or len(self.turns) <= self.keep_recent:
            return None

        before_tokens = self.token_count
        evicted: list[Turn] = []
        # 一直淘汰到降回 75% 水位 —— 留出缓冲, 免得每加一轮就触发一次压缩
        target = self.max_tokens * 0.75
        while self.token_count > target and len(self.turns) > self.keep_recent:
            evicted.append(self.turns.popleft())

        if not evicted:
            return None
        self.evicted_count += len(evicted)

        if on_evict:  # 原文下沉到情景记忆, 保证可追溯
            result = on_evict(evicted)
            if hasattr(result, "__await__"):
                await result

        evicted_text = "\n".join(f"{t.role}: {t.content}" for t in evicted)
        if summarizer:
            prior = f"【已有摘要】{self.rolling_summary}\n\n" if self.rolling_summary else ""
            self.rolling_summary = await summarizer(prior + evicted_text)
        else:
            merged = (self.rolling_summary + "\n" + evicted_text).strip()
            self.rolling_summary = merged[-800:]

        return {
            "evicted_turns": len(evicted),
            "tokens_before": before_tokens,
            "tokens_after": self.token_count,
            "summary_tokens": count_tokens(self.rolling_summary),
        }

    # ------------------------------------------------------------------

    def to_messages(self, include_summary: bool = True) -> list[Message]:
        msgs: list[Message] = []
        if include_summary and self.rolling_summary:
            msgs.append(
                Message(
                    role="system",
                    content=f"【早前对话摘要】{self.rolling_summary}",
                    meta={"zone": "history", "compressed": True},
                )
            )
        msgs.extend(t.to_message() for t in self.turns)
        return msgs

    def recent(self, n: int = 6) -> list[Turn]:
        return list(self.turns)[-n:]

    def clear(self) -> None:
        self.turns.clear()
        self.rolling_summary = ""
        self.evicted_count = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": [t.to_dict() for t in self.turns],
            "rolling_summary": self.rolling_summary,
            "token_count": self.token_count,
            "max_tokens": self.max_tokens,
            "evicted_count": self.evicted_count,
            "utilization": round(self.token_count / self.max_tokens, 3),
        }

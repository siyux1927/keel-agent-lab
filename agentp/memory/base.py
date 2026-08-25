"""记忆的数据模型与强度计算。

四层记忆的划分来自认知心理学, 但落到工程上每层的**读写时机和淘汰策略**完全不同,
这才是分层的真正理由:

  working     工作记忆   当前会话的近期轮次。容量硬上限, 溢出即摘要。
  episodic    情景记忆   "什么时候发生了什么", 带时间戳的事件流。会衰减。
  semantic    语义记忆   从事件里抽出来的、脱离时间的事实与知识。衰减慢。
  procedural  程序性记忆 "这类任务该怎么做", 成功轨迹固化成的技能。靠成功率强化。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from agentp.util import hours_between, new_id, now_ts


class MemoryLayer(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


# 各层的遗忘速度不一样: 事实比事件记得久, 技能几乎不忘。
LAYER_HALF_LIFE_FACTOR = {
    MemoryLayer.WORKING: 0.15,
    MemoryLayer.EPISODIC: 1.0,
    MemoryLayer.SEMANTIC: 6.0,
    MemoryLayer.PROCEDURAL: 20.0,
}


@dataclass
class MemoryRecord:
    content: str
    layer: MemoryLayer = MemoryLayer.EPISODIC
    id: str = field(default_factory=lambda: new_id("mem"))
    importance: float = 5.0            # 0~10, 写入时判定
    created_at: float = field(default_factory=now_ts)
    last_access: float = field(default_factory=now_ts)
    access_count: int = 0
    session_id: str = ""
    source: str = ""
    tags: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    archived: bool = False
    derived_from: list[str] = field(default_factory=list)  # 反思固化的记忆指回原始记忆

    # -- 强度 / 遗忘 -------------------------------------------------------

    def strength(self, half_life_hours: float, at: Optional[float] = None) -> float:
        """记忆强度 ∈ (0, 1]。

        三个因子:
          importance  写入时的主观重要性 —— 决定初始高度
          指数衰减    艾宾浩斯遗忘曲线, 半衰期按层放大
          复习加成    被检索到的次数越多衰减越慢(对数增长, 避免热点记忆永不淘汰)

        用 last_access 而非 created_at 计算间隔: 每次被用到就等于"复习一次",
        计时重置 —— 这正是间隔重复的核心机制。
        """
        at = at or now_ts()
        effective_hl = max(half_life_hours * LAYER_HALF_LIFE_FACTOR.get(self.layer, 1.0), 0.01)
        elapsed = hours_between(self.last_access, at)
        decay = math.exp(-elapsed * math.log(2) / effective_hl)
        rehearsal = 1.0 + 0.35 * math.log1p(self.access_count)
        return min(1.0, (self.importance / 10.0) * decay * rehearsal)

    def touch(self) -> None:
        self.last_access = now_ts()
        self.access_count += 1

    def age_hours(self) -> float:
        return hours_between(self.created_at, now_ts())

    def to_dict(self, with_embedding: bool = False) -> dict[str, Any]:
        d = {
            "id": self.id,
            "layer": self.layer.value,
            "content": self.content,
            "importance": round(self.importance, 2),
            "created_at": self.created_at,
            "last_access": self.last_access,
            "access_count": self.access_count,
            "age_hours": round(self.age_hours(), 3),
            "session_id": self.session_id,
            "source": self.source,
            "tags": self.tags,
            "metadata": self.metadata,
            "archived": self.archived,
            "derived_from": self.derived_from,
        }
        if with_embedding:
            d["embedding"] = self.embedding
        return d


@dataclass
class RetrievalResult:
    record: MemoryRecord
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.record.to_dict(),
            "score": round(self.score, 4),
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
        }


# --------------------------------------------------------------------------


@dataclass
class Skill:
    """程序性记忆的载体: 一条被验证过的解题路径。"""

    goal_pattern: str
    steps: list[str]
    success_count: int = 1
    fail_count: int = 0
    avg_steps: float = 0.0

    @property
    def win_rate(self) -> float:
        """拉普拉斯平滑 —— 避免 1胜0负 就被当成 100% 可靠。"""
        total = self.success_count + self.fail_count
        return (self.success_count + 1) / (total + 2)

    def render(self) -> str:
        steps = " → ".join(self.steps[:6])
        return (
            f"面对「{self.goal_pattern}」这类任务, 历史有效路径: {steps} "
            f"(成功率 {self.win_rate:.0%}, 样本 {self.success_count + self.fail_count})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_pattern": self.goal_pattern,
            "steps": self.steps,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "win_rate": round(self.win_rate, 3),
            "avg_steps": round(self.avg_steps, 2),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skill":
        """to_dict 会附带 win_rate 这类派生字段, 反序列化时必须过滤掉。"""
        fields = {"goal_pattern", "steps", "success_count", "fail_count", "avg_steps"}
        return cls(**{k: v for k, v in data.items() if k in fields})


def judge_importance(content: str, source: str = "") -> float:
    """写入时的重要性打分(启发式版本)。

    真实系统这里应该用小模型打分, 但启发式有两个好处: 零延迟、完全可解释。
    实践中先用规则跑起来, 再拿线上数据决定值不值得上模型。
    """
    score = 4.0
    lowered = content.lower()
    if any(k in content for k in ("我叫", "我的", "偏好", "喜欢", "不要", "必须", "记住", "以后")):
        score += 3.0  # 用户画像类信息复用率最高
    if any(k in content for k in ("决定", "结论", "方案", "选择", "确认")):
        score += 1.5
    if any(k in lowered for k in ("error", "失败", "报错", "异常", "bug")):
        score += 1.5  # 失败经验的信息量大于成功
    if any(k in content for k in ("？", "?", "怎么", "如何")):
        score -= 0.5  # 提问本身通常不值得长期留存
    if len(content) < 12:
        score -= 1.5
    if source == "reflection":
        score += 2.5  # 反思产物是高层抽象, 优先保留
    return max(0.5, min(10.0, score))

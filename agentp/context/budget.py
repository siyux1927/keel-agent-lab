"""Token 预算调度。

上下文窗口是 Agent 最稀缺的资源。多数实现的做法是"塞满为止然后截断",
后果是历史对话把工具定义挤掉、或者检索结果把用户问题本身挤掉。

这里把它当成一个**带下界、上界和优先级的资源分配问题**:

  1. 先按优先级逐个满足 min_tokens(保命线)。预算不够时, 优先级最低的分区直接丢弃,
     而不是所有分区一起缩水 —— 半截的工具定义比没有工具定义更糟。
  2. 剩余预算按权重做**注水(water-filling)**: 已经吃饱(达到 max 或实际需求)的分区
     退出竞争, 把余量还给池子, 继续分给还饿着的分区。迭代到收敛。

这样保证: 关键分区永不被饿死, 同时空闲预算不会被浪费。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from agentp.config import settings


@dataclass
class Zone:
    """一个上下文分区的预算诉求。"""

    name: str
    priority: int = 5          # 数字越小越重要, 0 = 绝不可丢
    min_tokens: int = 0        # 保命线; 给不到就整块丢弃
    max_tokens: Optional[int] = None
    weight: float = 1.0        # 注水阶段的分配权重
    requested: int = 0         # 内容实际有多大
    droppable: bool = True     # False 表示宁可超预算也要保留(如系统提示词)

    def ceiling(self) -> int:
        """这个分区最多能吃下多少 —— 需求和硬上限的较小值。"""
        cap = self.requested if self.requested > 0 else (self.max_tokens or 0)
        if self.max_tokens is not None:
            cap = min(cap, self.max_tokens)
        return max(cap, 0)


@dataclass
class ZoneAllocation:
    name: str
    granted: int
    requested: int
    min_tokens: int
    dropped: bool = False
    reason: str = ""

    @property
    def deficit(self) -> int:
        return max(0, self.requested - self.granted)

    @property
    def needs_compression(self) -> bool:
        return not self.dropped and self.deficit > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "granted": self.granted,
            "requested": self.requested,
            "deficit": self.deficit,
            "dropped": self.dropped,
            "needs_compression": self.needs_compression,
            "reason": self.reason,
        }


@dataclass
class BudgetPlan:
    total_budget: int
    allocations: dict[str, ZoneAllocation] = field(default_factory=dict)
    rounds: int = 0
    overflow: int = 0  # 不可丢弃分区本身就超了预算, 只能靠换更大的模型解决

    @property
    def granted_total(self) -> int:
        return sum(a.granted for a in self.allocations.values())

    @property
    def requested_total(self) -> int:
        return sum(a.requested for a in self.allocations.values())

    def get(self, zone: str) -> int:
        alloc = self.allocations.get(zone)
        return alloc.granted if alloc else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "granted_total": self.granted_total,
            "requested_total": self.requested_total,
            "utilization": round(self.granted_total / self.total_budget, 3)
            if self.total_budget else 0.0,
            "pressure": round(self.requested_total / self.total_budget, 3)
            if self.total_budget else 0.0,
            "rounds": self.rounds,
            "overflow": self.overflow,
            "zones": [a.to_dict() for a in self.allocations.values()],
        }


class BudgetAllocator:
    def __init__(
        self,
        context_window: Optional[int] = None,
        reserve_for_output: Optional[int] = None,
        safety_margin: Optional[float] = None,
    ) -> None:
        cfg = settings.context
        self.context_window = context_window or cfg.model_context_window
        self.reserve_for_output = reserve_for_output or cfg.reserve_for_output
        self.safety_margin = cfg.safety_margin if safety_margin is None else safety_margin

    @property
    def total_budget(self) -> int:
        """可用于输入的预算 = 窗口 - 输出预留 - 安全边界。

        安全边界是给 token 估算误差留的; 估少了直接 400, 一次失败的代价
        远大于少用几百 token。
        """
        usable = self.context_window - self.reserve_for_output
        return max(0, int(usable * (1 - self.safety_margin)))

    def allocate(self, zones: Sequence[Zone]) -> BudgetPlan:
        budget = self.total_budget
        plan = BudgetPlan(total_budget=budget)
        ordered = sorted(zones, key=lambda z: (z.priority, -z.weight))

        # --- 第一阶段: 按优先级发放保命线 ---------------------------------
        remaining = budget
        active: list[Zone] = []
        for zone in ordered:
            # 不可丢弃的分区(系统提示词/用户任务/工具定义)是**逐字原样输出**的,
            # 所以它们的保命线就是全部需求 —— 让它们去按权重抢, 报表会失真:
            # 报告说 task 只分到 14 token, 实际却完整发出去了。
            need_min = zone.requested if not zone.droppable else min(zone.min_tokens, zone.ceiling())
            # 没有内容就不占预算, 哪怕它配了保命线 —— 保命线是"有内容时至少给这么多"
            if zone.requested == 0:
                plan.allocations[zone.name] = ZoneAllocation(
                    zone.name, 0, 0, zone.min_tokens, dropped=True, reason="内容为空"
                )
                continue
            if need_min > remaining:
                if zone.droppable:
                    plan.allocations[zone.name] = ZoneAllocation(
                        zone.name, 0, zone.requested, zone.min_tokens,
                        dropped=True,
                        reason=f"预算不足以满足保命线({need_min} > 剩余 {remaining}), 按优先级丢弃",
                    )
                    continue
                # 不可丢弃的分区放行并如实记账: 它一定会被原样发出去,
                # 谎报成"已压缩"会让人以为没事, 实际是要撞窗口的
                plan.overflow += need_min - remaining
                plan.allocations[zone.name] = ZoneAllocation(
                    zone.name, need_min, zone.requested, zone.min_tokens,
                    reason=f"不可丢弃分区, 超出剩余预算 {need_min - remaining} tokens(存在超窗风险)",
                )
                remaining = 0
                continue
            plan.allocations[zone.name] = ZoneAllocation(
                zone.name, need_min, zone.requested, zone.min_tokens
            )
            remaining -= need_min
            if zone.ceiling() > need_min:
                active.append(zone)

        # --- 第二阶段: 注水 ------------------------------------------------
        rounds = 0
        while remaining > 0 and active and rounds < 32:
            rounds += 1
            total_weight = sum(z.weight for z in active)
            if total_weight <= 0:
                break
            pool = remaining
            still_active: list[Zone] = []
            for zone in active:
                alloc = plan.allocations[zone.name]
                share = int(pool * zone.weight / total_weight)
                room = zone.ceiling() - alloc.granted
                give = max(0, min(share, room, remaining))
                alloc.granted += give
                remaining -= give
                if zone.ceiling() > alloc.granted:
                    still_active.append(zone)
            if len(still_active) == len(active) and remaining == pool:
                break  # 一轮下来没分出去任何东西(全是整除误差), 收手
            active = still_active

        # 整除误差导致的零头, 全给优先级最高的那个还饿着的分区
        if remaining > 0 and active:
            top = min(active, key=lambda z: z.priority)
            alloc = plan.allocations[top.name]
            alloc.granted = min(alloc.granted + remaining, top.ceiling())

        for alloc in plan.allocations.values():
            if alloc.needs_compression and not alloc.reason:
                alloc.reason = f"需压缩 {alloc.deficit} tokens"
        plan.rounds = rounds
        return plan


# --------------------------------------------------------------------------
# 默认分区配置。priority 的排序体现了产品判断:
#   系统提示词 > 当前任务 > 工具定义 > 长期记忆 > 检索文档 > 对话历史
# 对话历史排最后, 是因为它最容易被"滚动摘要"无损压缩。
# --------------------------------------------------------------------------

DEFAULT_ZONES: list[dict[str, Any]] = [
    {"name": "system",      "priority": 0, "min_tokens": 0,   "weight": 0.5, "droppable": False},
    {"name": "task",        "priority": 0, "min_tokens": 0,   "weight": 1.0, "droppable": False},
    {"name": "tools",       "priority": 1, "min_tokens": 0,   "weight": 0.8, "droppable": False},
    {"name": "procedural",  "priority": 3, "min_tokens": 80,  "weight": 0.6},
    {"name": "semantic",    "priority": 3, "min_tokens": 120, "weight": 1.4},
    {"name": "episodic",    "priority": 4, "min_tokens": 80,  "weight": 0.8},
    {"name": "retrieved",   "priority": 4, "min_tokens": 150, "weight": 1.8},
    {"name": "scratchpad",  "priority": 2, "min_tokens": 200, "weight": 2.0},
    {"name": "history",     "priority": 5, "min_tokens": 100, "weight": 1.2},
]


def build_zones(requested: dict[str, int], overrides: Optional[dict[str, dict]] = None) -> list[Zone]:
    """把"每个分区的实际内容大小"变成一组 Zone 诉求。"""
    overrides = overrides or {}
    zones: list[Zone] = []
    for spec in DEFAULT_ZONES:
        merged = {**spec, **overrides.get(spec["name"], {})}
        zones.append(
            Zone(
                name=merged["name"],
                priority=merged["priority"],
                min_tokens=merged.get("min_tokens", 0),
                max_tokens=merged.get("max_tokens"),
                weight=merged.get("weight", 1.0),
                droppable=merged.get("droppable", True),
                requested=requested.get(merged["name"], 0),
            )
        )
    return zones

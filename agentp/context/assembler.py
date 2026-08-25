"""上下文组装器 —— 把"我有的一切"变成"这次该发给模型的那些"。

这是上下文工程的收口: 切片、检索、记忆、预算、压缩全在这里汇合。

两个非显而易见的设计:

1. **排布顺序对抗 lost-in-the-middle**
   顺序是 system → 记忆/知识 → 历史 → scratchpad → **任务复述**。
   用户的真实指令在最开头出现一次, 在最末尾再复述一次 —— 两端是模型注意力最强的位置。
   中间放最能容忍被略读的背景材料。

2. **每个片段都带 provenance**
   每条注入的内容都标注了来源(哪层记忆、哪个块、什么分数), 出问题时能回答
   "模型为什么会这么说" —— 这是 Agent 系统能不能上线的分水岭。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from agentp.config import settings
from agentp.context.budget import BudgetAllocator, BudgetPlan, build_zones
from agentp.context.compressor import (
    ExtractiveCompressor,
    LLMSummaryCompressor,
    MiddleOutCompressor,
    compress_items,
)
from agentp.llm.base import Message
from agentp.observability.trace import current_tracer
from agentp.util import count_tokens


@dataclass
class ContextItem:
    """一条待注入的上下文, 带来源与打分。"""

    text: str
    zone: str
    score: float = 1.0
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


@dataclass
class AssembledContext:
    messages: list[Message]
    plan: BudgetPlan
    zone_report: dict[str, dict[str, Any]]
    total_tokens: int
    provenance: list[dict[str, Any]]

    def to_dict(self, with_messages: bool = False) -> dict[str, Any]:
        d = {
            "total_tokens": self.total_tokens,
            "budget": self.plan.to_dict(),
            "zones": self.zone_report,
            "provenance": self.provenance,
        }
        if with_messages:
            d["messages"] = [m.to_dict() for m in self.messages]
        return d


ZONE_TITLES = {
    "procedural": "## 可复用经验(程序性记忆)",
    "semantic": "## 已知事实(语义记忆)",
    "episodic": "## 相关历史事件(情景记忆)",
    "retrieved": "## 检索到的资料",
    "history": "## 对话历史",
    "scratchpad": "## 当前推理过程",
}


def _with_header(text: str, zone: str) -> int:
    """正文 + 分区标题 + 消息封装的总开销。"""
    if not text.strip():
        return 0
    return count_tokens(text) + count_tokens(ZONE_TITLES.get(zone, "")) + 8


class ContextAssembler:
    def __init__(self, provider: Any = None, allocator: Optional[BudgetAllocator] = None) -> None:
        self.provider = provider
        self.allocator = allocator or BudgetAllocator()

    async def build(
        self,
        task: str,
        system_prompt: str = "",
        tools_schema: Optional[list[dict[str, Any]]] = None,
        items: Optional[Sequence[ContextItem]] = None,
        history: Optional[Sequence[Message]] = None,
        scratchpad: str = "",
        zone_overrides: Optional[dict[str, dict]] = None,
    ) -> AssembledContext:
        tracer = current_tracer()
        with tracer.span("context.assemble", kind="context") as span:
            items = list(items or [])
            history = list(history or [])
            tools_schema = tools_schema or []

            by_zone: dict[str, list[ContextItem]] = {}
            for item in items:
                by_zone.setdefault(item.zone, []).append(item)

            history_text = self._render_history(history)
            tools_text = json.dumps(tools_schema, ensure_ascii=False) if tools_schema else ""

            # --- 1. 统计各分区的实际需求 ---------------------------------
            # 分区标题本身也要占预算。漏算的后果很隐蔽: granted 永远比正文少几个 token,
            # 于是明明有余量却被判定需要压缩, 把 scratchpad 里的动作记录压没了 —— 模型
            # 看不到自己做过什么, 就会重复调同一个工具。
            requested = {
                "system": count_tokens(system_prompt),
                "task": count_tokens(task) * 2 + 40,  # 头尾各出现一次
                "tools": count_tokens(tools_text),
                "scratchpad": _with_header(scratchpad, "scratchpad"),
                "history": _with_header(history_text, "history"),
            }
            for zone, zone_items in by_zone.items():
                header = count_tokens(ZONE_TITLES.get(zone, ""))
                requested[zone] = sum(i.tokens + 12 for i in zone_items) + header

            # --- 2. 预算分配 ---------------------------------------------
            plan = self.allocator.allocate(build_zones(requested, zone_overrides))
            span.attributes["budget"] = plan.to_dict()

            # --- 3. 逐分区按配额压缩 --------------------------------------
            report: dict[str, dict[str, Any]] = {}
            provenance: list[dict[str, Any]] = []
            rendered: dict[str, str] = {}

            for zone in ("procedural", "semantic", "episodic", "retrieved"):
                zone_items = by_zone.get(zone, [])
                alloc = plan.allocations.get(zone)
                if not zone_items or not alloc or alloc.dropped or alloc.granted <= 0:
                    report[zone] = {
                        "granted": alloc.granted if alloc else 0,
                        "requested": requested.get(zone, 0),
                        "kept": 0,
                        "dropped": len(zone_items),
                        "method": "drop_zone",
                    }
                    continue
                header = ZONE_TITLES.get(zone, f"## {zone}")
                body_budget = alloc.granted - count_tokens(header)
                kept, dropped = await compress_items(
                    [(self._render_item(i), i.score) for i in zone_items], max(body_budget, 0)
                )
                rendered[zone] = f"{header}\n" + "\n".join(kept) if kept else ""
                report[zone] = {
                    "granted": alloc.granted,
                    "requested": requested.get(zone, 0),
                    "kept": len(kept),
                    "dropped": dropped,
                    "method": "score_based_drop",
                }
                for item in zone_items[: len(kept)]:
                    provenance.append(
                        {"zone": zone, "source": item.source, "score": round(item.score, 4),
                         "tokens": item.tokens, "preview": item.text[:80]}
                    )

            # 历史: 可以无损度较高地做滚动摘要, 所以给它上 LLM 压缩
            rendered["history"], report["history"] = await self._compress_zone(
                history_text, plan, "history", task,
                LLMSummaryCompressor(self.provider), requested,
            )
            # scratchpad: 头部是计划、尾部是最新观察, 都要留 → middle-out
            rendered["scratchpad"], report["scratchpad"] = await self._compress_zone(
                scratchpad, plan, "scratchpad", task,
                MiddleOutCompressor(head_ratio=0.28, tail_ratio=0.55,
                                    inner=ExtractiveCompressor(self.provider)),
                requested,
            )

            # --- 4. 按抗遗忘顺序排布 ---------------------------------------
            messages: list[Message] = []
            if system_prompt:
                messages.append(Message(role="system", content=system_prompt, meta={"zone": "system"}))
                report["system"] = {"granted": plan.get("system"), "requested": requested["system"],
                                    "method": "verbatim"}

            background = [rendered.get(z, "") for z in
                          ("procedural", "semantic", "episodic", "retrieved", "history")]
            background_text = "\n\n".join(b for b in background if b.strip())
            if background_text:
                messages.append(
                    Message(role="system", content="# 背景材料\n" + background_text,
                            meta={"zone": "background"})
                )

            messages.append(Message(role="user", content=task, meta={"zone": "task"}))

            if rendered.get("scratchpad", "").strip():
                messages.append(
                    Message(role="assistant",
                            content=f"{ZONE_TITLES['scratchpad']}\n{rendered['scratchpad']}",
                            meta={"zone": "scratchpad"})
                )
                # 末尾复述任务: 长 scratchpad 之后模型很容易忘记初衷
                messages.append(
                    Message(role="user",
                            content=f"[任务复述] 请继续完成: {task}",
                            meta={"zone": "task_restate"})
                )

            total = sum(count_tokens(m.content) + 4 for m in messages) + count_tokens(tools_text)
            report["task"] = {"granted": plan.get("task"), "requested": requested["task"],
                              "method": "verbatim_x2"}
            report["tools"] = {"granted": plan.get("tools"), "requested": requested["tools"],
                               "method": "verbatim"}

            span.attributes.update({"total_tokens": total, "messages": len(messages)})
            return AssembledContext(messages, plan, report, total, provenance)

    # ------------------------------------------------------------------

    async def _compress_zone(
        self, text: str, plan: BudgetPlan, zone: str, query: str,
        compressor: Any, requested: dict[str, int],
    ) -> tuple[str, dict[str, Any]]:
        alloc = plan.allocations.get(zone)
        if not text.strip() or not alloc or alloc.dropped or alloc.granted <= 0:
            return "", {"granted": alloc.granted if alloc else 0,
                        "requested": requested.get(zone, 0), "method": "drop_zone"}
        header = ZONE_TITLES.get(zone, f"## {zone}")
        budget = alloc.granted - count_tokens(header)
        if count_tokens(text) <= budget:
            return f"{header}\n{text}" if zone != "scratchpad" else text, {
                "granted": alloc.granted, "requested": requested.get(zone, 0),
                "final": count_tokens(text), "method": "verbatim",
            }
        result = await compressor.compress(text, max(budget, 32), query=query)
        body = result.text
        return (f"{header}\n{body}" if zone != "scratchpad" else body), {
            "granted": alloc.granted, "requested": requested.get(zone, 0),
            "final": result.final_tokens, **result.to_dict(),
        }

    @staticmethod
    def _render_item(item: ContextItem) -> str:
        tag = f"[{item.source}]" if item.source else ""
        return f"- {tag} {item.text}".strip()

    @staticmethod
    def _render_history(history: Sequence[Message]) -> str:
        role_cn = {"user": "用户", "assistant": "助手", "tool": "工具", "system": "系统"}
        lines = []
        for msg in history:
            if not msg.content.strip():
                continue
            lines.append(f"{role_cn.get(msg.role, msg.role)}: {msg.content}")
        return "\n".join(lines)


def default_system_prompt(agent_name: str = "AgentP", extra: str = "") -> str:
    base = f"""你是 {agent_name}, 一个严谨的任务型智能体。

行为准则:
1. 先判断信息是否充分。不足时调用工具获取证据, 而不是猜测。
2. 每次只做一件事: 一步一个动作, 依据观察结果再决定下一步。
3. 引用背景材料中的事实时, 明确指出来源。
4. 如果同一个动作已经做过且结果无用, 换一种思路, 不要重复。
5. 证据足够时立即给出最终答案, 不要为了显得努力而多调工具。
6. 无法完成时, 明确说明卡在哪里以及还缺什么, 不要编造。"""
    return f"{base}\n\n{extra}".strip() if extra else base


__all__ = [
    "ContextItem", "AssembledContext", "ContextAssembler",
    "default_system_prompt", "ZONE_TITLES",
]

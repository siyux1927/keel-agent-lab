"""离线 Mock Provider: 一个规则驱动的"假模型"。

它不是用来糊弄人的占位符 —— 它必须真的能驱动 ReAct 循环走完
think → tool_call → observe → answer, 还要能扮演 planner / critic / reflector,
否则离线状态下整套框架都测不了。

价值有三:
  1. 断网也能演示 + 单元测试完全确定性(CI 里不需要 API Key)
  2. 逼着上层代码只依赖 Provider 抽象, 不泄漏任何厂商细节
  3. 通过特殊触发词可以复现"死循环""连续失败"等异常路径, 方便展示护栏
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Optional, Sequence

from keel.config import settings
from keel.llm.base import LLMResponse, Message, ToolCall, Usage, estimate_cost
from keel.util import count_message_tokens, count_tokens, hash_embed

_MATH = re.compile(r"[-+]?[(\d][\d\s,.()+\-*/^%×÷]{1,60}[)\d]")
_OPERATOR = re.compile(r"[+\-*/^%×÷]")
# 引擎注入的是渲染后的 scratchpad 文本而非原始 tool_call 消息,
# 所以要从文本里反解出"已经执行过哪些动作"
_ACTION_LINE = re.compile(
    r"(?:动作:\s*|^-\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", re.M
)
_OBS_LINE = re.compile(
    r"观察:\s*\[([a-zA-Z_][a-zA-Z0-9_]*)[^\]]*\]\s*(.*?)"
    r"(?=\n(?:动作:|观察:|思考:|反思:|草稿答案:|###)|\Z)",
    re.S,
)


def _clean_goal(text: str) -> str:
    """剥掉上层拼进来的包装(目标前缀、背景段、上游产出), 只留纯粹的目标句。"""
    for marker in ("\n\n已知背景", "\n\n可直接使用的上游结果", "\n\n上游"):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
    return re.sub(r"^(目标|任务)[:：]\s*", "", text.strip()).strip()


def _find_math(text: str) -> str:
    """抠出文本里的数学表达式。要求含运算符且括号配平, 否则丢弃。"""
    for match in _MATH.finditer(text):
        expr = match.group(0).strip().rstrip(",.， ")
        if not _OPERATOR.search(expr) or not any(c.isdigit() for c in expr):
            continue
        if expr.count("(") != expr.count(")"):
            expr = expr.strip("()")
            if expr.count("(") != expr.count(")"):
                continue
        return expr
    return ""
_TIME_HINT = re.compile(r"现在|几点|今天|日期|时间|date|today|now|time", re.I)
_SEARCH_HINT = re.compile(r"搜索|查询|查一下|最新|新闻|search|latest|news|谁是|什么是", re.I)
_KB_HINT = re.compile(r"知识库|文档|资料|记忆|里面|提到|根据|kb|doc|memory", re.I)
# 演示护栏用的触发词: 让模型故意犯病。
# 必须用不会在正常文档里出现的词 —— 早先用的是"死循环", 结果知识库里
# 「死循环检测」这句话经上游结果流进了子任务目标, 把演示开关误触发了。
_LOOP_TRIGGER = re.compile(r"@loopdemo|死循环演示", re.I)
_FAIL_TRIGGER = re.compile(r"@faildemo|熔断演示", re.I)


class MockProvider:
    name = "mock"

    def __init__(self, model: str = "mock-medium", latency_ms: float = 30.0) -> None:
        self.model = model
        self.latency_ms = latency_ms
        self.call_count = 0

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        hint: str = "react",
    ) -> LLMResponse:
        self.call_count += 1
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000.0)

        prompt_tokens = count_message_tokens([m.to_api() for m in messages])
        tool_names = [t.get("function", {}).get("name", "") for t in (tools or [])]

        handler = {
            "plan": self._do_plan,
            "critic": self._do_critic,
            "reflect": self._do_reflect,
            "summarize": self._do_summarize,
            "extract_facts": self._do_extract_facts,
            "insight": self._do_insight,
            "final": self._do_final,
            "aggregate": self._do_aggregate,
        }.get(hint, self._do_react)

        content, tool_calls = handler(list(messages), tool_names)

        completion_tokens = count_tokens(content) + sum(
            count_tokens(str(tc.arguments)) + 8 for tc in tool_calls
        )
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=estimate_cost(self.model, prompt_tokens, completion_tokens),
            ),
            model=self.model,
            latency_ms=self.latency_ms,
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        dim = settings.context.embedding_dim
        return [hash_embed(t, dim) for t in texts]

    # ------------------------------------------------------------------
    # 各角色的"推理"
    # ------------------------------------------------------------------

    def _do_react(
        self, messages: list[Message], tool_names: list[str]
    ) -> tuple[str, list[ToolCall]]:
        goal = self._extract_goal(messages)
        executed = self._executed_tools(messages)
        observations = self._observations(messages)

        # 只在目标开头找触发词, 避免被检索结果里的同名内容带偏
        head = goal[:120]
        # 护栏演示: 强制反复调同一个工具, 让死循环检测器有活干
        if _LOOP_TRIGGER.search(head) and "now" in tool_names:
            return "我再确认一次当前时间。", [ToolCall.create("now", {"tz": "Asia/Shanghai"})]

        if _FAIL_TRIGGER.search(head) and "flaky" in tool_names:
            return "调用不稳定的下游服务。", [ToolCall.create("flaky", {"fail": True})]

        for name, args in self._plan_actions(goal, tool_names):
            if name not in executed:
                thought = f"目标「{goal[:40]}」还缺 {name} 的结果, 先调它。"
                return thought, [ToolCall.create(name, args)]

        return self._compose_answer(goal, observations), []

    def _do_final(self, messages: list[Message], _tools: list[str]) -> tuple[str, list[ToolCall]]:
        """强制收口: 只汇总已有观察, 绝不再提工具。"""
        goal = self._extract_goal(messages)
        observations = self._observations(messages)
        text = self._compose_answer(goal, observations)
        return text + "\n\n注意: 本次执行被资源护栏提前中止, 以上结论仅基于已获取的证据。", []

    def _plan_actions(self, goal: str, tool_names: list[str]) -> list[tuple[str, dict]]:
        """把目标映射成一串工具调用。真实模型靠权重, 这里靠正则 —— 接口一致就行。"""
        actions: list[tuple[str, dict]] = []
        expr = _find_math(goal)
        if expr and "calculator" in tool_names:
            actions.append(("calculator", {"expression": expr}))
        if _TIME_HINT.search(goal) and "now" in tool_names:
            actions.append(("now", {"tz": "Asia/Shanghai"}))
        if "kb_search" in tool_names and (_KB_HINT.search(goal) or not actions):
            actions.append(("kb_search", {"query": goal[:80], "top_k": 4}))
        if _SEARCH_HINT.search(goal) and "web_search" in tool_names:
            actions.append(("web_search", {"query": goal[:80]}))
        return actions

    def _compose_answer(self, goal: str, observations: list[tuple[str, str]]) -> str:
        goal = _clean_goal(goal)[:60]
        if not observations:
            return f"针对「{goal}」: 当前没有取到外部证据, 以下是基于既有上下文的回答。"
        lines = [f"针对「{goal}」, 我综合了 {len(observations)} 条工具证据:"]
        for i, (tool, obs) in enumerate(observations, 1):
            snippet = obs.strip().replace("\n", " ")
            lines.append(f"{i}. [{tool}] {snippet[:180]}")
        lines.append("\n结论: 以上证据已覆盖问题的关键点, 可以给出最终答复。")
        return "\n".join(lines)

    def _do_plan(self, messages: list[Message], _tools: list[str]) -> tuple[str, list[ToolCall]]:
        goal = _clean_goal(self._extract_goal(messages))
        parts = self._split_goal(goal)
        subtasks = []
        for i, part in enumerate(parts):
            subtasks.append(
                {
                    "id": f"t{i + 1}",
                    "title": part[:60],
                    "goal": part,
                    # 简单的链式依赖; 最后一步汇总时依赖前面全部, 好演示 DAG 并发
                    "depends_on": [],
                }
            )
        if len(subtasks) > 1:
            subtasks.append(
                {
                    "id": f"t{len(subtasks) + 1}",
                    "title": "汇总各子任务结论",
                    "goal": f"综合前面各子任务的结果, 回答: {goal}",
                    "depends_on": [s["id"] for s in subtasks],
                }
            )
        import json

        return json.dumps({"subtasks": subtasks}, ensure_ascii=False), []

    def _do_aggregate(self, messages: list[Message], _tools: list[str]) -> tuple[str, list[ToolCall]]:
        """把各子任务的【标题】块整合成一份答复。"""
        text = messages[-1].content if messages else ""
        goal_match = re.search(r"用户目标:\s*\n?(.+)", text)
        goal = _clean_goal(goal_match.group(1).strip() if goal_match else "")[:70]
        # 只认行首的块标记 —— 否则子任务正文里引用的上游标记会被当成新块
        blocks = re.findall(r"^【([^】]+)】\s*\n?(.*?)(?=\n^【|\Z)", text, re.S | re.M)
        if not blocks:
            return f"针对「{goal}」: 未能获取到子任务产出。", []
        lines = [f"关于「{goal}」, 综合 {len(blocks)} 个子任务的执行结果:"]
        unfinished = []
        for i, (title, body) in enumerate(blocks, 1):
            body = " ".join(body.split())
            if "未完成" in body[:12]:
                unfinished.append(title)
                continue
            lines.append(f"\n{i}. {title}\n   {body[:400]}")
        if unfinished:
            lines.append(f"\n以下子任务未能完成, 相关结论存在缺口: {', '.join(unfinished)}")
        if "必须修正" in text:
            lines.append("\n(已根据质检意见补充了各项诉求的逐条回应)")
        return "\n".join(lines), []

    def _do_critic(self, messages: list[Message], _tools: list[str]) -> tuple[str, list[ToolCall]]:
        import json

        draft = messages[-1].content if messages else ""
        issues: list[str] = []
        if len(draft) < 40:
            issues.append("回答过短, 缺少必要展开")
        if "没有取到外部证据" in draft:
            issues.append("缺少事实依据, 需要补充检索")
        accepted = not issues
        return json.dumps(
            {
                "accepted": accepted,
                "score": 0.9 if accepted else 0.5,
                "issues": issues,
                "suggestion": "补充证据后重写" if issues else "可以交付",
            },
            ensure_ascii=False,
        ), []

    def _do_reflect(self, messages: list[Message], _tools: list[str]) -> tuple[str, list[ToolCall]]:
        import json

        text = messages[-1].content if messages else ""
        repeated = "重复" in text or "loop" in text.lower()
        return json.dumps(
            {
                "diagnosis": "检测到重复动作, 当前策略无法推进" if repeated else "进展偏慢, 证据不足",
                "should_replan": True,
                "avoid_actions": ["now"] if repeated else [],
                "new_plan": [
                    "换一个信息源重新检索",
                    "缩小问题范围, 先解决可验证的子问题",
                    "基于已有证据直接产出结论",
                ],
            },
            ensure_ascii=False,
        ), []

    def _do_summarize(self, messages: list[Message], _tools: list[str]) -> tuple[str, list[ToolCall]]:
        """抽取式摘要: 按位置+长度打分选句。无 LLM 也能压缩上下文。"""
        text = messages[-1].content if messages else ""
        sentences = [s.strip() for s in re.split(r"(?<=[。!?！？\n])", text) if len(s.strip()) > 8]
        if not sentences:
            return text[:200], []
        scored = []
        for i, sent in enumerate(sentences):
            position_score = 1.0 / (1 + i * 0.12)          # 前面的句子更重要
            info_score = min(len(sent) / 60.0, 1.0)        # 太短的多半是废话
            digit_bonus = 0.3 if re.search(r"\d", sent) else 0.0  # 带数字的通常是事实
            scored.append((position_score + info_score + digit_bonus, i, sent))
        scored.sort(reverse=True)
        keep = sorted(scored[: max(2, len(sentences) // 3)], key=lambda x: x[1])
        return "【摘要】" + " ".join(s for _, _, s in keep), []

    def _do_extract_facts(
        self, messages: list[Message], _tools: list[str]
    ) -> tuple[str, list[ToolCall]]:
        import json

        text = messages[-1].content if messages else ""
        facts = []
        for sent in re.split(r"[。\n!?！？]", text):
            sent = sent.strip()
            if len(sent) < 6:
                continue
            # 只留看起来像"事实陈述"的句子: 含定义/数字/偏好
            if re.search(r"是|为|叫|喜欢|需要|使用|等于|\d", sent):
                importance = 7.0 if re.search(r"我|偏好|必须|要求", sent) else 4.0
                facts.append({"content": sent[:120], "importance": importance})
        return json.dumps({"facts": facts[:5]}, ensure_ascii=False), []

    def _do_insight(self, messages: list[Message], _tools: list[str]) -> tuple[str, list[ToolCall]]:
        import json

        text = messages[-1].content if messages else ""
        topics = [t for t in ("上下文", "记忆", "检索", "编排", "循环", "工具") if t in text]
        insights = [f"用户持续关注 {t} 相关问题, 后续可主动提供该方向的细节" for t in topics[:3]]
        if not insights:
            insights = ["本轮交互未形成稳定模式, 暂不固化高层记忆"]
        return json.dumps({"insights": insights}, ensure_ascii=False), []

    # ------------------------------------------------------------------
    # 从消息里反推状态
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_goal(messages: list[Message]) -> str:
        for msg in messages:  # 组装器标了 zone 的那条才是真正的任务
            if msg.meta.get("zone") == "task" and msg.content.strip():
                return msg.content.strip()
        for msg in reversed(messages):
            if msg.role == "user" and msg.content.strip():
                return msg.content.strip().replace("[任务复述] 请继续完成: ", "")
        return messages[0].content.strip() if messages else ""

    @staticmethod
    def _executed_tools(messages: list[Message]) -> set[str]:
        """已经执行过哪些工具。三个来源都要看, 因为上下文形态不止一种。"""
        executed: set[str] = set()
        for msg in messages:
            executed.update(tc.name for tc in msg.tool_calls)
            if msg.role == "tool" and msg.name:
                executed.add(msg.name)
            executed.update(_ACTION_LINE.findall(msg.content))
        return executed

    @staticmethod
    def _observations(messages: list[Message]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for msg in messages:
            if msg.role == "tool":
                out.append((msg.name or "tool", msg.content))
                continue
            for tool, content in _OBS_LINE.findall(msg.content):
                if content.strip():
                    out.append((tool, content.strip()))
        # 同一条观察可能同时出现在 scratchpad 和降级提示里, 去重保序
        seen: set[str] = set()
        deduped = []
        for tool, content in out:
            key = f"{tool}:{content[:80]}"
            if key not in seen:
                seen.add(key)
                deduped.append((tool, content))
        return deduped

    @staticmethod
    def _split_goal(goal: str) -> list[str]:
        """把复合目标拆成子目标 —— 找并列连接词。"""
        parts = re.split(r"[;；]|,\s*(?=然后|并且|同时)|然后|并且|同时|以及", goal)
        parts = [p.strip(" ,，。") for p in parts if len(p.strip(" ,，。")) > 3]
        if len(parts) <= 1:
            return [goal]
        return parts[:4]

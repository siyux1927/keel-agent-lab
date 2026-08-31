"""编排角色: Planner / Worker / Critic。

角色分工的本质是**给每个模型调用一个尽可能窄的职责**, 因为窄职责意味着:
提示词更短(省预算)、输出格式更容易约束(好解析)、失败更容易定位(好调试)。

  Planner  只做一件事: 把目标拆成带依赖的子任务图。不执行、不回答。
  Worker   一个独立的 ReAct 智能体, 只对自己那个子目标负责, 工具集可裁剪。
  Critic   只做质检: 收敛判定的把关人, 输出结构化裁决而不是新答案。
"""

from __future__ import annotations

from typing import Any, Optional

from keel.config import settings
from keel.llm.base import Message
from keel.loop.engine import ReActEngine
from keel.observability.trace import current_tracer
from keel.orchestrator.graph import TaskGraph, TaskNode
from keel.util import new_id, safe_json_loads

PLANNER_SYSTEM = """你是任务规划器。把用户目标拆解成可并行执行的子任务图。

严格输出 JSON:
{"subtasks":[{"id":"t1","title":"简短标题","goal":"完整的子任务描述","depends_on":[]}]}

规则:
- 最多 __MAX_SUBTASKS__ 个子任务。目标本身很简单时, 就只给 1 个子任务。
- depends_on 只在**确实需要上游产出**时才填, 能并行的绝不串行。
- 每个 goal 必须自包含: 执行者看不到用户原话, 只能看到这句 goal。
- 若存在需要汇总的场景, 最后加一个汇总节点, 依赖所有前置节点。"""


class Planner:
    """目标 → 任务 DAG。"""

    def __init__(self, provider: Any = None, max_subtasks: Optional[int] = None) -> None:
        self._provider = provider
        self.max_subtasks = max_subtasks or settings.orchestrator.max_subtasks

    @property
    def provider(self) -> Any:
        if self._provider is None:
            from keel.llm import get_provider

            self._provider = get_provider()
        return self._provider

    async def plan(self, goal: str, context: str = "") -> TaskGraph:
        with current_tracer().span("orchestrator.plan", kind="agent", goal=goal[:80]) as span:
            resp = await self.provider.chat(
                [
                    # 用字面替换而不是 str.format: 提示词里全是 JSON 花括号, format 会当成占位符
                    Message(role="system",
                            content=PLANNER_SYSTEM.replace("__MAX_SUBTASKS__", str(self.max_subtasks))),
                    Message(role="user",
                            content=f"目标: {goal}" + (f"\n\n已知背景:\n{context}" if context else "")),
                ],
                hint="plan", max_tokens=900,
            )
            data = safe_json_loads(resp.content) or {}
            graph = self._build_graph(goal, data.get("subtasks") or [])
            span.attributes.update({"subtasks": len(graph.nodes),
                                    "levels": graph.critical_path_hint})
            return graph

    def _build_graph(self, goal: str, subtasks: list[dict[str, Any]]) -> TaskGraph:
        graph = TaskGraph(goal=goal)
        if not subtasks:
            # 规划失败就退化成单节点直跑, 而不是报错 —— 编排是加速手段, 不该成为单点故障
            graph.add(TaskNode(id="t1", title=goal[:40], goal=goal))
            return graph
        for i, raw in enumerate(subtasks[: self.max_subtasks]):
            node_id = str(raw.get("id") or f"t{i + 1}")
            sub_goal = str(raw.get("goal") or raw.get("title") or "").strip()
            if not sub_goal:
                continue
            graph.add(
                TaskNode(
                    id=node_id,
                    title=str(raw.get("title") or sub_goal)[:60],
                    goal=sub_goal,
                    depends_on=[str(d) for d in (raw.get("depends_on") or [])],
                    tools=raw.get("tools"),
                )
            )
        if not graph.nodes:
            graph.add(TaskNode(id="t1", title=goal[:40], goal=goal))
        return graph


class Worker:
    """子任务执行器。每个节点一个独立 ReAct 实例, 状态互不污染。"""

    def __init__(self, provider: Any = None, memory: Any = None, tools: Optional[list[str]] = None) -> None:
        self.provider = provider
        self.memory = memory
        self.tools = tools

    async def execute(self, node: TaskNode, upstream: dict[str, str], session_id: str) -> str:
        engine = ReActEngine(
            provider=self.provider,
            memory=self.memory,
            tools=node.tools or self.tools,
            name=f"Worker[{node.id}]",
            verify=False,  # 质检统一交给 Critic, 每个 worker 都自检太贵
        )
        goal = node.goal
        if upstream:
            # 用方括号而不是【】: 汇总环节按【】切块, 标记撞车会导致上游内容被误当成独立子任务
            context = "\n\n".join(f"[上游 {k} 的产出]\n{v[:1200]}" for k, v in upstream.items())
            goal = f"{node.goal}\n\n可直接使用的上游结果:\n{context}"

        # 子任务用独立 session: 避免多个 worker 的工作记忆互相串味,
        # 但长期记忆(语义/程序性)仍然共享, 该复用的知识不会丢。
        # tracer 必须显式传下去: 不传的话 engine 会自建一条新 trace, 于是 worker 的
        # token 和耗时全部漏出编排的账本 —— 成本核算会凭空少掉一大半。
        result = await engine.run(
            goal, session_id=f"{session_id}::{node.id}", tracer=current_tracer()
        )
        node.meta.update({
            "steps": result.state.step_count,
            "tools": result.state.successful_tools(),
            "tokens": result.state.total_tokens,
            "status": result.state.status.value,
            "stop_reason": result.state.stop_reason.value if result.state.stop_reason else None,
        })
        return result.answer


CRITIC_SYSTEM = """你是终审质检员。判断答案是否真正解决了用户目标。

严格输出 JSON:
{"accepted":true/false,"score":0.0-1.0,"issues":["具体问题"],"suggestion":"如何改进"}

判定标准:
- 是否逐条回应了目标中的**每一个**诉求
- 关键结论是否有证据支撑, 有没有明显编造
- 是否存在自相矛盾或答非所问
宁可放行也不要吹毛求疵: 只有存在实质缺陷时才 accepted=false。"""


class Critic:
    def __init__(self, provider: Any = None) -> None:
        self._provider = provider

    @property
    def provider(self) -> Any:
        if self._provider is None:
            from keel.llm import get_provider

            self._provider = get_provider()
        return self._provider

    async def review(self, goal: str, answer: str, evidence: str = "") -> dict[str, Any]:
        with current_tracer().span("orchestrator.critic", kind="agent") as span:
            resp = await self.provider.chat(
                [
                    Message(role="system", content=CRITIC_SYSTEM),
                    Message(role="user", content=(
                        f"用户目标:\n{goal}\n\n候选答案:\n{answer}\n\n"
                        f"可核查的证据:\n{evidence[:2000] or '(无)'}"
                    )),
                ],
                hint="critic", max_tokens=500,
            )
            verdict = safe_json_loads(resp.content)
            if not isinstance(verdict, dict):
                # 解析不出来就默认放行: 质检失败不该阻断交付
                verdict = {"accepted": True, "score": 0.6, "issues": [],
                           "suggestion": "质检输出解析失败, 默认放行"}
            span.attributes.update({"accepted": verdict.get("accepted"),
                                    "score": verdict.get("score")})
            return verdict


AGGREGATOR_SYSTEM = """你是汇总者。把多个子任务的产出整合成一份面向用户的完整答案。

要求:
- 直接回答用户目标, 不要罗列"子任务1说...子任务2说..."这种流水账
- 子任务结论冲突时明确指出并说明取舍依据
- 有子任务失败时, 说明缺失了什么, 不要假装完整
- 不要引入子任务产出中不存在的新事实"""


class Aggregator:
    def __init__(self, provider: Any = None) -> None:
        self._provider = provider

    @property
    def provider(self) -> Any:
        if self._provider is None:
            from keel.llm import get_provider

            self._provider = get_provider()
        return self._provider

    async def aggregate(
        self, goal: str, graph: TaskGraph, feedback: str = ""
    ) -> str:
        from keel.orchestrator.graph import NodeStatus

        with current_tracer().span("orchestrator.aggregate", kind="agent"):
            parts = []
            for node in graph.nodes.values():
                if node.status == NodeStatus.DONE:
                    parts.append(f"【{node.title}】\n{node.result[:1500]}")
                else:
                    parts.append(f"【{node.title}】未完成({node.status.value}): {node.error}")
            body = "\n\n".join(parts)
            user = f"用户目标:\n{goal}\n\n各子任务产出:\n{body}"
            if feedback:
                user += f"\n\n上一版被质检打回, 必须修正:\n{feedback}"
            resp = await self.provider.chat(
                [
                    Message(role="system", content=AGGREGATOR_SYSTEM),
                    Message(role="user", content=user),
                ],
                hint="aggregate", max_tokens=1200,
            )
            return resp.content

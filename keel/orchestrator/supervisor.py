"""编排总控。

完整流程:

    Planner 规划 DAG
        → 单节点? 直接走 ReAct(不为编排而编排)
        → 多节点? DAGRunner 分层并发执行 Worker
        → Aggregator 汇总
        → Critic 质检 →(不合格)→ 带着意见重新汇总, 最多 N 轮

两个容易被忽略但很重要的判断:

  1. **单节点直跑**。编排本身有成本(多一次规划调用 + 一次汇总调用)。
     简单问题走编排就是纯浪费, 所以规划出来只有一个节点时直接降级。

  2. **质检只重汇总, 不重跑执行**。多数质量问题出在整合环节而不是取证环节,
     重跑所有 Worker 又慢又贵。只有当证据确实不足时才值得重新规划。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from keel.config import settings
from keel.observability.events import EventBus, NullBus
from keel.observability.trace import Tracer, trace_store, use_tracer
from keel.orchestrator.graph import CycleError, DAGRunner, NodeStatus, TaskGraph, TaskNode
from keel.orchestrator.roles import Aggregator, Critic, Planner, Worker


@dataclass
class OrchestratorResult:
    answer: str
    graph: TaskGraph
    tracer: Tracer
    dag_summary: dict[str, Any] = field(default_factory=dict)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "dag"

    @property
    def success(self) -> bool:
        return any(n.status == NodeStatus.DONE for n in self.graph.nodes.values())

    def to_dict(self, with_trace: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "answer": self.answer,
            "success": self.success,
            "mode": self.mode,
            "graph": self.graph.to_dict(),
            "dag_summary": self.dag_summary,
            "verdicts": self.verdicts,
            "usage": self.tracer.totals(),
        }
        if with_trace:
            d["trace"] = self.tracer.to_dict()
        return d


class Orchestrator:
    def __init__(
        self,
        provider: Any = None,
        memory: Any = None,
        tools: Optional[list[str]] = None,
        max_critic_rounds: Optional[int] = None,
    ) -> None:
        self.provider = provider
        self.memory = memory
        self.tools = tools
        self.planner = Planner(provider)
        self.worker = Worker(provider, memory, tools)
        self.aggregator = Aggregator(provider)
        self.critic = Critic(provider)
        self.runner = DAGRunner()
        self.max_critic_rounds = max_critic_rounds or settings.orchestrator.max_critic_rounds

    async def run(
        self,
        goal: str,
        session_id: str = "default",
        bus: Optional[EventBus] = None,
        tracer: Optional[Tracer] = None,
    ) -> OrchestratorResult:
        bus = bus or NullBus()
        tracer = tracer or Tracer(name=f"orchestrate:{goal[:40]}")

        with use_tracer(tracer):
            with tracer.span("orchestrator.run", kind="agent", goal=goal) as root:
                bus.emit("orchestrator.start", goal=goal, trace_id=tracer.trace_id)

                # --- 1. 规划 ---------------------------------------------
                memory_context = await self._recall_context(goal, session_id)
                graph = await self.planner.plan(goal, context=memory_context)
                try:
                    graph.validate()
                except CycleError as exc:
                    # 规划器产出了环: 剪掉全部依赖降级成全并行, 而不是让整轮失败
                    bus.emit("plan.repaired", error=str(exc))
                    for node in graph.nodes.values():
                        node.depends_on = []
                bus.emit("plan.ready", **graph.to_dict())

                # --- 2. 单节点直跑 ----------------------------------------
                if len(graph.nodes) <= 1:
                    node = next(iter(graph.nodes.values()), None) or graph.add(
                        TaskNode(id="t1", title=goal[:40], goal=goal)
                    )
                    bus.emit("orchestrator.direct", reason="任务无需拆分, 降级为单 Agent 直跑")
                    node.result = await self.worker.execute(node, {}, session_id)
                    node.status = NodeStatus.DONE
                    root.attributes["mode"] = "direct"
                    bus.emit("orchestrator.finish", answer=node.result, mode="direct",
                             usage=tracer.totals())
                    trace_store.put(tracer)
                    return OrchestratorResult(node.result, graph, tracer, mode="direct")

                # --- 3. DAG 并发执行 --------------------------------------
                async def executor(node: TaskNode, upstream: dict[str, str]) -> str:
                    return await self.worker.execute(node, upstream, session_id)

                dag_summary = await self.runner.run(graph, executor, bus=bus)

                # --- 4. 汇总 + 质检回路 -----------------------------------
                evidence = "\n".join(
                    f"{n.title}: {n.result[:400]}"
                    for n in graph.nodes.values() if n.status == NodeStatus.DONE
                )
                answer = await self.aggregator.aggregate(goal, graph)
                verdicts: list[dict[str, Any]] = []
                for round_i in range(self.max_critic_rounds):
                    verdict = await self.critic.review(goal, answer, evidence)
                    verdicts.append(verdict)
                    bus.emit("critic.verdict", round=round_i + 1, **verdict)
                    if verdict.get("accepted", True):
                        break
                    feedback = "; ".join(verdict.get("issues", [])) or verdict.get("suggestion", "")
                    answer = await self.aggregator.aggregate(goal, graph, feedback=feedback)

                await self._writeback(goal, answer, graph, session_id, bus)

                root.attributes.update({"mode": "dag", **dag_summary})
                bus.emit("orchestrator.finish", answer=answer, mode="dag",
                         usage=tracer.totals(), **dag_summary)

        trace_store.put(tracer)
        return OrchestratorResult(answer, graph, tracer, dag_summary, verdicts, mode="dag")

    # ------------------------------------------------------------------

    async def _recall_context(self, goal: str, session_id: str) -> str:
        """规划前先查记忆: 同类任务以前怎么拆的, 有没有已知约束。"""
        memory = self.memory
        if memory is None:
            from keel.memory import get_memory

            memory = get_memory()
            self.memory = memory
        try:
            results = await memory.retrieve(goal, session_id=session_id, top_k=4)
            return "\n".join(f"- {r.record.content[:200]}" for r in results)
        except Exception:
            return ""

    async def _writeback(
        self, goal: str, answer: str, graph: TaskGraph, session_id: str, bus: EventBus
    ) -> None:
        """把这次编排的成功拆解方式固化成技能, 下次遇到同类目标可以直接复用。"""
        try:
            await self.memory.observe_turn(session_id, "user", goal)
            await self.memory.observe_turn(session_id, "assistant", answer)
            done = [n for n in graph.nodes.values() if n.status == NodeStatus.DONE]
            if done:
                await self.memory.record_trajectory(
                    goal=goal,
                    steps=[f"{n.title}" for n in done],
                    success=len(done) == len(graph.nodes),
                    session_id=session_id,
                )
            insight = await self.memory.maybe_reflect(session_id)
            if insight:
                bus.emit("memory.reflected", **insight)
        except Exception as exc:  # noqa: BLE001
            bus.emit("memory.error", error=str(exc))

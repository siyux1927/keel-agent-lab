"""任务 DAG 与并发调度器。

为什么用 DAG 而不是 Supervisor 动态派发:

  Supervisor 每派一次任务都要过一遍模型, 决策成本高, 而且天然是串行的 ——
  主控必须等子任务回来才能决定下一步。DAG 把"规划"和"执行"彻底分开:
  规划一次性产出依赖图, 执行时同一层的节点可以真并发。

  代价是灵活性: 执行到一半发现规划错了, DAG 改不了。所以上层用 Critic 兜底 ——
  产出不合格就带着意见重新规划。相当于在"可控"和"灵活"之间做了一次分层。

调度器本身要处理的三件事: 环检测(规划器会生成非法图)、层内并发上限、
失败传播(依赖失败的节点应该跳过而不是拿着空输入硬跑)。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from agentp.config import settings
from agentp.observability.events import EventBus, NullBus
from agentp.observability.trace import current_tracer
from agentp.util import new_id


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"   # 上游失败, 不再执行
    TIMEOUT = "timeout"


@dataclass
class TaskNode:
    id: str
    title: str
    goal: str
    depends_on: list[str] = field(default_factory=list)
    tools: Optional[list[str]] = None
    retry: int = 1
    status: NodeStatus = NodeStatus.PENDING
    result: str = ""
    error: str = ""
    duration_ms: float = 0.0
    attempts: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "goal": self.goal,
            "depends_on": self.depends_on, "status": self.status.value,
            "result": self.result[:2000], "error": self.error,
            "duration_ms": round(self.duration_ms, 2), "attempts": self.attempts,
            "meta": self.meta,
        }


class CycleError(ValueError):
    pass


class TaskGraph:
    def __init__(self, goal: str = "") -> None:
        self.id = new_id("dag")
        self.goal = goal
        self.nodes: dict[str, TaskNode] = {}

    def add(self, node: TaskNode) -> TaskNode:
        self.nodes[node.id] = node
        return node

    def validate(self) -> None:
        """丢掉不存在的依赖 + 环检测。

        规划器是模型产出的, 生成非法图是常态而不是异常, 所以这里必须能自愈:
        指向不存在节点的依赖直接剪掉, 有环则抛错交给上层降级成串行。
        """
        for node in self.nodes.values():
            node.depends_on = [d for d in node.depends_on if d in self.nodes and d != node.id]

        color: dict[str, int] = {}  # 0=未访问 1=在栈上 2=已完成

        def visit(nid: str, path: list[str]) -> None:
            state = color.get(nid, 0)
            if state == 1:
                cycle = path[path.index(nid):] + [nid]
                raise CycleError(f"任务图存在环: {' → '.join(cycle)}")
            if state == 2:
                return
            color[nid] = 1
            for dep in self.nodes[nid].depends_on:
                visit(dep, path + [nid])
            color[nid] = 2

        for nid in self.nodes:
            visit(nid, [])

    def levels(self) -> list[list[TaskNode]]:
        """拓扑分层。同一层内无依赖关系, 可以并发。"""
        self.validate()
        remaining = {nid: set(node.depends_on) for nid, node in self.nodes.items()}
        out: list[list[TaskNode]] = []
        done: set[str] = set()
        while remaining:
            ready = [nid for nid, deps in remaining.items() if not (deps - done)]
            if not ready:
                raise CycleError("拓扑排序卡住, 任务图不合法")
            ready.sort()
            out.append([self.nodes[nid] for nid in ready])
            done.update(ready)
            for nid in ready:
                remaining.pop(nid)
        return out

    def upstream_results(self, node: TaskNode) -> dict[str, str]:
        return {
            dep: self.nodes[dep].result
            for dep in node.depends_on
            if dep in self.nodes and self.nodes[dep].status == NodeStatus.DONE
        }

    @property
    def critical_path_hint(self) -> int:
        return len(self.levels())

    def to_dict(self) -> dict[str, Any]:
        try:
            levels = [[n.id for n in level] for level in self.levels()]
        except CycleError:
            levels = []
        return {
            "id": self.id, "goal": self.goal,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "levels": levels,
        }


# --------------------------------------------------------------------------

NodeExecutor = Callable[[TaskNode, dict[str, str]], Awaitable[str]]


class DAGRunner:
    def __init__(
        self,
        max_concurrency: Optional[int] = None,
        node_timeout_s: Optional[float] = None,
    ) -> None:
        cfg = settings.orchestrator
        self.max_concurrency = max_concurrency or cfg.max_concurrency
        self.node_timeout_s = node_timeout_s or cfg.node_timeout_s

    async def run(
        self, graph: TaskGraph, executor: NodeExecutor, bus: Optional[EventBus] = None
    ) -> dict[str, Any]:
        bus = bus or NullBus()
        semaphore = asyncio.Semaphore(self.max_concurrency)
        levels = graph.levels()
        started = time.perf_counter()
        bus.emit("dag.start", nodes=len(graph.nodes), levels=len(levels),
                 plan=[[n.id for n in lv] for lv in levels])

        async def run_node(node: TaskNode) -> None:
            # 上游有任何一个没成功, 就跳过 —— 拿着残缺输入硬跑只会产出垃圾并浪费预算
            failed_deps = [
                d for d in node.depends_on
                if graph.nodes[d].status != NodeStatus.DONE
            ]
            if failed_deps:
                node.status = NodeStatus.SKIPPED
                node.error = f"上游任务未成功: {failed_deps}"
                bus.emit("dag.node", id=node.id, status=node.status.value, error=node.error)
                return

            async with semaphore:
                node.status = NodeStatus.RUNNING
                bus.emit("dag.node", id=node.id, title=node.title, status="running")
                node_started = time.perf_counter()
                upstream = graph.upstream_results(node)
                for attempt in range(node.retry + 1):
                    node.attempts = attempt + 1
                    try:
                        node.result = await asyncio.wait_for(
                            executor(node, upstream), timeout=self.node_timeout_s
                        )
                        node.status = NodeStatus.DONE
                        break
                    except asyncio.TimeoutError:
                        node.status = NodeStatus.TIMEOUT
                        node.error = f"节点执行超时(>{self.node_timeout_s}s)"
                    except Exception as exc:  # noqa: BLE001
                        node.status = NodeStatus.FAILED
                        node.error = f"{type(exc).__name__}: {exc}"
                    if attempt < node.retry:
                        await asyncio.sleep(0.4 * (2 ** attempt))
                node.duration_ms = (time.perf_counter() - node_started) * 1000
                bus.emit("dag.node", id=node.id, status=node.status.value,
                         duration_ms=round(node.duration_ms, 1),
                         preview=node.result[:200], error=node.error)

        tracer = current_tracer()
        for depth, level in enumerate(levels):
            with tracer.span(f"dag.level[{depth}]", kind="agent", nodes=[n.id for n in level]):
                await asyncio.gather(*(run_node(n) for n in level))

        wall_ms = (time.perf_counter() - started) * 1000
        serial_ms = sum(n.duration_ms for n in graph.nodes.values())
        summary = {
            "nodes": len(graph.nodes),
            "levels": len(levels),
            "done": sum(1 for n in graph.nodes.values() if n.status == NodeStatus.DONE),
            "failed": sum(1 for n in graph.nodes.values()
                          if n.status in (NodeStatus.FAILED, NodeStatus.TIMEOUT)),
            "skipped": sum(1 for n in graph.nodes.values() if n.status == NodeStatus.SKIPPED),
            "wall_ms": round(wall_ms, 2),
            "serial_ms": round(serial_ms, 2),
            # 并发到底省了多少时间 —— 编排值不值得做, 看这个数
            "speedup": round(serial_ms / wall_ms, 2) if wall_ms > 0 else 1.0,
        }
        bus.emit("dag.finish", **summary)
        return summary

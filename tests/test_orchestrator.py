"""编排: DAG 校验、分层并发、失败传播、端到端。"""

from __future__ import annotations

import asyncio

import pytest

from agentp.orchestrator import (
    CycleError,
    DAGRunner,
    NodeStatus,
    Orchestrator,
    TaskGraph,
    TaskNode,
)


def _graph(edges: dict[str, list[str]]) -> TaskGraph:
    graph = TaskGraph(goal="test")
    for node_id, deps in edges.items():
        graph.add(TaskNode(id=node_id, title=node_id, goal=f"完成 {node_id}", depends_on=list(deps)))
    return graph


# ==========================================================================
# 图结构
# ==========================================================================


def test_levels_group_independent_nodes():
    graph = _graph({"a": [], "b": [], "c": ["a", "b"], "d": ["c"]})
    levels = [sorted(n.id for n in level) for level in graph.levels()]
    assert levels == [["a", "b"], ["c"], ["d"]]


def test_cycle_is_detected():
    graph = _graph({"a": ["b"], "b": ["a"]})
    with pytest.raises(CycleError):
        graph.levels()


def test_self_dependency_is_stripped():
    """模型生成自环是常见错误, 直接剪掉比报错更实用。"""
    graph = _graph({"a": ["a"]})
    graph.validate()
    assert graph.nodes["a"].depends_on == []


def test_dangling_dependency_is_stripped():
    graph = _graph({"a": ["ghost"]})
    graph.validate()
    assert graph.nodes["a"].depends_on == []
    assert len(graph.levels()) == 1


# ==========================================================================
# 调度
# ==========================================================================


async def test_same_level_nodes_run_concurrently():
    graph = _graph({"a": [], "b": [], "c": []})

    async def executor(node, upstream):
        await asyncio.sleep(0.08)
        return f"{node.id} done"

    started = asyncio.get_event_loop().time()
    summary = await DAGRunner(max_concurrency=3).run(graph, executor)
    elapsed = asyncio.get_event_loop().time() - started

    assert summary["done"] == 3
    assert elapsed < 0.2                # 串行需要 0.24s
    assert summary["speedup"] > 1.8


async def test_concurrency_limit_is_respected():
    graph = _graph({f"n{i}": [] for i in range(6)})
    peak = 0
    current = 0

    async def executor(node, upstream):
        nonlocal peak, current
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.05)
        current -= 1
        return "ok"

    await DAGRunner(max_concurrency=2).run(graph, executor)
    assert peak <= 2


async def test_upstream_results_are_passed_downstream():
    graph = _graph({"a": [], "b": ["a"]})
    seen: dict[str, dict] = {}

    async def executor(node, upstream):
        seen[node.id] = dict(upstream)
        return f"{node.id} 的产出"

    await DAGRunner().run(graph, executor)
    assert seen["a"] == {}
    assert seen["b"] == {"a": "a 的产出"}


async def test_failed_node_skips_dependents():
    """依赖失败时跳过而不是硬跑 —— 拿着残缺输入只会产出垃圾还烧预算。"""
    graph = _graph({"a": [], "b": ["a"], "c": []})

    async def executor(node, upstream):
        if node.id == "a":
            raise RuntimeError("下游服务不可用")
        return "ok"

    summary = await DAGRunner().run(graph, executor)
    assert graph.nodes["a"].status == NodeStatus.FAILED
    assert graph.nodes["b"].status == NodeStatus.SKIPPED
    assert graph.nodes["c"].status == NodeStatus.DONE  # 无关分支不受牵连
    assert summary["failed"] == 1 and summary["skipped"] == 1


async def test_node_retries_before_failing():
    graph = _graph({"a": []})
    graph.nodes["a"].retry = 2
    attempts = 0

    async def executor(node, upstream):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("暂时失败")
        return "终于成功"

    await DAGRunner().run(graph, executor)
    assert graph.nodes["a"].status == NodeStatus.DONE
    assert graph.nodes["a"].attempts == 3


async def test_node_timeout_is_enforced():
    graph = _graph({"a": []})
    graph.nodes["a"].retry = 0

    async def executor(node, upstream):
        await asyncio.sleep(5)
        return "永远到不了"

    await DAGRunner(node_timeout_s=0.05).run(graph, executor)
    assert graph.nodes["a"].status == NodeStatus.TIMEOUT


# ==========================================================================
# 端到端
# ==========================================================================


async def test_orchestrator_splits_and_aggregates(memory):
    orchestrator = Orchestrator(memory=memory)
    result = await orchestrator.run(
        "查一下知识库里的发布规定; 同时计算 (99+1)*3 的结果", session_id="o1"
    )
    assert result.success
    assert result.mode == "dag"
    assert len(result.graph.nodes) >= 2
    assert result.dag_summary["failed"] == 0
    assert "300" in result.answer or any("300" in n.result for n in result.graph.nodes.values())


async def test_orchestrator_falls_back_to_direct_for_simple_goal(memory):
    """简单任务不该为编排付出额外的规划 + 汇总成本。"""
    orchestrator = Orchestrator(memory=memory)
    result = await orchestrator.run("计算 6*7", session_id="o2")
    assert result.mode == "direct"
    assert len(result.graph.nodes) == 1


async def test_orchestrator_records_skill(memory):
    orchestrator = Orchestrator(memory=memory)
    await orchestrator.run("查一下发布规定; 再计算 2+2", session_id="o3")
    from agentp.memory import MemoryLayer

    assert memory.store.all(MemoryLayer.PROCEDURAL)


async def test_worker_spans_land_in_orchestrator_trace(memory):
    """Worker 的 span 必须挂在编排这一条 trace 上。

    回归用例。原先 Worker 调 engine.run() 没把 tracer 传下去, 引擎于是自建了一条
    新 trace —— 子任务里所有的 LLM 调用和 token 全部漏出编排的账本, 实测编排模式的
    总 token 被少记了 80% 以上。成本核算错到这个程度, 就没法回答"钱花在哪"了。
    """
    result = await Orchestrator(memory=memory).run(
        "查一下知识库里的发布规定; 同时计算 (99+1)*3 的结果", session_id="o5"
    )
    names = [s.name for s in result.tracer.spans]
    assert any(n.startswith("agent.run") for n in names), "worker 的 agent.run 不在编排 trace 里"
    assert any(n.startswith("loop.step") for n in names), "worker 的步级 span 不在编排 trace 里"
    # token 必须覆盖 worker 的消耗, 而不只是规划/汇总/质检那几次
    assert result.tracer.totals()["total_tokens"] > 0
    worker_tokens = sum(n.meta.get("tokens", 0) for n in result.graph.nodes.values())
    assert result.tracer.totals()["total_tokens"] >= worker_tokens


async def test_orchestrator_survives_cyclic_plan(memory):
    """规划器吐出环时降级成全并行, 而不是让整轮失败。"""
    orchestrator = Orchestrator(memory=memory)
    original = orchestrator.planner.plan

    async def bad_plan(goal, context=""):
        graph = _graph({"a": ["b"], "b": ["a"]})
        graph.goal = goal
        return graph

    orchestrator.planner.plan = bad_plan
    try:
        result = await orchestrator.run("任意目标", session_id="o4")
        assert result.answer
        assert result.dag_summary["done"] == 2
    finally:
        orchestrator.planner.plan = original

"""上下文工程: 切片、预算、压缩、组装。"""

from __future__ import annotations

import pytest

from keel.context import (
    BudgetAllocator,
    ContextAssembler,
    ContextItem,
    ExtractiveCompressor,
    MiddleOutCompressor,
    build_zones,
    chunk_stats,
    get_chunker,
)
from keel.context.budget import Zone
from keel.llm.base import Message
from keel.util import count_tokens


# ==========================================================================
# 切片
# ==========================================================================


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "structure", "semantic"])
async def test_chunk_respects_token_budget(sample_doc, strategy):
    """所有策略都必须尊重 token 上限, 否则下游预算调度就是空谈。"""
    chunks = await get_chunker(strategy, chunk_tokens=120, overlap=20).split(sample_doc)
    assert chunks
    # 重叠会让块略微超出目标值, 但不能失控
    assert all(c.tokens <= 120 + 40 for c in chunks), [c.tokens for c in chunks]
    assert all(c.tokens > 0 for c in chunks)


async def test_chunks_cover_original_content(sample_doc):
    chunks = await get_chunker("recursive", chunk_tokens=100, overlap=0).split(sample_doc)
    joined = "".join(c.text for c in chunks)
    assert "遗忘曲线半衰期默认 72 小时" in joined.replace("\n", "")


async def test_structure_chunker_carries_heading_path(sample_doc):
    """结构切片必须把标题路径下推到每个块 —— 这是块脱离上下文后仍可理解的关键。"""
    chunks = await get_chunker("structure", chunk_tokens=200).split(sample_doc)
    memory_chunks = [c for c in chunks if "遗忘曲线" in c.text]
    assert memory_chunks
    assert memory_chunks[0].heading_path == ["Agent 系统设计", "记忆系统"]


async def test_structure_chunker_keeps_code_fence_atomic(sample_doc):
    chunks = await get_chunker("structure", chunk_tokens=500).split(sample_doc)
    code_chunks = [c for c in chunks if "def allocate" in c.text]
    assert len(code_chunks) == 1
    assert code_chunks[0].text.count("```") == 2  # 围栏成对, 没被切开


async def test_hierarchical_links_children_to_parents(sample_doc):
    chunks = await get_chunker("hierarchical", parent_tokens=400, child_tokens=80).split(sample_doc)
    parents = {c.id for c in chunks if c.meta.get("role") == "parent"}
    children = [c for c in chunks if c.meta.get("role") == "child"]
    assert parents and children
    assert all(c.parent_id in parents for c in children)
    # 子块用于检索, 必须受更小的预算约束(父块可以远大于此)
    assert all(c.tokens <= 80 + 40 for c in children)
    assert max(c.tokens for c in chunks if c.meta.get("role") == "parent") >= max(
        c.tokens for c in children
    )


async def test_chunk_stats_reports_variance(sample_doc):
    chunks = await get_chunker("structure", chunk_tokens=200).split(sample_doc)
    stats = chunk_stats(chunks)
    assert stats["count"] == len(chunks)
    assert stats["min_tokens"] <= stats["avg_tokens"] <= stats["max_tokens"]


async def test_empty_input_yields_no_chunks():
    for strategy in ("fixed", "recursive", "structure", "semantic", "hierarchical"):
        assert await get_chunker(strategy).split("   \n  ") == []


# ==========================================================================
# 预算
# ==========================================================================


def test_budget_never_exceeds_when_all_droppable():
    allocator = BudgetAllocator(context_window=4000, reserve_for_output=500)
    zones = [
        Zone("a", priority=2, min_tokens=50, weight=1.0, requested=5000),
        Zone("b", priority=3, min_tokens=50, weight=2.0, requested=5000),
    ]
    plan = allocator.allocate(zones)
    assert plan.granted_total <= plan.total_budget
    assert plan.overflow == 0
    # 权重更高的分区拿得更多
    assert plan.get("b") > plan.get("a")


def test_budget_drops_low_priority_zones_first():
    plan = BudgetAllocator(context_window=700, reserve_for_output=100).allocate(
        build_zones({
            "system": 200, "task": 70, "tools": 300,
            "history": 4000, "retrieved": 9000, "semantic": 3000,
        })
    )
    assert not plan.allocations["system"].dropped
    assert not plan.allocations["task"].dropped
    assert plan.allocations["history"].dropped
    assert plan.allocations["retrieved"].dropped


def test_budget_reports_overflow_for_undroppable_zones():
    """不可丢弃分区撑爆预算时必须如实上报, 不能假装压缩过了。"""
    plan = BudgetAllocator(context_window=400, reserve_for_output=100).allocate(
        build_zones({"system": 500, "task": 200, "tools": 400})
    )
    assert plan.overflow > 0
    assert plan.allocations["task"].granted == 200  # 用户问题绝不被裁


def test_budget_fills_all_available_capacity():
    plan = BudgetAllocator(context_window=8192).allocate(
        build_zones({
            "system": 200, "task": 60, "tools": 500, "semantic": 3000,
            "retrieved": 9000, "history": 4000, "scratchpad": 2000,
            "episodic": 800, "procedural": 400,
        })
    )
    # 注水算法必须把预算用满, 剩余不超过分区数(整除误差)
    assert plan.total_budget - plan.granted_total <= len(plan.allocations)


def test_budget_zero_content_zone_is_marked_dropped():
    plan = BudgetAllocator().allocate(build_zones({"system": 100, "task": 50}))
    assert plan.allocations["retrieved"].dropped
    assert plan.allocations["retrieved"].reason == "内容为空"


# ==========================================================================
# 压缩
# ==========================================================================


async def test_extractive_compressor_fits_budget(provider):
    text = "。".join(f"这是第 {i} 个句子, 讲的是上下文预算与切片策略的关系" for i in range(60))
    result = await ExtractiveCompressor(provider).compress(text, budget=150, query="切片策略")
    assert result.final_tokens <= 150
    assert result.ratio < 1.0
    assert result.text  # 不能压没了


async def test_middle_out_keeps_both_ends(provider):
    head, tail = "开头的关键结论是必须保留的", "结尾的行动项也必须保留"
    text = head + "。" + "。".join(f"中间第 {i} 段无关内容" for i in range(200)) + "。" + tail
    result = await MiddleOutCompressor(inner=ExtractiveCompressor(provider)).compress(
        text, budget=200
    )
    assert head[:8] in result.text
    assert tail[-8:] in result.text
    assert result.final_tokens <= 260  # 含标注文字的少量开销


async def test_compressor_noop_when_within_budget(provider):
    result = await ExtractiveCompressor(provider).compress("短文本", budget=1000)
    assert result.text == "短文本"
    assert result.ratio == 1.0


# ==========================================================================
# 组装
# ==========================================================================


async def test_assembler_puts_task_at_both_ends(provider):
    assembler = ContextAssembler(provider)
    ctx = await assembler.build(
        task="计算季度增长率",
        system_prompt="你是助手",
        items=[ContextItem("一些背景事实", zone="semantic", score=0.9, source="kb")],
        scratchpad="### 第 1 步\n动作: calculator(x=1)\n观察: [calculator] 1",
    )
    zones = [m.meta.get("zone") for m in ctx.messages]
    assert zones[0] == "system"
    assert "task" in zones
    # 有 scratchpad 时必须在末尾复述任务, 对抗 lost-in-the-middle
    assert zones[-1] == "task_restate"


async def test_assembler_respects_tight_window(provider):
    assembler = ContextAssembler(provider, BudgetAllocator(context_window=900, reserve_for_output=200))
    items = [
        ContextItem(f"第 {i} 条检索结果, 内容比较长需要占据不少 token 预算" * 3,
                    zone="retrieved", score=1.0 - i * 0.01, source=f"doc{i}")
        for i in range(40)
    ]
    ctx = await assembler.build(task="总结", system_prompt="你是助手", items=items)
    assert ctx.total_tokens <= 900
    assert ctx.zone_report["retrieved"]["dropped"] > 0  # 低分条目被整条丢弃


async def test_assembler_records_provenance(provider):
    ctx = await ContextAssembler(provider).build(
        task="问题",
        items=[ContextItem("事实A", zone="semantic", score=0.8, source="kb:handbook")],
    )
    assert ctx.provenance
    assert ctx.provenance[0]["source"] == "kb:handbook"
    assert ctx.provenance[0]["zone"] == "semantic"


async def test_assembler_compresses_long_history(provider):
    history = [
        Message(role="user" if i % 2 == 0 else "assistant",
                content=f"第 {i} 轮对话内容, 包含了一些需要被摘要压缩的细节信息")
        for i in range(120)
    ]
    ctx = await ContextAssembler(
        provider, BudgetAllocator(context_window=3000, reserve_for_output=500)
    ).build(task="继续", system_prompt="你是助手", history=history)
    report = ctx.zone_report["history"]
    assert report["method"] != "verbatim"
    assert report["final"] <= report["granted"] + 5

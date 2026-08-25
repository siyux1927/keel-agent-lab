"""工具安全边界、基础设施、可观测性。"""

from __future__ import annotations

import asyncio

import pytest

from agentp.observability.trace import Tracer, use_tracer
from agentp.tools import get_registry
from agentp.util import (
    BM25,
    cosine,
    count_tokens,
    hash_embed,
    mmr,
    safe_json_loads,
    stable_hash,
    truncate_to_tokens,
)


# ==========================================================================
# 工具安全
# ==========================================================================


async def test_calculator_computes_correctly():
    result = await get_registry().call("calculator", {"expression": "(128*7+56)/4"})
    assert result.ok
    assert "238" in result.content


async def test_calculator_rejects_arbitrary_code():
    """用 AST 白名单而不是 eval —— 这是能不能上线的分界线。"""
    for payload in ("__import__('os').system('echo hacked')",
                    "open('/etc/passwd').read()",
                    "[].__class__.__base__"):
        result = await get_registry().call("calculator", {"expression": payload})
        assert not result.ok


async def test_calculator_rejects_compute_bomb():
    result = await get_registry().call("calculator", {"expression": "9**999999999"})
    assert not result.ok


async def test_sandbox_blocks_import_and_attribute_access():
    registry = get_registry()
    for code in ("import os", "print((1).__class__)", "eval('1+1')", "while True: pass"):
        result = await registry.call("python_exec", {"code": code})
        assert not result.ok, code


async def test_sandbox_runs_safe_code():
    result = await get_registry().call(
        "python_exec", {"code": "result = sum(x*x for x in range(5))"}
    )
    assert result.ok
    assert "30" in result.content


async def test_read_file_blocks_path_traversal():
    result = await get_registry().call("read_file", {"path": "../../../etc/passwd"})
    assert not result.ok
    assert "拒绝访问" in result.error or "不存在" in result.error


async def test_missing_required_arg_returns_readable_error():
    """参数错误要变成可读的 observation 回灌给模型, 而不是抛异常中断循环。"""
    result = await get_registry().call("calculator", {})
    assert not result.ok
    assert "缺少必填参数" in result.error
    assert result.meta.get("validation")


async def test_numeric_string_arg_is_coerced():
    """小模型经常把数字包成字符串, 能转就别报错。"""
    result = await get_registry().call("web_search", {"query": "agent", "top_k": "2"})
    assert result.ok


async def test_unknown_tool_lists_available_ones():
    result = await get_registry().call("nonexistent_tool", {})
    assert not result.ok
    assert "calculator" in result.error


async def test_enum_violation_is_rejected():
    result = await get_registry().call("now", {"tz": "Mars/Olympus"})
    assert not result.ok
    assert "取值非法" in result.error


# ==========================================================================
# Token / 向量
# ==========================================================================


def test_token_count_monotonic_and_nonzero():
    assert count_tokens("") == 0
    assert count_tokens("上下文") > 0
    assert count_tokens("短") < count_tokens("短" * 50)


def test_truncate_respects_budget():
    text = "上下文工程决定了智能体的上限。" * 40
    for budget in (10, 50, 200):
        assert count_tokens(truncate_to_tokens(text, budget)) <= budget


def test_truncate_from_end_keeps_tail():
    text = "开头内容" + "x" * 500 + "结尾内容"
    assert "结尾内容" in truncate_to_tokens(text, 40, from_end=True)


def test_hash_embed_is_deterministic_and_normalized():
    a, b = hash_embed("上下文工程"), hash_embed("上下文工程")
    assert a == b
    assert abs(sum(x * x for x in a) ** 0.5 - 1.0) < 1e-6


def test_cosine_reflects_lexical_similarity():
    base = hash_embed("切片策略决定检索质量")
    similar = hash_embed("切片策略影响检索质量上限")
    unrelated = hash_embed("今天天气不错适合出门散步")
    assert cosine(base, similar) > cosine(base, unrelated)


def test_bm25_ranks_exact_match_first():
    index = BM25()
    index.add("d1", "熔断器在连续失败后打开, 拒绝后续调用")
    index.add("d2", "上下文窗口是最稀缺的资源")
    top = index.search("熔断器", top_k=1)
    assert top and top[0][0] == "d1"


def test_mmr_prefers_diverse_results():
    """纯 top-k 会选出一堆重复内容, 白白吃掉上下文预算。"""
    query = hash_embed("切片策略")
    dup_a = hash_embed("切片策略决定检索质量")
    dup_b = hash_embed("切片策略决定检索质量上限")
    other = hash_embed("熔断器保护下游服务不被重试打垮")
    picked = mmr(query, [("a", dup_a, 0.9), ("b", dup_b, 0.88), ("c", other, 0.6)],
                 top_k=2, lambda_=0.5)
    assert "c" in picked


def test_stable_hash_ignores_key_order():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


@pytest.mark.parametrize("raw,expected", [
    ('{"ok": true}', {"ok": True}),
    ('```json\n{"ok": 1}\n```', {"ok": 1}),
    ('模型的解释文字 {"ok": 2} 后面还有话', {"ok": 2}),
    ('完全不是 JSON', None),
])
def test_safe_json_loads_handles_llm_noise(raw, expected):
    """模型很少乖乖只输出 JSON, 解析器必须能从噪声里抠出来。"""
    assert safe_json_loads(raw) == expected


# ==========================================================================
# 可观测
# ==========================================================================


def test_span_tree_reflects_nesting():
    tracer = Tracer(name="t")
    with use_tracer(tracer):
        with tracer.span("outer", kind="agent"):
            with tracer.span("inner", kind="llm"):
                tracer.record_usage(100, 50, 0.001)
    tree = tracer.tree()
    assert len(tree) == 1
    assert tree[0]["name"] == "outer"
    assert tree[0]["children"][0]["name"] == "inner"
    assert tracer.totals()["total_tokens"] == 150


def test_error_span_is_marked_and_reraised():
    tracer = Tracer()
    with use_tracer(tracer), pytest.raises(ValueError):
        with tracer.span("boom"):
            raise ValueError("炸了")
    assert tracer.spans[0].status == "error"
    assert tracer.totals()["errors"] == 1


async def test_concurrent_spans_do_not_corrupt_parentage():
    """并发编排下 span 栈必须按任务隔离, 否则父子关系会串。"""
    tracer = Tracer()

    async def branch(name: str):
        with tracer.span(f"branch.{name}", kind="agent"):
            await asyncio.sleep(0.01)
            with tracer.span(f"child.{name}", kind="tool"):
                await asyncio.sleep(0.01)

    with use_tracer(tracer):
        with tracer.span("root", kind="agent") as root:
            await asyncio.gather(*(branch(n) for n in "abc"))

    by_name = {s.name: s for s in tracer.spans}
    for n in "abc":
        assert by_name[f"branch.{n}"].parent_id == root.id
        assert by_name[f"child.{n}"].parent_id == by_name[f"branch.{n}"].id


async def test_event_bus_streams_and_terminates():
    from agentp.observability.events import EventBus

    bus = EventBus()
    queue = bus.register()
    bus.emit("a", v=1)
    bus.emit("b", v=2)
    bus.close()
    received = [e.type async for e in bus.stream(queue)]
    assert received == ["a", "b"]


async def test_event_bus_drops_instead_of_blocking():
    """消费者跟不上时宁可丢展示事件, 也不能拖慢推理主循环。"""
    from agentp.observability.events import EventBus

    bus = EventBus(maxsize=2)
    bus.register()
    for i in range(50):
        bus.emit("flood", i=i)  # 不该抛异常也不该卡住
    assert len(bus.history) == 50

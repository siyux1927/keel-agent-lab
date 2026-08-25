"""命令行演示脚本 —— 五个场景, 每个对应一块核心能力。

    python scripts/demo.py all          # 全部跑一遍(约 30 秒)
    python scripts/demo.py chunk        # 切片策略对比
    python scripts/demo.py context      # 上下文预算调度与压缩
    python scripts/demo.py memory       # 四层记忆 / 混合检索 / 遗忘曲线
    python scripts/demo.py loop         # 循环护栏: 死循环、熔断、降级
    python scripts/demo.py orchestrate  # DAG 多智能体并发编排

默认跑在离线 Mock Provider 上, 不需要任何 API Key。
配好 .env 里的 AGENTP_PROVIDER=openai + AGENTP_API_KEY 就会自动切到真实模型。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentp.config import settings
from agentp.context import BudgetAllocator, ContextAssembler, chunk_stats, get_chunker
from agentp.context.assembler import default_system_prompt
from agentp.llm import get_provider
from agentp.loop import ReActEngine
from agentp.memory import MemoryLayer, get_memory
from agentp.orchestrator import Orchestrator
from agentp.tools import get_registry
from agentp.util import now_ts

DOC = """# Agent 平台工程手册

## 上下文工程
上下文窗口是智能体最稀缺的资源。切片策略决定检索质量的上限, 切错了后面再好的向量模型也救不回来。
定长切片实现简单但会把句子拦腰砍断。递归切片按分隔符优先级下切, 是通用场景的默认选择。

### 预算调度
预算分配应建模为带下界和优先级的资源分配问题, 而不是先到先得。
系统默认上下文窗口 8192 tokens, 输出预留 1024 tokens, 安全边界 5%。

```python
def allocate(zones, budget):
    # 先满足保命线, 再按权重注水
    return plan
```

## 记忆系统
记忆分为工作、情景、语义、程序性四层。遗忘曲线半衰期默认 72 小时。
被检索命中相当于复习一次, 会重置衰减计时。语义记忆的衰减速度是情景记忆的六分之一。

## 发布流程
发布窗口为每周二下午 14:00 到 16:00, 需要两人评审通过。紧急发布需要 SRE 值班同学签字。

## 循环工程
ReAct 循环必须配备预算护栏、死循环检测和熔断器。反思机制在连续失败两次后触发。
周期振荡通常意味着模型在两个错误假设之间横跳, 反思救不回来, 应当直接终止。
"""


def title(text: str) -> None:
    print(f"\n{'=' * 78}\n  {text}\n{'=' * 78}")


def sub(text: str) -> None:
    print(f"\n--- {text} " + "-" * max(0, 72 - len(text)))


# ==========================================================================


async def demo_chunk() -> None:
    title("切片策略对比 —— 同一份文档, 五种切法")
    print("关注 std_tokens(块大小方差): 方差大意味着检索打分失去可比性。\n")
    print(f"{'策略':<14}{'块数':>6}{'平均':>8}{'最小':>8}{'最大':>8}{'标准差':>10}  说明")
    notes = {
        "fixed": "定长, 会切断句子",
        "recursive": "按分隔符递归, 通用默认",
        "structure": "标题路径下推, 代码块原子",
        "semantic": "句向量找话题突变点",
        "hierarchical": "小块检索 + 大块喂模型",
    }
    for name in ("fixed", "recursive", "structure", "semantic", "hierarchical"):
        kwargs = ({"parent_tokens": 600, "child_tokens": 150}
                  if name == "hierarchical" else {"chunk_tokens": 200, "overlap": 30})
        chunks = await get_chunker(name, **kwargs).split(DOC, source="handbook.md")
        s = chunk_stats(chunks)
        print(f"{name:<14}{s['count']:>6}{s['avg_tokens']:>8}{s['min_tokens']:>8}"
              f"{s['max_tokens']:>8}{s['std_tokens']:>10}  {notes[name]}")

    sub("结构感知切片的标题路径(解决『块脱离上下文』)")
    for chunk in (await get_chunker("structure", chunk_tokens=200).split(DOC))[:4]:
        print(f"  [{' > '.join(chunk.heading_path)}]  {chunk.tokens} tok")
        print(f"     {chunk.text.strip()[:80].replace(chr(10), ' ')}...")


async def demo_context() -> None:
    title("上下文预算调度 —— 窗口收缩时会发生什么")
    from agentp.context.budget import build_zones

    demand = {"system": 220, "task": 80, "tools": 620, "procedural": 400,
              "semantic": 3000, "episodic": 900, "retrieved": 9000,
              "scratchpad": 2200, "history": 4200}
    print(f"各分区原始需求合计 {sum(demand.values())} tokens\n")

    for window in (16384, 8192, 2000):
        plan = BudgetAllocator(context_window=window).allocate(build_zones(demand))
        print(f"\n[上下文窗口 {window}] 可用预算 {plan.total_budget}, "
              f"压力 {plan.to_dict()['pressure']:.1f}x, 注水 {plan.rounds} 轮"
              + (f", 超额 {plan.overflow}" if plan.overflow else ""))
        for alloc in sorted(plan.allocations.values(), key=lambda a: -a.granted):
            state = ("丢弃" if alloc.dropped else
                     ("压缩" if alloc.needs_compression else "原样"))
            print(f"    {alloc.name:<12} 需求 {alloc.requested:>5} → 分配 {alloc.granted:>5}  [{state}]")

    sub("真实组装: 看清楚模型到底收到了什么")
    memory = get_memory()
    await memory.ingest_document(DOC, source="handbook.md", strategy="structure")
    items = await memory.build_context_items("发布流程和上下文预算的规定", session_id="demo")
    assembled = await ContextAssembler(get_provider()).build(
        task="总结发布流程和上下文预算的规定",
        system_prompt=default_system_prompt(),
        tools_schema=get_registry().schemas(),
        items=items,
    )
    print(f"  最终上下文 {assembled.total_tokens} tokens, {len(assembled.messages)} 条消息")
    print("  消息排布(对抗 lost-in-the-middle):")
    for msg in assembled.messages:
        print(f"    [{msg.meta.get('zone', msg.role):<12}] {len(msg.content):>5} 字符")
    print("\n  注入内容的来源(provenance) —— 出问题时能回答『模型为什么这么说』:")
    for p in assembled.provenance[:5]:
        print(f"    {p['zone']:<10} {p['source']:<28} score={p['score']:.3f}  {p['preview'][:40]}")


async def demo_memory() -> None:
    title("四层记忆 —— 写入、混合检索、遗忘、固化")
    memory = get_memory()

    sub("1. 文档入库(切片 → 向量化 → 语义记忆)")
    report = await memory.ingest_document(DOC, source="handbook.md", strategy="structure")
    print(f"  索引 {report['indexed']} 个块, 平均 {report['chunk_stats']['avg_tokens']} tokens")

    sub("2. 写入用户事实 + 语义去重")
    await memory.remember("用户偏好使用中文回答, 且要求先给结论",
                          layer=MemoryLayer.SEMANTIC, session_id="demo")
    dup = await memory.remember("用户偏好使用中文回答, 且要求先给结论",
                                layer=MemoryLayer.SEMANTIC, session_id="demo")
    print(f"  重复写入同一事实 → 未新建记录, 重要性提升至 {dup.importance:.1f} "
          f"(命中 {dup.metadata.get('duplicate_hits')} 次)")

    sub("3. 混合检索: 向量 + BM25 + 新鲜度 + 重要性")
    for query in ("发布窗口是什么时候", "遗忘曲线怎么设计的"):
        print(f"\n  查询「{query}」")
        print(f"    {'总分':>7}{'向量':>8}{'BM25':>8}{'新鲜':>8}{'重要':>8}   内容")
        for res in await memory.retrieve(query, top_k=3):
            b = res.breakdown
            print(f"    {res.score:>7.3f}{b['vector']:>8.3f}{b['bm25']:>8.3f}"
                  f"{b['recency']:>8.3f}{b['importance']:>8.3f}   "
                  f"{res.record.content.strip()[:52].replace(chr(10), ' ')}")

    sub("4. 程序性记忆: 成功轨迹固化成技能")
    await memory.record_trajectory("查询发布相关规定", ["kb_search"], success=True)
    await memory.record_trajectory("查询发布相关规定", ["kb_search", "web_search"], success=True)
    for rec in memory.store.all(MemoryLayer.PROCEDURAL):
        print(f"  {rec.content}")

    sub("5. 遗忘曲线: 强度随时间衰减, 检索命中会『复习』")
    old = await memory.remember("一条很久没被用到的琐碎记录",
                                layer=MemoryLayer.EPISODIC, importance=2.0)
    hl = memory.store.half_life_hours
    print(f"  半衰期 {hl}h, 各层衰减系数不同(语义记忆是情景记忆的 6 倍慢)")
    for hours in (0, 24, 72, 240):
        print(f"    {hours:>4}h 后强度: {old.strength(hl, now_ts() + hours * 3600):.4f}")
    old.last_access = now_ts() - 2000 * 3600
    print(f"  归档结果: {memory.decay_and_prune()}")

    sub("6. 反思固化: 从零散事件里长出高层抽象")
    for topic in ("用户询问了切片策略的选择依据", "用户关心遗忘曲线怎么调参",
                  "用户希望检索结果给出打分依据", "用户对编排并发度有疑问"):
        await memory.remember(topic, layer=MemoryLayer.EPISODIC, session_id="demo", importance=8.0)
    result = await memory.maybe_reflect("demo", force=True)
    for insight in (result or {}).get("insights", []):
        print(f"  → {insight}")

    print(f"\n  记忆统计: {memory.stats()['by_layer']}")


async def demo_loop() -> None:
    title("Loop Engineering —— 护栏怎么救场")
    memory = get_memory()
    await memory.ingest_document(DOC, source="handbook.md", strategy="structure")

    sub("场景 1: 正常收敛(多工具协作)")
    engine = ReActEngine(memory=memory, verify=False)
    result = await engine.run("帮我算一下 (128*7+56)/4, 再查一下知识库里的发布窗口", session_id="d1")
    for step in result.state.steps:
        tools = ", ".join(a.tool for a in step.actions) or "(给出答案)"
        print(f"  第 {step.index + 1} 步: {tools:<28} 信息增量 {step.progress:.2f}  "
              f"上下文 {step.context_tokens} tok")
    print(f"  终止: {result.state.stop_reason.value} | 步数 {result.state.step_count} | "
          f"tokens {result.state.total_tokens}")
    print(f"  答案: {result.answer[:120].replace(chr(10), ' ')}...")

    sub("场景 2: 死循环 —— 检测 → 反思 → 拉黑 → 终止")
    engine2 = ReActEngine(memory=memory, verify=False)
    result2 = await engine2.run("死循环演示 请反复确认当前时间", session_id="d2")
    for entry in result2.guard_log:
        print(f"  [第 {entry['step']} 步] {entry['action']:<14} {entry['reason']}")
    print(f"  终止: {result2.state.stop_reason.value} | 步数 {result2.state.step_count} "
          f"(上限 {settings.loop.max_steps}) | 重规划 {result2.state.replans} 次")
    print(f"  即便被中止, 仍产出降级答案: {result2.answer[:100].replace(chr(10), ' ')}...")

    sub("场景 3: 工具熔断 —— 连续失败后直接拒绝调用, 不再烧 token")
    registry = get_registry()
    for i in range(5):
        r = await registry.call("flaky", {"fail": True})
        print(f"  第 {i + 1} 次调用: {'成功' if r.ok else r.error[:52]}  "
              f"[熔断器 {registry.get('flaky').breaker.state}]")

    sub("成本核算(全链路 token 与耗时按 span 类型汇总)")
    for kind, stat in result.tracer.totals()["by_kind"].items():
        print(f"  {kind:<10} 调用 {stat['count']:>3} 次, {stat['duration_ms']:>8.1f} ms, "
              f"{stat['tokens']:>6} tokens, ${stat['cost_usd']:.6f}")


async def demo_orchestrate() -> None:
    title("多智能体编排 —— DAG 分层并发")
    memory = get_memory()
    await memory.ingest_document(DOC, source="handbook.md", strategy="structure")

    orchestrator = Orchestrator(memory=memory)
    result = await orchestrator.run(
        "查一下知识库里的发布流程规定; 同时搜索 ReAct 论文的核心思想; 然后计算 (99+1)*3 的结果",
        session_id="orc",
    )

    sub("规划出的任务图")
    for level_i, level in enumerate(result.graph.levels()):
        print(f"  第 {level_i + 1} 层(并发): " + " | ".join(f"{n.id} {n.title[:26]}" for n in level))

    sub("执行结果")
    for node in result.graph.nodes.values():
        print(f"  {node.id:<4} {node.status.value:<8} {node.duration_ms:>7.0f}ms  "
              f"deps={node.depends_on}  tools={node.meta.get('tools')}")

    s = result.dag_summary
    print(f"\n  并发收益: 串行需 {s['serial_ms']:.0f}ms, 实际 {s['wall_ms']:.0f}ms, "
          f"加速 {s['speedup']}×")
    print(f"  质检: {result.verdicts}")

    sub("最终答案")
    print("  " + result.answer[:600].replace("\n", "\n  "))

    sub("单智能体 vs 编排 —— 什么时候不该编排")
    simple = await Orchestrator(memory=memory).run("计算 6*7", session_id="orc2")
    print(f"  简单任务「计算 6*7」→ mode={simple.mode}(自动降级, 省掉规划与汇总两次模型调用)")


async def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    provider = get_provider()
    print(f"Provider: {provider.name} / {provider.model}"
          f"{'  (离线 Mock, 无需 API Key)' if provider.name == 'mock' else '  (真实模型)'}")

    demos = {
        "chunk": demo_chunk, "context": demo_context, "memory": demo_memory,
        "loop": demo_loop, "orchestrate": demo_orchestrate,
    }
    if which == "all":
        for fn in demos.values():
            await fn()
    elif which in demos:
        await demos[which]()
    else:
        print(f"未知场景: {which}。可选: all, {', '.join(demos)}")
        return
    print("\n" + "=" * 78)
    print("  演示结束。运行 `python -m uvicorn agentp.server.app:app --reload` 打开可视化控制台。")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())

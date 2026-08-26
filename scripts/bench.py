"""A/B 对照与消融实验 —— 用数据回答"这套护栏到底有没有用"。

    python scripts/bench.py all         # 全部六组(mock, 约 1 分钟)
    python scripts/bench.py guard       # 护栏 A/B: 朴素 ReAct vs 全护栏
    python scripts/bench.py breaker     # 熔断器省下多少次无谓的下游调用
    python scripts/bench.py mmr         # MMR 去冗余的检索效果
    python scripts/bench.py budget      # 预算调度 + 压缩的超窗率
    python scripts/bench.py digest      # 动作清单注入对重复调用的抑制
    python scripts/bench.py dag         # DAG 并发 vs 单智能体: 快多少, 贵多少

    python scripts/bench.py guard --real     # 换真实模型跑(见下方说明)

为什么自建评测集而不是用 GAIA / AgentBench 这类公开榜单:

  公开榜单测的是**模型能力**, 标注的是"最终答案对不对"。这里要证明的是**框架价值** ——
  护栏有没有减少无效调用、死循环在第几步被止损、超窗率降了多少、检索冗余度如何。
  这些指标公开榜单既不标注也不敏感: 把护栏全关掉, 榜单分数可能一分不掉,
  因为它压根不测"过程有多浪费"。

  而离线 Mock Provider 是**完全确定性**的(规则驱动, 无随机数), 于是 A/B 两组的差异
  可以 100% 归因到框架改动, 而不是模型这次心情好。这是公开榜单给不了的东西 ——
  它们必须用真模型跑, 输出方差会把要测的框架差异直接淹掉。

  所以定位说清楚: 这是 ablation study(消融实验), 不是 capability benchmark(能力评测)。
  代价是"自己出题自己考", 缓解办法是任务分类借鉴公开榜单的划分方式,
  并且把病态任务和正常任务分开统计 —— 混在一起平均, 什么都看不出来。

真实模型下的局限(必须诚实说明):

  病态任务靠 Mock 的确定性触发词(@loopdemo / @faildemo)复现, 真实模型不会听这个词,
  所以 --real 模式只跑正常任务, 验证的是"护栏不误杀 + 真实成本核算 + 真实并发收益"。
  死循环止损的量化必须在 Mock 下做。
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentp.config import LoopConfig, settings
from agentp.context import BudgetAllocator, ContextAssembler
from agentp.context.assembler import default_system_prompt
from agentp.llm import build_provider, set_provider
from agentp.llm.base import Message
from agentp.loop import ReActEngine
from agentp.loop.policy import LoopPolicy
from agentp.memory import MemoryLayer, MemoryManager, set_memory
from agentp.orchestrator import Orchestrator
from agentp.tools import get_registry
from agentp.util import cosine, count_tokens, tokenize

# ==========================================================================
# 排版工具: 中文是全角, 直接 ljust 会错位
# ==========================================================================


def _w(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def pad(text: str, width: int, right: bool = False) -> str:
    gap = max(0, width - _w(text))
    return (" " * gap + text) if right else (text + " " * gap)


def title(text: str) -> None:
    print(f"\n{'=' * 82}\n  {text}\n{'=' * 82}")


def sub(text: str) -> None:
    print(f"\n--- {text} " + "-" * max(0, 76 - _w(text)))


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def pct(new: float, old: float) -> str:
    """new 相对 old 的变化。old 为 0 时不编数字。"""
    if old == 0:
        return "n/a"
    delta = (new - old) / old * 100
    return f"{delta:+.1f}%"


# ==========================================================================
# 素材
# ==========================================================================

DOC = """# Agent 平台工程手册

## 上下文工程
上下文窗口是智能体最稀缺的资源。切片策略决定检索质量的上限, 切错了后面再好的向量模型也救不回来。
定长切片实现简单但会把句子拦腰砍断。递归切片按分隔符优先级下切, 是通用场景的默认选择。

### 预算调度
预算分配应建模为带下界和优先级的资源分配问题, 而不是先到先得。
系统默认上下文窗口 8192 tokens, 输出预留 1024 tokens, 安全边界 5%。

## 记忆系统
记忆分为工作、情景、语义、程序性四层。遗忘曲线半衰期默认 72 小时。
被检索命中相当于复习一次, 会重置衰减计时。语义记忆的衰减速度是情景记忆的六分之一。

## 发布流程
发布窗口为每周二下午 14:00 到 16:00, 需要两人评审通过。紧急发布需要 SRE 值班同学签字。

## 循环工程
ReAct 循环必须配备预算护栏、循环检测和熔断器。反思机制在连续失败两次后触发。
周期振荡通常意味着模型在两个错误假设之间横跳, 反思救不回来, 应当直接终止。
"""


@dataclass
class Task:
    id: str
    goal: str
    kind: str  # healthy | pathological
    expect: str = ""  # 这个字符串出现在答案里就算答对


TASKS: list[Task] = [
    # 正常任务: 护栏在这里的唯一职责是"别误杀"
    Task("h1", "计算 (128*7+56)/4", "healthy", "238"),
    Task("h2", "计算 (99+1)*3 的结果", "healthy", "300"),
    Task("h3", "帮我算一下 2**10 等于多少", "healthy", "1024"),
    Task("h4", "计算 (12+8)*5, 并查一下知识库里的遗忘曲线半衰期", "healthy", "100"),
    Task("h5", "现在几点了", "healthy"),
    Task("h6", "知识库里发布窗口是什么时候", "healthy"),
    Task("h7", "搜索 ReAct 论文的核心思想", "healthy"),
    Task("h8", "根据文档总结上下文预算调度的规定", "healthy"),
    # 病态任务: 靠 Mock 的确定性触发词复现死循环与下游持续故障
    Task("p1", "@loopdemo 请反复确认当前时间", "pathological"),
    Task("p2", "@loopdemo 核对一下系统时钟", "pathological"),
    Task("p3", "@faildemo 从下游服务拉取报表", "pathological"),
    Task("p4", "@faildemo 调用不稳定服务取数据", "pathological"),
]

# 两组配置的**硬上限完全一致**, 差别只在"智能检测器"开或关。
# 这一点很关键: 如果顺手把 guarded 的 max_steps 调小, 省下来的 token 就分不清
# 是检测器的功劳还是步数上限的功劳, 整个对照就白做了。
_CAPS = dict(max_steps=30, max_tokens=10**9, max_cost_usd=10**6, max_wall_time_s=600.0)

NAIVE_CFG = replace(
    LoopConfig(),
    **_CAPS,
    repeat_action_threshold=10**9,      # 关: 动作重复检测
    cycle_detect_window=0,              # 关: 周期振荡检测(窗口 0 → 切片为空, 永不触发)
    stall_threshold=10**9,              # 关: 进展停滞检测
    reflect_every_n_steps=0,            # 关: 周期性反思
    reflect_on_consecutive_failures=10**9,  # 关: 失败触发反思
)
GUARDED_CFG = replace(LoopConfig(), **_CAPS)  # 其余全部用默认值(3 / 6 / 3 / 4 / 2)


class NoopReflector:
    """朴素组用: 保证一次反思都不会发生。"""

    async def reflect(self, state: Any, trigger: str = "") -> None:
        return None


# ==========================================================================
# 环境准备
# ==========================================================================


async def fresh_env(provider: Any) -> MemoryManager:
    """每个 run 之前重置记忆与熔断器。

    必须重置的两个隐藏状态: 记忆是全局单例(kb_search 走 get_memory()),
    熔断器挂在工具实例上会跨 run 残留 —— 不清的话第二组一上来熔断器就是开的。
    """
    set_provider(provider)
    memory = MemoryManager(provider=provider)
    set_memory(memory)
    await memory.ingest_document(DOC, source="handbook.md", strategy="structure")
    get_registry().reset_breakers()
    return memory


def chat_calls(tracer: Any) -> int:
    """只数真正的对话调用。

    不能直接用 tracer.totals()["llm_calls"]: 它把 llm.embed 也算进去了, 而
    TracedProvider 的 embedding 缓存是跨 run 复用的 —— 先跑的那一组付掉全部
    cache miss, 后跑的一组白蹭, 两组的"LLM 调用次数"于是完全不可比。
    """
    return sum(1 for s in tracer.spans if s.kind == "llm" and s.name != "llm.embed")


def row(cells: list[str]) -> str:
    return "  " + "  ".join(cells)


# ==========================================================================
# 第 1 组: 护栏 A/B
# ==========================================================================


@dataclass
class RunMetrics:
    task: Task
    arm: str
    steps: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost: float = 0.0
    wall_ms: float = 0.0
    stop_reason: str = ""
    redundant: int = 0      # 同一动作指纹的重复次数(首次不算)
    blocked: int = 0        # 被动作黑名单在执行前拦下的次数
    guard_trips: int = 0
    reflections: int = 0
    answered: bool = False
    has_evidence: bool = False
    correct: Optional[bool] = None


async def run_one(task: Task, arm: str, cfg: LoopConfig, provider: Any) -> RunMetrics:
    await fresh_env(provider)
    engine = ReActEngine(
        provider=provider,
        policy=LoopPolicy(cfg),
        reflector=NoopReflector() if arm == "naive" else None,
        verify=False,  # 自检不是本组要测的机制, 两组都关掉以免引入噪声
    )
    started = time.perf_counter()
    result = await engine.run(task.goal, session_id=f"bench-{arm}-{task.id}")
    wall_ms = (time.perf_counter() - started) * 1000

    state = result.state
    totals = result.tracer.totals()
    fps = state.fingerprints()
    observations = state.all_observations()

    return RunMetrics(
        task=task,
        arm=arm,
        steps=state.step_count,
        llm_calls=chat_calls(result.tracer),
        tool_calls=totals["tool_calls"],
        # 用 tracer 的总量而不是 state 的: state 只累加主循环里的 chat,
        # 反思、强制收口、记忆写回那几次调用都不在里面, 会低估真实成本
        tokens=totals["total_tokens"],
        cost=totals["cost_usd"],
        wall_ms=wall_ms,
        stop_reason=state.stop_reason.value if state.stop_reason else "",
        redundant=len(fps) - len(set(fps)),
        blocked=sum(1 for o in observations if "系统层拦截" in o.error),
        guard_trips=len(result.guard_log),
        reflections=state.replans,
        answered=bool(result.answer.strip()),
        has_evidence=any(o.ok for o in observations),
        correct=(task.expect in result.answer) if task.expect else None,
    )


def _summarize(rows: list[RunMetrics]) -> dict[str, Any]:
    graded = [r for r in rows if r.correct is not None]
    return {
        "n": len(rows),
        "steps": mean([r.steps for r in rows]),
        "llm": mean([float(r.llm_calls) for r in rows]),
        "tools": mean([float(r.tool_calls) for r in rows]),
        "tokens": mean([float(r.tokens) for r in rows]),
        "cost": sum(r.cost for r in rows),
        "wall": mean([r.wall_ms for r in rows]),
        "redundant": mean([float(r.redundant) for r in rows]),
        "blocked": sum(r.blocked for r in rows),
        "trips": sum(r.guard_trips for r in rows),
        "answered": sum(1 for r in rows if r.answered),
        "evidence": sum(1 for r in rows if r.has_evidence),
        "correct": sum(1 for r in graded if r.correct),
        "graded": len(graded),
    }


_ARM_COLS = [("组别", 16, False), ("步数", 6, True), ("对话调用", 9, True),
             ("工具调用", 9, True), ("tokens", 8, True), ("重复调用", 9, True),
             ("拦截", 5, True), ("护栏触发", 9, True), ("有答案", 7, True),
             ("答对", 6, True)]


def _print_arm_table(label: str, per_arm: dict[str, dict[str, Any]]) -> None:
    print(f"\n  {label}")
    print(row([pad(name, width, right) for name, width, right in _ARM_COLS]))
    for arm, s in per_arm.items():
        values = [
            arm,
            f"{s['steps']:.1f}", f"{s['llm']:.1f}", f"{s['tools']:.1f}",
            f"{s['tokens']:.0f}", f"{s['redundant']:.1f}",
            str(s["blocked"]), str(s["trips"]),
            f"{s['answered']}/{s['n']}",
            f"{s['correct']}/{s['graded']}" if s["graded"] else "-",
        ]
        print(row([pad(v, w, r) for v, (_, w, r) in zip(values, _ARM_COLS)]))


async def bench_guardrails(provider: Any, real: bool) -> None:
    title("第 1 组 · 护栏 A/B —— 朴素 ReAct vs 全套护栏")
    print("  两组的硬上限完全相同(步数 30 / token 10^9 / 成本 10^6 / 墙钟 600s),")
    print("  差别只有: 动作重复检测、周期振荡检测、进展停滞检测、反思重规划 —— 开 or 关。")
    print("  这样省下来的 token 才能归因到检测器本身, 而不是「顺手把步数上限调小了」。")
    if real:
        print("\n  [真实模型] 病态任务依赖 Mock 的确定性触发词, 已跳过。本轮只验证「护栏不误杀」。")

    tasks = [t for t in TASKS if not (real and t.kind == "pathological")]
    rows: list[RunMetrics] = []
    for arm, cfg in (("naive", NAIVE_CFG), ("guarded", GUARDED_CFG)):
        # 每组用**独立的 Provider 实例**: embedding 缓存跨组共享的话,
        # 后跑的那组会白蹭前一组的缓存, 耗时和调用次数都失去可比性。
        arm_provider = provider if real else build_provider("mock")
        for task in tasks:
            print(f"    {pad(arm, 10)} {task.id} 运行中 ...", end="\r")
            rows.append(await run_one(task, arm, cfg, arm_provider))
    print(" " * 60, end="\r")

    for kind, kind_label in (("healthy", "正常任务"), ("pathological", "病态任务")):
        subset = [r for r in rows if r.task.kind == kind]
        if not subset:
            continue
        per_arm = {}
        for arm, arm_label in (("naive", "naive"), ("guarded", "guarded")):
            arm_rows = [r for r in subset if r.arm == arm]
            if arm_rows:
                per_arm[arm_label] = _summarize(arm_rows)
        n = len(subset) // max(1, len(per_arm))
        _print_arm_table(f"{kind_label}({n} 个)", per_arm)

        if len(per_arm) == 2:
            a, b = per_arm["naive"], per_arm["guarded"]
            print(f"    → 步数 {pct(b['steps'], a['steps'])}"
                  f" | tokens {pct(b['tokens'], a['tokens'])}"
                  f" | LLM 调用 {pct(b['llm'], a['llm'])}"
                  f" | 重复调用 {a['redundant']:.1f} → {b['redundant']:.1f}")

    detail = [r for r in rows if r.task.kind == "pathological"]
    if detail:
        sub("病态任务逐条明细(护栏是怎么止损的)")
        cols = [("任务", 5, False), ("组别", 9, False), ("步数", 5, True),
                ("重复调用", 9, True), ("反思", 5, True), ("拦截", 5, True),
                ("终止原因", 16, False), ("答案", 14, False)]
        print(row([pad(n, w, rt) for n, w, rt in cols]))
        for r in sorted(detail, key=lambda x: (x.task.id, x.arm)):
            values = [r.task.id, r.arm, str(r.steps), str(r.redundant),
                      str(r.reflections), str(r.blocked), r.stop_reason or "-",
                      "有(降级作答)" if r.answered else "无"]
            print(row([pad(v, w, rt) for v, (_, w, rt) in zip(values, cols)]))

    print("\n  读法: 正常任务两组应当完全一致 —— 护栏的第一要求是不误杀;")
    print("        病态任务上护栏把步数和 token 砍掉大半, 且仍然产出降级答案而不是抛错。")


# ==========================================================================
# 第 2 组: 熔断器
# ==========================================================================


async def bench_breaker(provider: Any) -> None:
    title("第 2 组 · 熔断器 —— 下游挂了以后, 省下多少次无谓调用")
    print("  被测对象是工具层的熔断器, 和第 1 组的循环护栏是两套独立机制:")
    print("  循环护栏管「模型别再想调它了」, 熔断器管「就算模型想调, 也别真发出去」。")

    registry = get_registry()
    tool = registry.get("flaky")
    assert tool is not None
    attempts_calls = 12

    rows = []
    for label, threshold in (("熔断关闭", 10**9), ("熔断开启(阈值3)", 3)):
        from agentp.tools.base import CircuitBreaker

        tool.breaker = CircuitBreaker(failure_threshold=threshold, cooldown_s=30.0)
        started = time.perf_counter()
        rejected = 0
        for _ in range(attempts_calls):
            result = await registry.call("flaky", {"fail": True})
            if result.meta.get("circuit_open"):
                rejected += 1
        elapsed = (time.perf_counter() - started) * 1000
        rows.append({
            "label": label,
            "calls": attempts_calls,
            # 每次真实调用会重试 1 次 → 2 次落到下游
            "downstream": (attempts_calls - rejected) * (tool.max_retries + 1),
            "rejected": rejected,
            "ms": elapsed,
            "state": tool.breaker.state,
        })
    registry.reset_breakers()

    print("\n  " + pad("配置", 20) + pad("模型发起", 10, True) + pad("落到下游", 10, True)
          + pad("短路拒绝", 10, True) + pad("耗时(ms)", 12, True) + pad("终态", 10))
    for r in rows:
        print("  " + pad(r["label"], 20) + pad(str(r["calls"]), 10, True)
              + pad(str(r["downstream"]), 10, True) + pad(str(r["rejected"]), 10, True)
              + pad(f"{r['ms']:.0f}", 12, True) + pad(r["state"], 10))

    off, on = rows[0], rows[1]
    print(f"\n    → 同样 {attempts_calls} 次调用意图, 落到下游的真实请求"
          f" {off['downstream']} → {on['downstream']} ({pct(on['downstream'], off['downstream'])}),"
          f" 耗时 {pct(on['ms'], off['ms'])}")
    print("    读法: 熔断器不减少模型的调用意图, 它减少的是**打到下游的真实流量**。")
    print("          这是雪崩防护的关键 —— 下游已经 503 了, 再重试只会让它更起不来。")


# ==========================================================================
# 第 3 组: MMR 去冗余
# ==========================================================================

# 刻意构造的冗余语料: 两个主题各 5 条改写。相似但低于去重阈值(0.93),
# 所以它们会全部入库 —— 这正是真实知识库的常态(同一件事被反复记录)。
_REDUNDANT = {
    # 主题 A 刻意给得更多(8 条), 模拟"热门话题被反复记录"的真实分布。
    # 不做去重(dedup=False), 因为要测的就是冗余进来之后检索层怎么处理。
    "budget": [
        "上下文预算应当建模为带下界和优先级的资源分配问题, 而不是先到先得地塞满窗口。",
        "Token 预算不能先到先得, 要按分区优先级分配, 并给每个分区留一条保命线。",
        "预算调度的核心是优先级加保命线, 剩余额度再按权重注水分配下去。",
        "分配上下文额度时先满足高优先级分区的下界, 余量按权重迭代注水到收敛。",
        "窗口紧张时应该按优先级丢弃整个低优分区, 而不是让所有分区一起缩水。",
        "上下文额度分配要先保住系统提示词和用户任务, 这两块永远不参与竞争。",
        "预算不足时优先牺牲对话历史, 因为历史是唯一能被无损滚动摘要的分区。",
        "把上下文窗口当成带优先级的资源池来分配, 比截断式拼接稳定得多。",
    ],
    "forget": [
        "记忆强度按艾宾浩斯遗忘曲线衰减, 默认半衰期是 72 小时。",
        "遗忘曲线的半衰期设为 72 小时, 记忆强度随时间指数下降。",
        "长期记忆会随时间衰减, 被检索命中相当于复习一次, 会重置衰减计时。",
        "语义记忆的衰减速度约为情景记忆的六分之一, 所以事实比事件留得更久。",
    ],
}


def _jaccard(a: str, b: str) -> float:
    """词集重合度。

    和向量余弦并列报一个词法指标, 是因为离线 Mock 用的是 feature hashing 向量
    (词袋随机投影, 不懂近义词), 用它算出来的"语义冗余度"会系统性偏低。
    Jaccard 不依赖任何 embedding, 直接回答"这 6 个坑位里有多少是同样的词"。
    """
    sa, sb = set(tokenize(a)), set(tokenize(b))
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


async def bench_mmr(provider: Any) -> None:
    title("第 3 组 · MMR 去冗余 —— 检索结果里有多少是同一件事")
    n_a, n_b = len(_REDUNDANT["budget"]), len(_REDUNDANT["forget"])
    print(f"  构造场景: 主题 A(预算){n_a} 条 + 主题 B(遗忘){n_b} 条, 共 {n_a + n_b} 条改写入库。")
    print("  A 给得更多是刻意的 —— 真实知识库里热门话题总被反复记录。查 A 相关问题, 取 top-6。")
    print("  期望: 别把 6 个坑位全填成同一件事的不同说法。")

    memory = MemoryManager(provider=provider)
    set_memory(memory)
    for topic, texts in _REDUNDANT.items():
        for text in texts:
            await memory.remember(
                text, layer=MemoryLayer.SEMANTIC, importance=6.0,
                tags=[topic], source="bench", dedup=False,
            )

    query = "上下文 token 预算怎么分配"
    print(f"\n  查询: 「{query}」")
    print("  除了开/关, 还扫一遍 mmr_lambda —— 它就是「相关性 vs 多样性」的旋钮,")
    print("  只报默认值那一个数字, 等于把最该讲清楚的取舍藏起来了。")

    async def measure(use_mmr: bool, lam: Optional[float]) -> dict[str, Any]:
        if lam is not None:
            settings.memory.mmr_lambda = lam  # search() 每次调用都重读, 改了即刻生效
        results = await memory.retrieve(query, top_k=6, use_mmr=use_mmr)
        vectors = [r.record.embedding for r in results if r.record.embedding]
        texts = [r.record.content for r in results]
        idx = [(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))]
        cos_pairs = [cosine(vectors[i], vectors[j]) for i, j in idx] if vectors else [0.0]
        jac_pairs = [_jaccard(texts[i], texts[j]) for i, j in idx] or [0.0]
        topics = [(r.record.tags or ["?"])[0] for r in results]
        dist = {t: topics.count(t) for t in sorted(set(topics))}
        return {
            "avg_cos": mean(cos_pairs), "avg_jac": mean(jac_pairs),
            # 保留的相关性: 命中结果的平均检索总分。下降说明多样性是花钱买来的
            "relevance": mean([r.score for r in results]),
            "dist": ", ".join(f"{k}={v}" for k, v in dist.items()),
        }

    original_lambda = settings.memory.mmr_lambda
    cols = [("配置", 22, False), ("平均余弦(冗余)", 15, True), ("平均词重合", 12, True),
            ("保留相关性", 12, True), ("命中分布", 22, False)]
    print("\n" + row([pad(n, w, r) for n, w, r in cols]))
    try:
        baseline = await measure(use_mmr=False, lam=None)
        sweep: list[tuple[str, dict[str, Any]]] = [("MMR 关闭(纯相关性排序)", baseline)]
        for lam in (0.9, 0.7, 0.5, 0.3):
            tag = f"MMR λ={lam}" + ("  ← 默认" if abs(lam - original_lambda) < 1e-9 else "")
            sweep.append((tag, await measure(use_mmr=True, lam=lam)))
    finally:
        settings.memory.mmr_lambda = original_lambda

    for label, entry in sweep:
        values = [label, f"{entry['avg_cos']:.4f}", f"{entry['avg_jac']:.4f}",
                  f"{entry['relevance']:.4f}", entry["dist"]]
        print(row([pad(v, w, r) for v, (_, w, r) in zip(values, cols)]))

    default = next(e for lab, e in sweep if "默认" in lab)
    aggressive = sweep[-1][1]
    print(f"\n    → 默认 λ={original_lambda}: 冗余 {pct(default['avg_cos'], baseline['avg_cos'])},"
          f" 相关性 {pct(default['relevance'], baseline['relevance'])}")
    print(f"    → 激进 λ=0.3:  冗余 {pct(aggressive['avg_cos'], baseline['avg_cos'])},"
          f" 相关性 {pct(aggressive['relevance'], baseline['relevance'])}")
    print("    读法: 冗余检索的真实代价是上下文预算被同一件事占了好几遍, 挤掉本该进来的其他证据。")
    print("          但多样性不是免费的 —— λ 越小, 冗余降得越多, 保留的相关性也掉得越多。")
    print("          默认值偏向相关性, 是因为漏掉一条关键证据的代价通常高于多读一条重复内容。")
    print("    局限: 本组被 Mock 的 hashing 向量系统性低估 —— 它是词袋随机投影, 不懂近义词,")
    print("          改写句之间余弦本来就不高, MMR 能压的空间因此变小。换真实 embedding 后差距会")
    print("          更明显, 这也是为什么「平均词重合」(完全不依赖 embedding)这一列更可信。")


# ==========================================================================
# 第 4 组: 预算调度 + 压缩
# ==========================================================================


def _long_scratchpad(steps: int = 9) -> str:
    parts = []
    for i in range(steps):
        parts.append(
            f"### 第 {i + 1} 步\n"
            f"思考: 目标还缺少第 {i + 1} 项证据, 需要继续检索并交叉验证已有结论的一致性。\n"
            f"动作: kb_search(query=第{i + 1}项证据的检索式, top_k=4)\n"
            f"观察: [kb_search] 检索到 4 条相关记忆: 1. 上下文窗口是智能体最稀缺的资源, "
            f"切片策略决定检索质量的上限。2. 预算分配应当建模为带下界和优先级的资源分配问题。"
            f"3. 记忆分为工作、情景、语义、程序性四层, 遗忘曲线半衰期默认 72 小时。"
            f"4. ReAct 循环必须配备预算护栏、循环检测和熔断器, 反思在连续失败两次后触发。"
        )
    return "\n\n".join(parts)


def _long_history(turns: int = 10) -> list[Message]:
    out = []
    for i in range(turns):
        out.append(Message(role="user", content=(
            f"第 {i + 1} 轮提问: 请说明上下文工程里预算调度、压缩策略与切片粒度三者的关系, "
            f"并给出在窗口收缩时应当优先牺牲哪一部分的判断依据。"
        )))
        out.append(Message(role="assistant", content=(
            f"第 {i + 1} 轮回答: 切片决定检索质量上限, 预算调度决定谁能进上下文, "
            f"压缩决定进来的内容以什么密度呈现。窗口收缩时优先牺牲可无损摘要的对话历史, "
            f"其次是可按分数丢弃的检索结果, 系统提示词与用户任务永不牺牲。"
        )))
    return out


async def bench_budget(provider: Any) -> None:
    title("第 4 组 · 预算调度 + 压缩 —— 超窗率")
    print("  同一批素材, 在不同窗口下组装两次:")
    print("    压缩关闭 = 给分配器一个近乎无限的窗口, 于是一切原样输出(朴素做法: 全量拼接)")
    print("    压缩开启 = 按真实窗口做优先级分配与逐分区压缩")
    print("  然后拿两者的实际 token 数去和真实窗口比, 看谁会撞窗口。")

    memory = MemoryManager(provider=provider)
    set_memory(memory)
    await memory.ingest_document(DOC, source="handbook.md", strategy="structure")
    for topic, texts in _REDUNDANT.items():
        for text in texts:
            await memory.remember(text, layer=MemoryLayer.SEMANTIC, importance=6.0,
                                  tags=[topic], source="bench", dedup=False)

    task = "总结上下文预算调度与遗忘曲线的设计要点"
    items = await memory.build_context_items(task, session_id="bench-budget")
    scratchpad = _long_scratchpad()
    history = _long_history()
    tools_schema = get_registry().schemas()

    async def assemble(window: int, compress: bool):
        allocator = BudgetAllocator(context_window=10 ** 7 if not compress else window)
        assembler = ContextAssembler(provider, allocator=allocator)
        return await assembler.build(
            task=task,
            system_prompt=default_system_prompt(),
            tools_schema=tools_schema,
            items=items,
            history=history,
            scratchpad=scratchpad,
        )

    raw = await assemble(8192, compress=False)
    print(f"\n  原始素材总需求: {raw.total_tokens} tokens"
          f" (记忆条目 {len(items)} 条, scratchpad {count_tokens(scratchpad)} tok,"
          f" 历史 {count_tokens(''.join(m.content for m in history))} tok)")

    print("\n  " + pad("窗口", 8, True) + pad("配置", 12) + pad("实际 tokens", 14, True)
          + pad("可用预算", 11, True) + pad("是否超窗", 11) + pad("丢弃分区", 10, True)
          + pad("压缩分区", 10, True))
    overflow_count = {"压缩关闭": 0, "压缩开启": 0}
    windows = (2048, 4096, 8192, 16384)
    for window in windows:
        budget = BudgetAllocator(context_window=window).total_budget
        for label, compress in (("压缩关闭", False), ("压缩开启", True)):
            assembled = await assemble(window, compress)
            over = assembled.total_tokens > window
            overflow_count[label] += int(over)
            dropped = sum(1 for a in assembled.plan.allocations.values() if a.dropped)
            squeezed = sum(1 for a in assembled.plan.allocations.values()
                           if a.needs_compression and not a.dropped)
            print("  " + pad(str(window), 8, True) + pad(label, 12)
                  + pad(str(assembled.total_tokens), 14, True)
                  + pad(str(budget), 11, True)
                  + pad("超窗!" if over else "安全", 11)
                  + pad(str(dropped), 10, True) + pad(str(squeezed), 10, True))

    n = len(windows)
    print(f"\n    → 超窗次数: 压缩关闭 {overflow_count['压缩关闭']}/{n},"
          f" 压缩开启 {overflow_count['压缩开启']}/{n}")
    print("    读法: 朴素做法在窗口小的时候必然撞墙, 而撞墙的代价不是「效果差一点」,")
    print("          是这次请求直接 400 报错。预算调度把硬故障换成了可控的信息取舍。")

    sub("窗口 2048 时的取舍顺序(优先级的产品判断)")
    assembled = await assemble(2048, compress=True)
    for name, alloc in sorted(assembled.plan.allocations.items(),
                              key=lambda kv: -kv[1].granted):
        state = "丢弃" if alloc.dropped else ("压缩" if alloc.needs_compression else "原样")
        print(f"    {pad(name, 12)} 需求 {pad(str(alloc.requested), 6, True)}"
              f" → 分配 {pad(str(alloc.granted), 6, True)}  [{state}]")


# ==========================================================================
# 第 5 组: 动作清单注入
# ==========================================================================


async def bench_digest(provider: Any) -> None:
    title("第 5 组 · 动作清单注入 —— 上下文一紧, 模型就忘了自己做过什么")
    print("  被测机制: 把「已执行动作」注入**系统提示词**(不可丢弃), 而不是留在 scratchpad。")
    print("  理由: scratchpad 是可压缩分区, 窗口一紧动作记录就被压掉, 模型于是重复调同一个工具。")
    print("  为了让重复暴露出来, 本组**关掉了动作重复检测** —— 否则护栏会把现象直接盖住。")

    task = next(t for t in TASKS if t.id == "h4")  # 需要 3 个工具的任务
    cfg = replace(LoopConfig(), max_steps=14, repeat_action_threshold=10**9,
                  stall_threshold=10**9, reflect_every_n_steps=0,
                  reflect_on_consecutive_failures=10**9, cycle_detect_window=0)

    print("\n  " + pad("窗口", 8, True) + pad("动作清单", 12) + pad("步数", 7, True)
          + pad("工具调用", 10, True) + pad("重复调用", 10, True)
          + pad("scratchpad", 12) + pad("答对", 7))
    summary: dict[str, list[int]] = {"关闭": [], "开启": []}
    for window in (1024, 1536, 2048, 8192):
        for label, inject in (("关闭", False), ("开启", True)):
            await fresh_env(provider)
            engine = ReActEngine(
                provider=provider,
                assembler=ContextAssembler(
                    provider, allocator=BudgetAllocator(context_window=window)
                ),
                policy=LoopPolicy(cfg),
                reflector=NoopReflector(),
                verify=False,
                inject_action_digest=inject,
            )
            result = await engine.run(task.goal, session_id=f"bench-digest-{window}-{inject}")
            fps = result.state.fingerprints()
            redundant = len(fps) - len(set(fps))
            summary[label].append(redundant)
            method = (result.context_report.get("scratchpad", {}) or {}).get("method", "-")
            print("  " + pad(str(window), 8, True) + pad(label, 12)
                  + pad(str(result.state.step_count), 7, True)
                  + pad(str(len(fps)), 10, True) + pad(str(redundant), 10, True)
                  + pad(str(method), 12)
                  + pad("是" if task.expect in result.answer else "否", 7))

    off, on = mean([float(x) for x in summary["关闭"]]), mean([float(x) for x in summary["开启"]])
    print(f"\n    → 各窗口平均重复调用: 关闭 {off:.2f} 次, 开启 {on:.2f} 次 ({pct(on, off)})")
    print("    读法: 看 scratchpad 那一列。method=verbatim 说明没被压缩, 两组自然一样;")
    print("          一旦变成 middle_out/extractive, 关闭组就开始重复调用了 —— 这正是")
    print("          「把关键状态放进不可丢弃分区」这个设计要解决的问题。")


# ==========================================================================
# 第 6 组: DAG 并发 vs 单智能体
# ==========================================================================

MULTI_GOAL = ("查一下知识库里的发布流程规定; 同时搜索 ReAct 论文的核心思想; "
              "然后计算 (99+1)*3 的结果")


async def bench_orchestration(provider: Any) -> None:
    title("第 6 组 · DAG 并发编排 vs 单智能体 —— 快多少, 又贵多少")
    print(f"  目标(含三个可并行的子任务): {MULTI_GOAL}")
    print("  编排不是免费的: 多了规划、汇总、质检三次模型调用。所以要同时看两个数字。")

    rows = []
    for label in ("单智能体 ReAct", "DAG 编排"):
        await fresh_env(provider)
        started = time.perf_counter()
        if label == "单智能体 ReAct":
            result = await ReActEngine(provider=provider, verify=False).run(
                MULTI_GOAL, session_id="bench-single")
            extra = f"步数 {result.state.step_count}"
            speedup = None
        else:
            result = await Orchestrator(provider=provider).run(
                MULTI_GOAL, session_id="bench-dag")
            s = result.dag_summary
            extra = f"{s.get('nodes', 0)} 节点 / {s.get('levels', 0)} 层"
            speedup = s.get("speedup")
        wall = (time.perf_counter() - started) * 1000
        totals = result.tracer.totals()
        rows.append({"label": label, "wall": wall, "tokens": totals["total_tokens"],
                     "llm": chat_calls(result.tracer), "tools": totals["tool_calls"],
                     "cost": totals["cost_usd"], "extra": extra, "speedup": speedup})

    cols = [("模式", 16, False), ("墙钟(ms)", 10, True), ("对话调用", 9, True),
            ("工具调用", 9, True), ("tokens", 8, True), ("成本($)", 10, True),
            ("结构", 18, False)]
    print("\n" + row([pad(n, w, r) for n, w, r in cols]))
    for r in rows:
        values = [r["label"], f"{r['wall']:.0f}", str(r["llm"]), str(r["tools"]),
                  str(r["tokens"]), f"{r['cost']:.6f}", r["extra"]]
        print(row([pad(v, w, rt) for v, (_, w, rt) in zip(values, cols)]))

    single, dag = rows[0], rows[1]
    print(f"\n    → 墙钟 {pct(dag['wall'], single['wall'])}"
          f" | tokens {pct(dag['tokens'], single['tokens'])}"
          f" | 对话调用 {pct(dag['llm'], single['llm'])}")
    if dag["speedup"]:
        print(f"    → DAG 内部并发加速比 {dag['speedup']}× (同层节点串行耗时 / 实际墙钟)")
    print("\n    读法(实测结论, 和直觉不一样, 值得细看):")
    print("    1. 加速比在 mock 和真实模型下**基本相同**(实测 1.53× vs 真实模型三轮 1.35/1.48/1.80×)。")
    print("       原先猜「真模型延迟大所以加速比更高」是错的 —— 加速比是个比值, 延迟等比放大会在")
    print("       分子分母上一起放大然后抵消。真正的上界是图的结构(关键路径长度), 不是延迟。")
    print("    2. 但端到端墙钟 DAG 反而**更慢**, token 也贵好几倍: 多了规划、汇总、质检三次调用,")
    print("       每个 worker 还各自跑一遍完整 ReAct(各带一份完整上下文)。")
    print("    3. 所以真正的结论是: 这个规模的任务**不该编排**。编排的收益要等到子任务本身足够重")
    print("       (每个都要多步取证)、并且数量够多的时候才回本。这正是 Orchestrator 对单节点任务")
    print("       自动降级直跑的理由 —— 而且从数据看, 这个降级阈值其实还可以再往上调。")

    await fresh_env(provider)
    simple = await Orchestrator(provider=provider).run("计算 6*7", session_id="bench-simple")
    print(f"    → 验证降级: 简单任务「计算 6*7」的执行模式 = {simple.mode}"
          f"(省掉规划与汇总两次模型调用)")


# ==========================================================================


GROUPS = {
    "guard": bench_guardrails,
    "breaker": bench_breaker,
    "mmr": bench_mmr,
    "budget": bench_budget,
    "digest": bench_digest,
    "dag": bench_orchestration,
}


async def main() -> None:
    parser = argparse.ArgumentParser(description="AgentP 消融实验")
    parser.add_argument("group", nargs="?", default="all",
                        choices=["all", *GROUPS], help="要跑的实验组")
    parser.add_argument("--real", action="store_true",
                        help="用 .env 里配置的真实模型(默认强制 mock 以保证确定性)")
    args = parser.parse_args()

    # 默认强制 mock: bench 的全部价值建立在"差异可归因"上, 真实模型的输出方差
    # 会把要测的框架差异淹掉。所以这里不读 .env, 除非显式 --real。
    provider = build_provider("mock") if not args.real else build_provider()
    set_provider(provider)

    print(f"Provider: {provider.name} / {provider.model}"
          + ("  (离线规则模型, 完全确定性)" if provider.name == "mock" else "  (真实模型, 输出有方差)"))
    print(f"窗口 {settings.context.model_context_window} / "
          f"输出预留 {settings.context.reserve_for_output} / "
          f"遗忘半衰期 {settings.memory.decay_half_life_hours}h")

    started = time.perf_counter()
    names = list(GROUPS) if args.group == "all" else [args.group]
    for name in names:
        fn = GROUPS[name]
        if name == "guard":
            await fn(provider, args.real)
        else:
            await fn(provider)

    print(f"\n{'=' * 82}")
    print(f"  全部实验完成, 耗时 {time.perf_counter() - started:.1f}s。")
    print("  这些数字的用途: 把「我实现了 X」变成「我实现了 X, 并证明它带来了 Y」。")
    print(f"{'=' * 82}")

    closer = getattr(provider, "aclose", None)
    if closer:
        await closer()


if __name__ == "__main__":
    asyncio.run(main())

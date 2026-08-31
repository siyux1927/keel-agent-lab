"""记忆系统: 分层、混合检索、遗忘曲线、去重、技能强化。"""

from __future__ import annotations

from keel.config import settings
from keel.memory import MemoryLayer, MemoryRecord, Skill, judge_importance
from keel.memory.working import WorkingMemory
from keel.util import now_ts


# ==========================================================================
# 强度与遗忘
# ==========================================================================


def test_strength_decays_over_time():
    rec = MemoryRecord(content="一条普通记忆", layer=MemoryLayer.EPISODIC, importance=8.0)
    fresh = rec.strength(half_life_hours=24)
    aged = rec.strength(half_life_hours=24, at=now_ts() + 24 * 3600)
    assert aged < fresh
    assert abs(aged / fresh - 0.5) < 0.02  # 一个半衰期后正好减半


def test_semantic_layer_forgets_slower_than_episodic():
    """分层的意义就在这里: 事实该比事件记得久。"""
    at = now_ts() + 72 * 3600
    episodic = MemoryRecord(content="x", layer=MemoryLayer.EPISODIC, importance=6.0)
    semantic = MemoryRecord(content="x", layer=MemoryLayer.SEMANTIC, importance=6.0)
    assert semantic.strength(48, at) > episodic.strength(48, at)


def test_access_count_slows_decay():
    at = now_ts() + 48 * 3600
    cold = MemoryRecord(content="x", layer=MemoryLayer.EPISODIC, importance=6.0)
    hot = MemoryRecord(content="x", layer=MemoryLayer.EPISODIC, importance=6.0, access_count=10)
    assert hot.strength(24, at) > cold.strength(24, at)


def test_touch_resets_decay_clock():
    rec = MemoryRecord(content="x", layer=MemoryLayer.EPISODIC, importance=6.0)
    rec.last_access = now_ts() - 100 * 3600
    before = rec.strength(24)
    rec.touch()
    assert rec.strength(24) > before
    assert rec.access_count == 1


async def test_decay_and_prune_archives_weak_memories(memory):
    rec = await memory.remember("一条很久没被用到的琐碎记录", layer=MemoryLayer.EPISODIC,
                                importance=1.0)
    rec.last_access = now_ts() - 1000 * 3600
    report = memory.decay_and_prune()
    assert rec.id in report["ids"]
    assert rec.archived
    # 归档而非删除: 记录仍在, 只是不再参与检索
    assert memory.store.get(rec.id) is not None


async def test_procedural_memory_never_decays(memory):
    await memory.record_trajectory("部署服务", ["kb_search", "now"], success=True)
    for rec in memory.store.all(MemoryLayer.PROCEDURAL):
        rec.last_access = now_ts() - 10000 * 3600
    memory.decay_and_prune()
    assert all(not r.archived for r in memory.store.all(MemoryLayer.PROCEDURAL))


# ==========================================================================
# 写入
# ==========================================================================


async def test_duplicate_memory_strengthens_instead_of_duplicating(memory):
    first = await memory.remember("用户偏好使用 Python 而不是 Java", layer=MemoryLayer.SEMANTIC)
    before = first.importance
    second = await memory.remember("用户偏好使用 Python 而不是 Java", layer=MemoryLayer.SEMANTIC)
    assert second.id == first.id
    assert second.importance > before
    assert second.metadata["duplicate_hits"] == 1
    assert len(memory.store.all(MemoryLayer.SEMANTIC)) == 1


def test_importance_heuristic_ranks_user_profile_highest():
    profile = judge_importance("我的偏好是必须使用中文回复")
    trivial = judge_importance("好的")
    question = judge_importance("这个怎么做？")
    assert profile > question
    assert profile > trivial
    assert 0 < trivial <= 10


async def test_ingest_document_indexes_leaf_chunks_only(memory, sample_doc):
    report = await memory.ingest_document(sample_doc, source="doc.md", strategy="hierarchical")
    records = memory.store.all(MemoryLayer.SEMANTIC)
    # 分层切片下只索引子块, 父块索引会造成重复召回
    assert report["indexed"] == len(records)
    assert report["indexed"] < report["chunk_stats"]["count"]
    assert all(r.metadata.get("chunk_id") for r in records)


# ==========================================================================
# 检索
# ==========================================================================


async def test_hybrid_retrieval_returns_score_breakdown(memory):
    await memory.remember("发布窗口为每周二下午 14:00 到 16:00", layer=MemoryLayer.SEMANTIC)
    await memory.remember("代码评审需要两人通过", layer=MemoryLayer.SEMANTIC)
    results = await memory.retrieve("发布窗口是什么时候", top_k=2)
    assert results
    top = results[0]
    assert "发布窗口" in top.record.content
    # 每一路信号都要能被单独看到, 否则检索效果没法调
    assert {"vector", "bm25", "recency", "importance"} <= set(top.breakdown)


async def test_bm25_channel_catches_exact_terms(memory):
    """向量检索对生僻标识符不敏感, BM25 是来兜这个底的。"""
    await memory.remember("错误码 ERR_7731 表示上游配额耗尽", layer=MemoryLayer.SEMANTIC)
    for i in range(6):
        await memory.remember(f"第 {i} 条关于系统错误与异常处理的一般性说明",
                              layer=MemoryLayer.SEMANTIC)
    results = await memory.retrieve("ERR_7731", top_k=3)
    assert any("ERR_7731" in r.record.content for r in results)


async def test_retrieval_marks_records_as_rehearsed(memory):
    rec = await memory.remember("会被检索到的内容", layer=MemoryLayer.SEMANTIC)
    await memory.retrieve("会被检索到的内容", top_k=1)
    assert rec.access_count >= 1


async def test_archived_memories_are_excluded(memory):
    rec = await memory.remember("已归档的内容不该被召回", layer=MemoryLayer.SEMANTIC)
    rec.archived = True
    results = await memory.retrieve("已归档的内容不该被召回", top_k=5)
    assert all(r.record.id != rec.id for r in results)


async def test_build_context_items_routes_layers_to_zones(memory, sample_doc):
    await memory.ingest_document(sample_doc, source="doc.md", strategy="structure")
    await memory.remember("用户偏好简洁回答", layer=MemoryLayer.SEMANTIC, session_id="s")
    items = await memory.build_context_items("切片策略", session_id="s")
    zones = {i.zone for i in items}
    # 文档块进 retrieved, 模型自己记住的事实进 semantic —— 预算上互相隔离
    assert "retrieved" in zones


# ==========================================================================
# 工作记忆
# ==========================================================================


async def test_working_memory_compacts_on_overflow():
    wm = WorkingMemory("s", max_tokens=200, keep_recent=2)
    for i in range(30):
        wm.add("user", f"第 {i} 轮对话内容, 需要占据一定的 token 预算才能触发压缩")
    assert wm.is_overflowing()
    evicted_seen: list = []
    report = await wm.maybe_compact(
        summarizer=lambda text: _echo(f"摘要({len(text)}字)"),
        on_evict=lambda turns: evicted_seen.extend(turns),
    )
    assert report["evicted_turns"] > 0
    assert wm.token_count <= 200
    assert len(wm.turns) >= 2  # keep_recent 得到尊重
    assert evicted_seen  # 原文下沉到情景记忆, 可追溯
    assert wm.rolling_summary


async def _echo(value: str) -> str:
    return value


async def test_working_memory_no_compact_when_within_budget():
    wm = WorkingMemory("s", max_tokens=5000)
    wm.add("user", "短消息")
    assert await wm.maybe_compact() is None


async def test_observe_turn_sinks_overflow_to_episodic(memory):
    memory.working("s").max_tokens = 160
    memory.working("s").keep_recent = 2
    for i in range(20):
        await memory.observe_turn("s", "user", f"第 {i} 轮内容, 这里需要足够长才能触发溢出压缩")
    episodic = memory.store.all(MemoryLayer.EPISODIC)
    assert episodic
    assert any(r.source == "working_overflow" for r in episodic)


# ==========================================================================
# 程序性记忆
# ==========================================================================


def test_skill_win_rate_uses_laplace_smoothing():
    """1 胜 0 负不该被当成 100% 可靠。"""
    rookie = Skill("任务", ["a"], success_count=1, fail_count=0)
    veteran = Skill("任务", ["a"], success_count=20, fail_count=0)
    assert rookie.win_rate < 1.0
    assert veteran.win_rate > rookie.win_rate


async def test_trajectory_reinforces_existing_skill(memory):
    first = await memory.record_trajectory("查询发布规定", ["kb_search"], success=True)
    again = await memory.record_trajectory("查询发布规定", ["kb_search", "web_search"], success=True)
    assert again.id == first.id
    assert again.metadata["skill"]["success_count"] == 2
    assert len(memory.store.all(MemoryLayer.PROCEDURAL)) == 1


async def test_low_win_rate_skills_are_filtered_out(memory):
    rec = await memory.record_trajectory("会失败的任务", ["flaky"], success=False)
    for _ in range(5):
        await memory.record_trajectory("会失败的任务", ["flaky"], success=False)
    assert rec.metadata["skill"]["win_rate"] < 0.4
    assert await memory.recall_skills("会失败的任务") == []


# ==========================================================================
# 反思固化
# ==========================================================================


async def test_reflection_creates_higher_level_memories(memory):
    # 内容必须彼此不同, 否则会被语义去重合并成一条(这本身是正确行为)
    topics = [
        "用户询问了上下文切片的策略选择",
        "用户关心记忆的遗忘曲线怎么调参",
        "用户希望检索结果能给出打分依据",
        "用户对多智能体编排的并发度有疑问",
        "用户要求循环护栏必须能拦住死循环",
        "用户提到工具调用失败时需要熔断",
    ]
    for topic in topics:
        await memory.remember(topic, layer=MemoryLayer.EPISODIC, session_id="s", importance=8.0)
    result = await memory.maybe_reflect("s", force=True)
    assert result and result["insights"]
    insights = [r for r in memory.store.all(MemoryLayer.SEMANTIC) if r.source == "reflection"]
    assert insights
    assert insights[0].importance >= 8.0        # 抽象结论优先级高于原始事件
    assert insights[0].derived_from             # 可回溯到来源记忆


async def test_reflection_skipped_below_threshold(memory):
    await memory.remember("孤立的一条记录", layer=MemoryLayer.EPISODIC, session_id="s")
    assert await memory.maybe_reflect("s") is None


async def test_reflection_threshold_accumulates(memory):
    settings.memory.reflection_importance_threshold = 1.0
    try:
        for topic in ("我要求所有回复使用中文", "我必须先看结论再看推导",
                      "我偏好用 Python 而不是 Java", "我不要过长的解释"):
            await memory.remember(topic, layer=MemoryLayer.SEMANTIC, session_id="s")
        assert await memory.maybe_reflect("s") is not None
    finally:
        settings.memory.reflection_importance_threshold = 6.0

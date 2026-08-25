"""记忆管理器 —— 四层记忆的统一读写入口。

它负责三件上层不该操心的事:
  写入路由   一条信息该进哪层、给多少重要性、是不是重复
  混合召回   跨层检索并按分区打包成上下文条目
  记忆演化   遗忘衰减、反思固化、技能强化

其中"反思固化"最值得说: 借鉴 Generative Agents 的做法, 当累积的新信息重要性
超过阈值, 就触发一次反思, 让模型从一堆零散事件里抽出高层结论存进语义记忆。
这让记忆不只是越堆越多, 而是会**逐渐变得更抽象、更有用**。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from agentp.config import settings
from agentp.context.assembler import ContextItem
from agentp.context.chunker import get_chunker
from agentp.llm.base import Message
from agentp.memory.base import (
    MemoryLayer,
    MemoryRecord,
    RetrievalResult,
    Skill,
    judge_importance,
)
from agentp.memory.store import MemoryStore
from agentp.memory.working import Turn, WorkingMemory
from agentp.observability.trace import current_tracer
from agentp.util import cosine, safe_json_loads


class MemoryManager:
    def __init__(self, provider: Any = None, store: Optional[MemoryStore] = None) -> None:
        self._provider = provider
        self.store = store or MemoryStore()
        self._working: dict[str, WorkingMemory] = {}
        self._pending_importance: dict[str, float] = {}
        self.reflection_count = 0

    @property
    def provider(self) -> Any:
        if self._provider is None:
            from agentp.llm import get_provider

            self._provider = get_provider()
        return self._provider

    # ==================================================================
    # 工作记忆
    # ==================================================================

    def working(self, session_id: str) -> WorkingMemory:
        if session_id not in self._working:
            self._working[session_id] = WorkingMemory(session_id)
        return self._working[session_id]

    async def observe_turn(self, session_id: str, role: str, content: str, **meta: Any) -> None:
        """记录一轮对话, 必要时触发工作记忆压缩。"""
        wm = self.working(session_id)
        wm.add(role, content, **meta)
        report = await wm.maybe_compact(
            summarizer=self._summarize,
            on_evict=lambda turns: self._sink_to_episodic(session_id, turns),
        )
        if report:
            tracer = current_tracer()
            if tracer.spans:
                tracer.spans[-1].add_event("working_memory.compacted", **report)

    async def _summarize(self, text: str) -> str:
        resp = await self.provider.chat(
            [
                Message(role="system", content="把对话压缩成要点摘要, 保留决策、事实、待办, 去掉寒暄。"),
                Message(role="user", content=text),
            ],
            hint="summarize",
            max_tokens=400,
        )
        return resp.content

    async def _sink_to_episodic(self, session_id: str, turns: list[Turn]) -> None:
        for turn in turns:
            if turn.role in ("user", "assistant") and len(turn.content) > 15:
                await self.remember(
                    content=f"{turn.role}: {turn.content}",
                    layer=MemoryLayer.EPISODIC,
                    session_id=session_id,
                    source="working_overflow",
                )

    # ==================================================================
    # 写入
    # ==================================================================

    async def remember(
        self,
        content: str,
        layer: MemoryLayer = MemoryLayer.EPISODIC,
        importance: Optional[float] = None,
        session_id: str = "",
        source: str = "",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        derived_from: Optional[list[str]] = None,
        dedup: bool = True,
    ) -> MemoryRecord:
        content = content.strip()
        if not content:
            raise ValueError("记忆内容不能为空")

        embedding = (await self.provider.embed([content]))[0]
        importance = judge_importance(content, source) if importance is None else importance

        if dedup:
            dup = self.store.find_duplicate(embedding, layer, settings.memory.dedup_threshold)
            if dup is not None:
                # 重复出现 = 这条信息更重要, 而不是要多存一份
                dup.importance = min(10.0, dup.importance + 0.6)
                dup.touch()
                dup.metadata["duplicate_hits"] = dup.metadata.get("duplicate_hits", 0) + 1
                return dup

        record = MemoryRecord(
            content=content, layer=layer, importance=importance, session_id=session_id,
            source=source, tags=tags or [], embedding=embedding, metadata=metadata or {},
            derived_from=derived_from or [],
        )
        self.store.add(record)
        if layer in (MemoryLayer.EPISODIC, MemoryLayer.SEMANTIC):
            self._pending_importance[session_id] = (
                self._pending_importance.get(session_id, 0.0) + importance / 10.0
            )
        return record

    async def extract_facts(self, text: str, session_id: str = "") -> list[MemoryRecord]:
        """从自然语言里抽取值得长期保存的事实, 存进语义记忆。"""
        with current_tracer().span("memory.extract_facts", kind="memory"):
            resp = await self.provider.chat(
                [
                    Message(
                        role="system",
                        content=(
                            "从文本中抽取值得长期记住的事实(用户偏好、身份、约束、结论)。"
                            '严格输出 JSON: {"facts":[{"content":"...","importance":0-10}]}。'
                            "没有就返回空数组。"
                        ),
                    ),
                    Message(role="user", content=text),
                ],
                hint="extract_facts",
                max_tokens=500,
            )
            data = safe_json_loads(resp.content) or {}
            out: list[MemoryRecord] = []
            for fact in (data.get("facts") or [])[:8]:
                content = str(fact.get("content", "")).strip()
                if len(content) < 5:
                    continue
                out.append(
                    await self.remember(
                        content=content,
                        layer=MemoryLayer.SEMANTIC,
                        importance=float(fact.get("importance", 5.0)),
                        session_id=session_id,
                        source="fact_extraction",
                    )
                )
            return out

    async def ingest_document(
        self, text: str, source: str = "doc", strategy: str = "structure", **chunk_kwargs: Any
    ) -> dict[str, Any]:
        """文档入库: 切片 → 批量 embed → 写语义记忆。切片策略可选, 方便对比效果。"""
        with current_tracer().span("memory.ingest", kind="memory", source=source, strategy=strategy) as span:
            chunker = get_chunker(strategy, **chunk_kwargs)
            chunks = await chunker.split(text, source=source)
            # 只索引叶子块: 分层切片下父块只作展开用, 索引它会造成重复召回
            indexable = [c for c in chunks if c.meta.get("role") != "parent"]
            embeddings = await self.provider.embed([c.text for c in indexable])

            created = 0
            skipped = 0
            for chunk, emb in zip(indexable, embeddings):
                # 入库必须幂等: 同一份文档被重复上传是常态, 不去重会让检索结果
                # 被同一段内容占满, MMR 也救不回来(它们本来就是同一个东西)
                dup = self.store.find_duplicate(
                    emb, MemoryLayer.SEMANTIC, settings.memory.dedup_threshold
                )
                if dup is not None:
                    dup.touch()
                    dup.metadata["duplicate_hits"] = dup.metadata.get("duplicate_hits", 0) + 1
                    skipped += 1
                    continue
                record = MemoryRecord(
                    content=chunk.text,
                    layer=MemoryLayer.SEMANTIC,
                    importance=6.0,
                    source=source,
                    embedding=emb,
                    tags=chunk.heading_path,
                    metadata={
                        "chunk_id": chunk.id,
                        "parent_id": chunk.parent_id,
                        "strategy": strategy,
                        "tokens": chunk.tokens,
                        "heading_path": chunk.heading_path,
                    },
                )
                self.store.add(record)
                created += 1

            from agentp.context.chunker import chunk_stats

            stats = chunk_stats(chunks)
            span.attributes.update({"chunks": len(chunks), "indexed": created, "skipped": skipped})
            return {"source": source, "strategy": strategy, "indexed": created,
                    "skipped_duplicates": skipped, "chunk_stats": stats}

    # ==================================================================
    # 检索
    # ==================================================================

    async def retrieve(
        self,
        query: str,
        session_id: str = "",
        top_k: Optional[int] = None,
        layers: Optional[Sequence[MemoryLayer]] = None,
    ) -> list[RetrievalResult]:
        top_k = top_k or settings.memory.retrieval_top_k
        query_vec = (await self.provider.embed([query]))[0]
        return self.store.search(
            query=query, query_embedding=query_vec, top_k=top_k,
            layers=layers, session_id=session_id or None,
        )

    async def build_context_items(
        self, query: str, session_id: str = "", top_k: Optional[int] = None
    ) -> list[ContextItem]:
        """检索并按记忆层拆到对应的上下文分区。

        分层给的不只是标签, 更是**预算隔离**: 检索结果再多也挤不掉程序性记忆的配额,
        因为它们在预算调度器里是两个独立分区。
        """
        top_k = top_k or settings.memory.retrieval_top_k
        results = await self.retrieve(query, session_id=session_id, top_k=top_k * 2)

        zone_of = {
            MemoryLayer.SEMANTIC: "semantic",
            MemoryLayer.EPISODIC: "episodic",
            MemoryLayer.PROCEDURAL: "procedural",
            MemoryLayer.WORKING: "history",
        }
        items: list[ContextItem] = []
        for res in results:
            rec = res.record
            zone = zone_of.get(rec.layer, "retrieved")
            # 文档切片进 retrieved 分区, 和"模型自己记住的事实"区分开
            if rec.layer == MemoryLayer.SEMANTIC and rec.metadata.get("chunk_id"):
                zone = "retrieved"
            label = " > ".join(rec.tags[:2]) if rec.tags else rec.source or rec.layer.value
            items.append(
                ContextItem(
                    text=rec.content,
                    zone=zone,
                    score=res.score,
                    source=label,
                    meta={"memory_id": rec.id, "breakdown": res.breakdown,
                          "layer": rec.layer.value},
                )
            )
        return items

    # ==================================================================
    # 程序性记忆(技能)
    # ==================================================================

    async def record_trajectory(
        self, goal: str, steps: list[str], success: bool, session_id: str = ""
    ) -> Optional[MemoryRecord]:
        """把一次任务执行固化成技能。失败也记 —— 用来降低这条路径的成功率。"""
        if not steps:
            return None
        goal_pattern = goal.strip()[:80]
        embedding = (await self.provider.embed([goal_pattern]))[0]

        existing = None
        for rec in self.store.all(MemoryLayer.PROCEDURAL):
            if rec.embedding and cosine(embedding, rec.embedding) > 0.86:
                existing = rec
                break

        if existing is not None:
            skill = Skill.from_dict(existing.metadata["skill"])
            skill.success_count += int(success)
            skill.fail_count += int(not success)
            total = skill.success_count + skill.fail_count
            skill.avg_steps = (skill.avg_steps * (total - 1) + len(steps)) / total
            if success and skill.win_rate >= 0.5:
                skill.steps = steps[:8]  # 用最新的成功路径覆盖
            existing.metadata["skill"] = skill.to_dict()
            existing.content = skill.render()
            existing.importance = min(10.0, 4.0 + skill.win_rate * 5)
            existing.touch()
            return existing

        skill = Skill(
            goal_pattern=goal_pattern, steps=steps[:8],
            success_count=int(success), fail_count=int(not success), avg_steps=len(steps),
        )
        record = MemoryRecord(
            content=skill.render(), layer=MemoryLayer.PROCEDURAL,
            importance=6.5 if success else 4.0, session_id=session_id,
            source="trajectory", embedding=embedding, metadata={"skill": skill.to_dict()},
        )
        return self.store.add(record)

    async def recall_skills(self, goal: str, top_k: int = 2) -> list[RetrievalResult]:
        """召回可用技能。低胜率的直接过滤 —— 给模型一条烂路径不如不给。"""
        query_vec = (await self.provider.embed([goal]))[0]
        return self.store.search(
            query=goal, query_embedding=query_vec, top_k=top_k,
            layers=[MemoryLayer.PROCEDURAL],
            predicate=lambda r: (r.metadata.get("skill", {}).get("win_rate", 0) or 0) >= 0.4,
            use_mmr=False,
        )

    # ==================================================================
    # 记忆演化
    # ==================================================================

    async def maybe_reflect(self, session_id: str, force: bool = False) -> Optional[dict[str, Any]]:
        """累积重要性超阈值时, 从零散事件里提炼高层结论。

        这一步是记忆系统区别于"日志数据库"的地方: 存下来的东西会自己长出抽象层。
        """
        pending = self._pending_importance.get(session_id, 0.0)
        if not force and pending < settings.memory.reflection_importance_threshold:
            return None

        recent = sorted(
            [r for r in self.store.all() if r.session_id == session_id and not r.derived_from],
            key=lambda r: r.created_at, reverse=True,
        )[:15]
        if len(recent) < 3:
            return None

        with current_tracer().span("memory.reflect", kind="memory", records=len(recent)) as span:
            digest = "\n".join(f"- {r.content}" for r in recent)
            resp = await self.provider.chat(
                [
                    Message(
                        role="system",
                        content=(
                            "你在整理智能体的长期记忆。从下列零散记录中提炼 1-3 条高层洞察"
                            "(用户模式、反复出现的约束、可复用结论), 不要复述原文。"
                            '严格输出 JSON: {"insights":["...","..."]}'
                        ),
                    ),
                    Message(role="user", content=digest),
                ],
                hint="insight",
                max_tokens=400,
            )
            data = safe_json_loads(resp.content) or {}
            insights = [str(i).strip() for i in (data.get("insights") or []) if str(i).strip()]

            created: list[str] = []
            for insight in insights[:3]:
                rec = await self.remember(
                    content=insight, layer=MemoryLayer.SEMANTIC, importance=8.0,
                    session_id=session_id, source="reflection",
                    derived_from=[r.id for r in recent], tags=["insight"],
                )
                created.append(rec.id)

            self._pending_importance[session_id] = 0.0
            self.reflection_count += 1
            span.attributes["insights"] = len(created)
            return {"insights": insights[:3], "record_ids": created, "from_records": len(recent)}

    def decay_and_prune(self) -> dict[str, Any]:
        return self.store.decay_and_prune()

    # ==================================================================

    def stats(self) -> dict[str, Any]:
        return {
            **self.store.stats(),
            "sessions": len(self._working),
            "reflections": self.reflection_count,
            "working": {sid: wm.to_dict()["utilization"] for sid, wm in self._working.items()},
        }

    def dump(self, session_id: Optional[str] = None, limit: int = 100) -> dict[str, Any]:
        """给 UI 用的记忆浏览器数据。"""
        records = self.store.all(include_archived=True)
        if session_id:
            records = [r for r in records if r.session_id in ("", session_id)]
        records.sort(key=lambda r: r.strength(self.store.half_life_hours), reverse=True)
        by_layer: dict[str, list[dict[str, Any]]] = {layer.value: [] for layer in MemoryLayer}
        for rec in records[:limit]:
            item = rec.to_dict()
            item["strength"] = round(rec.strength(self.store.half_life_hours), 4)
            by_layer[rec.layer.value].append(item)
        if session_id and session_id in self._working:
            by_layer["working"] = [
                {**t.to_dict(), "layer": "working"} for t in self._working[session_id].turns
            ]
        return {"stats": self.stats(), "by_layer": by_layer}


_default_manager: Optional[MemoryManager] = None


def get_memory() -> MemoryManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = MemoryManager()
    return _default_manager


def reset_memory() -> None:
    global _default_manager
    _default_manager = None

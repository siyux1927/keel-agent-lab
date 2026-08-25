"""记忆存储 + 混合检索。

检索是记忆系统的核心难点。这里用的是四路信号融合:

  向量相似度  语义匹配, 抗改写("怎么切片" ↔ "分块策略")
  BM25       词法匹配, 抗术语("HierarchicalChunker" 这种词向量模型常常认不准)
  时间新鲜度  艾宾浩斯衰减后的记忆强度
  重要性     写入时的主观打分

前两路做**混合召回**(RRF 思路的加权版), 后两路做**排序调权**, 最后用 MMR 去冗余。

单靠向量检索的典型翻车场景: 用户问某个具体报错码, 向量把所有"报错相关"的记忆
都拉出来了, 但那个精确的错误码反而排在后面 —— BM25 就是来治这个的。
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from agentp.config import settings
from agentp.memory.base import MemoryLayer, MemoryRecord, RetrievalResult
from agentp.observability.trace import current_tracer
from agentp.util import BM25, cosine, minmax_normalize, mmr, now_ts


class MemoryStore:
    """进程内存储。接口刻意做成可替换成 Milvus / pgvector 的形状。"""

    def __init__(self, half_life_hours: Optional[float] = None) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._bm25 = BM25()
        self.half_life_hours = half_life_hours or settings.memory.decay_half_life_hours

    # -- 写 ----------------------------------------------------------------

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self._records[record.id] = record
        self._bm25.add(record.id, record.content)
        return record

    def delete(self, record_id: str) -> bool:
        if self._records.pop(record_id, None) is None:
            return False
        self._bm25.remove(record_id)
        return True

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        return self._records.get(record_id)

    def all(self, layer: Optional[MemoryLayer] = None, include_archived: bool = False) -> list[MemoryRecord]:
        out = []
        for rec in self._records.values():
            if layer and rec.layer != layer:
                continue
            if rec.archived and not include_archived:
                continue
            out.append(rec)
        return out

    def find_duplicate(
        self, embedding: Sequence[float], layer: MemoryLayer, threshold: float
    ) -> Optional[MemoryRecord]:
        """语义去重。同一个事实被反复说, 应该是"加强"而不是"多存一份"。"""
        best, best_sim = None, threshold
        for rec in self._records.values():
            if rec.layer != layer or rec.archived or not rec.embedding:
                continue
            sim = cosine(list(embedding), rec.embedding)
            if sim >= best_sim:
                best, best_sim = rec, sim
        return best

    # -- 读 ----------------------------------------------------------------

    def search(
        self,
        query: str,
        query_embedding: Optional[Sequence[float]] = None,
        top_k: int = 6,
        layers: Optional[Sequence[MemoryLayer]] = None,
        session_id: Optional[str] = None,
        predicate: Optional[Callable[[MemoryRecord], bool]] = None,
        use_mmr: bool = True,
        touch: bool = True,
    ) -> list[RetrievalResult]:
        cfg = settings.memory
        tracer = current_tracer()
        with tracer.span("memory.search", kind="memory", query=query[:60], top_k=top_k) as span:
            candidates = [
                rec for rec in self._records.values()
                if not rec.archived
                and (not layers or rec.layer in layers)
                and (session_id is None or rec.session_id in ("", session_id))
                and (predicate is None or predicate(rec))
            ]
            span.attributes["candidates"] = len(candidates)
            if not candidates:
                return []

            # --- 两路召回 -------------------------------------------------
            if query_embedding:
                vec_raw = [
                    cosine(list(query_embedding), rec.embedding) if rec.embedding else 0.0
                    for rec in candidates
                ]
            else:
                vec_raw = [0.0] * len(candidates)
            bm25_raw = [self._bm25.score(query, rec.id) for rec in candidates]

            # 两路打分量纲不同(余弦 ∈[-1,1], BM25 ∈[0,∞)), 必须各自归一化再融合
            vec_scores = minmax_normalize(vec_raw)
            bm25_scores = minmax_normalize(bm25_raw)
            w = cfg.hybrid_vector_weight
            relevance = [w * v + (1 - w) * b for v, b in zip(vec_scores, bm25_scores)]

            # --- 排序调权 -------------------------------------------------
            at = now_ts()
            results: list[RetrievalResult] = []
            for rec, rel, vr, br in zip(candidates, relevance, vec_scores, bm25_scores):
                recency = rec.strength(self.half_life_hours, at)
                importance = rec.importance / 10.0
                final = (
                    cfg.weight_relevance * rel
                    + cfg.weight_recency * recency
                    + cfg.weight_importance * importance
                )
                results.append(
                    RetrievalResult(
                        record=rec,
                        score=final,
                        breakdown={
                            "vector": vr, "bm25": br, "relevance": rel,
                            "recency": recency, "importance": importance,
                        },
                    )
                )
            results.sort(key=lambda r: r.score, reverse=True)

            # --- 去冗余 ---------------------------------------------------
            if use_mmr and query_embedding and len(results) > top_k:
                pool = results[: top_k * 4]  # 只在头部候选里做 MMR, 省算力
                selected = mmr(
                    list(query_embedding),
                    [(r, r.record.embedding or [], r.score) for r in pool],
                    top_k=top_k,
                    lambda_=cfg.mmr_lambda,
                )
                results = selected
            else:
                results = results[:top_k]

            if touch:
                for r in results:
                    r.record.touch()  # 被检索到 = 复习一次, 重置遗忘计时
            span.attributes["returned"] = len(results)
            return results

    # -- 遗忘 --------------------------------------------------------------

    def decay_and_prune(self, threshold: Optional[float] = None) -> dict[str, Any]:
        """把强度跌破阈值的记忆归档。

        注意是**归档而非删除**: 记忆不参与检索但仍可审计。
        线上删数据是不可逆操作, 而遗忘策略几乎必然要调参。
        """
        threshold = threshold if threshold is not None else settings.memory.prune_threshold
        at = now_ts()
        archived: list[str] = []
        for rec in self._records.values():
            if rec.archived or rec.layer == MemoryLayer.PROCEDURAL:
                continue  # 技能不遗忘, 只靠成功率淘汰
            if rec.strength(self.half_life_hours, at) < threshold:
                rec.archived = True
                archived.append(rec.id)
        return {"archived": len(archived), "ids": archived[:20], "remaining": len(self.all())}

    def stats(self) -> dict[str, Any]:
        at = now_ts()
        by_layer: dict[str, dict[str, Any]] = {}
        for layer in MemoryLayer:
            recs = [r for r in self._records.values() if r.layer == layer]
            active = [r for r in recs if not r.archived]
            strengths = [r.strength(self.half_life_hours, at) for r in active]
            by_layer[layer.value] = {
                "total": len(recs),
                "active": len(active),
                "archived": len(recs) - len(active),
                "avg_importance": round(sum(r.importance for r in active) / len(active), 2)
                if active else 0.0,
                "avg_strength": round(sum(strengths) / len(strengths), 3) if strengths else 0.0,
            }
        return {
            "total": len(self._records),
            "active": len(self.all()),
            "half_life_hours": self.half_life_hours,
            "by_layer": by_layer,
        }

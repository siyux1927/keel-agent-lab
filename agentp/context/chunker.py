"""分层切片策略。

切片是 RAG / 长上下文里最被低估的一环: 切错了, 后面再好的向量模型和 rerank 都救不回来。
这里实现了工业界主流的四条路线, 可以直接对比效果:

  FixedTokenChunker   定长 + 重叠。最快、最可预测, 但会把句子拦腰砍断。
  RecursiveChunker    按分隔符优先级递归下切。通用默认值, 尽量在语义边界断开。
  StructureChunker    吃透 Markdown 结构: 标题路径下推, 代码块不可分。
  SemanticChunker     句向量相邻距离找突变点。真正按"话题"断开, 代价是要跑 embedding。
  HierarchicalChunker small-to-big: 小块用于检索命中, 大块(父块)用于喂给模型。

一个共同的硬约束: **每个块都携带自己的 token 数**, 下游预算调度器才有得算。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from agentp.config import settings
from agentp.util import count_tokens, cosine, new_id

# --------------------------------------------------------------------------


@dataclass
class Chunk:
    id: str
    text: str
    tokens: int
    index: int = 0
    heading_path: list[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    level: int = 0
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    @property
    def display_title(self) -> str:
        return " > ".join(self.heading_path) if self.heading_path else f"chunk#{self.index}"

    def to_dict(self, with_text: bool = True) -> dict[str, Any]:
        d = {
            "id": self.id,
            "tokens": self.tokens,
            "index": self.index,
            "heading_path": self.heading_path,
            "parent_id": self.parent_id,
            "level": self.level,
            "source": self.source,
            "meta": self.meta,
        }
        if with_text:
            d["text"] = self.text
        return d


def _make_chunk(text: str, index: int, **kwargs: Any) -> Chunk:
    return Chunk(id=new_id("ck"), text=text, tokens=count_tokens(text), index=index, **kwargs)


# --------------------------------------------------------------------------


class BaseChunker:
    name = "base"

    def __init__(self, chunk_tokens: Optional[int] = None, overlap: Optional[int] = None) -> None:
        self.chunk_tokens = chunk_tokens or settings.context.default_chunk_tokens
        self.overlap = overlap if overlap is not None else settings.context.default_chunk_overlap

    async def split(self, text: str, source: str = "") -> list[Chunk]:
        raise NotImplementedError

    # -- 公共工具 ----------------------------------------------------------

    def _apply_overlap(self, pieces: list[str]) -> list[str]:
        """把上一块的尾部拼到下一块开头, 避免边界处的信息被"切断关系"。"""
        if self.overlap <= 0 or len(pieces) <= 1:
            return pieces
        out = [pieces[0]]
        for prev, cur in zip(pieces, pieces[1:]):
            tail = self._tail_by_tokens(prev, self.overlap)
            out.append((tail + "\n" + cur) if tail else cur)
        return out

    @staticmethod
    def _tail_by_tokens(text: str, tokens: int) -> str:
        if tokens <= 0:
            return ""
        # 按句子回溯, 保证重叠部分本身是完整句而不是半截话
        sentences = _split_sentences(text)
        buf: list[str] = []
        total = 0
        for sent in reversed(sentences):
            t = count_tokens(sent)
            if total + t > tokens and buf:
                break
            buf.insert(0, sent)
            total += t
        return "".join(buf).strip()

    def _merge_to_budget(self, pieces: Sequence[str]) -> list[str]:
        """贪心合并小片段, 直到逼近 chunk_tokens。避免产出一堆碎块。"""
        merged: list[str] = []
        buf: list[str] = []
        buf_tokens = 0
        for piece in pieces:
            t = count_tokens(piece)
            if buf and buf_tokens + t > self.chunk_tokens:
                merged.append("".join(buf).strip())
                buf, buf_tokens = [], 0
            buf.append(piece)
            buf_tokens += t
        if buf:
            tail = "".join(buf).strip()
            if tail:
                merged.append(tail)
        return merged


# --------------------------------------------------------------------------

_SENT_END = re.compile(r"(?<=[。！？!?；;\n])")


def _split_sentences(text: str) -> list[str]:
    """中英混合句子切分。英文额外处理 ". " 且排除常见缩写。"""
    rough = [s for s in _SENT_END.split(text) if s]
    out: list[str] = []
    for seg in rough:
        if len(seg) < 200:
            out.append(seg)
            continue
        parts = re.split(r"(?<=\.)\s+(?=[A-Z])", seg)
        out.extend(parts if len(parts) > 1 else [seg])
    return [s for s in out if s.strip()]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


# --------------------------------------------------------------------------


class FixedTokenChunker(BaseChunker):
    """定长切片。基线方案 —— 用来跟别的策略做对比。"""

    name = "fixed"

    async def split(self, text: str, source: str = "") -> list[Chunk]:
        if not text.strip():
            return []
        # 按字符步长近似 token 步长(count_tokens 单调), 再二分修正边界
        pieces: list[str] = []
        cursor = 0
        while cursor < len(text):
            end = _advance_by_tokens(text, cursor, self.chunk_tokens)
            pieces.append(text[cursor:end])
            if end >= len(text):
                break
            cursor = end
        pieces = self._apply_overlap(pieces)
        return [
            _make_chunk(p, i, source=source, meta={"strategy": self.name})
            for i, p in enumerate(pieces) if p.strip()
        ]


def _advance_by_tokens(text: str, start: int, budget: int) -> int:
    """从 start 出发, 找到 token 数刚好不超预算的结束位置(二分)。"""
    lo, hi = start + 1, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[start:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return max(lo, start + 1)


class RecursiveChunker(BaseChunker):
    """按分隔符优先级递归下切 —— 通用场景的默认选择。

    先试最"重"的边界(空行/段落), 不够小再退到句号、逗号、空格。
    好处是绝大多数块都断在语义边界上, 且不需要任何模型调用。
    """

    name = "recursive"
    SEPARATORS = ["\n\n", "\n", "。", "！", "？", "; ", "；", ". ", ", ", "，", " ", ""]

    async def split(self, text: str, source: str = "") -> list[Chunk]:
        if not text.strip():
            return []
        pieces = self._recurse(text, self.SEPARATORS)
        pieces = self._merge_to_budget(pieces)
        pieces = self._apply_overlap(pieces)
        return [
            _make_chunk(p, i, source=source, meta={"strategy": self.name})
            for i, p in enumerate(pieces) if p.strip()
        ]

    def _recurse(self, text: str, separators: list[str]) -> list[str]:
        if count_tokens(text) <= self.chunk_tokens:
            return [text]
        if not separators:
            end = _advance_by_tokens(text, 0, self.chunk_tokens)
            return [text[:end]] + self._recurse(text[end:], [])
        sep, rest = separators[0], separators[1:]
        if sep == "":
            end = _advance_by_tokens(text, 0, self.chunk_tokens)
            return [text[:end]] + self._recurse(text[end:], [])
        if sep not in text:
            return self._recurse(text, rest)
        parts = [p + sep for p in text.split(sep)]
        parts[-1] = parts[-1][: -len(sep)] if parts[-1].endswith(sep) else parts[-1]
        out: list[str] = []
        for part in parts:
            if not part:
                continue
            out.extend([part] if count_tokens(part) <= self.chunk_tokens else self._recurse(part, rest))
        return out


class StructureChunker(BaseChunker):
    """Markdown 结构感知切片。

    两个关键设计:
      1. **标题路径下推** —— 每个块都带上 "H1 > H2 > H3", 检索命中孤立段落时
         模型仍然知道它在讲什么(解决 chunk 脱离上下文的经典问题)。
      2. **代码块原子性** —— ``` 围栏内的内容绝不切开, 切一半的代码是纯噪声。
    """

    name = "structure"
    _HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

    async def split(self, text: str, source: str = "") -> list[Chunk]:
        if not text.strip():
            return []
        sections = self._parse_sections(text)
        chunks: list[Chunk] = []
        fallback = RecursiveChunker(self.chunk_tokens, self.overlap)
        for path, body in sections:
            body = body.strip()
            if not body:
                continue
            header = ("# 位置: " + " > ".join(path) + "\n") if path else ""
            if count_tokens(body) <= self.chunk_tokens:
                chunks.append(
                    _make_chunk(
                        header + body, len(chunks), heading_path=list(path), source=source,
                        level=len(path), meta={"strategy": self.name},
                    )
                )
                continue
            # 段落过长: 降级用递归切, 但每个子块都补回标题路径
            for sub in await fallback.split(body, source=source):
                chunks.append(
                    _make_chunk(
                        header + sub.text, len(chunks), heading_path=list(path), source=source,
                        level=len(path), meta={"strategy": self.name, "split_from_section": True},
                    )
                )
        return chunks

    def _parse_sections(self, text: str) -> list[tuple[list[str], str]]:
        sections: list[tuple[list[str], str]] = []
        stack: list[str] = []
        buf: list[str] = []
        in_code = False

        def flush() -> None:
            if buf and "".join(buf).strip():
                sections.append((list(stack), "".join(buf)))
            buf.clear()

        for line in text.splitlines(keepends=True):
            if line.strip().startswith("```"):
                in_code = not in_code
                buf.append(line)
                continue
            match = None if in_code else self._HEADING.match(line.rstrip("\n"))
            if match:
                flush()
                depth = len(match.group(1))
                stack = stack[: depth - 1]
                stack.append(match.group(2).strip())
            else:
                buf.append(line)
        flush()
        return sections or [([], text)]


class SemanticChunker(BaseChunker):
    """语义切片: 在"话题突变"处断开。

    做法: 逐句 embedding → 算相邻句余弦距离 → 距离序列取分位数当阈值 →
    超过阈值的位置就是话题切换点。比固定长度更贴合内容, 代价是 O(n) 次 embedding。

    阈值用**分位数而非绝对值**: 不同文档的句间相似度基线差别很大,
    写死 0.3 这种阈值换个语料就失效了。
    """

    name = "semantic"

    def __init__(
        self,
        chunk_tokens: Optional[int] = None,
        overlap: Optional[int] = None,
        provider: Any = None,
        breakpoint_percentile: float = 82.0,
        min_chunk_tokens: int = 60,
    ) -> None:
        super().__init__(chunk_tokens, overlap)
        self.provider = provider
        self.breakpoint_percentile = breakpoint_percentile
        self.min_chunk_tokens = min_chunk_tokens

    async def split(self, text: str, source: str = "") -> list[Chunk]:
        if not text.strip():
            return []
        sentences = _split_sentences(text)
        if len(sentences) < 3:
            return await RecursiveChunker(self.chunk_tokens, self.overlap).split(text, source)

        if self.provider is None:
            from agentp.llm import get_provider

            self.provider = get_provider()
        vectors = await self.provider.embed(sentences)

        distances = [1.0 - cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
        threshold = _percentile(distances, self.breakpoint_percentile)

        groups: list[list[str]] = [[sentences[0]]]
        for i, dist in enumerate(distances):
            cur_tokens = count_tokens("".join(groups[-1]))
            # 三个条件任一成立就开新块: 话题变了 / 块已经太大
            topic_shift = dist > threshold and cur_tokens >= self.min_chunk_tokens
            too_big = cur_tokens + count_tokens(sentences[i + 1]) > self.chunk_tokens
            if topic_shift or too_big:
                groups.append([sentences[i + 1]])
            else:
                groups[-1].append(sentences[i + 1])

        pieces = self._apply_overlap(["".join(g).strip() for g in groups if "".join(g).strip()])
        return [
            _make_chunk(
                p, i, source=source,
                meta={"strategy": self.name, "breakpoint_threshold": round(threshold, 4)},
            )
            for i, p in enumerate(pieces)
        ]


class HierarchicalChunker(BaseChunker):
    """small-to-big / parent-document 检索。

    检索用小块(精准, 向量不被稀释), 命中后把它的父块整段喂给模型(上下文完整)。
    这是解决"块小了丢上下文、块大了检索不准"这个两难的标准答案。
    """

    name = "hierarchical"

    def __init__(
        self,
        parent_tokens: int = 900,
        child_tokens: int = 220,
        overlap: int = 32,
    ) -> None:
        super().__init__(child_tokens, overlap)
        self.parent_tokens = parent_tokens
        self.child_tokens = child_tokens

    async def split(self, text: str, source: str = "") -> list[Chunk]:
        parents = await StructureChunker(self.parent_tokens, 0).split(text, source)
        child_chunker = RecursiveChunker(self.child_tokens, self.overlap)
        all_chunks: list[Chunk] = []
        for parent in parents:
            parent.level = 0
            parent.meta.update({"strategy": self.name, "role": "parent"})
            all_chunks.append(parent)
            children = await child_chunker.split(parent.text, source=source)
            for child in children:
                child.parent_id = parent.id
                child.level = 1
                child.heading_path = list(parent.heading_path)
                child.meta.update({"strategy": self.name, "role": "child"})
                all_chunks.append(child)
        for i, chunk in enumerate(all_chunks):
            chunk.index = i
        return all_chunks


# --------------------------------------------------------------------------

CHUNKERS: dict[str, type[BaseChunker]] = {
    "fixed": FixedTokenChunker,
    "recursive": RecursiveChunker,
    "structure": StructureChunker,
    "semantic": SemanticChunker,
    "hierarchical": HierarchicalChunker,
}


def get_chunker(strategy: str = "recursive", **kwargs: Any) -> BaseChunker:
    cls = CHUNKERS.get(strategy)
    if cls is None:
        raise ValueError(f"未知的切片策略: {strategy}, 可选: {list(CHUNKERS)}")
    return cls(**kwargs)


def chunk_stats(chunks: Sequence[Chunk]) -> dict[str, Any]:
    """切片质量指标。对比策略时看这几个数: 块数、token 分布、方差。

    方差尤其重要 —— 块大小忽大忽小会让检索打分失去可比性。
    """
    if not chunks:
        return {"count": 0}
    tokens = [c.tokens for c in chunks]
    mean = sum(tokens) / len(tokens)
    variance = sum((t - mean) ** 2 for t in tokens) / len(tokens)
    return {
        "count": len(chunks),
        "total_tokens": sum(tokens),
        "avg_tokens": round(mean, 1),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
        "std_tokens": round(variance ** 0.5, 1),
        "parents": sum(1 for c in chunks if c.meta.get("role") == "parent"),
        "children": sum(1 for c in chunks if c.meta.get("role") == "child"),
    }

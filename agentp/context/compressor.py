"""上下文压缩。预算不够时, 怎么把内容塞进去而少丢信息。

四种手段, 成本从低到高:
  TruncateCompressor    直接截断。零成本, 信息损失不可控, 只当兜底。
  ExtractiveCompressor  查询感知的抽取式压缩。无需模型调用, 保留原文措辞(不会幻觉)。
  MiddleOutCompressor   保头保尾、压中间。针对"lost in the middle"效应。
  LLMSummaryCompressor  map-reduce 摘要。压缩率最高, 但要花钱花时间, 且可能引入幻觉。

选型原则: 能不调模型就不调。检索结果用抽取式, 长对话历史才值得上 LLM 摘要。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from agentp.util import (
    count_tokens,
    cosine,
    minmax_normalize,
    truncate_to_tokens,
)
from agentp.context.chunker import _split_sentences


@dataclass
class CompressionResult:
    text: str
    method: str
    original_tokens: int
    final_tokens: int
    dropped_units: int = 0

    @property
    def ratio(self) -> float:
        return self.final_tokens / self.original_tokens if self.original_tokens else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "original_tokens": self.original_tokens,
            "final_tokens": self.final_tokens,
            "ratio": round(self.ratio, 3),
            "dropped_units": self.dropped_units,
        }


class BaseCompressor:
    name = "base"

    async def compress(
        self, text: str, budget: int, query: str = "", **kwargs: Any
    ) -> CompressionResult:
        raise NotImplementedError

    @staticmethod
    def _passthrough(text: str, method: str) -> CompressionResult:
        n = count_tokens(text)
        return CompressionResult(text, f"{method}:noop", n, n)


class TruncateCompressor(BaseCompressor):
    name = "truncate"

    def __init__(self, keep: str = "head") -> None:
        self.keep = keep  # head | tail

    async def compress(
        self, text: str, budget: int, query: str = "", **kwargs: Any
    ) -> CompressionResult:
        original = count_tokens(text)
        if original <= budget:
            return self._passthrough(text, self.name)
        out = truncate_to_tokens(text, budget - 8, from_end=(self.keep == "tail"))
        marker = "\n...[前文已截断]" if self.keep == "tail" else "\n...[后文已截断]"
        out = (marker + out) if self.keep == "tail" else (out + marker)
        return CompressionResult(out, f"{self.name}:{self.keep}", original, count_tokens(out))


class ExtractiveCompressor(BaseCompressor):
    """查询感知的抽取式压缩。

    给每个句子打分(与查询的相关度 + 位置先验 + 信息密度), 取 top-N 后**按原顺序**重排。
    保持原顺序很重要, 打乱会破坏叙述逻辑, 模型读起来像碎片。

    相比 LLM 摘要的最大优势: 输出全是原文句子, 不可能产生幻觉。
    """

    name = "extractive"

    def __init__(self, provider: Any = None, use_embedding: bool = True) -> None:
        self.provider = provider
        self.use_embedding = use_embedding

    async def compress(
        self, text: str, budget: int, query: str = "", **kwargs: Any
    ) -> CompressionResult:
        original = count_tokens(text)
        if original <= budget:
            return self._passthrough(text, self.name)

        sentences = _split_sentences(text)
        if len(sentences) <= 2:
            return await TruncateCompressor().compress(text, budget)

        relevance = await self._relevance_scores(sentences, query)
        n = len(sentences)
        scores = []
        for i, sent in enumerate(sentences):
            position = 1.0 - (i / n) * 0.5           # 靠前的句子通常是主旨
            density = min(count_tokens(sent) / 40.0, 1.0)
            scores.append(0.6 * relevance[i] + 0.25 * position + 0.15 * density)

        # 贪心按分数收句子, 直到预算用完
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        chosen: set[int] = set()
        used = 0
        for idx in ranked:
            t = count_tokens(sentences[idx])
            if used + t > budget - 12:
                continue
            chosen.add(idx)
            used += t
            if used >= budget - 12:
                break

        parts: list[str] = []
        last = -2
        for idx in sorted(chosen):
            if idx != last + 1 and parts:
                parts.append(" […] ")  # 明确标出被省略的位置, 别让模型误以为原文连续
            parts.append(sentences[idx])
            last = idx
        out = "".join(parts).strip()
        return CompressionResult(out, self.name, original, count_tokens(out), n - len(chosen))

    async def _relevance_scores(self, sentences: list[str], query: str) -> list[float]:
        if not query:
            return [0.5] * len(sentences)
        if self.use_embedding:
            if self.provider is None:
                from agentp.llm import get_provider

                self.provider = get_provider()
            vectors = await self.provider.embed([query] + sentences)
            qv, svs = vectors[0], vectors[1:]
            return minmax_normalize([cosine(qv, sv) for sv in svs])
        from agentp.util import tokenize

        qt = set(tokenize(query))
        overlaps = [len(qt & set(tokenize(s))) / (len(qt) or 1) for s in sentences]
        return minmax_normalize(overlaps)


class MiddleOutCompressor(BaseCompressor):
    """保头保尾, 压中间。

    "Lost in the Middle"(Liu et al. 2023): 模型对超长上下文中段的信息利用率显著低于两端。
    既然中段本来就容易被忽略, 那要压缩时就优先压它, 把预算让给两端。
    """

    name = "middle_out"

    def __init__(self, head_ratio: float = 0.4, tail_ratio: float = 0.35, inner: Optional[BaseCompressor] = None) -> None:
        self.head_ratio = head_ratio
        self.tail_ratio = tail_ratio
        self.inner = inner or ExtractiveCompressor()

    async def compress(
        self, text: str, budget: int, query: str = "", **kwargs: Any
    ) -> CompressionResult:
        original = count_tokens(text)
        if original <= budget:
            return self._passthrough(text, self.name)

        head_budget = int(budget * self.head_ratio)
        tail_budget = int(budget * self.tail_ratio)
        middle_budget = max(0, budget - head_budget - tail_budget - 20)

        head = truncate_to_tokens(text, head_budget)
        tail = truncate_to_tokens(text, tail_budget, from_end=True)
        middle = text[len(head): len(text) - len(tail)]

        if middle.strip() and middle_budget > 0:
            mid_result = await self.inner.compress(middle, middle_budget, query=query)
            middle_out = f"\n\n[中段已压缩 {mid_result.original_tokens}→{mid_result.final_tokens} tokens]\n{mid_result.text}\n\n"
        else:
            middle_out = f"\n\n[中段 {count_tokens(middle)} tokens 已省略]\n\n"

        out = head + middle_out + tail
        return CompressionResult(out, self.name, original, count_tokens(out))


class LLMSummaryCompressor(BaseCompressor):
    """map-reduce 摘要: 分段摘要 → 再摘要, 突破单次上下文限制。

    对话历史的"滚动摘要"就靠它。注意摘要是有损且可能幻觉的,
    所以只用在**已经发生过的对话**上, 绝不用在用户当前指令或工具定义上。
    """

    name = "llm_summary"

    def __init__(self, provider: Any = None, segment_tokens: int = 900) -> None:
        self.provider = provider
        self.segment_tokens = segment_tokens

    async def compress(
        self, text: str, budget: int, query: str = "", **kwargs: Any
    ) -> CompressionResult:
        original = count_tokens(text)
        if original <= budget:
            return self._passthrough(text, self.name)

        if self.provider is None:
            from agentp.llm import get_provider

            self.provider = get_provider()

        from agentp.context.chunker import RecursiveChunker
        from agentp.llm.base import Message

        segments = await RecursiveChunker(self.segment_tokens, 0).split(text)
        summaries: list[str] = []
        for seg in segments:  # map
            resp = await self.provider.chat(
                [
                    Message(role="system", content="你是上下文压缩器。保留事实、数字、决策和未完成事项, 删除寒暄与重复。"),
                    Message(role="user", content=seg.text),
                ],
                hint="summarize",
                max_tokens=max(120, budget // max(len(segments), 1)),
            )
            summaries.append(resp.content)

        merged = "\n".join(summaries)
        if count_tokens(merged) > budget and len(summaries) > 1:  # reduce
            resp = await self.provider.chat(
                [
                    Message(role="system", content="把多段摘要合并成一段, 去重并保留全部关键结论。"),
                    Message(role="user", content=merged),
                ],
                hint="summarize",
                max_tokens=budget,
            )
            merged = resp.content
        if count_tokens(merged) > budget:
            merged = truncate_to_tokens(merged, budget - 4)

        return CompressionResult(merged, self.name, original, count_tokens(merged), len(segments))


# --------------------------------------------------------------------------


async def compress_items(
    items: Sequence[tuple[str, float]], budget: int
) -> tuple[list[str], int]:
    """对"打过分的条目列表"做预算裁剪(检索结果、记忆列表这类)。

    这里不做文本压缩, 而是按分数**整条丢弃** —— 半条记忆没有意义,
    与其每条都压得残缺, 不如保留完整的高分条目。
    """
    ordered = sorted(items, key=lambda x: x[1], reverse=True)
    kept: list[str] = []
    used = 0
    dropped = 0
    for text, _score in ordered:
        t = count_tokens(text)
        if used + t > budget:
            dropped += 1
            continue
        kept.append(text)
        used += t
    return kept, dropped


COMPRESSORS: dict[str, type[BaseCompressor]] = {
    "truncate": TruncateCompressor,
    "extractive": ExtractiveCompressor,
    "middle_out": MiddleOutCompressor,
    "llm_summary": LLMSummaryCompressor,
}


def get_compressor(name: str = "extractive", **kwargs: Any) -> BaseCompressor:
    cls = COMPRESSORS.get(name)
    if cls is None:
        raise ValueError(f"未知的压缩策略: {name}, 可选: {list(COMPRESSORS)}")
    return cls(**kwargs)

"""无依赖的基础设施: ID、时间、Token 计数、纯 Python 向量运算。

刻意不引入 numpy / tiktoken 作为硬依赖 —— 装了就自动走快路径, 没装也能跑,
这样面试现场断网的机器上 `pip install fastapi` 之后就能演示。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------
# ID / 时间
# --------------------------------------------------------------------------


def new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex[:12]
    return f"{prefix}_{raw}" if prefix else raw


def now_ts() -> float:
    return time.time()


def now_ms() -> float:
    return time.time() * 1000.0


def hours_between(a: float, b: float) -> float:
    return abs(b - a) / 3600.0


# --------------------------------------------------------------------------
# Token 计数
# --------------------------------------------------------------------------

_ENCODER = None
_ENCODER_TRIED = False

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _try_load_encoder():
    global _ENCODER, _ENCODER_TRIED
    if _ENCODER_TRIED:
        return _ENCODER
    _ENCODER_TRIED = True
    try:  # pragma: no cover - 取决于本机是否装了 tiktoken
        import tiktoken

        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODER = None
    return _ENCODER


def count_tokens(text: str) -> int:
    """估算 token 数。

    有 tiktoken 用 tiktoken; 否则用启发式: 中日韩字符 ≈ 1 token,
    其余字符 ≈ 1/4 token。实测对中英混排文本误差在 10% 量级,
    对"预算调度"这个用途足够 —— 关键是**单调且可复现**。
    """
    if not text:
        return 0
    enc = _try_load_encoder()
    if enc is not None:  # pragma: no cover
        return len(enc.encode(text))
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return int(cjk + other / 4.0) + 1


def count_message_tokens(messages: Sequence[dict[str, Any]]) -> int:
    """消息列表的 token 数, 每条消息加 4 token 的角色/分隔开销。"""
    total = 0
    for m in messages:
        total += 4
        for value in m.values():
            if isinstance(value, str):
                total += count_tokens(value)
            elif value is not None:
                total += count_tokens(json.dumps(value, ensure_ascii=False))
    return total


def truncate_to_tokens(text: str, max_tokens: int, from_end: bool = False) -> str:
    """按 token 预算裁剪文本(二分查找字符边界)。"""
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        piece = text[-mid:] if from_end else text[:mid]
        if count_tokens(piece) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[-lo:] if from_end else text[:lo]


# --------------------------------------------------------------------------
# 分词 (给 BM25 和 hashing embedding 用)
# --------------------------------------------------------------------------

_WORD = re.compile(r"[a-zA-Z0-9_]+")
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "and",
    "or", "for", "with", "that", "this", "it", "as", "at", "by", "be",
    "的", "了", "是", "在", "和", "与", "我", "你", "他", "它", "这", "那",
    "有", "就", "都", "而", "及", "或", "一个", "什么", "怎么",
}


def tokenize(text: str) -> list[str]:
    """中英混合分词: 英文按单词, 中文按单字 + 二元组(近似词)。"""
    text = text.lower()
    tokens: list[str] = [w for w in _WORD.findall(text) if w not in _STOP]
    cjk_chars = _CJK.findall(text)
    tokens.extend(c for c in cjk_chars if c not in _STOP)
    # 中文二元组: 弥补没有真正分词器带来的召回损失
    runs = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for run in runs:
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


# --------------------------------------------------------------------------
# 向量运算 (纯 Python, 语料规模小的时候完全够用)
# --------------------------------------------------------------------------

Vector = list[float]


def cosine(a: Vector, b: Vector) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def normalize(vec: Vector) -> Vector:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def mean_vector(vectors: Sequence[Vector]) -> Vector:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    return normalize([x / len(vectors) for x in acc])


def hash_embed(text: str, dim: int = 256) -> Vector:
    """离线确定性 embedding: feature hashing + 有符号哈希 + 亚线性词频。

    这不是语义模型, 本质是把稀疏词袋压到稠密低维空间(random projection),
    近义词捕捉不到, 但字面重合度能稳定反映出来 —— 离线自测/回归够用。
    生产环境把 provider 换成真 embedding 接口即可, 上层检索代码零改动。
    """
    vec = [0.0] * dim
    counts: dict[str, int] = {}
    for tok in tokenize(text):
        counts[tok] = counts.get(tok, 0) + 1
    for tok, cnt in counts.items():
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "big")
        idx = raw % dim
        sign = 1.0 if (raw >> 63) & 1 else -1.0
        vec[idx] += sign * (1.0 + math.log(cnt))
    return normalize(vec)


# --------------------------------------------------------------------------
# BM25 (混合检索的词法通道)
# --------------------------------------------------------------------------


class BM25:
    """极简 BM25。语料变动时惰性重建 —— demo 规模下重建成本可忽略。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[str, list[str]] = {}
        self._df: dict[str, int] = {}
        self._avg_len = 0.0
        self._dirty = True

    def add(self, doc_id: str, text: str) -> None:
        self._docs[doc_id] = tokenize(text)
        self._dirty = True

    def remove(self, doc_id: str) -> None:
        if self._docs.pop(doc_id, None) is not None:
            self._dirty = True

    def _rebuild(self) -> None:
        self._df = {}
        total = 0
        for tokens in self._docs.values():
            total += len(tokens)
            for tok in set(tokens):
                self._df[tok] = self._df.get(tok, 0) + 1
        self._avg_len = total / len(self._docs) if self._docs else 0.0
        self._dirty = False

    def score(self, query: str, doc_id: str) -> float:
        if self._dirty:
            self._rebuild()
        tokens = self._docs.get(doc_id)
        if not tokens:
            return 0.0
        n_docs = len(self._docs)
        freq: dict[str, int] = {}
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1
        doc_len = len(tokens)
        score = 0.0
        for qt in set(tokenize(query)):
            f = freq.get(qt, 0)
            if f == 0:
                continue
            df = self._df.get(qt, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = f + self.k1 * (1 - self.b + self.b * doc_len / (self._avg_len or 1.0))
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self._dirty:
            self._rebuild()
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in self._docs]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# --------------------------------------------------------------------------
# 排序辅助
# --------------------------------------------------------------------------


def minmax_normalize(values: Sequence[float]) -> list[float]:
    """把任意量纲的打分压到 [0,1], 好让向量分和 BM25 分能加权相加。"""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if math.isclose(hi, lo):
        return [1.0 if hi > 0 else 0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def mmr(
    query_vec: Vector,
    candidates: list[tuple[Any, Vector, float]],
    top_k: int,
    lambda_: float = 0.7,
) -> list[Any]:
    """Maximal Marginal Relevance: 在相关性和多样性之间取平衡。

    纯 top-k 会选出一堆内容重复的记忆, 白白吃掉上下文预算。
    MMR 每一轮挑 "相关性高但与已选结果不像" 的那条。
    """
    selected: list[Any] = []
    selected_vecs: list[Vector] = []
    pool = list(candidates)
    while pool and len(selected) < top_k:
        best_idx, best_score = 0, -1e9
        for i, (item, vec, relevance) in enumerate(pool):
            redundancy = max((cosine(vec, sv) for sv in selected_vecs), default=0.0)
            score = lambda_ * relevance - (1 - lambda_) * redundancy
            if score > best_score:
                best_idx, best_score = i, score
        item, vec, _ = pool.pop(best_idx)
        selected.append(item)
        selected_vecs.append(vec)
    return selected


# --------------------------------------------------------------------------
# 杂项
# --------------------------------------------------------------------------


def stable_hash(obj: Any) -> str:
    """结构无关的稳定哈希 —— 死循环检测靠它给 action 打指纹。"""
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def safe_json_loads(text: str) -> Any:
    """从 LLM 输出里抠 JSON: 先直接解析, 失败就找第一个平衡的 {} 或 []。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
    return None


def chunk_iter(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

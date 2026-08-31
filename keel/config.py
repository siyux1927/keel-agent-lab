"""集中式配置。所有可调参数都在这里, 便于面试现场改一行就换行为。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """迷你 .env 加载器, 省掉 python-dotenv 依赖。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")


def _env(key: str, default: Any, cast=str):
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.lower() in ("1", "true", "yes", "on")
    try:
        return cast(raw)
    except Exception:
        return default


@dataclass
class LLMConfig:
    provider: str = _env("KEEL_PROVIDER", "mock")  # mock | openai
    model: str = _env("KEEL_MODEL", "mock-medium")
    api_key: str = _env("KEEL_API_KEY", "")
    base_url: str = _env("KEEL_BASE_URL", "https://api.deepseek.com/v1")
    embedding_model: str = _env("KEEL_EMBEDDING_MODEL", "text-embedding-3-small")
    temperature: float = _env("KEEL_TEMPERATURE", 0.3, float)
    timeout_s: float = _env("KEEL_TIMEOUT", 60.0, float)
    max_retries: int = _env("KEEL_MAX_RETRIES", 2, int)


@dataclass
class ContextConfig:
    """上下文预算。model_context_window 是硬上限, 其余是软策略。"""

    model_context_window: int = _env("KEEL_CTX_WINDOW", 8192, int)
    reserve_for_output: int = _env("KEEL_RESERVE_OUTPUT", 1024, int)
    safety_margin: float = 0.05  # 留 5% 给 token 估算误差
    embedding_dim: int = 256
    default_chunk_tokens: int = 320
    default_chunk_overlap: int = 48


@dataclass
class MemoryConfig:
    working_max_tokens: int = 1600      # 工作记忆超过就触发滚动摘要
    working_keep_recent: int = 4        # 摘要时无条件保留的最近轮数
    decay_half_life_hours: float = 72.0  # 遗忘曲线半衰期
    prune_threshold: float = 0.05       # 强度低于此值的记忆被归档
    dedup_threshold: float = 0.93       # 语义去重的余弦阈值
    reflection_importance_threshold: float = 6.0  # 累积重要性触发反思固化
    retrieval_top_k: int = 6
    mmr_lambda: float = 0.7
    weight_relevance: float = 0.55
    weight_recency: float = 0.2
    weight_importance: float = 0.25
    hybrid_vector_weight: float = 0.65  # 剩下的给 BM25


@dataclass
class LoopConfig:
    """Loop Engineering 的全部护栏。这些数字是 Agent 不烧钱不失控的关键。"""

    max_steps: int = _env("KEEL_MAX_STEPS", 12, int)
    max_tokens: int = _env("KEEL_MAX_TOKENS", 60000, int)
    max_cost_usd: float = _env("KEEL_MAX_COST", 0.5, float)
    max_wall_time_s: float = _env("KEEL_MAX_WALL", 120.0, float)
    tool_timeout_s: float = 15.0
    tool_max_retries: int = 2
    # 死循环检测
    repeat_action_threshold: int = 3    # 同一 action 指纹重复次数上限
    cycle_detect_window: int = 6        # 检测 A-B-A-B 这类环的窗口
    stall_threshold: int = 3            # 连续无进展步数上限
    # 反思
    reflect_every_n_steps: int = 4
    reflect_on_consecutive_failures: int = 2
    max_replans: int = 3
    # 熔断
    breaker_failure_threshold: int = 3
    breaker_cooldown_s: float = 20.0


@dataclass
class OrchestratorConfig:
    max_concurrency: int = 4
    node_timeout_s: float = 90.0
    max_critic_rounds: int = 2
    max_subtasks: int = 6


@dataclass
class ServerConfig:
    """公网演示部署时的自我保护, 本地开发默认全关。

    把控制台挂到公网, 等于开放了一个任何人都能触发的 CPU 入口: 编排模式一次请求
    要跑十几次模型调用, 切片实验室能吃下任意大的文档。演示站不需要鉴权,
    但需要有个上限, 否则一个爬虫就能把它跑垮。
    """

    public_mode: bool = _env("KEEL_PUBLIC_MODE", False, bool)
    rate_limit_per_min: int = _env("KEEL_RATE_LIMIT", 20, int)
    max_input_chars: int = _env("KEEL_MAX_INPUT_CHARS", 4000, int)
    max_ingest_chars: int = _env("KEEL_MAX_INGEST_CHARS", 20000, int)


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    data_dir: Path = ROOT / "data"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["llm"]["api_key"] = "***" if self.llm.api_key else ""
        return d


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)

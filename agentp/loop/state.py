"""Agent 循环的状态模型。

把状态显式建模成数据结构(而不是散在闭包和局部变量里), 换来三样东西:
可断言(单测能直接检查)、可序列化(能存盘回放)、可观测(前端能直接渲染)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from agentp.util import now_ts, stable_hash


class AgentStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"  # 被护栏中止


class StopReason(str, Enum):
    GOAL_REACHED = "goal_reached"
    MAX_STEPS = "max_steps"
    MAX_TOKENS = "max_tokens"
    MAX_COST = "max_cost"
    TIMEOUT = "timeout"
    LOOP_DETECTED = "loop_detected"
    CYCLE_DETECTED = "cycle_detected"
    NO_PROGRESS = "no_progress"
    TOOL_CIRCUIT_OPEN = "tool_circuit_open"
    MAX_REPLANS = "max_replans"
    ERROR = "error"


@dataclass
class Action:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""

    @property
    def fingerprint(self) -> str:
        """动作指纹 = 工具名 + 规范化参数。死循环检测的基本单位。

        参数要规范化(排序 + 统一序列化), 否则 {"a":1,"b":2} 和 {"b":2,"a":1}
        会被当成两个不同动作, 检测直接失效。
        """
        return f"{self.tool}:{stable_hash(self.args)}"

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args, "fingerprint": self.fingerprint}


@dataclass
class Observation:
    action: Action
    content: str
    ok: bool = True
    error: str = ""
    latency_ms: float = 0.0

    def render(self) -> str:
        if self.ok:
            return f"[{self.action.tool}] {self.content}"
        return f"[{self.action.tool} 失败] {self.error}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(), "ok": self.ok,
            "content": self.content[:2000], "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class Step:
    index: int
    thought: str = ""
    actions: list[Action] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    answer: Optional[str] = None
    reflection: Optional[dict[str, Any]] = None
    progress: float = 0.0        # 本步带来多少新信息, 0~1
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_tokens: int = 0
    duration_ms: float = 0.0
    ts: float = field(default_factory=now_ts)

    @property
    def failed(self) -> bool:
        return bool(self.observations) and all(not o.ok for o in self.observations)

    def render(self) -> str:
        """渲染成注入上下文的 scratchpad 片段。"""
        lines = [f"### 第 {self.index + 1} 步"]
        if self.thought:
            lines.append(f"思考: {self.thought}")
        for obs in self.observations:
            lines.append(f"动作: {obs.action.tool}({_short_args(obs.action.args)})")
            lines.append(f"观察: {obs.render()[:600]}")
        if self.reflection:
            lines.append(f"反思: {self.reflection.get('diagnosis', '')}")
        if self.answer:
            lines.append(f"草稿答案: {self.answer[:300]}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "thought": self.thought,
            "actions": [a.to_dict() for a in self.actions],
            "observations": [o.to_dict() for o in self.observations],
            "answer": self.answer, "reflection": self.reflection,
            "progress": round(self.progress, 3),
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "context_tokens": self.context_tokens,
            "duration_ms": round(self.duration_ms, 2), "ts": self.ts,
        }


def _short_args(args: dict[str, Any]) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        parts.append(f"{k}={s[:60]}{'...' if len(s) > 60 else ''}")
    return ", ".join(parts)


@dataclass
class AgentState:
    goal: str
    session_id: str = "default"
    steps: list[Step] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    avoid_actions: set[str] = field(default_factory=set)  # 反思后拉黑的动作指纹
    status: AgentStatus = AgentStatus.RUNNING
    stop_reason: Optional[StopReason] = None
    stop_detail: str = ""
    final_answer: str = ""
    replans: int = 0
    consecutive_failures: int = 0
    no_progress_steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    started_at: float = field(default_factory=now_ts)

    # -- 派生量 ------------------------------------------------------------

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def elapsed_s(self) -> float:
        return now_ts() - self.started_at

    def fingerprints(self) -> list[str]:
        return [a.fingerprint for step in self.steps for a in step.actions]

    def successful_tools(self) -> list[str]:
        return [
            o.action.tool for step in self.steps for o in step.observations if o.ok
        ]

    def all_observations(self) -> list[Observation]:
        return [o for step in self.steps for o in step.observations]

    # -- 渲染 --------------------------------------------------------------

    def action_digest(self) -> str:
        """已执行动作的极简清单。

        它存在的理由是防御性的: scratchpad 是可压缩分区, 详细步骤随时可能被丢掉,
        而模型一旦看不到自己调过什么就会重复调用同一个工具。
        所以引擎把这份清单注入**系统提示词**(不可丢弃分区)而不是 scratchpad ——
        它很便宜(每个动作一行, 受 max_steps 上限约束), 但必须保证 100% 存活。
        """
        lines = []
        for step in self.steps:
            for obs in step.observations:
                mark = "成功" if obs.ok else "失败"
                lines.append(f"- {obs.action.tool}({_short_args(obs.action.args)}) → {mark}")
        return "### 已执行动作(不要重复)\n" + "\n".join(lines) if lines else ""

    def render_scratchpad(self, max_steps: Optional[int] = None) -> str:
        parts: list[str] = []
        if self.plan:
            parts.append("### 当前计划\n" + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(self.plan)))
        if self.avoid_actions:
            parts.append(f"### 已确认无效, 不要重复的动作\n{len(self.avoid_actions)} 个动作已被拉黑")
        steps = self.steps[-max_steps:] if max_steps else self.steps
        parts.extend(s.render() for s in steps)
        return "\n\n".join(parts)

    def new_step(self) -> Step:
        step = Step(index=len(self.steps))
        self.steps.append(step)
        return step

    def to_dict(self, with_steps: bool = True) -> dict[str, Any]:
        d = {
            "goal": self.goal,
            "session_id": self.session_id,
            "status": self.status.value,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "stop_detail": self.stop_detail,
            "final_answer": self.final_answer,
            "plan": self.plan,
            "step_count": self.step_count,
            "replans": self.replans,
            "consecutive_failures": self.consecutive_failures,
            "no_progress_steps": self.no_progress_steps,
            "avoid_actions": len(self.avoid_actions),
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "cost_usd": round(self.cost_usd, 6),
            },
            "elapsed_s": round(self.elapsed_s, 2),
        }
        if with_steps:
            d["steps"] = [s.to_dict() for s in self.steps]
        return d

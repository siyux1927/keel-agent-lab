"""循环护栏。

Agent "跑飞"的方式就那么几种, 每种都要有对应的检测器和处置动作:

  预算超支   步数 / token / 成本 / 墙钟时间 —— 硬停
  原地打转   同一动作反复调用 —— 先反思, 再犯就停
  周期振荡   A→B→A→B 的环 —— 直接停, 反思对它基本无效
  进展停滞   动作在变但获取不到新信息 —— 先反思, 再犯就逼它交答案
  下游熔断   依赖的工具全挂了 —— 停, 别再烧 token

关键设计: 检测到问题时**优先"降级"而不是"报错退出"**。
先给模型一次自我纠正的机会(反思), 不行就强制它用现有信息作答,
最后才放弃。用户拿到"基于已知信息的部分答案"远好过拿到一个超时错误。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from keel.config import settings
from keel.loop.state import AgentState, StopReason
from keel.util import cosine, hash_embed


@dataclass
class GuardDecision:
    """护栏判定结果。action 决定引擎下一步怎么走。"""

    action: str = "continue"  # continue | reflect | force_answer | stop
    reason: str = ""
    stop_reason: Optional[StopReason] = None
    detail: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.detail is None:
            self.detail = {}

    @property
    def should_stop(self) -> bool:
        return self.action == "stop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action, "reason": self.reason,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "detail": self.detail,
        }


class LoopPolicy:
    def __init__(self, config: Optional[Any] = None) -> None:
        self.cfg = config or settings.loop
        self.triggered: list[dict[str, Any]] = []  # 记录所有触发历史, 便于复盘
        # 已经为之反思过的动作指纹。重复计数是单调递增的, 不记住这个的话
        # 反思一结束护栏会立刻再次触发, 变成"反思死循环"
        self._reflected: set[str] = set()

    # ==================================================================

    def check(self, state: AgentState) -> GuardDecision:
        for detector in (
            self._check_budget,
            self._check_repeat,
            self._check_cycle,
            self._check_stall,
        ):
            decision = detector(state)
            if decision.action != "continue":
                self.triggered.append({"step": state.step_count, **decision.to_dict()})
                return decision
        return GuardDecision()

    # -- 预算 ------------------------------------------------------------

    def _check_budget(self, state: AgentState) -> GuardDecision:
        cfg = self.cfg
        if state.step_count >= cfg.max_steps:
            return GuardDecision(
                "stop", f"已达最大步数 {cfg.max_steps}", StopReason.MAX_STEPS,
                {"steps": state.step_count},
            )
        if state.total_tokens >= cfg.max_tokens:
            return GuardDecision(
                "stop", f"已达 token 预算 {cfg.max_tokens}", StopReason.MAX_TOKENS,
                {"tokens": state.total_tokens},
            )
        if state.cost_usd >= cfg.max_cost_usd:
            return GuardDecision(
                "stop", f"已达成本上限 ${cfg.max_cost_usd}", StopReason.MAX_COST,
                {"cost_usd": round(state.cost_usd, 6)},
            )
        if state.elapsed_s >= cfg.max_wall_time_s:
            return GuardDecision(
                "stop", f"已超时 {cfg.max_wall_time_s}s", StopReason.TIMEOUT,
                {"elapsed_s": round(state.elapsed_s, 2)},
            )
        # 逼近上限时提前收口, 留出产出答案所需的那一步
        if state.step_count == cfg.max_steps - 1:
            return GuardDecision(
                "force_answer", "只剩最后一步, 用现有信息作答",
                detail={"steps": state.step_count},
            )
        return GuardDecision()

    # -- 原地打转 --------------------------------------------------------

    def _check_repeat(self, state: AgentState) -> GuardDecision:
        fps = state.fingerprints()
        if not fps:
            return GuardDecision()
        counts: dict[str, int] = {}
        for fp in fps:
            counts[fp] = counts.get(fp, 0) + 1
        worst_fp, worst = max(counts.items(), key=lambda kv: kv[1])
        threshold = self.cfg.repeat_action_threshold
        if worst < threshold:
            return GuardDecision()
        if worst_fp in self._reflected:
            # 已经给过一次自我纠正的机会, 还在重复就说明反思无效, 直接停
            if worst >= threshold + 1:
                return GuardDecision(
                    "stop", f"同一动作已重复 {worst} 次, 反思未能纠正",
                    StopReason.LOOP_DETECTED, {"fingerprint": worst_fp, "count": worst},
                )
            return GuardDecision()
        self._reflected.add(worst_fp)
        return GuardDecision(
            "reflect", f"检测到动作重复 {worst} 次, 触发反思重规划",
            detail={"fingerprint": worst_fp, "count": worst},
        )

    # -- 周期振荡 --------------------------------------------------------

    def _check_cycle(self, state: AgentState) -> GuardDecision:
        """检测 A-B-A-B / A-B-C-A-B-C 这类周期。

        跟"重复"分开处理: 周期性振荡通常意味着模型在两个错误假设之间来回横跳,
        反思几乎救不回来, 直接停更划算。
        """
        window = self.cfg.cycle_detect_window
        # 必须显式判 <=0。踩过的坑: 想用 window=0 关掉这个检测器, 结果 fps[-0:]
        # 等价于 fps[0:] —— 取的是**全部**历史而不是空列表, 检测器反而变成了无限窗口。
        if window <= 0:
            return GuardDecision()
        fps = state.fingerprints()[-window:]
        n = len(fps)
        if n < 4:
            return GuardDecision()
        for period in range(2, n // 2 + 1):
            tail = fps[-period * 2:]
            if len(tail) == period * 2 and tail[:period] == tail[period:]:
                return GuardDecision(
                    "stop", f"检测到周期为 {period} 的动作环, 无法推进",
                    StopReason.CYCLE_DETECTED, {"period": period, "pattern": tail[:period]},
                )
        return GuardDecision()

    # -- 进展停滞 --------------------------------------------------------

    def _check_stall(self, state: AgentState) -> GuardDecision:
        if state.no_progress_steps >= self.cfg.stall_threshold + 1:
            return GuardDecision(
                "stop", f"连续 {state.no_progress_steps} 步无实质进展",
                StopReason.NO_PROGRESS, {"no_progress_steps": state.no_progress_steps},
            )
        if state.no_progress_steps >= self.cfg.stall_threshold:
            return GuardDecision(
                "force_answer", f"连续 {state.no_progress_steps} 步无进展, 用现有信息作答",
                detail={"no_progress_steps": state.no_progress_steps},
            )
        if state.consecutive_failures >= self.cfg.reflect_on_consecutive_failures:
            return GuardDecision(
                "reflect", f"连续 {state.consecutive_failures} 步失败, 触发反思",
                detail={"consecutive_failures": state.consecutive_failures},
            )
        return GuardDecision()

    # ==================================================================

    def should_reflect_periodically(self, state: AgentState) -> bool:
        n = self.cfg.reflect_every_n_steps
        return n > 0 and state.step_count > 0 and state.step_count % n == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "max_steps": self.cfg.max_steps, "max_tokens": self.cfg.max_tokens,
                "max_cost_usd": self.cfg.max_cost_usd, "max_wall_time_s": self.cfg.max_wall_time_s,
                "repeat_action_threshold": self.cfg.repeat_action_threshold,
                "stall_threshold": self.cfg.stall_threshold,
            },
            "triggered": self.triggered,
        }


# --------------------------------------------------------------------------


def compute_progress(new_observations: list[str], prior_observations: list[str]) -> float:
    """本步获得了多少**新**信息, 0(全是旧的) ~ 1(全新)。

    做法: 把新观察和历史观察都转成向量, 取最大相似度, 用 1-sim 当新颖度。
    这比"看动作变没变"可靠得多 —— 换个措辞搜同一个东西, 动作不同但信息为零。
    """
    if not new_observations:
        return 0.0
    if not prior_observations:
        return 1.0
    prior_vecs = [hash_embed(o) for o in prior_observations[-12:]]
    novelties = []
    for obs in new_observations:
        vec = hash_embed(obs)
        max_sim = max((cosine(vec, pv) for pv in prior_vecs), default=0.0)
        novelties.append(max(0.0, 1.0 - max_sim))
    return sum(novelties) / len(novelties)

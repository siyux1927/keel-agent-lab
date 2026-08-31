"""反思与重规划(Reflexion 思路的工程化)。

朴素 ReAct 的致命弱点: 它没有"跳出当前轨道"的机制 —— 每一步都只看上一步的观察,
一旦方向错了就会沿着错误方向一直走到预算耗尽。

反思是一次强制的**元认知中断**: 把整段轨迹摊开给模型看, 问它三个问题 ——
卡在哪、哪些动作已被证明无效、接下来换什么打法。产出会以两种形式回写状态:

  plan            新的计划, 注入 scratchpad 顶部
  avoid_actions   动作黑名单, 引擎在执行前拦截, 从机制上杜绝再犯

第二条尤其重要: 只把"别再这么做"写进 prompt, 模型经常照做不误;
在代码里拦下来才是真的拦下来了。
"""

from __future__ import annotations

from typing import Any, Optional

from keel.config import settings
from keel.llm.base import Message
from keel.loop.state import AgentState
from keel.observability.trace import current_tracer
from keel.util import safe_json_loads

REFLECT_SYSTEM = """你是智能体的元认知模块。分析下面这段执行轨迹, 判断它为什么没能推进, 并给出新策略。

严格输出 JSON:
{
  "diagnosis": "一句话说清卡点的根因",
  "should_replan": true/false,
  "avoid_actions": ["已被证明无效的工具名"],
  "new_plan": ["下一步该做什么", "再下一步", "..."]
}

要求:
- diagnosis 要具体, 不要说"需要更多信息"这类废话。
- new_plan 每一步必须是可执行的动作, 且与已尝试过的路径明显不同。
- 如果已有信息其实足够回答, 就把 new_plan 设为 ["直接基于已有证据给出最终答案"]。"""


class Reflector:
    def __init__(self, provider: Any = None, max_replans: Optional[int] = None) -> None:
        self._provider = provider
        self.max_replans = max_replans or settings.loop.max_replans

    @property
    def provider(self) -> Any:
        if self._provider is None:
            from keel.llm import get_provider

            self._provider = get_provider()
        return self._provider

    async def reflect(self, state: AgentState, trigger: str = "") -> Optional[dict[str, Any]]:
        if state.replans >= self.max_replans:
            return {
                "diagnosis": f"已重规划 {state.replans} 次仍未突破, 停止反思避免空转",
                "should_replan": False, "avoid_actions": [], "new_plan": [],
                "exhausted": True,
            }

        tracer = current_tracer()
        with tracer.span("loop.reflect", kind="loop", trigger=trigger) as span:
            trajectory = self._render_trajectory(state)
            resp = await self.provider.chat(
                [
                    Message(role="system", content=REFLECT_SYSTEM),
                    Message(
                        role="user",
                        content=(
                            f"原始目标: {state.goal}\n"
                            f"触发原因: {trigger}\n"
                            f"已执行 {state.step_count} 步, 连续失败 {state.consecutive_failures} 次, "
                            f"无进展 {state.no_progress_steps} 步。\n\n"
                            f"执行轨迹:\n{trajectory}"
                        ),
                    ),
                ],
                hint="reflect",
                max_tokens=600,
            )
            data = safe_json_loads(resp.content)
            if not isinstance(data, dict):
                # 反思本身失败不该让整轮崩掉, 降级成一条保守建议
                data = {
                    "diagnosis": "反思输出解析失败, 采用保守策略",
                    "should_replan": True,
                    "avoid_actions": [],
                    "new_plan": ["基于已获得的信息直接给出结论"],
                }

            self._apply(state, data)
            span.attributes.update({
                "diagnosis": str(data.get("diagnosis", ""))[:160],
                "replans": state.replans,
                "avoid_count": len(state.avoid_actions),
            })
            return data

    # ------------------------------------------------------------------

    def _apply(self, state: AgentState, data: dict[str, Any]) -> None:
        """把反思结论写回状态 —— 这一步才让反思真正产生约束力。"""
        new_plan = [str(p) for p in (data.get("new_plan") or []) if str(p).strip()]
        if data.get("should_replan") and new_plan:
            state.plan = new_plan[:6]
            state.replans += 1

        avoid_tools = {str(t) for t in (data.get("avoid_actions") or [])}
        if avoid_tools:
            # 工具名 → 具体动作指纹: 把该工具已经调过的所有参数组合全部拉黑
            for step in state.steps:
                for action in step.actions:
                    if action.tool in avoid_tools:
                        state.avoid_actions.add(action.fingerprint)

        # 反思本身就是一次"重新出发", 计数器清零, 否则刚反思完就被护栏判停
        state.consecutive_failures = 0
        state.no_progress_steps = 0

    @staticmethod
    def _render_trajectory(state: AgentState, max_steps: int = 8) -> str:
        steps = state.steps[-max_steps:]
        parts = []
        for step in steps:
            for obs in step.observations:
                status = "成功" if obs.ok else f"失败({obs.error[:60]})"
                parts.append(
                    f"步骤{step.index + 1}: 调用 {obs.action.tool}"
                    f"({list(obs.action.args.values())[:1]}) → {status}"
                    f" | 结果摘要: {obs.content[:120]}"
                )
            if not step.observations and step.thought:
                parts.append(f"步骤{step.index + 1}: 仅思考未行动 — {step.thought[:100]}")
        return "\n".join(parts) or "(尚无有效动作记录)"

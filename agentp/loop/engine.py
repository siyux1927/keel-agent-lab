"""ReAct 引擎 —— 把上下文、记忆、工具、护栏拧成一个可控的循环。

一轮循环的完整链路:

    护栏检查 → 记忆召回 → 上下文组装(切片/预算/压缩)
        → LLM 决策 → [动作黑名单拦截] → 并行执行工具
        → 计算信息增量 → 更新状态 → (必要时反思重规划)

对比朴素 ReAct, 这里多了五个东西, 每一个都对应一类线上事故:

  1. 上下文不是"历史全量拼接", 而是按预算重新组装 —— 治超窗
  2. 动作黑名单在**执行前**拦截 —— 治反思了也不改
  3. 多个工具调用并发执行 —— 治串行等待导致的延迟
  4. 信息增量(progress)驱动停滞检测 —— 治"看着在动其实没动"
  5. 被迫停止时仍产出尽力而为的答案 —— 治超时白跑

设计上引擎只负责"怎么循环", 不负责"装什么上下文"和"停不停",
后者分别由 ContextAssembler 和 LoopPolicy 决定 —— 三者可以独立替换和单测。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from agentp.config import settings
from agentp.context.assembler import ContextAssembler, ContextItem, default_system_prompt
from agentp.llm.base import Message
from agentp.loop.policy import GuardDecision, LoopPolicy, compute_progress
from agentp.loop.reflection import Reflector
from agentp.loop.state import (
    Action,
    AgentState,
    AgentStatus,
    Observation,
    Step,
    StopReason,
)
from agentp.observability.events import EventBus, NullBus
from agentp.observability.trace import Tracer, trace_store, use_tracer
from agentp.util import safe_json_loads


@dataclass
class AgentResult:
    answer: str
    state: AgentState
    tracer: Tracer
    context_report: dict[str, Any] = field(default_factory=dict)
    guard_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.state.status == AgentStatus.DONE

    def to_dict(self, with_trace: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "answer": self.answer,
            "success": self.success,
            "state": self.state.to_dict(),
            "context_report": self.context_report,
            "guard_log": self.guard_log,
            "usage": self.tracer.totals(),
        }
        if with_trace:
            d["trace"] = self.tracer.to_dict()
        return d


class ReActEngine:
    def __init__(
        self,
        provider: Any = None,
        memory: Any = None,
        registry: Any = None,
        assembler: Optional[ContextAssembler] = None,
        policy: Optional[LoopPolicy] = None,
        reflector: Optional[Reflector] = None,
        tools: Optional[list[str]] = None,
        name: str = "AgentP",
        system_extra: str = "",
        verify: bool = True,
    ) -> None:
        self._provider = provider
        self._memory = memory
        self._registry = registry
        self.assembler = assembler or ContextAssembler(provider)
        self.policy = policy or LoopPolicy()
        self.reflector = reflector or Reflector(provider)
        self.tools = tools
        self.name = name
        self.system_extra = system_extra
        self.verify = verify
        self.max_verify_rounds = 1

    # -- 惰性依赖 ----------------------------------------------------------

    @property
    def provider(self) -> Any:
        if self._provider is None:
            from agentp.llm import get_provider

            self._provider = get_provider()
        return self._provider

    @property
    def memory(self) -> Any:
        if self._memory is None:
            from agentp.memory import get_memory

            self._memory = get_memory()
        return self._memory

    @property
    def registry(self) -> Any:
        if self._registry is None:
            from agentp.tools import get_registry

            self._registry = get_registry()
        return self._registry

    # ==================================================================
    # 主循环
    # ==================================================================

    async def run(
        self,
        goal: str,
        session_id: str = "default",
        bus: Optional[EventBus] = None,
        tracer: Optional[Tracer] = None,
    ) -> AgentResult:
        bus = bus or NullBus()
        tracer = tracer or Tracer(name=f"react:{goal[:40]}")
        state = AgentState(goal=goal, session_id=session_id)
        last_context_report: dict[str, Any] = {}
        verify_rounds = 0

        with use_tracer(tracer):
            with tracer.span("agent.run", kind="agent", goal=goal, session=session_id) as root:
                await self.memory.observe_turn(session_id, "user", goal)
                await self._seed_plan_from_skills(state, bus)
                bus.emit("agent.start", goal=goal, session_id=session_id,
                         trace_id=tracer.trace_id, plan=state.plan)

                force_answer = False
                while state.status == AgentStatus.RUNNING:
                    decision = self.policy.check(state)
                    if decision.action != "continue":
                        bus.emit("guard", **decision.to_dict())
                    if decision.should_stop:
                        state.status = AgentStatus.STOPPED
                        state.stop_reason = decision.stop_reason
                        state.stop_detail = decision.reason
                        break
                    if decision.action == "reflect":
                        await self._do_reflect(state, decision.reason, bus)
                        continue
                    if decision.action == "force_answer":
                        force_answer = True

                    step_report = await self._step(state, force_answer, bus)
                    last_context_report = step_report or last_context_report
                    force_answer = False

                    # 收敛判定: 有答案了先自检一次, 不合格就带着意见回炉
                    if state.status == AgentStatus.DONE and self.verify and verify_rounds < self.max_verify_rounds:
                        verify_rounds += 1
                        verdict = await self._self_check(state, bus)
                        if verdict and not verdict.get("accepted", True):
                            state.status = AgentStatus.RUNNING
                            state.plan = [f"针对以下问题改进回答: {'; '.join(verdict.get('issues', []))}"]
                            bus.emit("verify.rejected", **verdict)

                await self._finalize(state, bus)
                root.attributes.update({
                    "steps": state.step_count,
                    "status": state.status.value,
                    "stop_reason": state.stop_reason.value if state.stop_reason else None,
                })

        trace_store.put(tracer)
        bus.emit("agent.finish", answer=state.final_answer, status=state.status.value,
                 stop_reason=state.stop_reason.value if state.stop_reason else None,
                 usage=tracer.totals())
        return AgentResult(
            answer=state.final_answer, state=state, tracer=tracer,
            context_report=last_context_report,
            guard_log=self.policy.triggered,
        )

    # ==================================================================
    # 单步
    # ==================================================================

    async def _step(
        self, state: AgentState, force_answer: bool, bus: EventBus
    ) -> Optional[dict[str, Any]]:
        from agentp.observability.trace import current_tracer

        tracer = current_tracer()
        step = state.new_step()
        started = time.perf_counter()

        with tracer.span(f"loop.step[{step.index}]", kind="loop", step=step.index) as span:
            bus.emit("step.start", index=step.index)

            # --- 1. 组装上下文 --------------------------------------------
            assembled = await self._build_context(state, force_answer)
            step.context_tokens = assembled.total_tokens
            bus.emit("context.built", index=step.index, tokens=assembled.total_tokens,
                     zones={k: v.get("granted", 0) for k, v in assembled.zone_report.items()})

            # --- 2. 模型决策 ----------------------------------------------
            tool_schemas = [] if force_answer else self.registry.schemas(allow=self.tools)
            resp = await self.provider.chat(
                assembled.messages, tools=tool_schemas or None, hint="react",
            )
            step.thought = resp.content
            step.prompt_tokens = resp.usage.prompt_tokens
            step.completion_tokens = resp.usage.completion_tokens
            state.prompt_tokens += resp.usage.prompt_tokens
            state.completion_tokens += resp.usage.completion_tokens
            state.cost_usd += resp.usage.cost_usd

            if resp.content.strip():
                bus.emit("thought", index=step.index, text=resp.content[:600])

            # --- 3. 没有动作 = 给出答案 ------------------------------------
            if not resp.tool_calls:
                step.answer = resp.content
                state.final_answer = resp.content
                state.status = AgentStatus.DONE
                state.stop_reason = StopReason.GOAL_REACHED
                step.duration_ms = (time.perf_counter() - started) * 1000
                span.attributes["outcome"] = "answer"
                bus.emit("step.end", index=step.index, outcome="answer")
                return assembled.zone_report

            # --- 4. 执行动作 ----------------------------------------------
            actions = [
                Action(tool=tc.name, args=tc.arguments, call_id=tc.id) for tc in resp.tool_calls
            ]
            step.actions = actions
            observations = await self._execute_actions(actions, state, bus, step.index)
            step.observations = observations

            # --- 5. 更新进展信号 ------------------------------------------
            prior = [o.content for o in state.all_observations()[: -len(observations)] if o.ok]
            new_contents = [o.content for o in observations if o.ok]
            step.progress = compute_progress(new_contents, prior)

            if step.failed:
                state.consecutive_failures += 1
            else:
                state.consecutive_failures = 0
            if step.progress < 0.15:
                state.no_progress_steps += 1
            else:
                state.no_progress_steps = 0

            step.duration_ms = (time.perf_counter() - started) * 1000
            span.attributes.update({
                "outcome": "act", "tools": [a.tool for a in actions],
                "progress": round(step.progress, 3),
                "ok": sum(1 for o in observations if o.ok),
            })
            bus.emit("step.end", index=step.index, outcome="act",
                     progress=round(step.progress, 3),
                     tools=[a.tool for a in actions])

            # 周期性反思: 不等出问题才反思, 定期抬头看看方向对不对
            if self.policy.should_reflect_periodically(state) and state.status == AgentStatus.RUNNING:
                await self._do_reflect(state, "周期性检查", bus)
            return assembled.zone_report

    async def _build_context(self, state: AgentState, force_answer: bool):
        # 检索 query 用"目标 + 最近观察", 而不是只用原始目标 ——
        # 多步任务里当前真正需要的信息往往由上一步的观察决定
        last_obs = state.all_observations()[-1].content[:200] if state.all_observations() else ""
        query = f"{state.goal}\n{last_obs}".strip()
        items: list[ContextItem] = await self.memory.build_context_items(
            query, session_id=state.session_id
        )

        wm = self.memory.working(state.session_id)
        history = [m for m in wm.to_messages() if m.content.strip() != state.goal]

        extra = self.system_extra
        # 已执行动作清单放进系统提示词而不是 scratchpad: 后者会被压缩,
        # 而"我做过什么"一旦丢失, 模型必然重复调用同一个工具
        digest = state.action_digest()
        if digest:
            extra += f"\n\n{digest}"
        if force_answer:
            extra += (
                "\n\n[强制收口] 预算即将耗尽, 本轮不要再调用任何工具, "
                "直接基于已有观察给出最完整的答案, 并明确标注哪些部分尚未验证。"
            )
        if state.avoid_actions:
            extra += f"\n\n[约束] 有 {len(state.avoid_actions)} 个动作已被判定无效并在系统层拦截, 请换思路。"

        return await self.assembler.build(
            task=state.goal,
            system_prompt=default_system_prompt(self.name, extra),
            tools_schema=[] if force_answer else self.registry.schemas(allow=self.tools),
            items=items,
            history=history,
            scratchpad=state.render_scratchpad(),
        )

    async def _execute_actions(
        self, actions: Sequence[Action], state: AgentState, bus: EventBus, step_index: int
    ) -> list[Observation]:
        """并发执行本步的所有动作, 黑名单动作直接拦截不下发。"""

        async def run_one(action: Action) -> Observation:
            if action.fingerprint in state.avoid_actions:
                bus.emit("tool.blocked", index=step_index, tool=action.tool)
                return Observation(
                    action=action, content="", ok=False,
                    error="该动作已被反思判定无效并在系统层拦截, 请改用其他方式。",
                )
            bus.emit("tool.start", index=step_index, tool=action.tool, args=action.args)
            result = await self.registry.call(action.tool, action.args)
            bus.emit("tool.end", index=step_index, tool=action.tool, ok=result.ok,
                     preview=(result.content or result.error)[:300],
                     latency_ms=round(result.latency_ms, 1))
            return Observation(
                action=action, content=result.content, ok=result.ok,
                error=result.error, latency_ms=result.latency_ms,
            )

        if len(actions) == 1:
            return [await run_one(actions[0])]
        return list(await asyncio.gather(*(run_one(a) for a in actions)))

    # ==================================================================
    # 反思 / 自检 / 收尾
    # ==================================================================

    async def _do_reflect(self, state: AgentState, trigger: str, bus: EventBus) -> None:
        data = await self.reflector.reflect(state, trigger=trigger)
        if state.steps:
            state.steps[-1].reflection = data
        bus.emit("reflect", trigger=trigger, diagnosis=(data or {}).get("diagnosis", ""),
                 new_plan=(data or {}).get("new_plan", []), replans=state.replans)
        if data and data.get("exhausted"):
            state.status = AgentStatus.STOPPED
            state.stop_reason = StopReason.MAX_REPLANS
            state.stop_detail = data.get("diagnosis", "")

    async def _self_check(self, state: AgentState, bus: EventBus) -> Optional[dict[str, Any]]:
        """交付前自检。只做一轮 —— 无限自检本身就是一种死循环。"""
        evidence = "\n".join(o.render()[:200] for o in state.all_observations() if o.ok)
        resp = await self.provider.chat(
            [
                Message(
                    role="system",
                    content=(
                        "你是质量审查员。判断候选回答是否已充分回应目标且有证据支撑。"
                        '严格输出 JSON: {"accepted":bool,"score":0-1,"issues":["..."],"suggestion":"..."}'
                    ),
                ),
                Message(
                    role="user",
                    content=f"目标: {state.goal}\n\n可用证据:\n{evidence or '(无)'}\n\n候选回答:\n{state.final_answer}",
                ),
            ],
            hint="critic", max_tokens=400,
        )
        verdict = safe_json_loads(resp.content)
        return verdict if isinstance(verdict, dict) else None

    async def _seed_plan_from_skills(self, state: AgentState, bus: EventBus) -> None:
        """开局先查程序性记忆: 这类任务以前是怎么做成的。"""
        skills = await self.memory.recall_skills(state.goal, top_k=1)
        if not skills:
            return
        skill = skills[0].record.metadata.get("skill", {})
        steps = skill.get("steps") or []
        if steps:
            state.plan = [f"参考历史成功路径: {s}" for s in steps[:5]]
            bus.emit("skill.recalled", pattern=skill.get("goal_pattern", ""),
                     win_rate=skill.get("win_rate", 0), steps=steps[:5])

    async def _finalize(self, state: AgentState, bus: EventBus) -> None:
        # 被护栏中止但还没有答案: 用已有观察做一次尽力而为的作答。
        # 返回半个答案 + 明确的未完成说明, 永远好过返回一个错误码。
        if not state.final_answer and state.all_observations():
            resp = await self.provider.chat(
                [
                    Message(role="system", content=(
                        "任务因资源限制提前结束。基于已获得的观察给出尽可能有用的部分答案, "
                        "并明确说明哪些部分未能完成、原因是什么。"
                    )),
                    Message(role="user", content=(
                        f"目标: {state.goal}\n中止原因: {state.stop_detail}\n\n已有观察:\n"
                        + "\n".join(o.render()[:300] for o in state.all_observations())
                    )),
                ],
                hint="final", max_tokens=800,
            )
            state.final_answer = resp.content
            bus.emit("answer.degraded", reason=state.stop_detail)
        elif not state.final_answer:
            state.final_answer = f"未能完成任务: {state.stop_detail or '没有获得任何有效信息'}"

        await self.memory.observe_turn(state.session_id, "assistant", state.final_answer)

        # 记忆写回。失败不能影响主流程 —— 记忆是增强项而不是关键路径
        try:
            await self.memory.extract_facts(
                f"用户目标: {state.goal}\n结论: {state.final_answer[:800]}",
                session_id=state.session_id,
            )
            tool_path = state.successful_tools()
            if tool_path:
                await self.memory.record_trajectory(
                    goal=state.goal, steps=tool_path,
                    success=(state.status == AgentStatus.DONE),
                    session_id=state.session_id,
                )
            insight = await self.memory.maybe_reflect(state.session_id)
            if insight:
                bus.emit("memory.reflected", **insight)
        except Exception as exc:  # noqa: BLE001
            bus.emit("memory.error", error=str(exc))

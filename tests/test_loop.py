"""Loop Engineering: 护栏、死循环、熔断、反思、端到端收敛。"""

from __future__ import annotations

import asyncio

import pytest

from agentp.config import settings
from agentp.loop import (
    Action,
    AgentState,
    AgentStatus,
    LoopPolicy,
    Observation,
    ReActEngine,
    StopReason,
    compute_progress,
)
from agentp.tools import CircuitBreaker, get_registry


def _state_with(actions: list[tuple[str, dict]], ok: bool = True) -> AgentState:
    state = AgentState(goal="测试目标")
    for tool, args in actions:
        step = state.new_step()
        action = Action(tool=tool, args=args)
        step.actions = [action]
        step.observations = [Observation(action=action, content="结果", ok=ok)]
    return state


# ==========================================================================
# 动作指纹
# ==========================================================================


def test_fingerprint_is_order_insensitive():
    """参数顺序不同但语义相同的动作必须同指纹, 否则死循环检测直接失效。"""
    a = Action("search", {"q": "x", "k": 3})
    b = Action("search", {"k": 3, "q": "x"})
    assert a.fingerprint == b.fingerprint
    assert Action("search", {"q": "y"}).fingerprint != a.fingerprint


# ==========================================================================
# 护栏
# ==========================================================================


def test_repeat_triggers_reflect_then_stop():
    """第一次重复给自我纠正的机会, 再犯才停 —— 降级而非直接失败。"""
    policy = LoopPolicy()
    state = _state_with([("now", {"tz": "UTC"})] * settings.loop.repeat_action_threshold)
    first = policy.check(state)
    assert first.action == "reflect"

    # 反思之后没改, 又调了一次
    step = state.new_step()
    action = Action("now", {"tz": "UTC"})
    step.actions = [action]
    step.observations = [Observation(action=action, content="x")]
    second = policy.check(state)
    assert second.action == "stop"
    assert second.stop_reason == StopReason.LOOP_DETECTED


def test_repeat_reflect_fires_only_once_per_fingerprint():
    """重复计数单调递增, 不做去重的话反思会被无限触发。"""
    policy = LoopPolicy()
    state = _state_with([("now", {"tz": "UTC"})] * settings.loop.repeat_action_threshold)
    assert policy.check(state).action == "reflect"
    assert policy.check(state).action == "continue"


def test_cycle_detection_catches_alternating_actions():
    policy = LoopPolicy()
    state = _state_with([("a", {"i": 1}), ("b", {"i": 2}), ("a", {"i": 1}), ("b", {"i": 2})])
    decision = policy.check(state)
    assert decision.action == "stop"
    assert decision.stop_reason == StopReason.CYCLE_DETECTED
    assert decision.detail["period"] == 2


def test_cycle_window_zero_disables_the_detector():
    """cycle_detect_window=0 必须是"关掉这个检测器", 而不是"窗口无限大"。

    回归用例。原实现写的是 fingerprints()[-window:], 而 Python 里 lst[-0:] 等价于
    lst[0:] —— 取的是全部历史。想关掉检测器的人会得到一个反而更激进的检测器,
    做消融实验时这种"看着关了其实开着"的开关会直接让结论反过来。
    """
    from dataclasses import replace

    from agentp.config import LoopConfig

    actions = [("a", {"i": 1}), ("b", {"i": 2}), ("a", {"i": 1}), ("b", {"i": 2})]
    assert LoopPolicy().check(_state_with(actions)).stop_reason == StopReason.CYCLE_DETECTED

    off = LoopPolicy(replace(LoopConfig(), cycle_detect_window=0,
                             repeat_action_threshold=10 ** 9, stall_threshold=10 ** 9,
                             reflect_on_consecutive_failures=10 ** 9))
    assert off.check(_state_with(actions)).action == "continue"


def test_budget_guards_stop_on_each_limit():
    cases = [
        ("step_count_stub", StopReason.MAX_STEPS),
        ("tokens", StopReason.MAX_TOKENS),
        ("cost", StopReason.MAX_COST),
    ]
    for kind, expected in cases:
        policy = LoopPolicy()
        state = AgentState(goal="g")
        if kind == "step_count_stub":
            for _ in range(settings.loop.max_steps):
                state.new_step()
        elif kind == "tokens":
            state.new_step()
            state.prompt_tokens = settings.loop.max_tokens + 1
        else:
            state.new_step()
            state.cost_usd = settings.loop.max_cost_usd + 1
        assert policy.check(state).stop_reason == expected


def test_penultimate_step_forces_answer():
    """留最后一步产出答案, 而不是用尽步数后什么都没有。"""
    policy = LoopPolicy()
    state = AgentState(goal="g")
    for _ in range(settings.loop.max_steps - 1):
        state.new_step()
    assert policy.check(state).action == "force_answer"


def test_stall_escalates_from_force_answer_to_stop():
    policy = LoopPolicy()
    state = AgentState(goal="g")
    state.new_step()
    state.no_progress_steps = settings.loop.stall_threshold
    assert policy.check(state).action == "force_answer"
    state.no_progress_steps += 1
    assert policy.check(state).stop_reason == StopReason.NO_PROGRESS


def test_consecutive_failures_trigger_reflection():
    policy = LoopPolicy()
    state = _state_with([("flaky", {})] * 2, ok=False)
    state.consecutive_failures = settings.loop.reflect_on_consecutive_failures
    assert policy.check(state).action == "reflect"


# ==========================================================================
# 进展度量
# ==========================================================================


def test_progress_is_zero_for_repeated_observation():
    obs = "发布窗口为每周二下午 14:00 到 16:00"
    assert compute_progress([obs], [obs]) < 0.05


def test_progress_is_high_for_novel_observation():
    assert compute_progress(["完全不同主题的全新观察内容"], ["旧的无关记录"]) > 0.5


def test_progress_without_history_is_full():
    assert compute_progress(["任何观察"], []) == 1.0


# ==========================================================================
# 熔断
# ==========================================================================


def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker(failure_threshold=3, cooldown_s=60)
    for _ in range(3):
        assert breaker.allow()[0]
        breaker.record(False)
    allowed, reason = breaker.allow()
    assert not allowed
    assert "熔断" in reason


def test_circuit_breaker_half_opens_after_cooldown():
    breaker = CircuitBreaker(failure_threshold=2, cooldown_s=0.0)
    breaker.record(False)
    breaker.record(False)
    assert breaker.state == "open"
    assert breaker.allow()[0]          # 冷却结束, 放一个探针
    assert breaker.state == "half_open"
    breaker.record(True)
    assert breaker.state == "closed"   # 探针成功则完全恢复


def test_success_resets_failure_counter():
    breaker = CircuitBreaker(failure_threshold=3)
    breaker.record(False)
    breaker.record(True)
    assert breaker.failures == 0


async def test_failing_tool_eventually_opens_breaker():
    registry = get_registry()
    tool = registry.get("flaky")
    for _ in range(6):
        await registry.call("flaky", {"fail": True})
    assert tool.breaker.state == "open"
    result = await registry.call("flaky", {"fail": True})
    assert not result.ok
    assert result.meta.get("circuit_open")


# ==========================================================================
# 端到端
# ==========================================================================


async def test_engine_converges_on_multi_tool_task(memory):
    engine = ReActEngine(memory=memory, verify=False)
    result = await engine.run("帮我计算 (12+8)*5 的结果", session_id="t1")
    assert result.success
    assert result.state.stop_reason == StopReason.GOAL_REACHED
    assert "100" in "".join(o.content for o in result.state.all_observations())
    # 不该为了显得努力而多调工具
    assert result.state.step_count <= 5


async def test_engine_stops_on_forced_loop(memory):
    """Mock 里有触发词能强制模型反复调同一个工具, 用来验证护栏真的生效。"""
    engine = ReActEngine(memory=memory, verify=False)
    result = await engine.run("死循环演示 请反复确认当前时间", session_id="t2")
    assert result.state.status == AgentStatus.STOPPED
    assert result.state.stop_reason in (StopReason.LOOP_DETECTED, StopReason.CYCLE_DETECTED)
    assert result.state.step_count < settings.loop.max_steps
    # 被中止也必须给出尽力而为的答案, 而不是抛错
    assert result.answer.strip()


async def test_engine_never_exceeds_step_budget(memory):
    original = settings.loop.max_steps
    settings.loop.max_steps = 4
    try:
        engine = ReActEngine(memory=memory, verify=False)
        result = await engine.run("死循环演示 一直查时间", session_id="t3")
        assert result.state.step_count <= 4
    finally:
        settings.loop.max_steps = original


async def test_blocked_action_is_intercepted_before_execution(memory):
    """反思拉黑的动作必须在代码层拦截, 光写进 prompt 模型经常照做不误。"""
    engine = ReActEngine(memory=memory, verify=False)
    state = AgentState(goal="g", session_id="t4")
    action = Action("now", {"tz": "Asia/Shanghai"})
    state.avoid_actions.add(action.fingerprint)

    from agentp.observability.events import NullBus

    observations = await engine._execute_actions([action], state, NullBus(), 0)
    assert not observations[0].ok
    assert "拦截" in observations[0].error


async def test_parallel_actions_execute_concurrently(memory):
    engine = ReActEngine(memory=memory, verify=False)
    state = AgentState(goal="g")
    actions = [Action("web_search", {"query": f"q{i}"}) for i in range(4)]

    from agentp.observability.events import NullBus

    started = asyncio.get_event_loop().time()
    observations = await engine._execute_actions(actions, state, NullBus(), 0)
    elapsed = asyncio.get_event_loop().time() - started

    assert all(o.ok for o in observations)
    # web_search 每次 sleep 50ms, 串行要 200ms 以上
    assert elapsed < 0.16


async def test_engine_writes_back_memory(memory):
    engine = ReActEngine(memory=memory, verify=False)
    await engine.run("计算 7*6 的结果", session_id="t5")
    from agentp.memory import MemoryLayer

    assert memory.store.all(MemoryLayer.PROCEDURAL)  # 成功轨迹固化成技能
    assert memory.working("t5").turns                # 对话进了工作记忆

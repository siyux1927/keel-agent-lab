"""工具抽象 + 注册表 + 熔断器。

工具层是 Agent 唯一能影响外部世界的地方, 所以三件事必须做在这里而不是散落各处:
超时、重试、熔断。少一个, 一个抽风的下游就能把整个 Agent 拖死。
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agentp.config import settings
from agentp.observability.trace import current_tracer
from agentp.util import now_ts


@dataclass
class ToolResult:
    ok: bool
    content: str
    error: str = ""
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "content": self.content, "error": self.error,
            "latency_ms": round(self.latency_ms, 2), "meta": self.meta,
        }


class CircuitBreaker:
    """三态熔断器: closed → open → half_open。

    连续失败到阈值就 open, 直接拒绝调用不再浪费时间和 token;
    冷却期过后放一个请求试探(half_open), 成功则恢复, 失败则重新 open。

    对 Agent 特别重要: 模型看到工具报错常常会"再试一次", 没有熔断的话
    一个挂掉的下游能吃光整轮的步数预算。
    """

    def __init__(self, failure_threshold: Optional[int] = None, cooldown_s: Optional[float] = None) -> None:
        cfg = settings.loop
        # 必须用 is None 判断: cooldown_s=0 是合法配置(立即半开), 用 `or` 会被当成未设置
        self.failure_threshold = (
            cfg.breaker_failure_threshold if failure_threshold is None else failure_threshold
        )
        self.cooldown_s = cfg.breaker_cooldown_s if cooldown_s is None else cooldown_s
        self.state = "closed"
        self.failures = 0
        self.opened_at = 0.0
        self.total_calls = 0
        self.total_failures = 0

    def allow(self) -> tuple[bool, str]:
        if self.state == "closed":
            return True, ""
        if self.state == "open":
            if now_ts() - self.opened_at >= self.cooldown_s:
                self.state = "half_open"
                return True, ""
            wait = self.cooldown_s - (now_ts() - self.opened_at)
            return False, f"熔断器开启中, 还需冷却 {wait:.1f}s (连续失败 {self.failures} 次)"
        return True, ""  # half_open: 放行一个探针

    def record(self, ok: bool) -> None:
        self.total_calls += 1
        if ok:
            self.failures = 0
            self.state = "closed"
            return
        self.total_failures += 1
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.failure_threshold:
            self.state = "open"
            self.opened_at = now_ts()

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state, "failures": self.failures,
            "total_calls": self.total_calls, "total_failures": self.total_failures,
        }


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]           # JSON Schema
    func: Callable[..., Any]
    timeout_s: Optional[float] = None
    max_retries: Optional[int] = None
    dangerous: bool = False              # 有副作用的工具, 可要求人工确认
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    def schema(self) -> dict[str, Any]:
        """OpenAI function-calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    # ------------------------------------------------------------------

    def validate(self, args: dict[str, Any]) -> Optional[str]:
        """轻量 JSON Schema 校验。

        把参数错误变成一条可读的 observation 回灌给模型, 比抛异常有用得多 ——
        模型看到"缺少参数 expression"下一轮基本能自己改对。
        """
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])
        missing = [k for k in required if k not in args or args[k] in (None, "")]
        if missing:
            return f"缺少必填参数: {', '.join(missing)}。参数规格: {list(props)}"
        type_map = {"string": str, "number": (int, float), "integer": int,
                    "boolean": bool, "array": list, "object": dict}
        for key, value in args.items():
            spec = props.get(key)
            if not spec:
                continue
            expected = type_map.get(spec.get("type", ""))
            if expected and not isinstance(value, expected):
                if expected in ((int, float), int) and isinstance(value, str):
                    try:  # LLM 经常把数字包成字符串, 能转就别报错
                        args[key] = float(value) if expected is not int else int(value)
                        continue
                    except ValueError:
                        pass
                return f"参数 {key} 类型错误: 期望 {spec.get('type')}, 实际 {type(value).__name__}"
            enum = spec.get("enum")
            if enum and value not in enum:
                return f"参数 {key} 取值非法: 只能是 {enum}"
        return None

    async def run(self, args: dict[str, Any]) -> ToolResult:
        tracer = current_tracer()
        with tracer.span(f"tool.{self.name}", kind="tool", args=args) as span:
            allowed, reason = self.breaker.allow()
            if not allowed:
                span.attributes["circuit"] = "rejected"
                return ToolResult(False, "", error=reason, meta={"circuit_open": True})

            err = self.validate(args)
            if err:
                self.breaker.record(True)  # 参数错是模型的问题, 不该算下游故障
                span.attributes["validation_error"] = err
                return ToolResult(False, "", error=err, meta={"validation": True})

            timeout = self.timeout_s or settings.loop.tool_timeout_s
            retries = self.max_retries if self.max_retries is not None else settings.loop.tool_max_retries

            last_error = ""
            started = time.perf_counter()
            for attempt in range(retries + 1):
                try:
                    result = self.func(**args)
                    if inspect.isawaitable(result):
                        result = await asyncio.wait_for(result, timeout=timeout)
                    content = result if isinstance(result, str) else _to_text(result)
                    latency = (time.perf_counter() - started) * 1000
                    self.breaker.record(True)
                    span.attributes.update({"attempts": attempt + 1, "ok": True})
                    return ToolResult(True, content, latency_ms=latency,
                                      meta={"attempts": attempt + 1})
                except asyncio.TimeoutError:
                    last_error = f"工具超时(>{timeout}s)"
                except Exception as exc:  # noqa: BLE001 - 工具异常一律转成可读结果
                    last_error = f"{type(exc).__name__}: {exc}"
                if attempt < retries:
                    await asyncio.sleep(0.3 * (2 ** attempt))

            latency = (time.perf_counter() - started) * 1000
            self.breaker.record(False)
            span.attributes.update({"attempts": retries + 1, "ok": False})
            span.status = "error"
            return ToolResult(False, "", error=last_error, latency_ms=latency,
                              meta={"attempts": retries + 1, "breaker": self.breaker.state})


def _to_text(value: Any) -> str:
    import json

    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(value)


# --------------------------------------------------------------------------


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., Any],
        **kwargs: Any,
    ) -> Tool:
        tool = Tool(name=name, description=description, parameters=parameters, func=func, **kwargs)
        self._tools[name] = tool
        return tool

    def tool(self, name: str, description: str, parameters: dict[str, Any], **kwargs: Any):
        """装饰器写法。"""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(name, description, parameters, func, **kwargs)
            return func

        return decorator

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self, allow: Optional[list[str]] = None, deny: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """按白/黑名单产出工具定义。

        按需暴露工具是有必要的: 工具定义直接占上下文预算, 而且候选越多模型选错的概率越高。
        """
        out = []
        for name, tool in self._tools.items():
            if allow is not None and name not in allow:
                continue
            if deny and name in deny:
                continue
            out.append(tool.schema())
        return out

    async def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                False, "",
                error=f"工具 {name} 不存在。可用工具: {self.names()}",
                meta={"unknown_tool": True},
            )
        return await tool.run(args)

    def health(self) -> dict[str, Any]:
        return {name: tool.breaker.to_dict() for name, tool in self._tools.items()}

    def reset_breakers(self) -> None:
        for tool in self._tools.values():
            tool.breaker = CircuitBreaker()

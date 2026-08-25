"""内置工具集。

一个刻意的设计取向: 宁可让工具"能力弱但边界清晰", 也不给 Agent 一把没有保险的枪。
calculator 用 AST 白名单而不是 eval, python_exec 禁掉 import 和属性访问,
read_file 锁死在 data 目录内 —— 这些都是被 prompt injection 打过之后的标准做法。
"""

from __future__ import annotations

import ast
import asyncio
import io
import math
import operator
import random
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agentp.config import settings
from agentp.tools.base import ToolRegistry
from agentp.util import BM25

registry = ToolRegistry()

# ==========================================================================
# calculator —— AST 白名单求值
# ==========================================================================

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "floor": math.floor, "ceil": math.ceil, "pi": math.pi, "e": math.e,
}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量类型: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if type(node.op) is ast.Pow and (abs(right) > 128 or abs(left) > 1e6):
            raise ValueError("幂运算规模过大, 已拒绝")  # 防 2**999999999 这种算力炸弹
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError("不支持的一元运算符")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _FUNCS and not callable(_FUNCS[node.id]):
            return _FUNCS[node.id]
        raise ValueError(f"未知标识符: {node.id}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("只允许调用白名单内的数学函数")
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(e) for e in node.elts]
    raise ValueError(f"表达式中包含不允许的语法: {type(node).__name__}")


@registry.tool(
    name="calculator",
    description="计算数学表达式。支持 + - * / // % ** 和 sqrt/log/sin/round 等函数。用于任何需要精确计算的场景。",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "数学表达式, 如 (3+5)*2/4"}},
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    cleaned = expression.replace("^", "**").replace("×", "*").replace("÷", "/").replace(",", "")
    tree = ast.parse(cleaned.strip(), mode="eval")
    value = _eval_node(tree)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{expression} = {value}"


# ==========================================================================
# now
# ==========================================================================

_TZ = {"Asia/Shanghai": 8, "UTC": 0, "America/New_York": -5, "Europe/London": 0}


@registry.tool(
    name="now",
    description="获取当前日期和时间。任何涉及'今天''现在''最近'的问题都应先调用它。",
    parameters={
        "type": "object",
        "properties": {"tz": {"type": "string", "description": "时区", "enum": list(_TZ)}},
        "required": [],
    },
)
def now_tool(tz: str = "Asia/Shanghai") -> str:
    offset = _TZ.get(tz, 8)
    dt = datetime.now(timezone(timedelta(hours=offset)))
    weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][dt.weekday()]
    return f"当前时间 {dt.strftime('%Y-%m-%d %H:%M:%S')} ({weekday}, {tz})"


# ==========================================================================
# kb_search —— 打到语义记忆里
# ==========================================================================


@registry.tool(
    name="kb_search",
    description="检索内部知识库与长期记忆。回答涉及已入库文档、历史结论、用户偏好时使用。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索问题"},
            "top_k": {"type": "integer", "description": "返回条数, 默认 4"},
        },
        "required": ["query"],
    },
)
async def kb_search(query: str, top_k: int = 4) -> str:
    from agentp.memory import MemoryLayer, get_memory

    memory = get_memory()
    results = await memory.retrieve(
        query, top_k=int(top_k), layers=[MemoryLayer.SEMANTIC, MemoryLayer.EPISODIC]
    )
    if not results:
        return "知识库中没有找到相关内容。"
    lines = [f"检索到 {len(results)} 条相关记忆:"]
    for i, res in enumerate(results, 1):
        tag = " > ".join(res.record.tags[:2]) or res.record.source or res.record.layer.value
        lines.append(f"{i}. [{tag}|score={res.score:.3f}] {res.record.content[:300]}")
    return "\n".join(lines)


# ==========================================================================
# web_search —— 离线语料模拟
# ==========================================================================

_WEB_CORPUS: list[dict[str, str]] = [
    {"title": "ReAct: Synergizing Reasoning and Acting in Language Models",
     "snippet": "ReAct 让模型交错生成推理轨迹(Thought)与动作(Action), 推理帮助规划和异常处理, "
                "动作从外部获取信息以缓解幻觉。是当前绝大多数 Agent 循环的基础范式。"},
    {"title": "Lost in the Middle: How Language Models Use Long Contexts",
     "snippet": "研究发现模型对长上下文首尾信息的利用率明显高于中间部分, 呈 U 型曲线。"
                "因此上下文排布应把关键信息放在两端, 而不是简单按时间顺序堆叠。"},
    {"title": "Generative Agents: Interactive Simulacra of Human Behavior",
     "snippet": "提出记忆流(Memory Stream) + 检索(相关性/新近性/重要性三因子加权) + 反思(Reflection) "
                "的记忆架构, 让智能体能从零散观察中提炼高层洞察并影响后续行为。"},
    {"title": "Reflexion: Language Agents with Verbal Reinforcement Learning",
     "snippet": "用语言化的自我反思代替梯度更新: 失败后让模型生成对失败原因的文字总结, "
                "存入记忆并在下次尝试时作为上下文注入, 显著提升多轮任务成功率。"},
    {"title": "RAG 分块策略实践综述",
     "snippet": "定长切分实现简单但破坏语义; 递归切分按分隔符优先级下切是通用默认值; "
                "语义切分用句向量距离找话题突变点; small-to-big 用小块检索、大块喂模型, "
                "兼顾检索精度与上下文完整性。"},
    {"title": "多智能体编排: Supervisor 与 DAG 模式对比",
     "snippet": "Supervisor 模式由一个主控 Agent 动态派发任务, 灵活但难以并行且成本高; "
                "DAG 模式先规划出依赖图再按层并发执行, 可控性和吞吐更好, 适合任务结构较明确的场景。"},
    {"title": "Agent 的成本与失败模式",
     "snippet": "生产环境 Agent 最常见的三类故障: 死循环重复调用同一工具、上下文超窗导致截断、"
                "下游工具抖动引发连锁重试。对应的工程手段是循环检测、预算调度和熔断降级。"},
    {"title": "Context Engineering 正在取代 Prompt Engineering",
     "snippet": "随着上下文窗口变长, 瓶颈从'怎么写提示词'转移到'该放什么进上下文'。"
                "核心工作变成检索、排序、压缩、预算分配, 以及为每条注入内容保留可追溯的来源。"},
]

_web_index = BM25()
for _i, _doc in enumerate(_WEB_CORPUS):
    _web_index.add(str(_i), _doc["title"] + " " + _doc["snippet"])


@registry.tool(
    name="web_search",
    description="搜索外部公开资料。用于知识库覆盖不到的通用知识、论文、技术方案。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "top_k": {"type": "integer", "description": "返回条数, 默认 3"},
        },
        "required": ["query"],
    },
)
async def web_search(query: str, top_k: int = 3) -> str:
    await asyncio.sleep(0.05)  # 模拟网络往返, 让并发编排的收益在 trace 上看得见
    hits = _web_index.search(query, top_k=int(top_k))
    if not hits:
        hits = [(str(i), 0.0) for i in range(min(int(top_k), len(_WEB_CORPUS)))]
    lines = [f"「{query}」的搜索结果:"]
    for rank, (doc_id, score) in enumerate(hits, 1):
        doc = _WEB_CORPUS[int(doc_id)]
        lines.append(f"{rank}. {doc['title']} (相关度 {score:.2f})\n   {doc['snippet']}")
    return "\n".join(lines)


# ==========================================================================
# python_exec —— 受限沙箱
# ==========================================================================

_FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.Attribute, ast.Global,
                    ast.Nonlocal, ast.Lambda, ast.ClassDef, ast.With, ast.AsyncWith)
_FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "input", "__import__",
                    "globals", "locals", "vars", "getattr", "setattr", "delattr", "breakpoint"}
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int, "len": len,
    "list": list, "map": map, "max": max, "min": min, "print": print, "range": range,
    "reversed": reversed, "round": round, "set": set, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "zip": zip,
}


def _audit(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise ValueError(f"沙箱禁止的语法: {type(node).__name__}(禁止 import / 属性访问 / 定义类)")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"沙箱禁止调用: {node.id}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise ValueError("沙箱禁止字符串中出现 dunder")
    # 循环上限: 静态扫一眼有没有 while True, 挡住最常见的一类无限循环
    for node in ast.walk(tree):
        if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value:
            raise ValueError("沙箱禁止 while True 无限循环")


def _run_sandbox(code: str) -> str:
    _audit(code)
    env: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, "math": None}
    env.update({k: v for k, v in _FUNCS.items() if callable(v)})
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exec(compile(code, "<sandbox>", "exec"), env)  # noqa: S102 - 已通过 AST 白名单审计
    output = buffer.getvalue().strip()
    result = env.get("result")
    if result is not None:
        output = (output + "\n" if output else "") + f"result = {result}"
    return output or "(代码执行完成, 无输出。把想返回的值赋给变量 result 或用 print 输出)"


@registry.tool(
    name="python_exec",
    description="在受限沙箱中执行 Python 代码做数据处理。禁止 import、文件与网络。把结果 print 出来或赋值给 result。",
    parameters={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python 代码片段"}},
        "required": ["code"],
    },
    timeout_s=5.0,
    max_retries=0,
    dangerous=True,
)
async def python_exec(code: str) -> str:
    # 放到线程里跑, 好让上层的 wait_for 超时能生效(注意: 超时只是让调用方返回,
    # 无法真正杀死线程 —— 生产环境这里应该换成子进程 + 资源限制)
    return await asyncio.to_thread(_run_sandbox, code)


# ==========================================================================
# read_file —— 沙箱目录内
# ==========================================================================


@registry.tool(
    name="read_file",
    description="读取 data 目录下的文本文件。",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "相对 data 目录的文件路径"}},
        "required": ["path"],
    },
)
def read_file(path: str) -> str:
    root = settings.data_dir.resolve()
    target = (root / path).resolve()
    # 目录穿越防护: 解析成绝对路径之后必须仍在 root 之内
    if not str(target).startswith(str(root)):
        raise ValueError("拒绝访问 data 目录之外的路径")
    if not target.exists():
        available = [p.name for p in root.glob("*") if p.is_file()][:20]
        raise FileNotFoundError(f"文件不存在: {path}。data 目录下现有: {available}")
    text = target.read_text(encoding="utf-8", errors="replace")
    return text[:8000] + ("\n...[文件过长已截断]" if len(text) > 8000 else "")


# ==========================================================================
# flaky —— 只为演示熔断
# ==========================================================================


@registry.tool(
    name="flaky",
    description="[演示专用] 一个不稳定的下游服务, 用于观察重试与熔断行为。",
    parameters={
        "type": "object",
        "properties": {"fail": {"type": "boolean", "description": "true 则必定失败"}},
        "required": [],
    },
    max_retries=1,
)
async def flaky(fail: bool = False) -> str:
    await asyncio.sleep(0.02)
    if fail or random.random() < 0.7:
        raise ConnectionError("下游服务 503 Service Unavailable")
    return "下游服务返回正常"


def get_registry() -> ToolRegistry:
    return registry


DEFAULT_TOOLS = ["calculator", "now", "kb_search", "web_search", "python_exec"]

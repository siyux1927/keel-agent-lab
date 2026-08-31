"""HTTP 服务层。

除了常规的对话接口, 这里刻意暴露了两个"实验室"接口:

  /api/lab/chunk    同一段文本, 五种切片策略的结果与质量指标并排对比
  /api/lab/context  给定任务, 看清楚上下文预算怎么分、什么被压缩了、什么被丢了

它们不是给终端用户用的, 是给开发者调试用的 —— Agent 系统最难的部分恰恰在于
"模型到底看到了什么", 把这一层做成可交互的界面, 调试效率完全是两个量级。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from keel.config import settings
from keel.context.assembler import ContextAssembler, default_system_prompt
from keel.context.chunker import CHUNKERS, chunk_stats, get_chunker
from keel.llm import get_provider
from keel.loop import ReActEngine
from keel.memory import MemoryLayer, get_memory
from keel.observability.events import EventBus
from keel.observability.trace import Tracer, trace_store
from keel.orchestrator import Orchestrator
from keel.tools import get_registry

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Keel", version="0.1.0",
              description="Agent 运行时: 上下文工程 / 记忆系统 / Loop Engineering / 多 Agent 编排")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ==========================================================================
# 公网模式的自我保护
# ==========================================================================

# 会真正烧 CPU 的端点: 每次请求要跑完整的 ReAct 或编排, 或者要吃下任意大的文本。
# 只读端点(健康检查、trace 查询、记忆浏览)不限流, 让界面能正常刷新。
_EXPENSIVE = ("/api/chat", "/api/memory/ingest", "/api/memory/reflect", "/api/lab/")


class _RateLimiter:
    """按客户端 IP 的滑动窗口限流。

    刻意做成进程内的: 多副本部署时每个副本各算各的, 精度不够, 但演示站够用了。
    要精确就得引入 Redis —— 为一个零外部依赖的 demo 挂一个状态存储不划算,
    这个取舍写在这里, 免得被当成实现疏漏。
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, limit: int, window_s: float = 60.0) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > window_s:
            hits.popleft()
        if len(hits) >= limit:
            return False
        hits.append(now)
        return True


_limiter = _RateLimiter()


@app.middleware("http")
async def _public_guard(request: Request, call_next):
    cfg = settings.server
    if cfg.public_mode and any(request.url.path.startswith(p) for p in _EXPENSIVE):
        client = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        client = client or (request.client.host if request.client else "unknown")
        if not _limiter.allow(client, cfg.rate_limit_per_min):
            return JSONResponse(
                status_code=429,
                content={"detail": f"请求过于频繁，公开演示站限制每分钟 {cfg.rate_limit_per_min} 次。"
                                   f"本地部署不受此限制。"},
            )
    return await call_next(request)


def _check_size(text: str, limit_attr: str, what: str) -> None:
    """公网模式下的输入长度上限。本地开发不设限 —— 实验室就是用来喂大文档的。"""
    cfg = settings.server
    if not cfg.public_mode:
        return
    limit = getattr(cfg, limit_attr)
    if len(text) > limit:
        raise HTTPException(413, f"{what}超长（{len(text)} 字符），公开演示站上限 {limit}。")


# ==========================================================================
# 请求模型
# ==========================================================================


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    mode: str = Field("react", description="react | orchestrate")
    tools: Optional[list[str]] = None


class IngestRequest(BaseModel):
    text: str
    source: str = "upload"
    strategy: str = "structure"


class ChunkLabRequest(BaseModel):
    text: str
    strategies: Optional[list[str]] = None
    chunk_tokens: int = 320
    overlap: int = 48


class ContextLabRequest(BaseModel):
    task: str
    session_id: str = "default"
    context_window: Optional[int] = None


class RememberRequest(BaseModel):
    content: str
    layer: str = "semantic"
    session_id: str = "default"
    importance: Optional[float] = None


# ==========================================================================
# 基础
# ==========================================================================


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    provider = get_provider()
    return {
        "status": "ok",
        "provider": provider.name,
        "model": provider.model,
        "offline_mode": provider.name == "mock",
        "tools": get_registry().names(),
        "memory": get_memory().stats(),
    }


@app.get("/api/config")
async def config() -> dict[str, Any]:
    return settings.to_dict()


@app.get("/api/tools")
async def tools() -> dict[str, Any]:
    registry = get_registry()
    return {
        "schemas": registry.schemas(),
        "health": registry.health(),  # 熔断器状态
    }


# ==========================================================================
# 对话
# ==========================================================================


def _build_runner(req: ChatRequest):
    if req.mode == "orchestrate":
        return Orchestrator(tools=req.tools)
    return ReActEngine(tools=req.tools)


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict[str, Any]:
    if not req.message.strip():
        raise HTTPException(400, "message 不能为空")
    _check_size(req.message, "max_input_chars", "提问")
    runner = _build_runner(req)
    result = await runner.run(req.message, session_id=req.session_id)
    return result.to_dict(with_trace=True)


@app.get("/api/chat/stream")
async def chat_stream(
    message: str, session_id: str = "default", mode: str = "react",
) -> StreamingResponse:
    """SSE 流式执行。每完成一个动作就推一次, 而不是等十几秒后一次性返回。"""
    if not message.strip():
        raise HTTPException(400, "message 不能为空")
    _check_size(message, "max_input_chars", "提问")

    bus = EventBus()
    queue = bus.register()  # 先注册再启动, 否则开头的事件会丢
    req = ChatRequest(message=message, session_id=session_id, mode=mode)
    runner = _build_runner(req)
    tracer = Tracer(name=f"{mode}:{message[:40]}")

    async def execute() -> None:
        try:
            await runner.run(message, session_id=session_id, bus=bus, tracer=tracer)
        except Exception as exc:  # noqa: BLE001
            bus.emit("error", error=f"{type(exc).__name__}: {exc}")
        finally:
            bus.close()

    async def generate():
        task = asyncio.create_task(execute())
        try:
            async for event in bus.stream(queue):
                yield f"data: {json.dumps(event.to_dict(), ensure_ascii=False, default=str)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'trace_id': tracer.trace_id}, ensure_ascii=False)}\n\n"
        finally:
            # 客户端断开时别让 Agent 继续在后台烧 token
            if not task.done():
                task.cancel()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ==========================================================================
# 追踪
# ==========================================================================


@app.get("/api/traces")
async def traces(limit: int = 20) -> dict[str, Any]:
    return {"traces": trace_store.list(limit)}


@app.get("/api/trace/{trace_id}")
async def trace_detail(trace_id: str) -> dict[str, Any]:
    item = trace_store.get(trace_id)
    if item is None:
        raise HTTPException(404, f"trace {trace_id} 不存在或已被环形缓冲淘汰")
    return item


# ==========================================================================
# 记忆
# ==========================================================================


@app.get("/api/memory")
async def memory_dump(session_id: Optional[str] = None, limit: int = 80) -> dict[str, Any]:
    return get_memory().dump(session_id=session_id, limit=limit)


@app.post("/api/memory/ingest")
async def memory_ingest(req: IngestRequest) -> dict[str, Any]:
    if not req.text.strip():
        raise HTTPException(400, "text 不能为空")
    if req.strategy not in CHUNKERS:
        raise HTTPException(400, f"未知切片策略: {req.strategy}, 可选 {list(CHUNKERS)}")
    _check_size(req.text, "max_ingest_chars", "文档")
    return await get_memory().ingest_document(req.text, source=req.source, strategy=req.strategy)


@app.post("/api/memory/remember")
async def memory_remember(req: RememberRequest) -> dict[str, Any]:
    try:
        layer = MemoryLayer(req.layer)
    except ValueError:
        raise HTTPException(400, f"未知记忆层: {req.layer}")
    record = await get_memory().remember(
        req.content, layer=layer, session_id=req.session_id, importance=req.importance,
        source="manual",
    )
    return record.to_dict()


@app.post("/api/memory/decay")
async def memory_decay() -> dict[str, Any]:
    """手动触发一次遗忘。演示时配合调小半衰期, 能直接看到记忆被归档。"""
    return get_memory().decay_and_prune()


@app.post("/api/memory/reflect")
async def memory_reflect(session_id: str = "default") -> dict[str, Any]:
    result = await get_memory().maybe_reflect(session_id, force=True)
    return result or {"insights": [], "note": "记录太少, 未触发反思"}


@app.get("/api/memory/search")
async def memory_search(query: str, session_id: Optional[str] = None, top_k: int = 6) -> dict[str, Any]:
    """检索接口会返回每一路信号的原始分数, 用来解释"为什么是这几条"。"""
    results = await get_memory().retrieve(query, session_id=session_id or "", top_k=top_k)
    return {"query": query, "results": [r.to_dict() for r in results]}


# ==========================================================================
# 实验室
# ==========================================================================


@app.post("/api/lab/chunk")
async def lab_chunk(req: ChunkLabRequest) -> dict[str, Any]:
    if not req.text.strip():
        raise HTTPException(400, "text 不能为空")
    _check_size(req.text, "max_ingest_chars", "文档")
    strategies = req.strategies or list(CHUNKERS)
    out: dict[str, Any] = {}
    for name in strategies:
        if name not in CHUNKERS:
            continue
        kwargs: dict[str, Any] = {}
        if name == "hierarchical":
            kwargs = {"parent_tokens": req.chunk_tokens * 3, "child_tokens": req.chunk_tokens}
        else:
            kwargs = {"chunk_tokens": req.chunk_tokens, "overlap": req.overlap}
        chunks = await get_chunker(name, **kwargs).split(req.text, source="lab")
        out[name] = {
            "stats": chunk_stats(chunks),
            "chunks": [
                {**c.to_dict(with_text=False), "preview": c.text[:220],
                 "title": c.display_title}
                for c in chunks[:40]
            ],
        }
    return {"strategies": out}


@app.post("/api/lab/context")
async def lab_context(req: ContextLabRequest) -> dict[str, Any]:
    """把一次真实的上下文组装过程摊开: 每个分区要了多少、给了多少、怎么压的。"""
    memory = get_memory()
    provider = get_provider()
    allocator = None
    if req.context_window:
        from keel.context.budget import BudgetAllocator

        allocator = BudgetAllocator(context_window=req.context_window)

    assembler = ContextAssembler(provider, allocator)
    items = await memory.build_context_items(req.task, session_id=req.session_id)
    history = memory.working(req.session_id).to_messages()
    assembled = await assembler.build(
        task=req.task,
        system_prompt=default_system_prompt(),
        tools_schema=get_registry().schemas(),
        items=items,
        history=history,
        scratchpad="",
    )
    return {
        **assembled.to_dict(with_messages=True),
        "retrieved_items": [
            {"zone": i.zone, "score": round(i.score, 4), "source": i.source,
             "tokens": i.tokens, "text": i.text[:300], "meta": i.meta}
            for i in items
        ],
    }


# ==========================================================================
# 消融实验
#
# 界面上能直接跑 A/B、看指标随 commit 的走向。原先 bench 只是个命令行脚本,
# 结果看完就没了 —— 把历次产物存下来并画成趋势, "这次改动让哪个指标变差了"
# 才第一次变成一个能当场回答的问题。
# ==========================================================================

BENCH_DIR = settings.data_dir / "bench"
HISTORY_DIR = BENCH_DIR / "history"
_BENCH_GROUPS = ("all", "guard", "breaker", "mmr", "budget", "digest", "dag")
# 同一时刻只允许一个实验在跑: bench 会重建全局单例并吃满 CPU, 并发跑两次
# 出来的数字互相干扰, 比不跑还糟
_bench_lock = asyncio.Lock()


def _sse(kind: str, payload: dict[str, Any]) -> str:
    return f"data: {json.dumps({'type': kind, **payload}, ensure_ascii=False, default=str)}\n\n"


def _load_runs(limit: int = 40) -> list[dict[str, Any]]:
    """按生成时间倒序读取历史产物。

    跳过读不动的文件而不是抛错: 历史目录里混进半截文件是常态(比如跑到一半被中断),
    一个坏文件不该让整个趋势图打不开。
    """
    if not HISTORY_DIR.exists():
        return []
    runs = []
    for path in HISTORY_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            runs.append({"file": path.name, "meta": data.get("meta", {}),
                         "metrics": data.get("metrics", {})})
        except Exception:  # noqa: BLE001
            continue
    runs.sort(key=lambda r: r["meta"].get("generated_at", ""), reverse=True)
    return runs[:limit]


@app.get("/api/bench/history")
async def bench_history(limit: int = 40) -> dict[str, Any]:
    runs = _load_runs(limit)
    names: list[str] = []
    for run_item in runs:
        for name in run_item["metrics"]:
            if name not in names:
                names.append(name)
    return {"runs": runs, "metric_names": sorted(names),
            "can_run": not settings.server.public_mode}


@app.get("/api/bench/stream")
async def bench_stream(group: str = "all") -> StreamingResponse:
    """跑一组消融实验, 把命令行输出实时推到界面。

    用子进程而不是在进程内 import bench: bench 会反复重建全局 Provider 和记忆单例,
    在服务进程里跑会污染正在进行的对话会话。子进程天然隔离, 代价只是多一次解释器启动。
    """
    if settings.server.public_mode:
        raise HTTPException(403, "公开演示站不开放实验触发（一次全量实验约 30 秒 CPU）。"
                                 "本地部署可用。")
    if group not in _BENCH_GROUPS:
        raise HTTPException(400, f"未知实验组: {group}, 可选 {list(_BENCH_GROUPS)}")

    async def generate():
        if _bench_lock.locked():
            yield _sse("error", {"error": "已有实验在跑，请等它结束。"})
            return
        async with _bench_lock:
            HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            out = HISTORY_DIR / f"{stamp}-{group}.json"
            root = Path(__file__).resolve().parent.parent.parent
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(root / "scripts" / "bench.py"), group,
                "--json", str(out),
                cwd=str(root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
            )
            yield _sse("start", {"group": group, "file": out.name})
            assert proc.stdout is not None
            async for raw in proc.stdout:
                yield _sse("log", {"line": raw.decode("utf-8", "replace").rstrip()})
            code = await proc.wait()
            if code != 0:
                yield _sse("error", {"error": f"实验进程退出码 {code}"})
                return
            # 同时刷新 latest, 让「最近一次」始终指向刚跑完这一份
            payload = json.loads(out.read_text(encoding="utf-8"))
            (BENCH_DIR / "latest.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            yield _sse("done", {"file": out.name, "metrics": payload.get("metrics", {}),
                                "meta": payload.get("meta", {})})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def run(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")

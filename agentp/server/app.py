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
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agentp.config import settings
from agentp.context.assembler import ContextAssembler, default_system_prompt
from agentp.context.chunker import CHUNKERS, chunk_stats, get_chunker
from agentp.llm import get_provider
from agentp.loop import ReActEngine
from agentp.memory import MemoryLayer, get_memory
from agentp.observability.events import EventBus
from agentp.observability.trace import Tracer, trace_store
from agentp.orchestrator import Orchestrator
from agentp.tools import get_registry

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="AgentP", version="0.1.0",
              description="Agent 运行时: 上下文工程 / 记忆系统 / Loop Engineering / 多 Agent 编排")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


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
        from agentp.context.budget import BudgetAllocator

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


def run(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")

"""OpenAI 兼容协议 Provider。

一份实现同时覆盖 OpenAI / DeepSeek / 通义千问(兼容模式) / Kimi / vLLM / Ollama,
改 base_url + model 就行。带指数退避重试和超时。
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any, Optional, Sequence

import httpx

from agentp.config import settings
from agentp.llm.base import LLMResponse, Message, ToolCall, Usage, estimate_cost
from agentp.util import count_message_tokens, count_tokens, hash_embed

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatProvider:
    name = "openai"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        cfg = settings.llm
        self.model = model or cfg.model
        self.api_key = api_key or cfg.api_key
        self.base_url = (base_url or cfg.base_url).rstrip("/")
        self.embedding_model = embedding_model or cfg.embedding_model
        self.timeout = cfg.timeout_s
        self.max_retries = cfg.max_retries
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """指数退避 + 抖动。抖动是为了避免多个并发 worker 被限流后同步重试。"""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._get_client().post(path, json=payload)
                if resp.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                    raise httpx.HTTPStatusError(
                        f"retryable {resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                backoff = (2 ** attempt) * 0.5 + random.random() * 0.3
                await asyncio.sleep(backoff)
        raise RuntimeError(f"LLM 请求失败({path}): {last_exc}") from last_exc

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        hint: str = "react",
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api() for m in messages],
            "temperature": settings.llm.temperature if temperature is None else temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        # planner/critic/reflect 都要求结构化输出, 能开 JSON mode 就开
        if hint in ("plan", "critic", "reflect", "extract_facts", "insight") and not tools:
            payload["response_format"] = {"type": "json_object"}

        started = asyncio.get_event_loop().time()
        data = await self._post("/chat/completions", payload)
        latency_ms = (asyncio.get_event_loop().time() - started) * 1000

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        content = msg.get("content") or ""

        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                # 小模型经常吐出坏 JSON; 保留原文交给工具层去报参数错误,
                # 这个错误信息会作为 observation 回灌, 模型下一轮通常能自我修正
                args = {"_raw": raw_args}
            tool_calls.append(ToolCall(id=tc.get("id") or "", name=fn.get("name", ""), arguments=args))

        usage_raw = data.get("usage") or {}
        prompt_tokens = usage_raw.get("prompt_tokens") or count_message_tokens(
            [m.to_api() for m in messages]
        )
        completion_tokens = usage_raw.get("completion_tokens") or count_tokens(content)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason") or ("tool_calls" if tool_calls else "stop"),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=estimate_cost(self.model, prompt_tokens, completion_tokens),
            ),
            model=data.get("model", self.model),
            latency_ms=latency_ms,
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            data = await self._post(
                "/embeddings", {"model": self.embedding_model, "input": list(texts)}
            )
            items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
            return [item["embedding"] for item in items]
        except Exception:
            # embedding 服务挂了不该让整个 Agent 停摆, 降级到本地 hashing 向量:
            # 召回质量下降但链路仍然可用
            dim = settings.context.embedding_dim
            return [hash_embed(t, dim) for t in texts]

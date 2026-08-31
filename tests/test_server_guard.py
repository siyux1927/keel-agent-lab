"""公网模式自我保护的测试。

演示站是要挂到公网的, 而 /api/chat 一次编排请求要跑十几次模型调用 ——
限流和长度上限如果只是"写了但没生效", 上线当天就会被一个爬虫跑垮。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from keel.config import settings
from keel.server.app import _limiter, app


@pytest.fixture
def public_mode():
    """打开公网模式并在结束后还原, 顺带清空限流器的历史。

    限流器是模块级单例, 不清的话用例之间会互相污染 —— 前一个用例打满了配额,
    后一个一上来就是 429。
    """
    cfg = settings.server
    before = (cfg.public_mode, cfg.rate_limit_per_min, cfg.max_input_chars, cfg.max_ingest_chars)
    cfg.public_mode = True
    cfg.rate_limit_per_min = 3
    cfg.max_input_chars = 50
    cfg.max_ingest_chars = 100
    _limiter._hits.clear()
    yield cfg
    (cfg.public_mode, cfg.rate_limit_per_min,
     cfg.max_input_chars, cfg.max_ingest_chars) = before
    _limiter._hits.clear()


def test_rate_limit_blocks_after_quota(public_mode):
    client = TestClient(app)
    codes = [client.post("/api/chat", json={"message": "计算 1+1"}).status_code
             for _ in range(5)]
    assert codes.count(429) == 2, f"配额 3 次之后应被拒绝, 实际 {codes}"


def test_readonly_endpoints_are_not_limited(public_mode):
    """健康检查和记忆浏览必须不受限流影响, 否则界面自身的轮询会把配额吃光。"""
    client = TestClient(app)
    codes = [client.get("/api/health").status_code for _ in range(10)]
    assert set(codes) == {200}


def test_oversized_input_rejected(public_mode):
    client = TestClient(app)
    resp = client.post("/api/chat", json={"message": "计" * 200})
    assert resp.status_code == 413


def test_oversized_document_rejected(public_mode):
    client = TestClient(app)
    resp = client.post("/api/memory/ingest", json={"text": "文" * 500})
    assert resp.status_code == 413


def test_local_mode_has_no_limits():
    """本地开发不该受任何限制 —— 实验室就是用来喂大文档的。"""
    settings.server.public_mode = False
    _limiter._hits.clear()
    client = TestClient(app)
    resp = client.post("/api/lab/chunk", json={"text": "# 标题\n\n" + "内容。" * 2000})
    assert resp.status_code == 200

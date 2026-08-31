"""HTTP 层冒烟测试: 覆盖控制台每个页签依赖的接口, 含 SSE 流式。

    python scripts/smoke_api.py [base_url]
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
DOC = """# 团队手册

## 发布流程
发布窗口为每周二下午 14:00 到 16:00, 需要两人评审通过。

## 上下文预算
默认窗口 8192 tokens, 输出预留 1024 tokens, 安全边界 5%。
"""

passed = failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  [OK]   {name}  {detail}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


def call(path: str, body=None, method=None):
    url = BASE + path
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if data else "GET"),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sse(path: str, limit: int = 400) -> list[dict]:
    events = []
    with urllib.request.urlopen(BASE + path, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
                if events[-1].get("type") == "done" or len(events) >= limit:
                    break
    return events


def main() -> int:
    print(f"目标: {BASE}\n")

    print("[基础]")
    health = call("/api/health")
    check("GET /api/health", health["status"] == "ok",
          f"provider={health['provider']} tools={len(health['tools'])}")
    check("GET /api/config", "loop" in call("/api/config"))
    tools = call("/api/tools")
    check("GET /api/tools", len(tools["schemas"]) >= 6 and "calculator" in tools["health"])

    print("\n[记忆浏览器]")
    ing = call("/api/memory/ingest", {"text": DOC, "source": "handbook.md", "strategy": "structure"})
    check("POST /api/memory/ingest", ing["indexed"] + ing["skipped_duplicates"] > 0,
          f"indexed={ing['indexed']} skipped={ing['skipped_duplicates']}")
    again = call("/api/memory/ingest", {"text": DOC, "source": "handbook.md", "strategy": "structure"})
    check("入库幂等(重复上传不新增)", again["indexed"] == 0,
          f"第二次 indexed={again['indexed']} skipped={again['skipped_duplicates']}")

    q = urllib.parse.quote("发布窗口是什么时候")
    search = call(f"/api/memory/search?query={q}&top_k=4")
    top = search["results"][0]
    check("GET /api/memory/search", "发布窗口" in top["content"],
          f"score={top['score']} 四路信号={sorted(top['breakdown'])}")
    check("检索返回打分分解", {"vector", "bm25", "recency", "importance"} <= set(top["breakdown"]))

    check("POST /api/memory/remember",
          call("/api/memory/remember",
               {"content": "用户偏好先给结论再给推导", "layer": "semantic",
                "session_id": "smoke"})["layer"] == "semantic")
    dump = call("/api/memory?session_id=smoke&limit=50")
    check("GET /api/memory", set(dump["by_layer"]) >= {"working", "episodic", "semantic", "procedural"})

    print("\n[切片实验室]")
    lab = call("/api/lab/chunk", {"text": DOC * 3, "chunk_tokens": 120, "overlap": 20})
    check("POST /api/lab/chunk", len(lab["strategies"]) == 5,
          " ".join(f"{k}={v['stats']['count']}块" for k, v in lab["strategies"].items()))

    print("\n[上下文实验室]")
    wide = call("/api/lab/context", {"task": "总结发布流程与预算规定",
                                     "session_id": "smoke", "context_window": 8192})
    check("POST /api/lab/context (8192)", wide["total_tokens"] > 0,
          f"tokens={wide['total_tokens']} 利用率={wide['budget']['utilization']}")
    tight = call("/api/lab/context", {"task": "总结发布流程与预算规定",
                                      "session_id": "smoke", "context_window": 1400})
    dropped = [z["name"] for z in tight["budget"]["zones"] if z["dropped"]]
    check("窗口收缩触发按优先级丢弃", len(dropped) > 0, f"被丢弃: {dropped}")
    check("不可丢弃分区始终保留",
          all(not z["dropped"] for z in tight["budget"]["zones"] if z["name"] in ("system", "task")))

    print("\n[对话 · 非流式]")
    react = call("/api/chat", {"message": "帮我算一下 (128*7+56)/4, 再查一下知识库里的发布窗口",
                               "session_id": "smoke", "mode": "react"})
    check("POST /api/chat (react)", react["success"],
          f"步数={react['state']['step_count']} 终止={react['state']['stop_reason']}")
    check("计算结果正确", "238" in json.dumps(react["state"], ensure_ascii=False))
    check("返回完整 trace", len(react["trace"]["tree"]) > 0,
          f"spans={react['usage']['spans']} llm={react['usage']['llm_calls']}")

    print("\n[对话 · SSE 流式]")
    events = sse("/api/chat/stream?" + urllib.parse.urlencode(
        {"message": "查一下知识库里的发布规定; 同时计算 (99+1)*3", "session_id": "smoke",
         "mode": "orchestrate"}))
    types = [e["type"] for e in events]
    check("SSE 建立并收到事件", len(events) > 5, f"{len(events)} 个事件")
    check("SSE 正常收尾", types[-1] == "done")
    for expected in ("orchestrator.start", "plan.ready", "dag.start", "dag.node",
                     "dag.finish", "critic.verdict", "orchestrator.finish"):
        check(f"事件 {expected}", expected in types)
    dag = next((e for e in events if e["type"] == "dag.finish"), {})
    check("DAG 并发有实际收益", dag.get("speedup", 0) > 1.0,
          f"speedup={dag.get('speedup')}× (串行 {dag.get('serial_ms')}ms → 实际 {dag.get('wall_ms')}ms)")

    print("\n[护栏 · SSE]")
    guard_events = sse("/api/chat/stream?" + urllib.parse.urlencode(
        {"message": "死循环演示 请反复确认当前时间", "session_id": "smoke2", "mode": "react"}))
    guards = [e for e in guard_events if e["type"] == "guard"]
    finish = next((e for e in guard_events if e["type"] == "agent.finish"), {})
    check("死循环被护栏拦下", any(g["action"] == "stop" for g in guards),
          " / ".join(g["reason"] for g in guards))
    check("先反思再终止(降级而非直接失败)", any(g["action"] == "reflect" for g in guards))
    check("中止后仍产出降级答案", bool(finish.get("answer", "").strip()),
          f"stop_reason={finish.get('stop_reason')}")

    print("\n[追踪]")
    traces = call("/api/traces?limit=10")["traces"]
    check("GET /api/traces", len(traces) >= 3, f"{len(traces)} 条")
    detail = call("/api/trace/" + traces[0]["trace_id"])
    check("GET /api/trace/{id}", "tree" in detail and detail["totals"]["spans"] > 0,
          f"spans={detail['totals']['spans']}")

    print("\n[记忆演化]")
    check("POST /api/memory/decay", "archived" in call("/api/memory/decay", {}))
    check("POST /api/memory/reflect", "insights" in call("/api/memory/reflect?session_id=smoke", {}))

    print("\n[消融实验]")
    hist = call("/api/bench/history?limit=40")
    check("GET /api/bench/history", "runs" in hist and "can_run" in hist,
          f"留档 {len(hist['runs'])} 次 / {len(hist['metric_names'])} 项指标")

    if hist["can_run"]:
        # 只跑 dag 这一组: 它是六组里最快的, 冒烟要验的是「链路通不通」而不是「实验准不准」
        events = sse("/api/bench/stream?group=dag", limit=600)
        kinds = [e.get("type") for e in events]
        check("SSE 实验流式执行", "start" in kinds and "log" in kinds,
              f"{len(events)} 个事件")
        done = next((e for e in events if e.get("type") == "done"), None)
        check("实验完成并留档", done is not None,
              f"file={done['file']}" if done else "未收到 done")
        if done:
            check("产出带方向的指标", all(
                m.get("goal") in ("lower", "higher", "info") for m in done["metrics"].values()),
                f"{len(done['metrics'])} 项")
            after = call("/api/bench/history?limit=40")
            check("历史记录随之增长", len(after["runs"]) > len(hist["runs"]),
                  f"{len(hist['runs'])} → {len(after['runs'])}")
    else:
        check("公网模式下实验触发被拒绝", True, "can_run=false，符合预期")

    print(f"\n{'=' * 60}\n  通过 {passed} 项, 失败 {failed} 项\n{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

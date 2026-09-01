"""把 bench 产出的 JSON 渲染成 Markdown 报告。

    python scripts/report.py data/bench/latest.json --out docs/BENCH.md
    python scripts/report.py data/bench/latest.json --inject 任意带标记对的文档.md

为什么要有这一步:

  实验结论最终要被人读到 —— README 里、PR 评论里、汇报材料里。手抄数字的问题不是抄错,
  是它**不会跟着代码变**: 改完实现文档还停在上个版本, 时间越久越不可信, 而读者无从分辨
  哪些数字是新的。所以这里划一条线: **数字由机器生成, 解读由人来写**。
  报告只渲染 bench 真实测出来的值, README 里的定位、取舍分析、踩坑记录仍是手写的 ——
  那些才是机器给不了的部分。

  注入采用标记对 <!-- BENCH:BEGIN --> ... <!-- BENCH:END -->, 只替换标记之间的内容,
  人写的段落不会被覆盖。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BEGIN = "<!-- BENCH:BEGIN -->"
END = "<!-- BENCH:END -->"

# 各组在报告里的标题与顺序。bench 只跑了部分组时, 缺失的组静默跳过。
SECTIONS = [
    ("guard", "1. 护栏 A/B：朴素 ReAct vs 全套护栏"),
    ("breaker", "2. 工具熔断器"),
    ("mmr", "3. MMR 去冗余（λ 扫描）"),
    ("budget", "4. 预算调度 + 压缩"),
    ("digest", "5. 动作清单注入系统提示词"),
    ("dag", "6. DAG 并发编排 vs 单智能体"),
]


def pct(new: float, old: float) -> str:
    if old == 0:
        return "n/a"
    return f"{(new - old) / old * 100:+.1f}%"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def render_guard(g: dict[str, Any]) -> list[str]:
    lines = ["两组的**硬上限完全相同**，差别只有动作重复检测、周期振荡检测、进展停滞检测、"
             "反思重规划的开与关——这样省下来的 token 才能归因到检测器本身。", ""]
    labels = {"healthy": "正常任务", "pathological": "病态任务"}
    rows = []
    for kind, per_arm in g.get("by_kind", {}).items():
        for arm, s in per_arm.items():
            arm_cn = "朴素" if arm == "naive" else "全护栏"
            rows.append([
                f"**{labels.get(kind, kind)} {s['n']} 个** · {arm_cn}",
                f"{s['steps']:.1f}", f"{s['llm']:.1f}", f"{s['tools']:.1f}",
                f"{s['tokens']:.0f}", f"{s['redundant']:.1f}", str(s["trips"]),
                f"{s['answered']}/{s['n']}",
                f"{s['correct']}/{s['graded']}" if s["graded"] else "—",
            ])
    lines += table(["分组", "步数", "对话调用", "工具调用", "tokens",
                    "重复调用", "护栏触发", "有答案", "答对"], rows)

    for kind, per_arm in g.get("by_kind", {}).items():
        if len(per_arm) != 2:
            continue
        a, b = per_arm["naive"], per_arm["guarded"]
        lines += ["", f"**{labels.get(kind, kind)}**：步数 {pct(b['steps'], a['steps'])}，"
                      f"token {pct(b['tokens'], a['tokens'])}，"
                      f"对话调用 {pct(b['llm'], a['llm'])}，"
                      f"重复调用 {a['redundant']:.1f} → {b['redundant']:.1f}。"]

    detail = g.get("pathological", [])
    if detail:
        lines += ["", "病态任务逐条明细（护栏是怎么止损的）：", ""]
        lines += table(
            ["任务", "组别", "步数", "重复调用", "反思", "拦截", "终止原因", "答案"],
            [[d["task"], d["arm"], str(d["steps"]), str(d["redundant"]),
              str(d["reflections"]), str(d["blocked"]), d["stop_reason"] or "—",
              "有（降级作答）" if d["answered"] else "无"] for d in detail],
        )
    return lines


def render_breaker(g: dict[str, Any]) -> list[str]:
    rows = g["rows"]
    lines = [f"同样 {g['intended_calls']} 次调用意图，看有多少真的打到了下游。", ""]
    lines += table(
        ["配置", "模型发起", "落到下游", "短路拒绝", "耗时(ms)", "终态"],
        [[r["label"], str(r["calls"]), str(r["downstream"]), str(r["rejected"]),
          f"{r['ms']:.0f}", r["state"]] for r in rows],
    )
    off, on = rows[0], rows[1]
    lines += ["", f"下游真实请求 {off['downstream']} → {on['downstream']} "
                  f"（{pct(on['downstream'], off['downstream'])}），"
                  f"耗时 {pct(on['ms'], off['ms'])}。"]
    return lines


def render_mmr(g: dict[str, Any]) -> list[str]:
    corpus = "，".join(f"{k} {v} 条" for k, v in g.get("corpus", {}).items())
    lines = [f"构造冗余语料（{corpus}），查询「{g['query']}」取 top-6。", ""]
    lines += table(
        ["配置", "平均余弦(冗余)", "平均词重合", "保留相关性", "命中分布"],
        [[s["label"], f"{s['avg_cos']:.4f}", f"{s['avg_jac']:.4f}",
          f"{s['relevance']:.4f}", s["dist"]] for s in g["sweep"]],
    )
    sweep = g["sweep"]
    base = sweep[0]
    default = next((s for s in sweep if "默认" in s["label"]), None)
    if default:
        lines += ["", f"默认 λ={g['default_lambda']}：冗余 {pct(default['avg_cos'], base['avg_cos'])}，"
                      f"相关性 {pct(default['relevance'], base['relevance'])}；"
                      f"最激进 λ 下冗余 {pct(sweep[-1]['avg_cos'], base['avg_cos'])}，"
                      f"相关性 {pct(sweep[-1]['relevance'], base['relevance'])}。"]
    return lines


def render_budget(g: dict[str, Any]) -> list[str]:
    lines = [f"同一批素材（原始需求 {g['raw_demand_tokens']} tokens）在不同窗口下组装两次。", ""]
    lines += table(
        ["窗口", "配置", "实际 tokens", "可用预算", "是否超窗", "丢弃分区", "压缩分区"],
        [[str(r["window"]), r["label"], str(r["tokens"]), str(r["budget"]),
          "**超窗**" if r["overflow"] else "安全", str(r["dropped"]), str(r["squeezed"])]
         for r in g["rows"]],
    )
    oc, n = g["overflow_count"], len(g["windows"])
    lines += ["", f"超窗次数：压缩关闭 {oc['压缩关闭']}/{n}，压缩开启 {oc['压缩开启']}/{n}。", "",
              "窗口 2048 时的取舍顺序：", ""]
    lines += table(
        ["分区", "需求", "分配", "处置"],
        [[z["zone"], str(z["requested"]), str(z["granted"]), z["state"]]
         for z in g["zones_at_2048"]],
    )
    return lines


def render_digest(g: dict[str, Any]) -> list[str]:
    lines = ["把「已执行动作」注入系统提示词（不可丢弃分区）而不是留在 scratchpad。"
             "本组关掉了动作重复检测，否则护栏会把现象直接盖住。", ""]
    lines += table(
        ["窗口", "动作清单", "步数", "工具调用", "重复调用", "scratchpad", "答对"],
        [[str(r["window"]), "开启" if r["inject"] else "关闭", str(r["steps"]),
          str(r["tool_calls"]), str(r["redundant"]), r["scratchpad_method"],
          "是" if r["correct"] else "否"] for r in g["rows"]],
    )
    avg = g["avg_redundant"]
    lines += ["", f"各窗口平均重复调用：关闭 {avg['off']:.2f} 次，开启 {avg['on']:.2f} 次"
                  f"（{pct(avg['on'], avg['off'])}）。"]
    return lines


def render_dag(g: dict[str, Any]) -> list[str]:
    rows = g["rows"]
    lines = [f"目标（含三个可并行的子任务）：{g['goal']}", ""]
    lines += table(
        ["模式", "墙钟(ms)", "对话调用", "工具调用", "tokens", "成本($)", "结构"],
        [[r["label"], f"{r['wall']:.0f}", str(r["llm"]), str(r["tools"]),
          str(r["tokens"]), f"{r['cost']:.6f}", r["extra"]] for r in rows],
    )
    single, dag = rows[0], rows[1]
    lines += ["", f"墙钟 {pct(dag['wall'], single['wall'])}，"
                  f"token {pct(dag['tokens'], single['tokens'])}，"
                  f"对话调用 {pct(dag['llm'], single['llm'])}。"]
    if dag.get("speedup"):
        lines += [f"DAG 内部并发加速比 {dag['speedup']}×（同层节点串行耗时 / 实际墙钟）。"]
    lines += [f"简单任务「计算 6*7」的执行模式 = `{g['simple_task_mode']}`（自动降级，省掉规划与汇总）。"]
    return lines


RENDERERS = {"guard": render_guard, "breaker": render_breaker, "mmr": render_mmr,
             "budget": render_budget, "digest": render_digest, "dag": render_dag}


def render(artifact: dict[str, Any], standalone: bool) -> str:
    meta = artifact["meta"]
    lines: list[str] = []
    if standalone:
        lines += ["# Keel 消融实验报告", ""]

    dirty = "，**工作区有未提交改动**" if meta.get("git_dirty") else ""
    lines += [
        f"> 由 `scripts/bench.py` 自动生成于 {meta['generated_at']}　·　"
        f"commit `{meta['git_commit'] or 'unknown'}`（{meta['git_branch'] or '?'}{dirty}）　·　"
        f"provider `{meta['provider']}/{meta['model']}`　·　"
        f"Python {meta['python']}　·　耗时 {meta['duration_s']}s",
        "",
        f"运行配置：上下文窗口 {meta['context_window']}　·　"
        f"遗忘半衰期 {meta['decay_half_life_hours']}h　·　MMR λ={meta['mmr_lambda']}",
        "",
    ]
    if meta.get("real_model"):
        lines += ["> 本轮使用**真实模型**，输出存在方差；病态任务依赖 Mock 的确定性触发词，已跳过。", ""]

    for key, heading in SECTIONS:
        group = artifact["groups"].get(key)
        if not group:
            continue
        lines += [f"### {heading}", ""]
        lines += RENDERERS[key](group)
        lines += [""]
    return "\n".join(lines).rstrip() + "\n"


def inject(target: Path, body: str) -> bool:
    text = target.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(
            f"{target} 里找不到标记对。请先手工插入：\n  {BEGIN}\n  {END}"
        )
    head, _, rest = text.partition(BEGIN)
    _, _, tail = rest.partition(END)
    merged = f"{head}{BEGIN}\n\n{body}\n{END}{tail}"
    if merged == text:
        return False
    target.write_text(merged, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="把 bench JSON 渲染成 Markdown 报告")
    parser.add_argument("artifact", help="bench.py --json 产出的文件")
    parser.add_argument("--out", metavar="PATH", help="写出独立报告")
    parser.add_argument("--inject", metavar="PATH", help="注入到已有文档的标记对之间")
    args = parser.parse_args()

    artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    if artifact.get("schema") != 1:
        print(f"警告: 未知的 schema 版本 {artifact.get('schema')}", file=sys.stderr)

    if not args.out and not args.inject:
        print(render(artifact, standalone=True))
        return

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(artifact, standalone=True), encoding="utf-8")
        print(f"报告已写出 → {out}")

    if args.inject:
        target = Path(args.inject)
        changed = inject(target, render(artifact, standalone=False))
        print(f"{'已更新' if changed else '无变化'} → {target}")


if __name__ == "__main__":
    main()

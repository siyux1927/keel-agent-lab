"""对比两份 bench 产物, 判断这次改动让哪些指标变好或变差。

    python scripts/bench_compare.py baseline.json current.json
    python scripts/bench_compare.py baseline.json current.json --out comment.md --tolerance 0.05

这是把评测当成回归测试用: 单元测试守的是"功能没坏", 这里守的是"效果没退"。
Agent 系统的退化通常不表现为异常, 而是悄悄多烧了 30% 的 token、多绕了两步 ——
没有基线对比的话, 这类退化永远不会被任何断言抓到。

判定规则:

  每个指标自带 goal(lower / higher / info), 由 bench 写进产物而不是写死在这里 ——
  历史产物因此可以自解释, 半年后指标定义变了也不会被反向解读。
  info 类指标只展示不判定(比如墙钟耗时, 它受 CI 机器负载影响, 拿来卡门槛只会制造噪声)。

  相对变化超过容差才算数。基线为 0 时退回看绝对变化 —— 0 → 3 的相对变化是无穷大,
  报出来只会淹没真正的信号。

退出码 1 表示存在超过容差的退化, 用于让 CI 失败。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

# 这些指标即便微小变化也值得关注: 它们的正常值就是 0, 任何非零都是行为改变。
_ZERO_IS_NORMAL = ("redundant", "overflow_compressed")


def _classify(name: str, goal: str, old: float, new: float, tol: float) -> str:
    """返回 improved / regressed / same。"""
    if goal == "info" or old == new:
        return "same"

    if old == 0:
        # 相对变化没有意义, 看绝对量。基线是 0 的指标通常是"应该保持为 0"的那一类,
        # 所以任何偏离都直接判定, 不设容差。
        significant = abs(new) > 1e-9
    elif any(k in name for k in _ZERO_IS_NORMAL):
        significant = abs(new - old) > 1e-9
    else:
        significant = abs(new - old) / abs(old) > tol

    if not significant:
        return "same"
    better = new < old if goal == "lower" else new > old
    return "improved" if better else "regressed"


def _fmt(value: float, unit: str) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".") if abs(value) < 100 else f"{value:.0f}"
    return f"{text}{unit}" if unit else text


def _delta(old: float, new: float) -> str:
    if old == 0:
        return f"{new - old:+.4g}（基线为 0）"
    return f"{(new - old) / abs(old) * 100:+.1f}%"


def compare(base: dict[str, Any], cur: dict[str, Any], tol: float) -> dict[str, Any]:
    bm, cm = base.get("metrics", {}), cur.get("metrics", {})
    rows = []
    for name in sorted(set(bm) | set(cm)):
        b, c = bm.get(name), cm.get(name)
        if b is None or c is None:
            rows.append({
                "name": name, "verdict": "added" if b is None else "removed",
                "old": None if b is None else b["value"],
                "new": None if c is None else c["value"],
                "goal": (c or b)["goal"], "unit": (c or b)["unit"], "delta": "—",
            })
            continue
        goal = c["goal"]
        rows.append({
            "name": name, "goal": goal, "unit": c["unit"],
            "old": b["value"], "new": c["value"],
            "delta": _delta(b["value"], c["value"]),
            "verdict": _classify(name, goal, b["value"], c["value"], tol),
        })
    return {"rows": rows, "tolerance": tol}


_ICON = {"improved": "🟢 改善", "regressed": "🔴 退化", "same": "⚪ 持平",
         "added": "🆕 新增", "removed": "⚠️ 缺失"}


def render(result: dict[str, Any], base: dict[str, Any], cur: dict[str, Any]) -> str:
    rows = result["rows"]
    changed = [r for r in rows if r["verdict"] in ("improved", "regressed", "added", "removed")]
    regressed = [r for r in rows if r["verdict"] == "regressed"]

    bmeta, cmeta = base["meta"], cur["meta"]
    out = ["## 消融实验回归对比", ""]
    out += [
        f"基线 `{bmeta.get('git_commit') or '?'}` → 本次 `{cmeta.get('git_commit') or '?'}`"
        f"　·　provider `{cmeta['provider']}/{cmeta['model']}`"
        f"　·　容差 {result['tolerance'] * 100:.0f}%",
        "",
    ]

    if not changed:
        out += [f"全部 {len(rows)} 项指标无显著变化。", ""]
    else:
        out += [f"{len(rows)} 项指标中 {len(changed)} 项发生变化"
                f"（{len(regressed)} 项退化）：", ""]
        out += ["| 指标 | 基线 | 本次 | 变化 | 判定 |", "|---|---|---|---|---|"]
        for r in sorted(changed, key=lambda x: (x["verdict"] != "regressed", x["name"])):
            old = "—" if r["old"] is None else _fmt(r["old"], r["unit"])
            new = "—" if r["new"] is None else _fmt(r["new"], r["unit"])
            out.append(f"| `{r['name']}` | {old} | {new} | {r['delta']} | {_ICON[r['verdict']]} |")
        out.append("")

    if regressed:
        out += ["> 存在退化指标。如果这是预期内的取舍（比如为了正确率主动多花 token），"
                "请在 PR 描述里说明原因；否则应当先修复。", ""]

    if cmeta.get("git_dirty") or bmeta.get("git_dirty"):
        out += ["> 注意：参与对比的产物中有工作区未提交的版本，结果不可完全复现。", ""]

    unchanged = len(rows) - len(changed)
    out += [f"<sub>持平 {unchanged} 项已折叠。`info` 类指标（墙钟耗时等）只展示不判定，"
            f"因为它们受 CI 机器负载影响。</sub>"]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="对比两份 bench 产物")
    parser.add_argument("baseline")
    parser.add_argument("current")
    parser.add_argument("--out", metavar="PATH", help="把 Markdown 结果写到文件")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="相对变化超过该比例才算显著, 默认 0.05")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="存在退化时退出码为 1")
    args = parser.parse_args()

    base_path, cur_path = Path(args.baseline), Path(args.current)
    if not base_path.exists():
        # 首次接入时 main 上还没有产物, 这不是错误。硬失败会让第一个 PR 永远合不进去。
        note = "## 消融实验回归对比\n\n基线产物不存在（可能是首次接入），本次跳过对比。\n"
        print(note)
        if args.out:
            Path(args.out).write_text(note, encoding="utf-8")
        return

    base = json.loads(base_path.read_text(encoding="utf-8"))
    cur = json.loads(cur_path.read_text(encoding="utf-8"))
    result = compare(base, cur, args.tolerance)
    body = render(result, base, cur)
    print(body)
    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")

    regressed = [r for r in result["rows"] if r["verdict"] == "regressed"]
    if regressed and args.fail_on_regression:
        names = ", ".join(r["name"] for r in regressed)
        print(f"\n检测到 {len(regressed)} 项指标退化: {names}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

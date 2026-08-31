"""把消融报告渲染成一个静态站点, 供 GitHub Pages 发布。

    python scripts/build_site.py site

只处理报告实际会用到的 Markdown 子集: 标题、表格、引用块、段落、列表,
以及行内的 **粗体**、`代码`、[链接](url)。

为什么不装 markdown 库: 这个项目的依赖清单是刻意压到最小的(连 python-dotenv 都自己实现了),
而这里的输入不是任意用户文档, 是 report.py 自己生成的、格式完全可控的 Markdown。
为一个已知形状的输入引入通用解析器不划算。代价是它撑不住任意 Markdown ——
所以下面显式列出了支持范围, 免得后来者以为它是通用的。
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def inline(text: str) -> str:
    out = html.escape(text)
    out = _CODE.sub(lambda m: f"<code>{m.group(1)}</code>", out)
    out = _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)
    out = _LINK.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', out)
    return out


def _table(rows: list[str]) -> str:
    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    head = cells(rows[0])
    body = [cells(r) for r in rows[2:]]  # rows[1] 是 |---|---| 分隔行
    out = ["<table><thead><tr>"]
    out += [f"<th>{inline(c)}</th>" for c in head]
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def md_to_html(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(_table(block) if len(block) >= 2 else "")
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            out.append(f"<h{level}>{inline(stripped[level:].strip())}</h{level}>")
            i += 1
            continue

        if stripped.startswith(">"):
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(block)) + "</blockquote>")
            continue

        if re.match(r"^\d+\.\s", stripped) or stripped.startswith(("- ", "* ")):
            ordered = bool(re.match(r"^\d+\.\s", stripped))
            tag = "ol" if ordered else "ul"
            items = []
            while i < len(lines):
                s = lines[i].strip()
                if ordered and re.match(r"^\d+\.\s", s):
                    items.append(re.sub(r"^\d+\.\s", "", s))
                elif not ordered and s.startswith(("- ", "* ")):
                    items.append(s[2:])
                else:
                    break
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue

        out.append(f"<p>{inline(stripped)}</p>")
        i += 1
    return "\n".join(out)


PAGE = """<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Keel · 消融实验报告</title>
<style>
  :root{{--bg:#0d1117;--panel:#151b23;--line:#232c37;--text:#e6edf3;--muted:#8b949e;
    --accent:#4493f8;--ok:#3fb950;--warn:#d29922}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--text);
    font:15px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK SC",sans-serif}}
  header{{padding:38px 22px 26px;border-bottom:1px solid var(--line);background:var(--panel)}}
  header div{{max-width:1080px;margin:0 auto}}
  h1{{margin:0 0 6px;font-size:30px;letter-spacing:-.4px}}
  .sub{{color:var(--muted);font-size:14px}}
  .sub a{{color:var(--accent)}}
  main{{max-width:1080px;margin:0 auto;padding:26px 22px 80px}}
  h2{{margin:34px 0 12px;font-size:21px;border-bottom:1px solid var(--line);padding-bottom:7px}}
  h3{{margin:30px 0 10px;font-size:17px;color:var(--accent)}}
  p{{margin:11px 0}}
  code{{background:#1c242e;padding:1px 6px;border-radius:4px;font-size:13px;
    font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}
  blockquote{{margin:14px 0;padding:11px 15px;border-left:3px solid var(--warn);
    background:#181f27;color:var(--muted);border-radius:0 6px 6px 0}}
  table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px;display:block;
    overflow-x:auto}}
  th,td{{border:1px solid var(--line);padding:7px 11px;text-align:left;white-space:nowrap}}
  th{{background:#1c242e;font-weight:600}}
  tbody tr:nth-child(even){{background:#12181f}}
  ol,ul{{padding-left:22px}} li{{margin:6px 0}}
  footer{{max-width:1080px;margin:0 auto;padding:22px;color:var(--muted);font-size:13px;
    border-top:1px solid var(--line)}}
</style>
<header><div>
  <h1>Keel · 消融实验报告</h1>
  <div class="sub">{subtitle}</div>
</div></header>
<main>{body}</main>
<footer>
  由 <code>scripts/bench.py</code> 在 CI 上自动生成并发布，每次 main 更新都会重跑。
  原始数据：<a href="bench.json">bench.json</a>
</footer>
</html>
"""


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "site")
    report = out_dir / "BENCH.md"
    if not report.exists():
        raise SystemExit(f"找不到报告 {report}，请先运行 scripts/report.py")

    md = report.read_text(encoding="utf-8")
    # 首个 h1 已经在页头呈现, 正文里再来一遍是重复的
    md = re.sub(r"^#\s+.*\n", "", md, count=1)

    subtitle = "离线 Mock Provider · 完全确定性 · 结果可复现"
    artifact = out_dir / "bench.json"
    if artifact.exists():
        meta = json.loads(artifact.read_text(encoding="utf-8"))["meta"]
        subtitle = (f"commit <code>{meta.get('git_commit', '?')}</code> · "
                    f"{meta['generated_at']} · provider {meta['provider']}/{meta['model']} · "
                    f"Python {meta['python']} · 耗时 {meta['duration_s']}s")

    (out_dir / "index.html").write_text(
        PAGE.format(subtitle=subtitle, body=md_to_html(md)), encoding="utf-8")
    print(f"站点已生成 → {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()

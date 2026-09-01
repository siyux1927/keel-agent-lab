// 记录构建产物是由哪份源码生成的。
//
// 产物提交进了版本库, 所以存在"改了源码忘了重新构建"这个失效模式, 且它不报错、
// 只是让仓库里的界面停在旧版本。最初的检查是在 CI 上重新构建一遍再逐字节比对产物,
// 但那要求跨平台可复现构建 —— Windows 与 Linux、Node 24 与 Node 22 之间并不成立,
// 结果是 CI 把"构建器输出格式的差异"报成了"你的产物过期了"。
//
// 这里改成对输入取指纹: 构建时把源码哈希写进 .build-stamp, CI 只重算一遍哈希做比对。
// 它精确回答"产物是否由当前源码构建", 且不依赖构建器逐字节可复现。

import { createHash } from 'node:crypto'
import { readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = fileURLToPath(new URL('..', import.meta.url))
const STAMP = join(ROOT, '.build-stamp')

// 决定产物内容的全部输入。漏掉任何一项, 改它就不会被检查到。
const FILES = ['index.html', 'vite.config.ts', 'tsconfig.json', 'package.json', 'package-lock.json']
const DIRS = ['src']

function walk(dir) {
  const out = []
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry)
    if (statSync(full).isDirectory()) out.push(...walk(full))
    else out.push(full)
  }
  return out
}

function fingerprint() {
  const paths = [
    ...FILES.map(f => join(ROOT, f)),
    ...DIRS.flatMap(d => walk(join(ROOT, d))),
  ]
  // 路径统一成正斜杠再排序, 否则 Windows 与 Linux 的遍历顺序可能不同
  const sorted = paths
    .map(p => [relative(ROOT, p).split('\\').join('/'), p])
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))

  const hash = createHash('sha256')
  for (const [rel, full] of sorted) {
    // 归一化换行符: 同一份文件在 Windows 检出是 CRLF、Linux 是 LF,
    // 不归一化的话这个指纹自己就会随平台漂移, 等于把要修的 bug 重犯一遍
    const content = readFileSync(full, 'utf8').split('\r\n').join('\n')
    hash.update(rel).update('\0').update(content).update('\0')
  }
  return hash.digest('hex')
}

const mode = process.argv[2]
const actual = fingerprint()

if (mode === 'write') {
  writeFileSync(STAMP, actual + '\n')
  console.log('build stamp:', actual.slice(0, 16))
} else if (mode === 'check') {
  let recorded = ''
  try {
    recorded = readFileSync(STAMP, 'utf8').trim()
  } catch {
    console.error('缺少 web/.build-stamp, 请执行 npm run build 后提交')
    process.exit(1)
  }
  if (recorded !== actual) {
    console.error('前端源码已改动但产物未重新构建。')
    console.error(`  stamp 记录: ${recorded.slice(0, 16)}`)
    console.error(`  当前源码:   ${actual.slice(0, 16)}`)
    console.error('请在 web/ 下执行 npm run build, 并把 keel/server/static 与 .build-stamp 一起提交。')
    process.exit(1)
  }
  console.log('产物与源码一致:', actual.slice(0, 16))
} else {
  console.error('用法: node scripts/stamp.mjs write|check')
  process.exit(2)
}

#!/usr/bin/env python3
"""把 lora_eval 的音檔打包成單一 HTML 聽辨頁。

音檔轉 64kbps mp3 後以 data URI 內嵌，整頁自帶音訊、不依賴外部路徑，
可直接發布成 Artifact 或用瀏覽器開啟。
"""

import argparse
import ast
import base64
import html
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_cases() -> dict[str, list[str]]:
    """從 eval_lora_novel.py 取出 CASES，但不觸發它的 import。

    直接 import 會拉進 soundfile / voxcpm，那些只在容器裡裝了；
    這支腳本跑在主機上，只需要文字。
    """
    src = (REPO / "scripts" / "eval_lora_novel.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "CASES":
            return ast.literal_eval(node.value)
    raise SystemExit("eval_lora_novel.py 找不到 CASES")


CASES = load_cases()

# 每組的驗收假設：樣本量決定 LoRA 學不學得到，這是全頁要回答的問題。
GROUP_META = {
    "seen": ("出現 2–4 次", "訓練集裡有幾個樣本，最有機會學到"),
    "once": ("只出現 1 次", "單樣本學習的極限"),
    "unseen": ("幾乎沒有", "對照組，預期與底模相同"),
    "terms": ("數十次", "專有名詞，樣本最充足"),
    "ctrl": ("不含生字", "確認音色沒被 LoRA 洗掉"),
}

VOICE_LABEL = {
    "cosy-young-female-01": "青年女聲 01",
    "cosy-young-male-01": "青年男聲 01",
}

# 生字表用於在句中標示重點字；順序不重要，逐字比對。
NOVEL_CHARS = set("倌刃劊喋嗝憫戾殯猩碘祢笅翩舟艇茁荊蓽褥襟頰飩餒餛")
TERM_WORDS = [
    "刀疤老五", "烏鴉老大", "劉經理", "桃花窟", "痱子粉",
    "風水師", "正龍爸", "罐茶葉", "兩仟萬", "白馬里",
]


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def to_mp3_data_uri(path: Path) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
             "-codec:a", "libmp3lame", "-b:a", "64k", "-ar", "24000", tmp.name],
            check=True,
        )
        raw = Path(tmp.name).read_bytes()
    return "data:audio/mpeg;base64," + base64.b64encode(raw).decode()


def mark_text(text: str, group: str) -> str:
    """把生字/專有名詞包起來，讓聽的人一眼看到該注意哪裡。"""
    esc = html.escape(text)
    if group == "terms":
        for w in TERM_WORDS:
            if w in esc:
                return esc.replace(w, f'<mark>{w}</mark>', 1)
        return esc
    if group == "ctrl":
        return esc
    return "".join(
        f"<mark>{c}</mark>" if c in NOVEL_CHARS else c for c in esc
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=REPO / "scratch" / "lora_eval")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", default="checkpoints/lora-novel-e4/best")
    args = ap.parse_args()

    voices = sorted(d.name for d in args.eval_dir.iterdir() if d.is_dir())
    if not voices:
        sys.exit(f"{args.eval_dir} 沒有任何參考音目錄")

    data = {}
    for v in voices:
        data[v] = {}
        for group in CASES:
            gdir = args.eval_dir / v / group
            if not gdir.exists():
                continue
            rows = []
            for i, text in enumerate(CASES[group], 1):
                lora, base = gdir / f"{i:02d}_lora.wav", gdir / f"{i:02d}_base.wav"
                if not (lora.exists() and base.exists()):
                    continue
                rows.append({
                    "text": text,
                    "marked": mark_text(text, group),
                    "lora": to_mp3_data_uri(lora),
                    "base": to_mp3_data_uri(base),
                    "dl": round(wav_duration(lora), 2),
                    "db": round(wav_duration(base), 2),
                })
                print(f"  {v}/{group}/{i:02d}", file=sys.stderr)
            data[v][group] = rows

    payload = json.dumps(
        {"voices": voices, "labels": VOICE_LABEL, "groups": GROUP_META, "data": data},
        ensure_ascii=False,
    )
    args.out.write_text(TEMPLATE.replace("__PAYLOAD__", payload), encoding="utf-8")
    n = sum(len(g) for v in data.values() for g in v.values())
    print(f"\n{args.out}  ({n} 句 × 2 版本, {args.out.stat().st_size/1024/1024:.1f} MB)")


TEMPLATE = r"""<title>台語 LoRA 生字聽辨</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root {
  --ink: #12262a;
  --ground: #f7f4ee;
  --surface: #fffdf9;
  --line: #ddd6ca;
  --muted: #6f7f80;
  --lora: #0f8f70;
  --base-c: #c2701c;
  --mark: #fbe6a2;
  --shadow: 0 1px 2px rgba(18,38,42,.06), 0 4px 16px rgba(18,38,42,.05);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ink: #e8e4db;
    --ground: #0d1b1e;
    --surface: #142529;
    --line: #263a3e;
    --muted: #859899;
    --lora: #35d6a4;
    --base-c: #f2a44f;
    --mark: #4a3d16;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 4px 16px rgba(0,0,0,.22);
  }
}
:root[data-theme="dark"] {
  --ink: #e8e4db;
  --ground: #0d1b1e;
  --surface: #142529;
  --line: #263a3e;
  --muted: #859899;
  --lora: #35d6a4;
  --base-c: #f2a44f;
  --mark: #4a3d16;
  --shadow: 0 1px 2px rgba(0,0,0,.3), 0 4px 16px rgba(0,0,0,.22);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: "Noto Serif TC", ui-serif, Georgia, serif;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 62rem; margin: 0 auto; padding: 3rem 1.25rem 6rem; }

header { margin-bottom: 2.5rem; }
h1 {
  font-size: clamp(1.75rem, 4vw, 2.5rem);
  font-weight: 700; letter-spacing: -.01em;
  margin: 0 0 .5rem; text-wrap: balance;
}
.sub { color: var(--muted); font-size: .95rem; max-width: 46rem; margin: 0; }
.meta {
  margin-top: 1.25rem; display: flex; flex-wrap: wrap; gap: .5rem 1.5rem;
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: .78rem; color: var(--muted);
}
.meta b { color: var(--ink); font-weight: 500; }

.tabs { display: flex; gap: .5rem; margin: 2rem 0 1.5rem; flex-wrap: wrap; }
.tab {
  font: inherit; font-size: .9rem; cursor: pointer;
  padding: .5rem 1.1rem; border-radius: 999px;
  border: 1px solid var(--line); background: var(--surface); color: var(--muted);
  transition: color .15s, border-color .15s;
}
.tab[aria-selected="true"] { color: var(--ink); border-color: var(--ink); }
.tab:focus-visible { outline: 2px solid var(--lora); outline-offset: 2px; }

.legend {
  display: flex; gap: 1.25rem; flex-wrap: wrap; align-items: center;
  font-family: "JetBrains Mono", monospace; font-size: .75rem;
  color: var(--muted); margin-bottom: 2rem;
}
.dot { display: inline-block; width: .6rem; height: .6rem; border-radius: 50%; margin-right: .4rem; vertical-align: -1px; }

section { margin-bottom: 3rem; }
.ghead {
  position: sticky; top: 0; z-index: 2;
  background: var(--ground); padding: .75rem 0 .6rem;
  border-bottom: 1px solid var(--line); margin-bottom: .25rem;
  display: flex; align-items: baseline; gap: .75rem; flex-wrap: wrap;
}
.gname { font-size: 1.05rem; font-weight: 600; }
.gcount {
  font-family: "JetBrains Mono", monospace; font-size: .72rem;
  color: var(--muted); border: 1px solid var(--line);
  padding: .1rem .5rem; border-radius: 999px;
}
.gnote { font-size: .82rem; color: var(--muted); }

.row {
  display: grid; grid-template-columns: 1fr auto;
  gap: 1rem 1.5rem; align-items: center;
  padding: .9rem 0; border-bottom: 1px solid var(--line);
}
.row:last-child { border-bottom: none; }
.line { font-size: 1.08rem; }
.n {
  font-family: "JetBrains Mono", monospace; font-size: .72rem;
  color: var(--muted); margin-right: .6rem;
}
mark { background: var(--mark); color: inherit; padding: 0 .1em; border-radius: 2px; }
.dur {
  font-family: "JetBrains Mono", monospace; font-size: .72rem;
  color: var(--muted); margin-top: .2rem;
  font-variant-numeric: tabular-nums;
}
.delta-neg { color: var(--lora); }

.btns { display: flex; gap: .5rem; }
.play {
  font: inherit; font-size: .82rem; cursor: pointer; white-space: nowrap;
  padding: .45rem .95rem; border-radius: .4rem;
  border: 1px solid currentColor; background: transparent;
  display: inline-flex; align-items: center; gap: .4rem;
  transition: background .15s;
}
.play.lora { color: var(--lora); }
.play.base { color: var(--base-c); }
.play:hover { background: color-mix(in srgb, currentColor 12%, transparent); }
.play:focus-visible { outline: 2px solid currentColor; outline-offset: 2px; }
.play[data-playing="1"] { background: color-mix(in srgb, currentColor 20%, transparent); }
.play svg { width: .7rem; height: .7rem; fill: currentColor; }

@media (max-width: 34rem) {
  .row { grid-template-columns: 1fr; }
  .btns { justify-content: flex-start; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header>
    <h1>台語 LoRA 生字聽辨</h1>
    <p class="sub">同一句話、同一個參考音，唯一差別是 LoRA 開或關。按樣本量分五組 —— 若 <b>terms</b> 有改善而 <b>unseen</b> 沒有，代表 LoRA 有效但受限於訓練樣本數。</p>
    <div class="meta" id="meta"></div>
  </header>

  <div class="tabs" id="tabs" role="tablist"></div>

  <div class="legend">
    <span><span class="dot" style="background:var(--lora)"></span>LoRA</span>
    <span><span class="dot" style="background:var(--base-c)"></span>原始底模</span>
    <span>黃底 = 該句要聽的生字或專有名詞</span>
  </div>

  <div id="body"></div>
</div>

<script>
const P = __PAYLOAD__;
let cur = P.voices[0], audio = null, curBtn = null;

const meta = document.getElementById('meta');
meta.innerHTML = [
  ['底模', 'ft-mixed-lr2e5-avgE-e12run-0820'],
  ['LoRA', 'r=32 alpha=32 · LM+DiT · val 0.922'],
  ['句數', Object.values(P.data[cur]).reduce((a,g)=>a+g.length,0) + ' 句 × 2 版本'],
].map(([k,v]) => `${k} <b>${v}</b>`).join('');

const tabs = document.getElementById('tabs');
P.voices.forEach(v => {
  const b = document.createElement('button');
  b.className = 'tab'; b.type = 'button'; b.role = 'tab';
  b.textContent = P.labels[v] || v;
  b.setAttribute('aria-selected', v === cur);
  b.onclick = () => { cur = v; stop(); render();
    [...tabs.children].forEach(c => c.setAttribute('aria-selected', c === b)); };
  tabs.appendChild(b);
});

function stop() {
  if (audio) { audio.pause(); audio = null; }
  if (curBtn) { curBtn.dataset.playing = '0'; curBtn = null; }
}

function play(src, btn) {
  const same = curBtn === btn;
  stop();
  if (same) return;
  audio = new Audio(src);
  btn.dataset.playing = '1'; curBtn = btn;
  audio.onended = () => { btn.dataset.playing = '0'; curBtn = null; audio = null; };
  audio.play();
}

const ICON = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1l9 5-9 5z"/></svg>';

function render() {
  const host = document.getElementById('body');
  host.innerHTML = '';
  for (const [g, rows] of Object.entries(P.data[cur])) {
    if (!rows.length) continue;
    const [count, note] = P.groups[g];
    const sec = document.createElement('section');
    sec.innerHTML = `<div class="ghead">
      <span class="gname">${g}</span>
      <span class="gcount">${count}</span>
      <span class="gnote">${note}</span></div>`;
    rows.forEach((r, i) => {
      const d = (r.dl - r.db).toFixed(2);
      const row = document.createElement('div');
      row.className = 'row';
      row.innerHTML = `<div>
          <div class="line"><span class="n">${String(i+1).padStart(2,'0')}</span>${r.marked}</div>
          <div class="dur">LoRA ${r.dl.toFixed(2)}s · 底模 ${r.db.toFixed(2)}s ·
            <span class="${d < 0 ? 'delta-neg' : ''}">${d > 0 ? '+' : ''}${d}s</span></div>
        </div>
        <div class="btns">
          <button class="play lora" type="button">${ICON} LoRA</button>
          <button class="play base" type="button">${ICON} 底模</button>
        </div>`;
      const [bl, bb] = row.querySelectorAll('.play');
      bl.onclick = () => play(r.lora, bl);
      bb.onclick = () => play(r.base, bb);
      sec.appendChild(row);
    });
    host.appendChild(sec);
  }
}
render();
</script>
"""


if __name__ == "__main__":
    main()

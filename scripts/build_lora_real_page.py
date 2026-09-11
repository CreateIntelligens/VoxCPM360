#!/usr/bin/env python3
"""把 lora_real 的三方對照音檔打包成單一 HTML 聽辨頁。

每句三個版本：真人原始錄音（標準答案）/ LoRA / 底模。
音檔轉 64kbps mp3 內嵌成 data URI，單檔即可播放。
"""

import argparse
import base64
import html
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

GROUP_META = {
    "terms": ("新詞", "舊語料 0 次、新集數數十次 — 底模沒學過"),
    "chars": ("生字", "整個語料庫只出現 1–4 次 — 樣本嚴重不足"),
    "ctrl": ("對照", "不含新詞新字，三版應該接近 — 你的耳朵基準線"),
}

# 互相關偵測出時間軸錯位的集數（見 Breeze-ASR-360 的稽核報告）。
# 這些集的原始錄音與逐字稿對不上，聽起來會怪 —— 是資料問題不是模型問題。
BAD_EPISODES = {81, 196, 241, 243, 245, 246, 255, 257, 258, 259}

VOICE_LABEL = {
    "cosy-young-female-01": "青年女聲",
    "cosy-young-male-01": "青年男聲",
}


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def mp3_data_uri(path: Path) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
             "-codec:a", "libmp3lame", "-b:a", "64k", "-ar", "24000", tmp.name],
            check=True,
        )
        return "data:audio/mpeg;base64," + base64.b64encode(
            Path(tmp.name).read_bytes()).decode()


def mark(text: str, term: str) -> str:
    esc = html.escape(text)
    if not term:
        return esc
    return esc.replace(term, f"<mark>{term}</mark>", 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=REPO / "scratch/lora_real",
                    help="主要版本（清理後）")
    ap.add_argument("--eval-dir-old", type=Path,
                    help="對照版本（清理前），給了就多一個播放鍵")
    ap.add_argument("--cases", type=Path, default=REPO / "scratch/real_cases.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cases = json.loads(args.cases.read_text())
    qpath = REPO / "scratch/case_quality.json"
    quality = json.loads(qpath.read_text()) if qpath.exists() else {}
    voices = sorted(d.name for d in args.eval_dir.iterdir() if d.is_dir())

    data = {}
    for v in voices:
        data[v] = {}
        for group, rows in cases.items():
            gdir = args.eval_dir / v / group
            if not gdir.exists():
                continue
            out = []
            for i, r in enumerate(rows, 1):
                paths = {k: gdir / f"{i:02d}_{k}.wav" for k in ("orig", "lora", "base")}
                if not all(p.exists() for p in paths.values()):
                    continue
                # 清理前的 LoRA 版本，用來對照資料清理的效果
                old = None
                if args.eval_dir_old:
                    cand = args.eval_dir_old / v / group / f"{i:02d}_lora.wav"
                    if cand.exists():
                        old = cand
                q = quality.get(f"{group}/{i}", {})
                ep = r.get("episode")
                out.append({
                    "term": r["term"],
                    "ep": ep,
                    "misaligned": ep in BAD_EPISODES,
                    "bad": q.get("bad", False),
                    "sil": q.get("sil"),
                    "cps": q.get("cps"),
                    "marked": mark(r["text"], r["term"]),
                    "seen": r["seen_in_train"],
                    "spk": r.get("speaker_id", ""),
                    **{k: mp3_data_uri(p) for k, p in paths.items()},
                    **{f"d_{k}": round(wav_seconds(p), 2) for k, p in paths.items()},
                    **({"old": mp3_data_uri(old), "d_old": round(wav_seconds(old), 2)}
                       if old else {}),
                })
                print(f"  {v}/{group}/{i:02d}", file=sys.stderr)
            data[v][group] = out

    payload = json.dumps({
        "voices": voices, "labels": VOICE_LABEL,
        "groups": GROUP_META, "data": data,
    }, ensure_ascii=False)

    args.out.write_text(TEMPLATE.replace("__PAYLOAD__", payload), encoding="utf-8")
    n = sum(len(g) for v in data.values() for g in v.values())
    print(f"\n{args.out} ({n} 句 × 3 版本, {args.out.stat().st_size/1024/1024:.1f} MB)")


TEMPLATE = r"""<title>台語 LoRA 三方聽辨</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root {
  --ink: #12262a; --ground: #f7f4ee; --surface: #fffdf9;
  --line: #ddd6ca; --muted: #6f7f80;
  --orig: #3a6ea5; --lora: #0f8f70; --base-c: #c2701c;
  --mark: #fbe6a2; --warn: #b4541c;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ink: #e8e4db; --ground: #0d1b1e; --surface: #142529;
    --line: #263a3e; --muted: #859899;
    --orig: #7aaede; --lora: #35d6a4; --base-c: #f2a44f;
    --mark: #4a3d16; --warn: #f2a44f;
  }
}
:root[data-theme="dark"] {
  --ink: #e8e4db; --ground: #0d1b1e; --surface: #142529;
  --line: #263a3e; --muted: #859899;
  --orig: #7aaede; --lora: #35d6a4; --base-c: #f2a44f;
  --mark: #4a3d16; --warn: #f2a44f;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--ground); color: var(--ink);
  font-family: "Noto Serif TC", ui-serif, Georgia, serif;
  line-height: 1.6; -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 66rem; margin: 0 auto; padding: 3rem 1.25rem 6rem; }
h1 { font-size: clamp(1.75rem,4vw,2.5rem); font-weight:700; margin:0 0 .5rem; text-wrap:balance; }
.sub { color: var(--muted); font-size:.95rem; max-width:48rem; margin:0; }
.meta {
  margin-top:1.25rem; display:flex; flex-wrap:wrap; gap:.5rem 1.5rem;
  font-family:"JetBrains Mono",monospace; font-size:.78rem; color:var(--muted);
}
.meta b { color: var(--ink); font-weight:500; }
.tabs { display:flex; gap:.5rem; margin:2rem 0 1.25rem; flex-wrap:wrap; }
.tab {
  font:inherit; font-size:.9rem; cursor:pointer; padding:.5rem 1.1rem;
  border-radius:999px; border:1px solid var(--line);
  background:var(--surface); color:var(--muted);
}
.tab[aria-selected="true"] { color:var(--ink); border-color:var(--ink); }
.tab:focus-visible { outline:2px solid var(--lora); outline-offset:2px; }
.legend {
  display:flex; gap:1.25rem; flex-wrap:wrap; align-items:center;
  font-family:"JetBrains Mono",monospace; font-size:.75rem;
  color:var(--muted); margin-bottom:2rem;
}
.dot { display:inline-block; width:.6rem; height:.6rem; border-radius:50%; margin-right:.4rem; vertical-align:-1px; }
section { margin-bottom:3rem; }
.ghead {
  position:sticky; top:0; z-index:2; background:var(--ground);
  padding:.75rem 0 .6rem; border-bottom:1px solid var(--line);
  margin-bottom:.25rem; display:flex; align-items:baseline; gap:.75rem; flex-wrap:wrap;
}
.gname { font-size:1.05rem; font-weight:600; }
.gnote { font-size:.82rem; color:var(--muted); }
.row {
  display:grid; grid-template-columns:1fr auto; gap:.75rem 1.5rem;
  align-items:center; padding:.9rem 0; border-bottom:1px solid var(--line);
}
.row:last-child { border-bottom:none; }
.line { font-size:1.08rem; }
.n { font-family:"JetBrains Mono",monospace; font-size:.72rem; color:var(--muted); margin-right:.6rem; }
mark { background:var(--mark); color:inherit; padding:0 .1em; border-radius:2px; }
.tag.warn { color:#b4541c; border-color:#b4541c; }
:root[data-theme="dark"] .tag.warn, @media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .tag.warn{color:#f2a44f;border-color:#f2a44f;}}
.tag.warn { color:var(--warn); border-color:var(--warn); }
.tag {
  font-family:"JetBrains Mono",monospace; font-size:.66rem;
  border:1px solid var(--line); padding:.05rem .4rem;
  border-radius:999px; color:var(--muted); margin-left:.5rem;
}
.dur {
  font-family:"JetBrains Mono",monospace; font-size:.72rem;
  color:var(--muted); margin-top:.25rem; font-variant-numeric:tabular-nums;
}
.btns { display:flex; gap:.5rem; }
.play {
  font:inherit; font-size:.82rem; cursor:pointer; white-space:nowrap;
  padding:.45rem .9rem; border-radius:.4rem; border:1px solid currentColor;
  background:transparent; display:inline-flex; align-items:center; gap:.4rem;
}
.play.orig { color:var(--orig); }
.play.lora { color:var(--lora); }
.play.base { color:var(--base-c); }
.play:hover { background:color-mix(in srgb,currentColor 12%,transparent); }
.play:focus-visible { outline:2px solid currentColor; outline-offset:2px; }
.play[data-playing="1"] { background:color-mix(in srgb,currentColor 22%,transparent); }
.play svg { width:.7rem; height:.7rem; fill:currentColor; }
@media (max-width:38rem) {
  .row { grid-template-columns:1fr; }
  .btns { justify-content:flex-start; }
}
@media (prefers-reduced-motion:reduce) { * { transition:none !important; } }
</style>

<div class="wrap">
  <header>
    <h1>台語 LoRA 三方聽辨</h1>
    <p class="sub">語料裡的真實句子，每句三個版本：<b>真人原始錄音</b>是標準答案，用來判斷 LoRA 念得對不對，而不只是跟底模不一樣。所有句子的參考音相同，差別只在模型。<br>帶橘色標記的句子，原始錄音本身就與逐字稿對不上（字幕時間軸錯位）—— 那些聽起來怪是資料問題，不是模型問題。</p>
    <div class="meta" id="meta"></div>
  </header>
  <div class="tabs" id="tabs" role="tablist"></div>
  <div class="legend">
    <span><span class="dot" style="background:var(--orig)"></span>真人錄音</span>
    <span><span class="dot" style="background:var(--lora)"></span>LoRA</span>
    <span><span class="dot" style="background:var(--base-c)"></span>底模</span>
    <span>黃底 = 要聽的新詞或生字</span>
    <span style="color:var(--warn)">橘標 = 原始資料本身有問題</span>
  </div>
  <div id="body"></div>
</div>

<script>
const P = __PAYLOAD__;
let cur = P.voices[0], audio = null, curBtn = null;

document.getElementById('meta').innerHTML = [
  ['底模','ft-mixed-lr2e5-avgE-e12run-0820'],
  ['LoRA','r=32 α=32 · LM+DiT · val 0.922'],
  ['題目','語料真實句 · 33 句 × 3 版本'],
].map(([k,v]) => `${k} <b>${v}</b>`).join('');

const tabs = document.getElementById('tabs');
P.voices.forEach(v => {
  const b = document.createElement('button');
  b.className = 'tab'; b.type = 'button'; b.role = 'tab';
  b.textContent = P.labels[v] || v;
  b.setAttribute('aria-selected', v === cur);
  b.onclick = () => {
    cur = v; stop(); render();
    [...tabs.children].forEach(c => c.setAttribute('aria-selected', c === b));
  };
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
  audio.onended = () => { btn.dataset.playing='0'; curBtn=null; audio=null; };
  audio.play();
}
const ICON = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1l9 5-9 5z"/></svg>';

function render() {
  const host = document.getElementById('body');
  host.innerHTML = '';
  for (const [g, rows] of Object.entries(P.data[cur])) {
    if (!rows.length) continue;
    const [name, note] = P.groups[g];
    const sec = document.createElement('section');
    sec.innerHTML = `<div class="ghead"><span class="gname">${name}</span><span class="gnote">${note}</span></div>`;
    rows.forEach((r, i) => {
      const row = document.createElement('div');
      row.className = 'row';
      row.innerHTML = `<div>
        <div class="line"><span class="n">${String(i+1).padStart(2,'0')}</span>${r.marked}${
          r.seen ? '' : '<span class="tag">未訓練</span>'}${
          r.bad ? `<span class="tag warn" title="原始錄音切點有問題：靜音 ${Math.round(r.sil*100)}%、${r.cps} 字/秒">切點異常</span>` : ''}${
          r.misaligned ? `<span class="tag warn" title="ep${r.ep} 的字幕時間軸整段錯位，原始錄音內容與逐字稿對不上 —— 這是資料問題，不是模型問題">ep${r.ep} 錯位</span>` : ''}</div>
        <div class="dur">真人 ${r.d_orig.toFixed(2)}s · LoRA ${r.d_lora.toFixed(2)}s · 底模 ${r.d_base.toFixed(2)}s</div>
      </div>
      <div class="btns">
        <button class="play orig" type="button">${ICON} 真人</button>
        <button class="play lora" type="button">${ICON} LoRA</button>
        <button class="play base" type="button">${ICON} 底模</button>
      </div>`;
      const [bo, bl, bb] = row.querySelectorAll('.play');
      bo.onclick = () => play(r.orig, bo);
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

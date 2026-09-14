#!/usr/bin/env python3
"""把多個 epoch 的驗收音檔打包成單一 HTML 聽辨頁。

台語的 val loss 與語音品質脫鉤（docs/model_registry.json 的 _comment 記過
同樣的教訓：val 最低的 ep2 完全不成句，val 較高的 ep12 骨架正確），所以
挑 checkpoint 只能靠耳朵。這頁把同一句在不同 epoch 的輸出並排，加上真人
原始錄音當標準答案，讓人直接判斷哪個 epoch 真的學到東西。
"""

import argparse
import base64
import html
import json
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

GROUP_META = {
    "terms": ("新詞", "舊語料 0 次、新集數數十次 — 底模沒學過"),
    "chars": ("生字", "整個語料庫只出現 1–4 次 — 樣本嚴重不足"),
    "ctrl": ("對照", "不含新詞新字，各版本應該接近 — 你的耳朵基準線"),
}

VOICE_LABEL = {
    "cosy-young-female-01": "青年女聲",
    "cosy-young-male-01": "青年男聲",
}

# 互相關偵測出時間軸錯位的集數；這些句子的原始錄音與逐字稿對不上。
BAD_EPISODES = {81, 196, 241, 243, 245, 246, 255, 257, 258, 259}

STEPS_PER_EPOCH = 1402


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def mp3_data_uri(path: Path) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
             "-codec:a", "libmp3lame", "-b:a", "56k", "-ar", "24000", tmp.name],
            check=True,
        )
        return "data:audio/mpeg;base64," + base64.b64encode(
            Path(tmp.name).read_bytes()).decode()


def mark(text: str, term: str) -> str:
    esc = html.escape(text)
    return esc.replace(term, f"<mark>{term}</mark>", 1) if term else esc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-dir", type=Path, default=REPO / "scratch/epochs")
    ap.add_argument("--cases", type=Path, default=REPO / "scratch/real_cases.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", nargs="*", help="只收這些 step 目錄；省略則全收")
    args = ap.parse_args()

    cases = json.loads(args.cases.read_text())
    step_dirs = sorted(
        d for d in args.epochs_dir.glob("step_*")
        if d.is_dir() and (not args.steps or d.name in args.steps)
    )
    # 每個 step 至少要有完整的 198 檔才納入，半成品會讓對照缺格
    step_dirs = [d for d in step_dirs if len(list(d.rglob("*.wav"))) >= 180]
    if not step_dirs:
        sys.exit("找不到完整的 step 目錄")

    voices = sorted({p.name for d in step_dirs for p in d.iterdir() if p.is_dir()})
    epochs = [(d, int(d.name.split("_")[1]) // STEPS_PER_EPOCH) for d in step_dirs]

    data = {}
    for voice in voices:
        data[voice] = {}
        for group, rows in cases.items():
            out_rows = []
            for i, r in enumerate(rows, 1):
                ref = step_dirs[0] / voice / group
                orig, base = ref / f"{i:02d}_orig.wav", ref / f"{i:02d}_base.wav"
                if not (orig.exists() and base.exists()):
                    continue
                ep = r.get("episode")
                # 錯位集數的真人錄音與逐字稿對不上，沒有標準答案就無從判斷
                # 對錯，整句略過（刃／碘／翩三個生字因此沒有樣本可測）。
                if ep in BAD_EPISODES:
                    continue
                item = {
                    "term": r["term"], "marked": mark(r["text"], r["term"]),
                    "seen": r["seen_in_train"], "ep": ep,
                    "orig": mp3_data_uri(orig), "d_orig": round(wav_seconds(orig), 2),
                    "base": mp3_data_uri(base), "d_base": round(wav_seconds(base), 2),
                    "loras": {},
                }
                for d, epoch in epochs:
                    wav = d / voice / group / f"{i:02d}_lora.wav"
                    if wav.exists():
                        item["loras"][str(epoch)] = {
                            "src": mp3_data_uri(wav),
                            "dur": round(wav_seconds(wav), 2),
                        }
                out_rows.append(item)
                print(f"  {voice}/{group}/{i:02d}", file=sys.stderr)
            data[voice][group] = out_rows

    payload = json.dumps({
        "voices": voices, "labels": VOICE_LABEL, "groups": GROUP_META,
        "epochs": [e for _, e in epochs], "data": data,
    }, ensure_ascii=False)

    args.out.write_text(TEMPLATE.replace("__PAYLOAD__", payload), encoding="utf-8")
    n = sum(len(g) for v in data.values() for g in v.values())
    print(f"\n{args.out} ({n} 句 × {2+len(epochs)} 版本, "
          f"{args.out.stat().st_size/1024/1024:.1f} MB)")


TEMPLATE = r"""<title>台語 LoRA Epoch 聽辨</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root {
  --ink:#12262a; --ground:#f7f4ee; --surface:#fffdf9; --line:#ddd6ca;
  --muted:#6f7f80; --orig:#3a6ea5; --base-c:#c2701c; --mark:#fbe6a2;
  --warn:#b4541c; --e1:#0f8f70; --e2:#7b5ea7;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ink:#e8e4db; --ground:#0d1b1e; --surface:#142529; --line:#263a3e;
    --muted:#859899; --orig:#7aaede; --base-c:#f2a44f; --mark:#4a3d16;
    --warn:#f2a44f; --e1:#35d6a4; --e2:#b49ae0;
  }
}
:root[data-theme="dark"] {
  --ink:#e8e4db; --ground:#0d1b1e; --surface:#142529; --line:#263a3e;
  --muted:#859899; --orig:#7aaede; --base-c:#f2a44f; --mark:#4a3d16;
  --warn:#f2a44f; --e1:#35d6a4; --e2:#b49ae0;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"Noto Serif TC",ui-serif,Georgia,serif;line-height:1.6;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:70rem;margin:0 auto;padding:3rem 1.25rem 6rem}
h1{font-size:clamp(1.75rem,4vw,2.5rem);font-weight:700;margin:0 0 .5rem;text-wrap:balance}
.sub{color:var(--muted);font-size:.95rem;max-width:50rem;margin:0}
.meta{margin-top:1.25rem;display:flex;flex-wrap:wrap;gap:.5rem 1.5rem;
  font-family:"JetBrains Mono",monospace;font-size:.78rem;color:var(--muted)}
.meta b{color:var(--ink);font-weight:500}
.tabs{display:flex;gap:.5rem;margin:2rem 0 1.25rem;flex-wrap:wrap}
.tab{font:inherit;font-size:.9rem;cursor:pointer;padding:.5rem 1.1rem;
  border-radius:999px;border:1px solid var(--line);background:var(--surface);color:var(--muted)}
.tab[aria-selected="true"]{color:var(--ink);border-color:var(--ink)}
.tab:focus-visible{outline:2px solid var(--e1);outline-offset:2px}
.legend{display:flex;gap:1.25rem;flex-wrap:wrap;align-items:center;
  font-family:"JetBrains Mono",monospace;font-size:.75rem;color:var(--muted);margin-bottom:2rem}
.dot{display:inline-block;width:.6rem;height:.6rem;border-radius:50%;margin-right:.4rem;vertical-align:-1px}
section{margin-bottom:3rem}
.ghead{position:sticky;top:0;z-index:2;background:var(--ground);padding:.75rem 0 .6rem;
  border-bottom:1px solid var(--line);margin-bottom:.25rem;
  display:flex;align-items:baseline;gap:.75rem;flex-wrap:wrap}
.gname{font-size:1.05rem;font-weight:600}
.gnote{font-size:.82rem;color:var(--muted)}
.row{display:grid;grid-template-columns:1fr auto;gap:.75rem 1.5rem;align-items:center;
  padding:.9rem 0;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:none}
.line{font-size:1.08rem}
.n{font-family:"JetBrains Mono",monospace;font-size:.72rem;color:var(--muted);margin-right:.6rem}
mark{background:var(--mark);color:inherit;padding:0 .1em;border-radius:2px}
.tag{font-family:"JetBrains Mono",monospace;font-size:.66rem;border:1px solid var(--line);
  padding:.05rem .4rem;border-radius:999px;color:var(--muted);margin-left:.5rem}
.tag.warn{color:var(--warn);border-color:var(--warn)}
.dur{font-family:"JetBrains Mono",monospace;font-size:.72rem;color:var(--muted);
  margin-top:.25rem;font-variant-numeric:tabular-nums}
.btns{display:flex;gap:.45rem;flex-wrap:wrap}
.play{font:inherit;font-size:.8rem;cursor:pointer;white-space:nowrap;
  padding:.4rem .8rem;border-radius:.4rem;border:1px solid currentColor;background:transparent;
  display:inline-flex;align-items:center;gap:.35rem}
.play.orig{color:var(--orig)} .play.base{color:var(--base-c)}
.play.ep0{color:var(--e1)} .play.ep1{color:var(--e2)}
.play:hover{background:color-mix(in srgb,currentColor 12%,transparent)}
.play:focus-visible{outline:2px solid currentColor;outline-offset:2px}
.play[data-playing="1"]{background:color-mix(in srgb,currentColor 22%,transparent)}
.play svg{width:.65rem;height:.65rem;fill:currentColor}
@media (max-width:44rem){.row{grid-template-columns:1fr}.btns{justify-content:flex-start}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
  <header>
    <h1>台語 LoRA Epoch 聽辨</h1>
    <p class="sub">同一句話的多個版本：<b>真人原始錄音</b>是標準答案，<b>底模</b>是未微調的 0820，其餘是不同 epoch 的 LoRA。台語的 val loss 與語音品質脫鉤，挑 checkpoint 只能靠耳朵 —— 這頁讓你直接比較。<br>真人錄音與逐字稿對不上的句子（字幕時間軸錯位）已排除。</p>
    <div class="meta" id="meta"></div>
  </header>
  <div class="tabs" id="tabs" role="tablist"></div>
  <div class="legend" id="legend"></div>
  <div id="body"></div>
</div>

<script>
const P = __PAYLOAD__;
let cur = P.voices[0], audio = null, curBtn = null;

document.getElementById('meta').innerHTML = [
  ['底模','ft-mixed-lr2e5-avgE-e12run-0820'],
  ['資料','2805 句 / 1.66h（已剔除時間軸錯位）'],
  ['Epoch', P.epochs.join(' · ')],
].map(([k,v]) => `${k} <b>${v}</b>`).join('');

const EPC = ['ep0','ep1','ep2','ep3'];
document.getElementById('legend').innerHTML = [
  `<span><span class="dot" style="background:var(--orig)"></span>真人錄音</span>`,
  `<span><span class="dot" style="background:var(--base-c)"></span>底模</span>`,
  ...P.epochs.map((e,i) => `<span><span class="dot" style="background:var(--${EPC[i]==='ep0'?'e1':'e2'})"></span>epoch ${e}</span>`),
  `<span>黃底 = 要聽的新詞或生字</span>`,
].join('');

const tabs = document.getElementById('tabs');
P.voices.forEach(v => {
  const b = document.createElement('button');
  b.className='tab'; b.type='button'; b.role='tab';
  b.textContent = P.labels[v] || v;
  b.setAttribute('aria-selected', v === cur);
  b.onclick = () => { cur=v; stop(); render();
    [...tabs.children].forEach(c => c.setAttribute('aria-selected', c===b)); };
  tabs.appendChild(b);
});

function stop(){ if(audio){audio.pause();audio=null;} if(curBtn){curBtn.dataset.playing='0';curBtn=null;} }
function play(src, btn){
  const same = curBtn===btn; stop(); if(same) return;
  audio=new Audio(src); btn.dataset.playing='1'; curBtn=btn;
  audio.onended=()=>{btn.dataset.playing='0';curBtn=null;audio=null;};
  audio.play();
}
const ICON='<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1l9 5-9 5z"/></svg>';

function render(){
  const host=document.getElementById('body'); host.innerHTML='';
  for(const [g,rows] of Object.entries(P.data[cur])){
    if(!rows.length) continue;
    const [name,note]=P.groups[g];
    const sec=document.createElement('section');
    sec.innerHTML=`<div class="ghead"><span class="gname">${name}</span><span class="gnote">${note}</span></div>`;
    rows.forEach((r,i)=>{
      const durs=[`真人 ${r.d_orig.toFixed(2)}s`,`底模 ${r.d_base.toFixed(2)}s`,
        ...P.epochs.filter(e=>r.loras[String(e)]).map(e=>`ep${e} ${r.loras[String(e)].dur.toFixed(2)}s`)].join(' · ');
      const row=document.createElement('div'); row.className='row';
      row.innerHTML=`<div>
          <div class="line"><span class="n">${String(i+1).padStart(2,'0')}</span>${r.marked}${
            r.seen?'':'<span class="tag">未訓練</span>'}</div>
          <div class="dur">${durs}</div>
        </div>
        <div class="btns">
          <button class="play orig" type="button">${ICON} 真人</button>
          <button class="play base" type="button">${ICON} 底模</button>
          ${P.epochs.filter(e=>r.loras[String(e)]).map((e,j)=>
            `<button class="play ${EPC[j]}" type="button" data-ep="${e}">${ICON} ep${e}</button>`).join('')}
        </div>`;
      const btns=[...row.querySelectorAll('.play')];
      btns[0].onclick=()=>play(r.orig,btns[0]);
      btns[1].onclick=()=>play(r.base,btns[1]);
      btns.slice(2).forEach(b=>{ b.onclick=()=>play(r.loras[String(b.dataset.ep)].src,b); });
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

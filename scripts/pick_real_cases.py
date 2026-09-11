#!/usr/bin/env python3
"""從訓練/驗證 manifest 挑出含新詞、新字的真實句子當驗收題目。

比起自編句子，真實句子帶原始錄音，可做「真人 / LoRA / 底模」三方對照 ——
能判斷 LoRA 念得對不對，而不只是跟底模不一樣。
"""

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TERMS = [
    "刀疤老五", "烏鴉老大", "劉經理", "桃花窟", "痱子粉",
    "風水師", "正龍爸", "白馬里", "珍珍", "三刀六眼",
]
NOVEL_CHARS = "倌刃劊喋嗝憫戾殯猩碘祢笅翩舟艇茁荊蓽褥襟頰飩餒餛"

# 太短的句子聽不出東西，太長的推論慢又難比對。
MIN_SEC, MAX_SEC = 1.5, 5.0


def load(name):
    p = REPO / "dataset/manifests" / name
    return [json.loads(l) for l in p.open()]


def usable(r):
    return MIN_SEC <= r.get("duration", 0) <= MAX_SEC


def main():
    rng = random.Random(20260909)
    train = load("lora_novel_train.jsonl")
    val = load("lora_novel_val.jsonl")

    cases = {}

    # terms: 每個新詞挑一句，優先用 val（模型沒背過）
    rows = []
    for t in TERMS:
        pool = [r for r in val if t in r["text"] and usable(r)]
        seen = False
        if not pool:
            pool = [r for r in train if t in r["text"] and usable(r)]
            seen = True
        if pool:
            r = dict(rng.choice(pool))
            r["term"] = t
            r["seen_in_train"] = seen
            rows.append(r)
    cases["terms"] = rows

    # chars: 生字每字一句，全部來自 train（val 沒有）
    rows = []
    for c in NOVEL_CHARS:
        pool = [r for r in train if c in r["text"] and usable(r)]
        if pool:
            r = dict(rng.choice(pool))
            r["term"] = c
            r["seen_in_train"] = True
            rows.append(r)
    cases["chars"] = rows

    # ctrl: 不含任何新詞新字的普通句，從 val 挑（基準線）
    plain = [
        r for r in val
        if usable(r)
        and not any(t in r["text"] for t in TERMS)
        and not any(c in r["text"] for c in NOVEL_CHARS)
    ]
    rows = []
    for r in rng.sample(plain, min(5, len(plain))):
        r = dict(r)
        r["term"] = ""
        r["seen_in_train"] = False
        rows.append(r)
    cases["ctrl"] = rows

    out = REPO / "scratch" / "real_cases.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=2))

    for g, rows in cases.items():
        n_val = sum(1 for r in rows if not r["seen_in_train"])
        print(f"{g}: {len(rows)} 句（{n_val} 句模型沒背過）")
        for r in rows[:3]:
            mark = "" if r["seen_in_train"] else " [val]"
            print(f"   {r['term'] or '—':6} {r['text'][:26]}  {r['duration']:.1f}s{mark}")
    print(f"\n{out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""把 eval_epochs.sh 產出的音檔整理成 score_audio.py 的輸入格式。

Breeze-ASR-360 的 score_audio.py 吃 {"text", "audio"} 配對，其餘欄位原樣
帶出當分組鍵。這裡把 checkpoint 與組別（terms/chars/ctrl）帶上，彙整時
就能按 checkpoint 分列、按組別分開看。

只取 lora 版本 —— orig 是真人錄音、base 是底模，那兩個不隨 checkpoint 變。
底模當基準線時另外跑一次即可。
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-dir", type=Path, default=REPO / "scratch/epochs")
    ap.add_argument("--cases", type=Path, default=REPO / "scratch/real_cases.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", nargs="*", default=["lora"],
                    help="要評分的版本；加 base 可一併納入底模基準線")
    args = ap.parse_args()

    cases = json.loads(args.cases.read_text())
    rows = []
    for ck_dir in sorted(args.epochs_dir.glob("step_*")):
        step = int(ck_dir.name.split("_")[1])
        for voice_dir in sorted(p for p in ck_dir.iterdir() if p.is_dir()):
            for group, items in cases.items():
                for i, r in enumerate(items, 1):
                    for var in args.variants:
                        wav = voice_dir / group / f"{i:02d}_{var}.wav"
                        if not wav.exists():
                            continue
                        rows.append({
                            "text": r["text"],
                            "audio": str(wav),
                            # score_audio.py 會把這些原樣帶出，供彙整分組
                            "checkpoint": f"{ck_dir.name}:{var}",
                            "step": step,
                            "group": group,
                            "voice": voice_dir.name,
                            "term": r.get("term", ""),
                        })

    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    from collections import Counter
    c = Counter(r["checkpoint"] for r in rows)
    print(f"{len(rows)} 筆 -> {args.out}")
    for k, n in sorted(c.items()):
        print(f"  {k:<26} {n}")


if __name__ == "__main__":
    main()

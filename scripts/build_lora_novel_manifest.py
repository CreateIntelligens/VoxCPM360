#!/usr/bin/env python3
"""把 lora_novel 的來源 jsonl 轉成 VoxCPM2 訓練用 manifest。

來源 jsonl（/mnt/nas/dataset202607_1/lora_novel/*.jsonl）只帶 corpus 內的
相對 `audio` 路徑、且沒有 `ref_audio`。訓練 loader 以執行時 CWD 解析相對
路徑，因此這裡一律輸出絕對路徑；ref_audio 則從同語者的其他語句挑一筆補上，
對齊底模 ft-mixed-* 的條件式多語者訓練條件。
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

TAI8_ROOT = Path("/mnt/nas/dataset202607_1/tai8")
PROFILE_16K = TAI8_ROOT / "profiles" / "16k_mono"

# ref_audio 太短的話 prompt 資訊量不足，太長則吃 max_batch_tokens 預算。
REF_MIN_SEC = 1.5
REF_MAX_SEC = 8.0


def to_16k_path(rec: dict) -> Path | None:
    """segments/drama1/1314/108/X.wav -> profiles/16k_mono/drama1_1314/108/X.wav

    來源的第二層是 drama 內的 speaker 編號，16k profile 則把它併進
    `drama1_1314` 這種單層目錄名，正好等於 manifest 的 speaker_id。
    """
    parts = Path(rec["audio"]).parts
    if len(parts) < 4:
        return None
    speaker = rec.get("speaker_id")
    if not speaker:
        return None
    return PROFILE_16K / speaker / parts[-2] / parts[-1]


def pick_refs(
    records: list[dict], rng: random.Random, pool: list[dict] | None = None
) -> list[dict]:
    """為每筆語句挑同語者、不同音檔的 ref_audio。

    pool 給定時改由該語料池挑 ref。小型分層檔（如 tier1 只有 61 筆）自己的
    語者配對不足，用整個 tai8 corpus 當池才留得住那些關鍵生字語句。
    """
    by_speaker = defaultdict(list)
    for rec in pool if pool is not None else records:
        by_speaker[rec["speaker_id"]].append(rec)

    out = []
    dropped_no_peer = 0
    for rec in records:
        peers = [
            p
            for p in by_speaker[rec["speaker_id"]]
            if p["audio"] != rec["audio"] and REF_MIN_SEC <= p["duration"] <= REF_MAX_SEC
        ]
        if not peers:
            dropped_no_peer += 1
            continue
        ref = rng.choice(peers)
        out.append(
            {
                "audio": rec["audio"],
                "text": rec["text"],
                "duration": rec["duration"],
                "ref_audio": ref["audio"],
                "ref_duration": ref["duration"],
                "speaker_id": rec["speaker_id"],
                "episode": rec.get("episode"),
                "utterance_id": rec.get("utterance_id"),
                "dataset_id": 0,
            }
        )
    if dropped_no_peer:
        print(f"  跳過 {dropped_no_peer} 筆：該語者沒有合用的 ref_audio")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="來源 jsonl")
    ap.add_argument("--out-dir", type=Path, default=Path("dataset/manifests"))
    ap.add_argument("--name", required=True, help="輸出檔名前綴")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--ref-pool",
        type=Path,
        help="改由這份 jsonl 挑 ref_audio（小型分層檔建議指向 tai8/metadata.jsonl）",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)

    def load(path: Path) -> tuple[list[dict], int]:
        rows, missing = [], 0
        for line in path.open():
            rec = json.loads(line)
            resolved = to_16k_path(rec)
            if resolved is None or not resolved.exists():
                missing += 1
                continue
            rec["audio"] = str(resolved)
            rows.append(rec)
        return rows, missing

    records, missing = load(args.source)
    print(f"{args.source.name}: 讀入 {len(records)} 筆（{missing} 筆找不到 16k 檔）")

    pool = None
    if args.ref_pool:
        pool, pool_missing = load(args.ref_pool)
        print(f"  ref 池 {args.ref_pool.name}: {len(pool)} 筆（{pool_missing} 筆缺檔）")

    records = pick_refs(records, rng, pool)
    rng.shuffle(records)

    n_val = max(1, int(len(records) * args.val_ratio))
    val, train = records[:n_val], records[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train), ("val", val)):
        dest = args.out_dir / f"{args.name}_{split}.jsonl"
        with dest.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        total = sum(r["duration"] for r in rows)
        print(f"  {dest}: {len(rows)} 筆, {total/3600:.2f} h")


if __name__ == "__main__":
    main()

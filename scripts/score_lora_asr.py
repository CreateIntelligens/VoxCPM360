#!/usr/bin/env python3
"""用 Breeze-ASR 替 LoRA checkpoint 的試聽音檔評分並排名。

台語的 val loss 跟「字念對沒」對不上（docs/model_registry.json 記過：val
最低的 ep2 完全不成句），所以挑 checkpoint 改看 ASR：把每個版本念的句子
辨識回文字，跟原文比 CER 與目標用語命中數，先篩出前幾名再交給人耳。

輸入是 eval 腳本產出的目錄：<out>/index.json 加上 <out>/<版本>/NN.wav。
只用標準函式庫，可直接在主機上跑（ASR 服務在主機的 :8801）。

Breeze 是「台語語音 → 華語漢字」，分不出念的是台語還是華語：模型若
飄成華語，分數反而變好。所以 CER 為 0 的句子會另外列出來要人耳確認。
"""

import argparse
import json
import re
import sys
import urllib.request
import uuid
from pathlib import Path

ASR_BATCH_LIMIT = 16
# 「一般句子」比底模差超過這個量就視為 LoRA 把其他聲音帶歪。定版原音
# 本身的 CER 約 0.12，0.03 以內的差距落在 ASR 雜訊裡。
CTRL_REGRESSION_MARGIN = 0.03
TARGET_GROUPS = ("novel", "holdout", "trained")


def normalize(text: str) -> str:
    text = text.replace("檯", "台").replace("臺", "台")
    return re.sub(r"[\W_]+", "", text).lower()


def cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    row = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        prev, row[0] = row[0], i
        for j, hc in enumerate(h, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (rc != hc))
    return row[len(h)] / max(1, len(r))


def term_hits(terms: list[str], hyp: str) -> int:
    rest = normalize(hyp)
    hits = 0
    for term in terms:
        if term in rest:
            hits += 1
            rest = rest.replace(term, "", 1)
    return hits


def transcribe(url: str, files: list[Path]) -> list[str]:
    texts: list[str] = []
    for start in range(0, len(files), ASR_BATCH_LIMIT):
        boundary = uuid.uuid4().hex
        body = b""
        for f in files[start : start + ASR_BATCH_LIMIT]:
            body += (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
                f"filename=\"{f.name}\"\r\nContent-Type: audio/wav\r\n\r\n"
            ).encode() + f.read_bytes() + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{url}/transcribe/batch",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            results = json.load(resp)
        for item in results:
            if item.get("error"):
                raise RuntimeError(f"ASR 失敗：{item['error']}")
            texts.append(item["text"])
    return texts


def summarize(rows: list[dict]) -> dict:
    out = {}
    for group in {r["group"] for r in rows}:
        sel = [r for r in rows if r["group"] == group]
        out[group] = {
            "cer": sum(r["cer"] for r in sel) / len(sel),
            "hits": sum(r["hits"] for r in sel),
            "terms": sum(r["terms"] for r in sel),
            "n": len(sel),
        }
    target = [r for r in rows if r["group"] in TARGET_GROUPS]
    out["target"] = {
        "cer": sum(r["cer"] for r in target) / max(1, len(target)),
        "hits": sum(r["hits"] for r in target),
        "terms": sum(r["terms"] for r in target),
        "n": len(target),
    }
    return out


def rank(summaries: dict[str, dict]) -> list[dict]:
    """依目標組（新句子＋題目）的 CER 排序；一般句子退步的版本排到最後。"""
    base_ctrl = summaries.get("base", {}).get("ctrl", {}).get("cer")
    ranked = []
    for tag, s in summaries.items():
        if tag == "base":
            continue
        ctrl = s.get("ctrl", {}).get("cer")
        regressed = (
            base_ctrl is not None
            and ctrl is not None
            and ctrl > base_ctrl + CTRL_REGRESSION_MARGIN
        )
        ranked.append({"tag": tag, "regressed": regressed, **s})
    ranked.sort(key=lambda x: (x["regressed"], x["target"]["cer"], -x["target"]["hits"]))
    return ranked


def fmt_group(s: dict | None) -> str:
    if not s:
        return "—"
    hits = f" ({s['hits']}/{s['terms']})" if s["terms"] else ""
    return f"{s['cer']:.3f}{hits}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", type=Path, help="eval 腳本的 --out-dir")
    ap.add_argument("--asr-url", default="http://localhost:8801")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    cases = json.loads((args.eval_dir / "index.json").read_text())
    cache_path = args.eval_dir / "asr_transcripts.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}

    tags = sorted(
        p.name for p in args.eval_dir.iterdir()
        if p.is_dir() and all((p / f"{c['n']:02d}.wav").is_file() for c in cases)
    )
    if "base" not in tags:
        sys.exit("缺少底模（base）的音檔，無法判斷一般句子有沒有退步")

    summaries, details = {}, {}
    for tag in tags:
        if tag not in cache:
            files = [args.eval_dir / tag / f"{c['n']:02d}.wav" for c in cases]
            cache[tag] = transcribe(args.asr_url, files)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
            print(f"辨識完成 {tag}", file=sys.stderr, flush=True)
        rows = []
        for c, hyp in zip(cases, cache[tag]):
            terms = c.get("terms", [])
            rows.append({
                "n": c["n"], "group": c["group"], "voice": c["voice"],
                "text": c["text"], "hyp": hyp, "cer": cer(c["text"], hyp),
                "terms": len(terms), "hits": term_hits(terms, hyp),
            })
        details[tag] = rows
        summaries[tag] = summarize(rows)

    ranked = rank(summaries)
    (args.eval_dir / "asr_scores.json").write_text(json.dumps(
        {"summaries": summaries, "ranking": [r["tag"] for r in ranked], "details": details},
        ensure_ascii=False, indent=1,
    ))

    groups = ("target", "novel", "holdout", "trained", "ctrl")
    lines = [
        "| 排名 | 版本 | " + " | ".join(groups) + " |",
        "|---|---|" + "---|" * len(groups),
        "| — | base | " + " | ".join(fmt_group(summaries["base"].get(g)) for g in groups) + " |",
    ]
    for i, r in enumerate(ranked, 1):
        flag = "（一般句子退步）" if r["regressed"] else ""
        lines.append(
            f"| {i} | {r['tag']}{flag} | "
            + " | ".join(fmt_group(r.get(g)) for g in groups) + " |"
        )
    report = ["# LoRA checkpoint ASR 排名", "",
              "CER 越低越好；括號是目標用語命中數。target = novel + holdout + trained。", ""]
    report += lines
    report += ["", f"## 前 {args.top} 名要人耳確認的句子", "",
               "CER 為 0 的句子可能是飄成華語，請確認還是台語。", ""]
    for r in [x for x in ranked if not x["regressed"]][: args.top]:
        exact = [d for d in details[r["tag"]] if d["cer"] == 0 and d["group"] in TARGET_GROUPS]
        report.append(f"- **{r['tag']}**：" + (
            "、".join(f"{d['n']:02d} {d['text']}" for d in exact) if exact else "沒有 CER 0 的句子"
        ))
    text = "\n".join(report) + "\n"
    (args.eval_dir / "asr_ranking.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

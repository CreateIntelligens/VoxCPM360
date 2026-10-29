#!/usr/bin/env python3
"""VH-387 Hayley LoRA 試聽音檔：底模 0820 與多個 checkpoint 逐句對照。

底模只載入一次，checkpoint 之間用 load_lora() 換權重。參考音的用法比照
線上 app.py 的 voice cloning（prompt_wav + prompt_text + reference_wav 都用同一
個 preset 檔）、步數同 VOXCPM_INFERENCE_TIMESTEPS，聽到的就是上線後的效果。

測試句分組：
  holdout — B 版沒進訓練的 4 題，看學到的字詞能不能用在沒見過的句子
  trained — 有進訓練的題目，確認確實學到
  novel   — 不在 40 題、但用到同樣醫院用語的新句子，最能看出是否過擬合
  ctrl    — 跟醫院無關的普通句，用其他聲音念，確認沒被帶歪
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

REPO = Path(__file__).resolve().parent.parent
REF_DIR = REPO / "assets" / "default_reference"
CSV_PATH = REPO / "dataset" / "vh387" / "定版對照_40題.csv"

# 共用逐字稿，定義在 gateway/presets.py 的 _COSY_PROMPT_TEXT。
COSY_PROMPT_TEXT = (
    "你好，歡迎使用創造智能台語生成服務，很高興為你服務，請輸入想要生成的文本內容"
)
HOLDOUT_IDS = (1, 11, 21, 31)
TRAINED_IDS = (2, 6, 17, 29)
NOVEL = [
    "請先到一樓櫃台掛號，記得帶健保卡。",
    "批價完成以後，再到領藥窗口拿藥。",
    "檢查報告大概三天後就可以拿了。",
    "抽血前八個小時要空腹，不能吃東西。",
    "住院的時候要準備換洗衣服和盥洗用品。",
    "出院前護理師會跟你說明藥要怎麼吃。",
    "哺乳室在二樓電梯旁邊。",
    "有重大傷病卡的話，部分負擔可以免繳。",
]
CTRL = [
    "你趕快打電話叫他不要來。",
    "我跟他的感情真的很好。",
    "這件事情我會處理好的。",
]
# ASR 評分時檢查的醫院用語；每句只算原文裡有出現的詞。長詞放前面，
# 避免「健保」在「健保卡」已計分後又重複計一次。
TERMS = (
    "重大傷病卡", "部分負擔", "檢查報告", "健康檢查", "健保卡", "哺乳室",
    "停車場", "護理師", "掛號", "批價", "領藥", "櫃台", "官網", "收據",
    "住院", "出院", "病房", "空腹", "健保", "補助", "疫苗", "盥洗", "電梯",
    "抽血", "手術", "預約", "退費",
)
HAYLEY = "cosy-young-female-02"
OTHER_VOICES = ("cosy-young-male-01", "cosy-senior-female-01")


def load_questions() -> dict[int, str]:
    rows = csv.DictReader(CSV_PATH.open(encoding="utf-8-sig"))
    return {int(r["新編號"]): r["送進TTS的文字"].strip() for r in rows}


def case_terms(text: str) -> list[str]:
    found, rest = [], text.replace("檯", "台")
    for term in TERMS:
        if term in rest:
            found.append(term)
            rest = rest.replace(term, "")
    return found


def build_cases() -> list[dict]:
    q = load_questions()
    raw = [(HAYLEY, "holdout", q[i]) for i in HOLDOUT_IDS]
    raw += [(HAYLEY, "trained", q[i]) for i in TRAINED_IDS]
    raw += [(HAYLEY, "novel", t) for t in NOVEL]
    for v in OTHER_VOICES:
        raw += [(v, "novel", t) for t in NOVEL[:3]]
        raw += [(v, "ctrl", t) for t in CTRL]
    return [
        {"n": n, "voice": v, "group": g, "text": t, "terms": case_terms(t)}
        for n, (v, g, t) in enumerate(raw, 1)
    ]


def ckpt_tag(ckpt: str) -> str:
    p = Path(ckpt)
    # 訓練目錄是 run/step_xxx；給服務用的扁平目錄（vh387-orig-ep15）本身就是名字。
    return f"{p.parent.name}__{p.name}" if p.name.startswith("step_") else p.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="LoRA checkpoint 目錄")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--timesteps", type=int, default=10)
    ap.add_argument("--cfg-value", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=387)
    ap.add_argument("--no-base", action="store_true", help="不生成底模（已生成過的版本本來就會略過）")
    args = ap.parse_args()

    cases = build_cases()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "index.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=1)
    )

    def done(tag: str) -> bool:
        return all((args.out_dir / tag / f"{c['n']:02d}.wav").is_file() for c in cases)

    todo = [ck for ck in args.ckpts if not done(ckpt_tag(ck))]
    need_base = not args.no_base and not done("base")
    if not todo and not need_base:
        print("全部已生成，略過", file=sys.stderr)
        return

    first = json.loads((Path(args.ckpts[0]) / "lora_config.json").read_text())
    model = VoxCPM.from_pretrained(
        hf_model_id=first["base_model"],
        load_denoiser=False,
        optimize=True,
        lora_config=LoRAConfig(**first["lora_config"]),
        lora_weights_path=args.ckpts[0],
    )

    def run(tag: str) -> None:
        for c in cases:
            ref = str(REF_DIR / f"{c['voice']}.mp3")
            # 每句固定 seed，底模與各 checkpoint 只差權重。
            torch.manual_seed(args.seed + c["n"])
            audio = model.generate(
                text=c["text"],
                prompt_wav_path=ref,
                prompt_text=COSY_PROMPT_TEXT,
                reference_wav_path=ref,
                cfg_value=args.cfg_value,
                inference_timesteps=args.timesteps,
            )
            dest = args.out_dir / tag / f"{c['n']:02d}.wav"
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dest), audio, model.tts_model.sample_rate)
        print(f"{tag}: {len(cases)} 句", file=sys.stderr, flush=True)

    if need_base:
        model.tts_model.set_lora_enabled(False)
        run("base")
        model.tts_model.set_lora_enabled(True)
    for ck in todo:
        model.load_lora(ck)
        run(ckpt_tag(ck))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""用語料裡的真實句子做三方對照：原始錄音 / LoRA / 底模。

自編句子只能比出「LoRA 跟底模不一樣」；真實句子帶原始錄音當標準答案，
才能判斷 LoRA 念得對不對。

模型只載入一次，靠 set_lora_enabled() 切換，避免每句重讀 9.2 GB 底模。
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import soundfile as sf

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm2 import LoRAConfig

REPO = Path(__file__).resolve().parent.parent

# 內建參考音共用的逐字稿；空字串會讓 barbet_runtime 整個丟掉參考音，
# 聲音克隆會失效（見 assets/default_reference/README.md）。
COSY_PROMPT_TEXT = (
    "你好，歡迎使用創造智能台語生成服務，很高興為你服務，請輸入想要生成的文本內容"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-ckpt", required=True)
    ap.add_argument("--cases", type=Path, default=REPO / "scratch/real_cases.json")
    ap.add_argument("--out-dir", type=Path, default=REPO / "scratch/lora_real")
    ap.add_argument("--voice", nargs="*", default=["cosy-young-female-01"])
    ap.add_argument("--cfg-value", type=float, default=2.0)
    args = ap.parse_args()

    ckpt = Path(args.lora_ckpt)
    info = json.loads((ckpt / "lora_config.json").read_text())
    lora_cfg = LoRAConfig(**info["lora_config"])

    cases = json.loads(args.cases.read_text())
    ref_dir = REPO / "assets/default_reference"

    print(f"載入底模 {info['base_model']}", file=sys.stderr)
    model = VoxCPM.from_pretrained(
        hf_model_id=info["base_model"],
        load_denoiser=False,
        optimize=True,
        lora_config=lora_cfg,
        lora_weights_path=str(ckpt),
    )
    sr = model.tts_model.sample_rate

    def synth(text, dest, wav, ptext):
        audio = model.generate(
            text=text, prompt_wav_path=wav, prompt_text=ptext,
            cfg_value=args.cfg_value,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), audio, sr)

    total = 0
    for voice in args.voice:
        vwav = str(ref_dir / f"{voice}.mp3")
        if not Path(vwav).exists():
            sys.exit(f"找不到參考音: {vwav}")
        print(f"\n===== {voice} =====")
        for group, rows in cases.items():
            print(f"\n[{group}]")
            for i, r in enumerate(rows, 1):
                d = args.out_dir / voice / group
                d.mkdir(parents=True, exist_ok=True)
                # 原始錄音直接複製，不重新編碼，保留真人發音的原貌
                shutil.copy(r["audio"], d / f"{i:02d}_orig.wav")
                model.tts_model.set_lora_enabled(True)
                synth(r["text"], d / f"{i:02d}_lora.wav", vwav, COSY_PROMPT_TEXT)
                model.tts_model.set_lora_enabled(False)
                synth(r["text"], d / f"{i:02d}_base.wav", vwav, COSY_PROMPT_TEXT)
                total += 1
                print(f"  {i:02d}. [{r['term'] or '—'}] {r['text'][:28]}")

    model.tts_model.set_lora_enabled(True)
    print(f"\n完成 {total} 句 × 3 版本 -> {args.out_dir}")


if __name__ == "__main__":
    main()

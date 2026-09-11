#!/usr/bin/env python3
"""批次生成 LoRA 驗收音檔，每句同時輸出 LoRA 版與原始底模版。

模型只載入一次，靠 set_lora_enabled() 切換開關 —— 逐句呼叫
test_voxcpm_lora_infer.py 會重複載入 9.2 GB 底模，36 句要跑很久。

測試句按「生字在訓練集出現幾次」分組，一聽就知道 LoRA 吃進去多少：
  seen   — 出現 2 次以上，最有機會學到
  once   — 只出現 1 次，單樣本學習的極限
  unseen — 整個 tai8 corpus 都幾乎沒有，對照組
  terms  — 高頻專有名詞（刀疤老五 62 次…），應該最明顯
  ctrl   — 不含生字的普通句，確認音色沒被洗掉

prompt 音檔從訓練 manifest 挑同一語者，避免音色差異干擾判斷。
"""

import argparse
import json
import random
import sys
from pathlib import Path

import soundfile as sf

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

REPO = Path(__file__).resolve().parent.parent

CASES = {
    "seen": [
        "他翩然轉身，衣角掠過門檻。",
        "廟裡擲筊，一正一反是聖笅。",
        "眾人喋喋不休，吵得人心煩。",
        "那件猩紅色的外套很顯眼。",
        "一葉扁舟順流而下。",
        "床上的被褥要拿去曬一曬。",
    ],
    "once": [
        "茶館裡的堂倌招呼客人。",
        "刀刃磨得雪亮。",
        "古時候的劊子手站在刑台上。",
        "他打了個嗝，忍不住笑出來。",
        "看他這樣，我心生憐憫。",
        "他一臉暴戾，誰也不敢靠近。",
        "殯儀館前停著一排車。",
        "傷口要先擦碘酒消毒。",
        "路上長滿荊棘，難以通行。",
        "蓽門圭竇，也能安身立命。",
        "他正了正衣襟，準備上台。",
        "她臉頰泛紅，低頭不語。",
        "晚餐煮了一鍋餛飩。",
        "輸了一場也不必氣餒。",
    ],
    "unseen": [
        "祢的名是聖潔的。",
        "快艇在海面上飛馳。",
        "新芽茁壯地生長。",
    ],
    "terms": [
        "刀疤老五又來收保護費了。",
        "烏鴉老大坐在角落抽菸。",
        "劉經理說這件事他來處理。",
        "桃花窟那邊最近不太平靜。",
        "小孩子身上要擦點痱子粉。",
        "他找了個風水師來看房子。",
        "正龍爸昨天才從醫院回來。",
        "那罐茶葉是別人送的。",
        "這筆生意要兩仟萬才談得成。",
        "白馬里的里長來過好幾次。",
    ],
    "ctrl": [
        "你趕快打電話叫他不要來。",
        "我跟他的感情真的很好。",
        "這件事情我會處理好的。",
    ],
}


# 內建參考音的逐字稿；14 個 cosy-* 檔共用同一句，定義在 gateway/presets.py。
# 逐字稿不能空 —— barbet_runtime.py 會在 prompt_text 為空時整個丟掉參考音，
# 聲音克隆會完全失效（見 assets/default_reference/README.md）。
COSY_PROMPT_TEXT = (
    "你好，歡迎使用創造智能台語生成服務，很高興為你服務，請輸入想要生成的文本內容"
)


def pick_prompt(manifest: Path, rng: random.Random) -> tuple[str, str]:
    """從訓練 manifest 挑一筆當 voice cloning 的 prompt。

    太短的 prompt 音色資訊不足，挑時長適中的。
    """
    rows = [json.loads(l) for l in manifest.open()]
    usable = [r for r in rows if 2.5 <= r.get("duration", 0) <= 6.0]
    pick = rng.choice(usable or rows)
    return pick["audio"], pick["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-ckpt", required=True)
    ap.add_argument("--out-dir", type=Path, default=REPO / "scratch" / "lora_eval")
    ap.add_argument(
        "--manifest", type=Path, default=REPO / "dataset/manifests/lora_novel_train.jsonl"
    )
    ap.add_argument("--groups", nargs="*", default=list(CASES))
    ap.add_argument("--no-clone", action="store_true", help="不用 prompt 音檔")
    ap.add_argument(
        "--voice",
        nargs="*",
        help="assets/default_reference 的 cosy 參考音 id（可給多個，各跑一組）；"
        "省略則從訓練 manifest 隨機挑語者",
    )
    ap.add_argument("--cfg-value", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    ckpt = Path(args.lora_ckpt)
    lora_info = json.loads((ckpt / "lora_config.json").read_text())
    base = lora_info["base_model"]
    lora_cfg = LoRAConfig(**lora_info["lora_config"])

    rng = random.Random(args.seed)
    # voices: [(輸出子目錄名, 參考音路徑, 逐字稿)]
    if args.voice:
        ref_dir = REPO / "assets" / "default_reference"
        voices = [(v, str(ref_dir / f"{v}.mp3"), COSY_PROMPT_TEXT) for v in args.voice]
        for name, path, _ in voices:
            if not Path(path).exists():
                sys.exit(f"找不到參考音: {path}")
            print(f"參考音 {name}: {path}")
    elif args.no_clone:
        voices = [("noclone", None, None)]
    else:
        wav, text = pick_prompt(args.manifest, rng)
        voices = [("manifest", wav, text)]
        print(f"prompt 語者範例: {text}  ({Path(wav).name})")
    print()

    print(f"載入底模 {base}", file=sys.stderr)
    model = VoxCPM.from_pretrained(
        hf_model_id=base,
        load_denoiser=False,
        optimize=True,
        lora_config=lora_cfg,
        lora_weights_path=str(ckpt),
    )

    def synth(text: str, dest: Path, wav: str | None, ptext: str | None) -> None:
        audio = model.generate(
            text=text,
            prompt_wav_path=wav,
            prompt_text=ptext,
            cfg_value=args.cfg_value,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        # tts_model.sample_rate 是輸出取樣率（48k）；audio_vae.sample_rate 是編碼端的 16k。
        sf.write(str(dest), audio, model.tts_model.sample_rate)

    total = 0
    for vname, vwav, vtext in voices:
        print(f"\n===== 參考音 {vname} =====")
        for group in args.groups:
            out_dir = args.out_dir / vname / group
            print(f"\n[{group}]")
            for i, text in enumerate(CASES[group], 1):
                # 同一句先後跑兩次，中間只切 LoRA 開關，其餘條件完全一致。
                model.tts_model.set_lora_enabled(True)
                synth(text, out_dir / f"{i:02d}_lora.wav", vwav, vtext)
                model.tts_model.set_lora_enabled(False)
                synth(text, out_dir / f"{i:02d}_base.wav", vwav, vtext)
                total += 1
                print(f"  {i:02d}. {text}")

    model.tts_model.set_lora_enabled(True)
    print(f"\n完成 {total} 句 × 2 版本，輸出於 {args.out_dir}")
    print("  *_lora.wav = 套用 LoRA   *_base.wav = 原始底模")


if __name__ == "__main__":
    main()

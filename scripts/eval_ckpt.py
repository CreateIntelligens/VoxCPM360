#!/usr/bin/env python3
"""在同一份驗證集上評測多個 LoRA checkpoint，產生可公平比較的 val loss。

存在的理由：訓練期的 val 有兩個問題讓 run 之間無法互比 ——
  1. `validate()` 寫死 `max_val_batches = 10`，只看 10 個 batch，
     結果隨機性大（run3 曲線 ±0.003 的震盪多半來自這裡）。
  2. 每個 run 各自用自己的 val_manifest，run3 的混合驗證集含 22.9% naer
     朗讀語音，比純戲劇對白好預測，低 val 有一部分是「尺變短」而非模型變強。

本腳本固定驗證集、跑完整資料、對每個 checkpoint 重複同一套 forward，
因此輸出的數字才可以並排比較。

用法：
    python eval_ckpt.py --config <yaml> --val-manifest <jsonl> \
        --ckpt <dir> [--ckpt <dir> ...] [--batch-size 16] [--max-batches 0]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from safetensors.torch import load_file  # noqa: E402

from voxcpm.model import VoxCPM2Model, VoxCPMModel  # noqa: E402
from voxcpm.model.voxcpm import LoRAConfig as LoRAConfigV1  # noqa: E402
from voxcpm.model.voxcpm2 import LoRAConfig as LoRAConfigV2  # noqa: E402
from voxcpm.training import (  # noqa: E402
    Accelerator,
    BatchProcessor,
    build_dataloader,
    load_audio_text_datasets,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="訓練用 YAML，取 pretrained_path 與 lora 設定")
    p.add_argument("--val-manifest", required=True, help="評測用 manifest；所有 ckpt 共用同一份")
    p.add_argument("--ckpt", action="append", required=True, help="checkpoint 目錄，可重複指定")
    p.add_argument("--batch-size", type=int, default=0, help="0 表示沿用 config")
    p.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="0 = 跑完整驗證集（預設）。設 N 則只取前 N 個 batch，僅供快速抽驗",
    )
    p.add_argument("--output", default="", help="輸出 JSON 路徑")
    return p.parse_args()


def load_lora_into(model, ckpt_dir, tracker_print):
    """把 checkpoint 的 LoRA 權重灌進已建好的模型。

    只載入 lora_weights.safetensors —— optimizer/scheduler 是續訓才需要的。
    strict=False 是必要的：state_dict 只含 LoRA 分支，base 權重不在裡面。
    """
    path = Path(ckpt_dir) / "lora_weights.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"找不到 LoRA 權重：{path}")

    state = load_file(str(path))
    missing, unexpected = model.load_state_dict(state, strict=False)

    if unexpected:
        raise RuntimeError(
            f"{ckpt_dir}: 有 {len(unexpected)} 個非預期的 key，"
            f"checkpoint 與模型結構不符：{unexpected[:3]}"
        )
    tracker_print(f"  載入 {len(state)} 個 LoRA tensor")
    return len(state)


@torch.no_grad()
def evaluate(model, val_loader, batch_processor, accelerator, lambdas, max_batches):
    """跑完整驗證集，回傳加權後的 total 與各分項 loss。

    刻意複製 train_voxcpm_finetune.validate() 的 forward 與加權方式，
    差別只在不限制 batch 數，以及不做音檔生成。
    """
    model.eval()
    total_losses = []
    sub_losses = defaultdict(list)
    n = 0

    for batch in val_loader:
        if max_batches and n >= max_batches:
            break
        processed = batch_processor(batch)
        with accelerator.autocast(dtype=torch.bfloat16):
            outputs = model(
                processed["text_tokens"],
                processed["text_mask"],
                processed["audio_feats"],
                processed["audio_mask"],
                processed["loss_mask"],
                processed["position_ids"],
                processed["labels"],
                progress=0.0,
                sample_generate=False,
            )
        total = 0.0
        for key, value in outputs.items():
            if key.startswith("loss/"):
                total += lambdas.get(key, 1.0) * value
                sub_losses[key].append(value.detach())
        total_losses.append(total.detach())
        n += 1

        if n % 50 == 0:
            print(f"    ... {n} batches", file=sys.stderr, flush=True)

    if not total_losses:
        raise RuntimeError("驗證集沒有產生任何 batch")

    mean_total = torch.stack(total_losses).mean()
    accelerator.all_reduce(mean_total)
    metrics = {"loss/total": mean_total.item()}
    for key, values in sub_losses.items():
        m = torch.stack(values).mean()
        accelerator.all_reduce(m)
        metrics[key] = m.item()
    metrics["num_batches"] = n
    return metrics


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pretrained_path = cfg["pretrained_path"]
    lora_cfg = cfg.get("lora")
    lambdas = cfg.get("lambdas", {"loss/diff": 1.0, "loss/stop": 1.0})
    sample_rate = cfg.get("sample_rate", 16000)
    batch_size = args.batch_size or cfg.get("batch_size", 16)
    num_workers = cfg.get("num_workers", 4)

    accelerator = Accelerator(amp=True)
    rank0 = accelerator.rank == 0

    def out(msg):
        if rank0:
            print(msg, file=sys.stderr, flush=True)

    out(f"評測 {len(args.ckpt)} 個 checkpoint")
    out(f"驗證集：{args.val_manifest}")
    out(f"batch_size={batch_size}，max_batches={args.max_batches or '完整'}\n")

    with open(os.path.join(pretrained_path, "config.json"), "r", encoding="utf-8") as f:
        arch = json.load(f).get("architecture", "voxcpm").lower()
    model_cls = VoxCPM2Model if arch == "voxcpm2" else VoxCPMModel
    lora_cls = LoRAConfigV2 if arch == "voxcpm2" else LoRAConfigV1

    base_model = model_cls.from_local(
        pretrained_path,
        optimize=False,
        training=True,
        lora_config=lora_cls(**lora_cfg) if lora_cfg else None,
    )
    tokenizer = base_model.text_tokenizer

    _, val_ds = load_audio_text_datasets(
        train_manifest=args.val_manifest,  # 只評測，train 側不會被使用
        val_manifest=args.val_manifest,
        sample_rate=sample_rate,
    )
    if val_ds is None:
        raise RuntimeError(f"無法載入驗證集：{args.val_manifest}")

    def tokenize(batch):
        return {"text_ids": [tokenizer(t) for t in batch["text"]]}

    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    out(f"驗證樣本數：{len(val_ds)}")

    # dataset_cnt 必須涵蓋 manifest 內所有 dataset_id，否則 embedding 會越界。
    # 混合資料 tai8=1／naer=0，純 tai8 則只有 1 種。
    dataset_cnt = (
        int(max(val_ds["dataset_id"])) + 1 if "dataset_id" in val_ds.column_names else 1
    )
    out(f"dataset_cnt={dataset_cnt}\n")

    val_loader = build_dataloader(
        val_ds,
        accelerator=accelerator,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    # batch_processor 需要 audio_vae，故必須在 prepare_model 之前建好 ——
    # 訓練腳本在 prepare 前會 del base_model.audio_vae。
    batch_processor = BatchProcessor(
        config=base_model.config,
        audio_vae=base_model.audio_vae,
        dataset_cnt=dataset_cnt,
        device=accelerator.device,
    )
    del base_model.audio_vae
    model = accelerator.prepare_model(base_model)

    results = {}
    for ckpt in args.ckpt:
        name = Path(ckpt).name
        out(f"[{name}]")
        n_tensors = load_lora_into(accelerator.unwrap(model), ckpt, out)
        metrics = evaluate(
            model, val_loader, batch_processor, accelerator, lambdas, args.max_batches
        )
        metrics["n_lora_tensors"] = n_tensors
        results[name] = metrics
        out(
            f"  total={metrics['loss/total']:.4f}  "
            + "  ".join(
                f"{k.split('/')[-1]}={v:.4f}"
                for k, v in metrics.items()
                if k.startswith("loss/") and k != "loss/total"
            )
            + f"  ({metrics['num_batches']} batches)\n"
        )

    if rank0:
        print("\n" + "=" * 62)
        print(f"驗證集：{args.val_manifest}")
        print(f"樣本數：{len(val_ds)}    batch_size：{batch_size}")
        print("=" * 62)
        print(f"{'checkpoint':<20} {'total':>9} {'diff':>9} {'stop':>9}")
        print("-" * 62)
        for name, m in sorted(results.items(), key=lambda kv: kv[1]["loss/total"]):
            print(
                f"{name:<20} {m['loss/total']:>9.4f} "
                f"{m.get('loss/diff', float('nan')):>9.4f} "
                f"{m.get('loss/stop', float('nan')):>9.4f}"
            )
        print("=" * 62)
        best = min(results.items(), key=lambda kv: kv[1]["loss/total"])
        print(f"最佳：{best[0]}（total={best[1]['loss/total']:.4f}）")

        if args.output:
            payload = {
                "val_manifest": args.val_manifest,
                "num_samples": len(val_ds),
                "batch_size": batch_size,
                "max_batches": args.max_batches,
                "results": results,
            }
            Path(args.output).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"已寫出：{args.output}")


if __name__ == "__main__":
    main()

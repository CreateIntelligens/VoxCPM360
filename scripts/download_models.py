import argparse
import sys
import os

# Ensure src/ is in Python path to import voxcpm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    import voxcpm
except ImportError:
    print("Error: Could not import voxcpm. Ensure PYTHONPATH is set correctly.")
    sys.exit(1)

def download_models(model_id: str):
    print("==================================================")
    print(f"Pre-downloading/Checking TTS Model: {model_id}...")
    print("==================================================")
    try:
        # If model_id is a local directory with required weights, skip CPU loading
        if os.path.isdir(model_id):
            model_dir = os.path.abspath(model_id)
            if (
                os.path.isfile(os.path.join(model_dir, "model.safetensors"))
                and os.path.isfile(os.path.join(model_dir, "config.json"))
            ):
                print(f"Local model checkpoint verified at {model_dir}.")
                return

        # Trigger huggingface/modelscope download by loading on CPU without optimization
        voxcpm.VoxCPM.from_pretrained(
            model_id,
            optimize=False,
            device="cpu"
        )
        print("TTS model weights verified/downloaded successfully.")
    except Exception as e:
        print(f"Warning: TTS model pre-download encountered an issue: {e}")
        print("The app will still attempt to load/download it during inference.")

    print("\n==================================================")
    print("Pre-downloading/Checking ASR Model: iic/SenseVoiceSmall...")
    print("==================================================")
    try:
        from funasr import AutoModel
        # Pre-download ASR model to cache
        AutoModel(
            model="iic/SenseVoiceSmall",
            disable_update=True,
            log_level="ERROR",
            device="cpu"
        )
        print("ASR model weights verified/downloaded successfully.")
    except Exception as e:
        print(f"Warning: ASR model pre-download encountered an issue: {e}")
        print("The app will still attempt to load/download it when ASR is triggered.")

def verify_flash_attn() -> None:
    """在載入模型前確認 flash-attn 可用。

    wheel 綁定編譯當時的 torch ABI，版本不符時 import 會拋出 undefined
    symbol 之類的底層錯誤 —— 那個訊息看不出真正的原因，而失敗會延到
    nano-vLLM 載入數 GB 權重之後才出現。
    """
    try:
        import flash_attn
        import torch
    except ImportError as exc:
        raise RuntimeError(
            f"無法匯入 flash-attn：{exc}\n"
            "預編 wheel 綁定特定 torch ABI，請確認 TORCH_VERSION 與 "
            "FLASH_ATTN_VERSION 是配對的組合（見 wheels/README.md 的對應表）。"
        ) from exc

    print(
        f"flash-attn {flash_attn.__version__} / torch {torch.__version__} ABI OK."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="/app/checkpoints/ft-mixed-lr2e5-avgE-e12run-0820",
    )
    args = parser.parse_args()
    verify_flash_attn()
    download_models(args.model_id)

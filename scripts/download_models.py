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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="openbmb/VoxCPM2")
    args = parser.parse_args()
    download_models(args.model_id)

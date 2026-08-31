"""模型 id 前綴常數（api.py 與 gateway 各模組共用，避免循環依賴）。"""

from voxcpm.lora_registry import BASE_MODEL_KEY

BASE_MODEL_PREFIX = "base::"
LORA_MODEL_PREFIX = "lora::"
PUBLIC_BASE_MODEL_ID = f"{BASE_MODEL_PREFIX}{BASE_MODEL_KEY}"

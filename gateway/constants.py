"""模型 id 前綴常數（api.py 與 gateway 各模組共用，避免循環依賴）。"""

from voxcpm.lora_registry import BASE_MODEL_KEY

BASE_MODEL_PREFIX = "base::"
LORA_MODEL_PREFIX = "lora::"
PUBLIC_BASE_MODEL_ID = f"{BASE_MODEL_PREFIX}{BASE_MODEL_KEY}"

# 每筆請求自選是否排隊：wait＝照舊排隊（預設），reject＝GPU slot 滿就立刻 429。
QUEUE_POLICY_WAIT = "wait"
QUEUE_POLICY_REJECT = "reject"
QUEUE_POLICIES = (QUEUE_POLICY_WAIT, QUEUE_POLICY_REJECT)
# queue=reject 被拒時的 Retry-After：GPU slot 通常數秒內就會空出。
QUEUE_REJECT_RETRY_AFTER_SECONDS = 1

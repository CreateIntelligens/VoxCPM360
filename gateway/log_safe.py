"""寫進 log 的請求欄位要先跳脫換行。

log 是一行一筆、靠 request_id 串起來的；欄位裡夾 \r 或 \n 就能偽造出一整行
看起來像真的紀錄（例如假的 stage=completed）。只跳脫、不刪除，原值仍看得出來。
"""

from typing import Any


def log_value(value: Any) -> str:
    return str(value).replace("\r", "\\r").replace("\n", "\\n")

"""把請求帶來的名稱接到固定目錄下時，確認結果仍在那個目錄裡。

呼叫端各自已有驗證（UUID、batch 是否存在），這裡是最後一道：日後若有人放寬
前面的驗證，`../` 也逃不出目錄。用 normpath＋前綴比對，是 CodeQL
（py/path-injection）認得的寫法。
"""

import os
from pathlib import Path


def child_path(base: Path, *parts: str) -> Path | None:
    """回傳 base 底下的路徑；組出來的路徑跑出 base 時回 None。"""
    root = os.path.normpath(os.path.abspath(base))
    full = os.path.normpath(os.path.join(root, *parts))
    if not full.startswith(root + os.sep):
        return None
    return Path(full)

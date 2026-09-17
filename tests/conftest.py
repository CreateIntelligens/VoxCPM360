"""把測試與真實部署內容隔離。

三個 registry 的根目錄預設指向 `/app/models/native:/app/checkpoints`，而
compose 以 `.:/app` 掛載整個工作目錄 —— 在部署容器裡跑測試時，registry 會掃到
實際存在的 9GB checkpoint 並列進 catalog，於是假設 registry 為空的測試會拿到
`full::ft-mixed-...` 而非 `base::__base__`。那是環境差異，不是程式缺陷，卻會
偽裝成大量失敗（實測 49 個）。

指向 tmp 下不存在的路徑，讓結果只取決於程式碼與各測試自己的 monkeypatch。
需要真 checkpoint 的測試自行覆寫這些變數即可。
"""

import pytest

# 在 import 期就讀取的模組層常數不受本 fixture 影響；這裡處理的是
# 建構期才讀 os.environ 的三個 registry。
_REGISTRY_ROOT_VARS = (
    "VOXCPM_LORA_ROOTS",
    "VOXCPM_LORA_ROOT",  # 舊變數，app.py 仍優先採用
    "VOXCPM_FULL_MODEL_ROOTS",
    "VOXCPM_BARBET_MODEL_ROOTS",
)


@pytest.fixture(autouse=True)
def isolate_model_registries(tmp_path, monkeypatch):
    empty = tmp_path / "empty-registry-root"
    empty.mkdir(exist_ok=True)
    for name in _REGISTRY_ROOT_VARS:
        monkeypatch.setenv(name, str(empty))
    return empty

# 預設參考音檔

前端會將本目錄內的白名單音檔顯示為下拉選單。未上傳 reference audio
時，`api.py` 會採用使用者選取的內建音檔；未指定時預設為
「青年女聲 01（臺灣台語）」。
使用者自行上傳音檔時，上傳檔案優先於下拉選擇。

舊版用戶端未傳送選項時，可用環境變數 `VOXCPM_DEFAULT_REFERENCE` 覆寫預設
路徑；設為空字串則停用預設參考音訊。

本目錄包含 `cosy-*` 系列 14 個音檔（孩童／少年／青年／年長 × 男女聲），
為同一句話由不同聲音錄製，逐字稿共用 `api.py` 的 `_COSY_PROMPT_TEXT`。

下拉項目與檔名的白名單定義於 `api.py`；新增檔案後仍須在該處加入項目。
資產完整性由 `tests/test_reference_presets.py` 檢查；啟動時若 preset
對應的音檔缺失，`create_app` 會記錄 warning 並自選單隱藏該項目。

## ⚠️ 逐字稿是必要的，不是選填

`api.py` 的 `_REFERENCE_AUDIO_PRESETS` 每筆都帶 `prompt_text`，未上傳自訂
音檔時會**自動帶入**。原因見 AGENTS.md 7.6.1：`barbet_runtime.py:148` 寫成
`prompt_wav_path=reference_path if prompt_text else ""`，逐字稿為空會讓
參考音整個被丟掉、聲音克隆完全失效（實測輸出 peak 0.973 vs 參考音 0.259）。

**新增內建參考音時，務必同時在 `api.py` 補上 `prompt_text`。**
使用者自行上傳的音檔不適用（我們不知道其逐字稿），需由使用者自行填寫。

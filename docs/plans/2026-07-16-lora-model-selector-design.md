# LoRA 模型切換設計

狀態：Draft
日期：2026-07-16

## 目標

在既有 Gradio 推論頁面加入模型下拉選單，讓操作人員可在 VoxCPM2 基礎模型與各次 LoRA 訓練的最新 checkpoint 之間切換，不新增服務、不複製 checkpoint，也不改變既有 nginx 單一入口。

## 範圍

- 僅掃描 `checkpoints/<訓練名稱>/latest/`。
- 只有同時包含 `lora_config.json` 與 `lora_weights.safetensors` 的目錄才顯示。
- 下拉選單預設為「基礎模型」，並提供重新掃描按鈕。
- 選擇 LoRA 後，在下一次生成時註冊 adapter；同一 adapter 在程序生命週期內只註冊一次。
- 沿用 nano-vLLM 的 `register_lora` 與 `generate(lora_name=...)`，保留現有加速推論。
- 不顯示中途 `step_*` checkpoint，避免選單膨脹及誤選未完成結果。

## 結構

`src/voxcpm/lora_registry.py` 負責 checkpoint 掃描、nano-vLLM LoRA 組態彙整與 adapter 註冊快取。純檔案與狀態邏輯與 Gradio 分離，便於單元測試。

`app.py` 負責：

- 啟動時用可用 checkpoint 建立 nano-vLLM LoRA 容量。
- 將模型選擇傳入生成流程。
- 顯示模型下拉選單、重新掃描按鈕與目前狀態。
- 基礎模型生成時傳入 `lora_name=None`。

## 錯誤處理

損壞或不完整的 checkpoint 不進入選單。選到已移除的 checkpoint 時，生成明確錯誤並提示重新掃描；LoRA 註冊失敗不影響之後切回基礎模型。

## 驗證

- 單元測試：只發現有效的 `latest`、忽略 `step_*`、彙整 LoRA 組態、adapter 僅註冊一次。
- 語法檢查與既有測試。
- Compose 啟動後確認前端列出基礎模型及三個現有訓練結果。
- 以基礎模型及 `trial_lora_20epochs / latest` 各執行一次短句生成。

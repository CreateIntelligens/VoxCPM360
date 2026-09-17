# DiT 推論步數

VoxCPM2 的一般與串流 API 接受逐請求 `inference_timesteps`，範圍為整數 1–50。
未指定時沿用 `VOXCPM_INFERENCE_TIMESTEPS`（預設 10）；Barbet 未指定維持 30。
一般及串流回應以 `X-Inference-Timesteps-Effective` 回報本次步數，生成歷史也記錄該值。

VoxCPM2 與部署步數相同的請求保留 CUDA graph；其他步數使用 eager，可能較慢。
同一批請求可以各自使用不同步數，不會修改共用引擎的部署設定。
前端使用新的儲存 key，初始值為 10，可調整為 1–50。

CastVoice 維持原有請求格式，不新增步數欄位。native 單筆及批次沿用部署步數，
Barbet 維持 30；兩者的預設步數均納入 `model_version` 指紋。

舊版 native 雖接受或記錄 30，實際仍使用部署步數。舊歷史保留原紀錄，
不能用其中的請求步數判定當時的實際步數或音質差異。比較 10／20／30 時，
應固定模型、文字、reference 與 seed，並記錄回應步數及生成時間。

## 部署與驗證（2026-09-17）

修補由 `scripts/patch_nanovllm_timesteps.py` 套用至
`nano-vllm-voxcpm==2.0.4`；Dockerfile 的 runner 階段會自動套用。
腳本以來源結構檢查相容性，不符合時直接停止建置，避免升級後靜默失效。

GB10 `10.9.0.37:8800` 已驗證一般、串流及 10／20／30 混合並行請求，
皆回傳 HTTP 200 與正確的 effective 步數。未指定時仍回報 10。
短句串流單次觀察約為 10 步 1.22 秒、20 步 1.68 秒、30 步 2.56 秒；
這是功能驗證，不代表固定效能或音質結論。

單元測試檢查 runner 分組、結果順序、異常時狀態還原，以及實際 CFM
solver 的取樣時間表與 estimator 呼叫次數。API 測試涵蓋邊界、預設值、
一般／串流 header 與歷史。完整 API suite 另有既存 catalog 測試仍期待
已移除的 base 模型，不能把本次針對性測試通過視為全套通過。

A4000 本次以原運作映像 `voxcpm360-app:pre-dit-20260917` 加入修補層，
保留既有 CUDA／PyTorch 版本；正式 Dockerfile 亦已同步修補，後續完整
建置仍會自動套用。臨時 Compose 建置覆寫位於
`scratch/dit-runtime.compose.yaml`，未納入版本控制。

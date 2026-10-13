# VoxCPM 參考音：Gemini 聲線，男女各 3 個

這批原始 WAV 是 Gemini TTS 預建聲線（檔名即聲線名）唸同一句台語，由 OpenVman 工作線經 Google Drive 送到 GB10，供 VoxCPM 當參考音。

## State

- 容器內位置：`/app/assets/default_reference/gemini/`
- GB10 主機對應：`/home/altos/VoxCPM360/assets/default_reference/gemini/`
- 2026-10-13 已登錄：`gateway/presets.py::_REFERENCE_AUDIO_PRESETS` 六筆（id 依 manifest 建議）、
  `gateway/castvoice.py::_CASTVOICE_DEFINITIONS` 六筆（`voxcpm2-gemini-*`）。預設音色未變。

## Verified

- 男聲候選（依來源檔名標示）：`gemini-fenrir.wav`、`gemini-alnilam.wav`、`gemini-sadaltager.wav`。
- 女聲候選（依來源檔名標示）：`gemini-autonoe.wav`、`gemini-despina.wav`、`gemini-erinome.wav`。
- 6 個都是原始 24 kHz、單聲道、16-bit PCM WAV，各約 4.76 秒。
- 檔案 SHA-256、Google Drive 來源及建議 ID 詳見 `manifest.json`；音檔未重編碼。
- 選音是 Gemini 3.5 Flash 對實際音檔隱藏檔名後的主觀音訊評估，並非人耳試聽，也不代表 VoxCPM 克隆後的品質。
- 目前 preset 登錄在 `gateway/presets.py::_REFERENCE_AUDIO_PRESETS`；CastVoice 列表另在 `gateway/castvoice.py::_CASTVOICE_DEFINITIONS`。`api.py` 是相容匯入層。
- `assets/default_reference/README.md` 說明新增 preset 需要正確逐字稿；不可將原有 cosy 系列的逐字稿套到這批音檔。

- 逐字稿：六個檔都是同一句「創作智能真歡喜，會使參加高雄的智慧時代。」
  Breeze-ASR-26（翻成華語「創作智能很高興 可以參加高雄的智慧時代」）與 SenseVoice
  （逐音「創作智靈真歡喜會使參加各項的智慧時代」）兩套結果一致；「真歡喜」「會使」
  是台語讀音，語言標 nan-TW。

## Unknowns

- 逐字稿只經兩套 ASR 交叉核對，尚未人耳逐字驗證。
- 尚未用這 6 個參考音做 VoxCPM 合成 A/B 或人耳驗收。

## Next

1. 人耳逐字核對逐字稿。
2. 用這 6 個聲音做合成人耳驗收，決定要不要調整預設音色。

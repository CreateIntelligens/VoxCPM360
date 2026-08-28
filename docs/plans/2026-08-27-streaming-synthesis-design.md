# 串流語音合成端點設計（Streaming TTS）

**狀態：Draft**
**日期：2026-08-27**
**範圍：`api.py`、`app.py`（gateway／demo 層），前端整合另案**

## 1. 背景

現有 `/api/v1/synthesize`（`api.py` 的 `synthesize` handler）等整段音訊
生成完才回傳完整 WAV。
但底層其實**每一層都已經是串流**，只是在最後被收攏：

- nano-vLLM 後端：`server.generate(**request)` 是同步 generator、
  `server.server_pool.generate(**request)` 是 async generator，逐段 yield
  `np.ndarray` 音訊 chunk。
- `app.py` 的 `VoxCPMDemo._generate_tts_requests` 與
  `generate_tts_audio_batch` 把
  chunks `np.concatenate` 成整段。
- `src/voxcpm/core.py:177` 另有 `VoxCPM.generate_streaming()`（非 vLLM
  路徑用，本案不走這條）。

目標是把 chunk 一路透傳到 HTTP 回應，讓客戶端在生成完成前就能開始播放，
降低長文本的首音延遲（TTFB 從「整段生成時間」降到「首個 chunk 生成時間」）。

## 2. Goals / Non-goals

**Goals**

1. 新端點 `POST /api/v1/synthesize/stream`，chunked transfer 回傳可邊收邊播的 WAV。
2. 僅支援 `engine_id=voxcpm2`（nano-vLLM 路徑）。
3. 生成完成後仍寫入 generation history（與非串流版行為一致）。
4. GPU 併發控制（admission／gate）語意與現有端點完全一致。

**Non-goals（明確不做）**

- Barbet 引擎串流（`src/voxcpm/barbet_runtime.py` 無串流介面）→ 422。
- `speed ≠ 1.0`（見 §5.1）→ 422。
- 批次串流、WebSocket／SSE 形式、`/api/v1/tts/*`（CastAgent voice_id
  是外部合約，一律不碰）。
- 前端 UI 整合。
- 既有 `/api/v1/synthesize` 的任何行為變更。

## 3. API 合約

### 請求

`POST /api/v1/synthesize/stream`，`multipart/form-data`，欄位與
`/api/v1/synthesize` 相同，但增加限制：

| 欄位 | 限制 | 違反時 |
|---|---|---|
| `engine_id` | 必須為 `voxcpm2` | 422，`detail="串流端點目前僅支援 voxcpm2 引擎"` |
| `speed` | 必須為 1.0（容差 ±1e-3） | 422，`detail="串流端點不支援語速調整"` |

其餘驗證（text 非空、cfg_value 1.0–5.0、inference_timesteps 1–50、
reference 解析、prompt_text 自動帶入、control_instruction 互斥邏輯）
**必須與現有端點一致**——直接把前置邏輯抽成
`_prepare_synthesis_request` 重用，不要複製貼上。特別注意兩段既有註解
描述的坑：

- 內建參考音未填 `prompt_text` 時必須自動帶入 preset 的逐字稿，否則
  克隆整個失效。
- cloning 模式（有 reference + prompt_text）必須清空
  `control_instruction`。

### 回應（成功）

- `200 OK`，`media_type="audio/wav"`，chunked transfer（無 `Content-Length`）。
- Body：一個 data/RIFF size 填 `0xFFFFFFFF` 的 WAV header（PCM16、mono、
  sample rate 取自 `server.get_model_info()["sample_rate"]`），後接 PCM16
  小端序音訊 chunk。播放器（瀏覽器 MediaSource、ffmpeg、VLC）對
  unknown-length WAV 均可邊收邊播。
- Headers（回應開頭即送出，因此**不含**生成結束才知道的
  `X-Queue-Wait` / `X-GPU-Job-Time` / `X-Synthesis-Time`）：
  - `X-Request-ID`、`X-Model-Engine`、`X-Model-Version`（同現有端點）
  - `X-Sample-Rate`：整數字串
  - `X-History-ID`：**預先產生**的 history id；串流正常完成後才會真的
    寫入 history。客戶端拿到 id 後若查無此筆，表示串流未完成。
  - `X-Accel-Buffering: no`：停用 nginx response buffering，確保公開入口
    不會把 chunks 收攏後才轉送。
  - `Content-Disposition: inline; filename="tts-output.wav"`

### 回應（失敗）

- 進入生成前的錯誤（驗證、404 模型、429 佇列滿、503 佇列逾時）：與現有
  端點相同的 HTTPException 語意（錯誤碼與 headers 對齊 `_run_gpu_job`
  既有行為）。
- **串流已開始後**發生的生成錯誤：HTTP status 已送出，無法改狀態碼。
  處理方式：記 `logger.error`（含 request_id）後中斷 body（客戶端會收到
  截斷的 WAV），**不寫 history**。文件中明確告知客戶端以「連線正常結束」
  判定成功，不能只看 200。

## 4. 實作設計

### 4.1 Gateway 層（`api.py`，`TTSGateway`）

新增 `synthesize_native_stream(...) -> AsyncIterator[np.ndarray]`
（async generator），參數同 `synthesize_native` 去掉 `speed`。

GPU 併發控制不能直接用 `_run_gpu_job`（形狀是「thread 跑完
回傳整包」）。新增 `_run_gpu_job_streaming`，保留同樣的 admission 檢查
（`_admission_lock`、`_inflight_jobs`、429/503 邏輯）與 `_gpu_lock`，差異：

1. admission 與 `_gpu_lock` 必須在 handler 建立 response 前取得，worker
   也要先回報 sample rate；如此 429／503／模型載入錯誤才能在送出 200
   headers 前維持正式 HTTP status。
2. worker thread 執行 sync generator（見 §4.2），每個 chunk 透過
   `asyncio.run_coroutine_threadsafe(queue.put(...), loop)` 推進
   `asyncio.Queue`；結束推 sentinel，例外推 exception 物件。
3. async 端從 queue 取 chunk yield 出去。
4. **`_gpu_lock` 必須等 worker thread 完全結束才釋放**——包含客戶端
   斷線的情況。`_run_gpu_job` 的 docstring 說明了
   原因：執行中的 CUDA thread 不可取消，提早放 gate 會讓下一個模型疊上
   未完成的 job。斷線時（async generator 收到 `GeneratorExit` /
   `CancelledError`）設一個 threading.Event 通知 worker 停止消費底層
   generator 並等待其 `close()`；底層 async pool 還必須明確等待
   `async_generator.aclose()` 的 cancel RPC 完成，之後才可釋放 gate。
5. queue 需設 `maxsize`（建議 8）做 backpressure，避免客戶端讀得慢時
   chunk 在記憶體堆積；worker 端 put 要用 threadsafe 的阻塞等待方式
   （例如以 `asyncio.run_coroutine_threadsafe(queue.put(...), loop)` 取代
   put_nowait）。
6. 模型切換沿用 `_switch_native_runtime`，在 worker
   thread 內、開始生成前執行一次。

### 4.2 Demo 層（`app.py`，`VoxCPMDemo`）

新增 `generate_tts_audio_stream(self, request: dict) -> Iterator[np.ndarray]`
（sync generator，單請求、不走 batch pool）：

1. 重用 `_prepare_tts_generation`（`app.py` 既有，含 reference 前處理、
   ZipEnhancer denoise、latent cache、`max_generate_length` 計算）。
2. nano-vLLM 有 `server_pool`／`loop` 時直接橋接 async generator，逐次
   `__anext__()`；`finally` 明確 `aclose()`，確保斷線會送 cancel 並等待
   回覆。舊版 runtime 才退回 `server.generate(**prepared)` 並 `close()`。
3. temp files 清理放 generator 的 `finally`（現有整批版也在
   `generate_tts_audio_batch` 的 `finally` 做）——
   串流版生命週期跟著 generator 走，`close()` 時也要清到。
4. 回傳的 sample_rate 由呼叫端另取 `server.get_model_info()`。

### 4.3 後處理（chunk 級）

現有 `_wav_response` 的三步在串流下的對應：

| 整段版 | 串流版 |
|---|---|
| `_apply_speed`（librosa time-stretch） | 不支援（§5.1），請求層已擋 |
| `_peak_normalize`（全域 peak，只縮不放） | 逐 chunk 執行 `_peak_normalize(chunk)`（每段獨立 cap 到 0.35）。理由：全域 peak 需生成完才知道；「只縮不放」性質下逐段 cap 的聽感差異可接受 |
| `sf.write` 完整 WAV | 手寫 header + 逐 chunk `(np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes()` |

新增 `_streaming_wav_header(sample_rate: int) -> bytes`：標準 44-byte
PCM WAV header，RIFF chunk size 與 data chunk size 均填 `0xFFFFFFFF`。

### 4.4 端點（`api.py`）

`POST /api/v1/synthesize/stream`：

1. 重用抽出的共用前置邏輯（驗證、reference 解析、prompt_text 帶入、
   control_instruction 互斥）。
2. 上傳 reference 的 temp file 清理：現有端點在 handler `finally` 刪；
   串流版 handler return 時生成還沒結束，
   **不能在 handler finally 刪**——改由 managed response 的 `on_close`
   清理；此 hook 涵蓋正常完成、生成失敗、headers/body send 失敗與斷線。
3. history：預先產 `history_id` 放 header；串流 generator 內累積所有
   chunk（PCM16 bytes 或 np.ndarray），正常耗盡後組完整 WAV 並沿用現有
   record 欄位寫入 `_save_generation_history`（配 `history_lock`）。history
   先原子寫入，再送 ASGI final body；final send 失敗就刪除該筆 rollback，
   使「客戶端正常 EOF」與「history 已存在」一致。生成失敗不寫。記憶體
   成本與現有整段版同量級，可接受。
4. 回傳 `_ManagedStreamingResponse(...)`，在 response-level `finally`
   等待 stream worker 完整退出並清理 upload temp。

### 4.5 catalog 標示

`/api/v1/catalog` 的 voxcpm2 engine `capabilities` dict 加
`"streaming": true`，barbet 加 `"streaming": false`，前端後續
據此顯示。

## 5. 設計決策記錄

### 5.1 為何不支援 speed

`_apply_speed` 是相位聲碼器 time-stretch，需要整段音訊；
逐 chunk 做會在邊界產生相位不連續的可聽雜音。模型本身無語速控制
（無 duration predictor）。要語速就用非串流端點。

### 5.2 為何另開端點而非 `stream=true` 參數

- 回應合約不同：現有端點帶 `Content-Length` 與生成結束才算得出的
  `X-Queue-Wait`／`X-GPU-Job-Time`／`X-Synthesis-Time`。
- 參數支援矩陣不同（speed、Barbet、批次）。分開端點讓 OpenAPI 與
  客戶端實作清晰。
- 既有客戶端與 CastAgent 外部合約零風險。

### 5.3 為何 Barbet 不做

`barbet_runtime.py` 的生成介面一次回傳整段，無 chunk 介面；且其
`synthesize_barbet` 路徑有 seed 等額外語意。等有實際需求再評估。

## 6. 測試（`tests/test_api_gateway.py` 慣例：fake demo + TestClient）

Fake 端讓 `generate_tts_audio_stream` yield 3 個固定 chunk，驗證：

1. 200、`transfer-encoding: chunked`（或無 content-length）、
   `X-Sample-Rate`／`X-History-ID`／`X-Request-ID` 存在。
2. Body 前 44 bytes 是合法 WAV header（RIFF 魔數、sample rate 正確、
   size 欄位為 0xFFFFFFFF），其後 PCM16 長度 = 各 chunk 樣本數總和 × 2。
3. 串流完成後 history 檔存在且 id 與 header 一致；chunk 中途丟例外時
   history 不寫入。
4. `engine_id=barbet` → 422；`speed=1.5` → 422。
5. 併發語意：串流進行中第二個請求會佇列（或依 `_max_pending_jobs`
   收到 429）——用 fake 的慢 generator 驗證 gate 在串流結束前不釋放。
6. 以 direct ASGI send 模擬中途斷線與 final send 失敗；worker 正常收尾、
   `_inflight_jobs` 歸零、history rollback，下一個請求可正常執行。
7. 逐 chunk peak cap：輸入 peak 0.9 的 chunk，輸出 PCM 峰值約
   0.35 × 32767。
8. 關閉 demo streaming generator 時，底層 async generator 的 `aclose()`
   必須確實執行；上傳 reference temp 也必須在 response 結束後刪除。

`TestClient` 只用於一般成功 body／headers；它會等 ASGI app 完整結束後才
回傳，不能拿來證明真實 TTFB 或中途斷線，因此後兩者使用 direct ASGI 與
第 7 節 live 驗收。

回歸：現有 `/api/v1/synthesize` 測試全數不變、全過。

## 7. 驗收條件

1. `curl -N -X POST .../api/v1/synthesize/stream -F engine_id=voxcpm2 ...`
   在生成完成前即開始輸出 bytes；`ffplay -i -`（stdin）可邊收邊播。
2. 首 byte 延遲（TTFB）顯著低於同文本非串流端點的總時間（長文本尤其）。
3. §6 測試全過；既有測試零回歸。
4. 串流中斷（Ctrl-C curl）後服務不需重啟即可處理下一個請求，
   GPU gate 無洩漏（可觀察 log 的 `stage=` 序列確認）。

## 8. 未來工作（不在本案）

- Barbet 串流、speed 的 chunked time-stretch（overlap-add）、
  WebSocket／SSE 介面、前端 MediaSource 播放整合、
  串流版 timing 資訊改用 HTTP trailer 回傳。

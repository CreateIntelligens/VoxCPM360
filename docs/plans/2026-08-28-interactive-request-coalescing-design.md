# 互動合成請求合批（Interactive Request Coalescing）設計

**狀態：Draft**
**日期：2026-08-28**
**範圍：`api.py`（TTSGateway 的 GPU gate／admission 層）、`app.py`（batch 收集路徑的錯誤隔離）**

## 1. 背景與動機

`/api/v1/synthesize` 的 GPU gate（`api.py` `_run_gpu_job`）一次只讓一個請求
上 GPU。2026-08-28 於 RTX A4000 實測（43 字台語文本、佇列深度 8）：

- 併發 6 個請求：全數 200、依序每 ~2.3s 完成一個，**總計 14 秒**；
  第 6 個請求等了 ~14 秒
- 底層 nano-vLLM 支援 continuous batching：多個語音序列共用同一批
  forward pass（AR 解碼是記憶體頻寬瓶頸，batch=1 時算力大量閒置）。
  同樣 6 個請求若合為一批，預期 4–6 秒全部完成，且每個請求的等待
  時間都下降

**引擎能力已存在、只是互動路徑接不到**：

- `app.py` `_generate_tts_requests`（約 :492）在 nano-vLLM 有
  `server_pool`／`loop` 時，會把一個 request list 以 async gather 併發
  丟進 pool——這就是 continuous batching 的入口
- 但這條路只有「單一呼叫者自帶 list」（castvoice batch 端點）才走得到；
  6 個獨立 HTTP 請求各自進 gate，永遠不會被湊在一起

### 1.1 已知陷阱：DynamicBatchSizer 在小 VRAM 卡上必然算出 1

castvoice batch 的 `DynamicBatchSizer`（`api.py:1504`）以
「GPU 可用記憶體 − reserve(2.0GiB)」÷「每項 1.5GiB」估 chunk size。
在 A4000（16GB、`VOXCPM_GPU_MEMORY_UTILIZATION=0.6`、與其他服務共卡）
上：可用 ≈ 4GB → (4−2)/1.5 → **size=1**，實測 log 也證實逐項執行。

根本問題是**記憶體重複計價**：nano-vLLM 的 KV cache 已在啟動時按
utilization 預先配置，continuous batching 的額外序列消耗的是**池內
既有配額**，不是新的 GPU 記憶體。以「池外可用記憶體」估「池內批量」
在小卡上必然過度保守。本設計的合批上限因此**不沿用** sizer，改用
固定可調上限（§4.3）；sizer 的修正列為獨立後續工作（§8）。

## 2. Goals / Non-goals

**Goals**

1. 多個併發的 `/api/v1/synthesize` 請求（同模型）自動合批送入
   nano-vLLM，吞吐接近線性提升
2. **延遲中性**：單獨到來的請求不因合批機制多等任何時間窗
3. API 合約完全不變（請求欄位、回應格式、headers、429/503 語意、
   history 行為）
4. 一個請求失敗不影響同批其他請求（錯誤隔離）

**Non-goals**

- 串流端點 `/api/v1/synthesize/stream` 不合批（chunk 交錯管理複雜，
  維持 batch-of-one 走原 gate）
- Barbet 引擎不合批（獨立 runtime，一次回傳整段）
- castvoice batch 端點（`/api/v1/tts/synthesize/batch`）行為不動
  （後續可改走同一合批器，見 §8）
- 不改 nano-vLLM 端（`max_num_seqs=16` 已足夠）

## 3. 設計總覽

把「gate 佇列裡等待的請求」從個別 job 改為**可被整批撈走的工作項**：

```
現在:  req──►admission──►gate 佇列──►gate──►單一 job──►GPU
設計:  req──►admission──►合批佇列──►drainer──►同模型批次──►GPU
                                        └─ 批內各請求的 future 各自 resolve
```

核心元件 `_NativeCoalescer`（gateway 內部）：

- **enqueue**：admission 檢查（沿用 `_inflight_jobs`／
  `VOXCPM_MAX_PENDING_SYNTHESIS`／429 邏輯，`api.py:656` 起）通過後，
  請求（參數 dict + `asyncio.Future`）進入合批佇列
- **drainer**：單一背景 task。gate（`_gpu_lock`）空出時，從佇列頭部
  取第一個請求，再**只撈與它同模型的後續請求**（保持 FIFO，遇到
  不同模型即停），湊成一批，上限 §4.3。model switch 每批一次
  （`_switch_native_runtime`）
- **執行**：批次丟 `asyncio.to_thread` 跑
  `VoxCPMDemo.generate_tts_audio_batch`（既有路徑），完成後把
  各項結果 set 到對應 future；HTTP handler 端 `await future` 後照舊
  做 `_wav_response`、history、headers

**延遲中性的關鍵**：不設收單時間窗。gate 空著時第一個請求立刻執行
（批量 = 當下佇列裡剛好在等的同模型請求數，沒人排隊就是 1）。批次
只在「本來就有人在排隊」時自然形成——這正是需要吞吐的時刻。

## 4. 實作細節

### 4.1 admission 與 429（不變）

`_read_int_setting("VOXCPM_MAX_PENDING_SYNTHESIS", default=2)` 的
容量檢查、429 回應（`Retry-After: 30`）、`X-Request-ID` 全部沿用。
佇列逾時 `VOXCPM_QUEUE_TIMEOUT_SECONDS`（120s）改為在合批佇列中
等待的上限：future 超時未被 drainer 撈走 → 503（語意與現行
`_gpu_lock` acquire 逾時一致），並從佇列移除。

### 4.2 同模型分組

批內所有請求的 canonical model id 必須相同（`resolve_native_model_id`
後比較）。drainer 從佇列頭開始收集，遇到第一個不同模型的請求即停
（**不跳過撿後面的**，保持 FIFO 公平、避免飢餓）。LoRA 選擇屬於
模型 id 的一部分（`lora::` 前綴不同即不同模型）。

### 4.3 批量上限

`VOXCPM_INTERACTIVE_BATCH_MAX`，預設 4，上限 clamp 到 16
（nano-vLLM `max_num_seqs`）。**設 1 即完全回到現行為**——這是
上線後的即時 rollback 開關，不用回滾程式碼。
預設保守取 4 的理由：KV cache 在 A4000/0.6 配額下約只有模型初始化
外的少量餘裕，4 序列 × 短文本經驗上安全；GB10（119GB 統一記憶體）
可透過環境變數放大。

### 4.4 錯誤隔離（需要動 app.py）

現行 `_generate_tts_requests` 的 async 收集在任一請求失敗時
raise 整批錯誤（`app.py:515-521`，錯誤時其他結果也被丟棄）。
新增變體（或加參數）`return_exceptions=True` 路徑：回傳
`list[np.ndarray | Exception]`，coalescer 把 Exception set 給對應
future（該請求回 500），其餘正常回 200。castvoice batch 呼叫端
維持原「整批失敗」語意不變。

### 4.5 Headers 與觀測性

- `X-Queue-Wait`：該請求 enqueue → 批次開始執行的等待
- `X-GPU-Job-Time`：整批的 GPU 執行時間（批內共用同值）
- 新增 `X-Batch-Size`：本請求所在批的大小（除錯／壓測用）
- log：`synthesis.request stage=coalesced batch_size=N models=...`，
  以及每批一筆 `stage=batch_completed elapsed items_per_second`

### 4.6 與串流／Barbet 的 gate 互動

串流與 Barbet 請求維持原 `_run_gpu_job`／`_run_gpu_job_streaming`
路徑，與 coalescer 共搶同一把 `_gpu_lock`：對 drainer 而言它們是
「別人拿著 gate」，對它們而言 coalescer 的一批是「一個 job」。
`_inflight_jobs` 計數涵蓋三者，429 容量全域一致。

## 5. 設計決策記錄

### 5.1 為何「無時間窗」而非固定 50ms 收單窗

時間窗讓**每個**請求都付固定延遲稅，換取低併發時偶爾多湊到一兩個；
無時間窗在單請求時零損耗，高併發時批量自然等於佇列深度。壓測
（§7）驗證：合批收益主要來自「已在排隊的請求」，時間窗的邊際收益
不值得延遲稅。

### 5.2 為何不沿用 DynamicBatchSizer

見 §1.1：它以池外可用記憶體估池內批量，在共卡小 VRAM 機器上恆為 1，
等於把合批靜默關閉。固定上限＋rollback 開關的行為可預期、可觀測。

### 5.3 為何串流不進批

串流的價值是首塊延遲，批內其他成員會拖慢它的首塊；且多路 chunk
的順序與背壓管理複雜度高。串流已有專屬路徑，維持 batch-of-one。

## 6. 測試（`tests/test_api_gateway.py` 慣例：fake demo + TestClient）

fake `generate_tts_audio_batch` 記錄每次收到的 list 大小：

1. 單請求：批大小 1、無額外延遲（fake 內斷言呼叫即刻發生）
2. 併發 6（同模型、fake 慢生成製造佇列）：批大小依序如 [1, 5] 或
   [1, 4, 1]（第一批執行中累積後續），總 GPU 呼叫次數 < 6
3. 混模型 FIFO：佇列 [A, A, B, A] → 批次 [A,A]、[B]、[A]，順序不亂
4. 錯誤隔離：批內第 2 項丟例外 → 該請求 500、其餘 200，history
   只寫成功者
5. `VOXCPM_INTERACTIVE_BATCH_MAX=1` → 行為與現行完全一致（回歸開關）
6. 429 語意：容量（1+`MAX_PENDING`）不因合批改變
7. 佇列逾時：drainer 被長批佔住、等待者超過
   `VOXCPM_QUEUE_TIMEOUT_SECONDS` → 503 且自佇列移除
8. 串流請求與批次交錯：gate 互斥正確、`_inflight_jobs` 歸零
9. `X-Batch-Size` header 正確反映

回歸：既有 120 項測試全過，castvoice batch 行為不變。

## 7. 驗收

1. A4000 實機：併發 6（43 字文本）總耗時從 ~14s 降到 **≤ 7s**，
   零 429，音訊可解碼
2. 單發延遲與合批前差異 < 5%（延遲中性）
3. `VOXCPM_INTERACTIVE_BATCH_MAX=1` 時數據回到基準
4. GB10 實機同型測試（上限可放大到 8–16 觀察曲線）

## 8. 後續工作（不在本案）

- `DynamicBatchSizer` 改以 nano-vLLM 池內餘裕（KV cache 使用率）
  估算，修正記憶體重複計價；屆時 castvoice batch 與本合批器可共用
  同一套上限邏輯
- castvoice batch 端點改走同一 coalescer（統一 GPU 排程視角）
- 串流合批（多路 chunk 交錯）評估

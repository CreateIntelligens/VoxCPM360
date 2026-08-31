# 併發串流（Concurrent Streaming）設計 v2

**狀態：Draft**
**日期：2026-08-31**
**範圍：`api.py`（GPU gate 語意）、`app.py`（event loop 橋接）**
**前置：interactive request coalescing 已 merge（同一套 admission 基礎設施）**

## 1. 背景與動機

nano-vLLM 的核心價值是 continuous batching——多個序列共用 forward pass。
但 gateway 的 `_gpu_lock` 是互斥鎖：一次一個 job，串流請求獨佔整段生成
（A4000 上 ~8 秒），第二路串流的 TTFB 要等第一路整段結束。引擎能力被
gate 掐成單線程。

**引擎端已就緒的證據**：

- `server_pool.generate()` 是 per-request 的 async generator，各請求
  chunk 流各自獨立、不混流
- castvoice batch 路徑（`app.py` `_generate_tts_requests`）已在單一
  `run_until_complete` 內 `asyncio.gather` 多個 generate——引擎同時
  服務多序列是既有事實
- GB10 實測 M=2~32 GEMM 同價：多一路串流的邊際成本趨近零；
  A4000 單流 4.6× 即時，2~4 路併發後每路仍可望 >1× 即時

## 2. Goals / Non-goals

**Goals**

1. 同模型的多路串流**同時生成**：第 N 路的 TTFB 不再等前面整段，
   只受 admission 容量限制
2. 串流與（合批後的）非串流批次共存於引擎，同時進行
3. `VOXCPM_ENGINE_CONCURRENCY=1` 時行為完全等同現行（rollback 開關）

**Non-goals**

- 跨模型併發（不同 model 仍需 drain + switch，維持現行語意）
- Barbet 併發（獨立 runtime，維持互斥）
- 優先權排程／搶佔（v1 先 FIFO admission）

## 3. 設計

### 3.1 前置工程：event loop 橋接改造（最關鍵、風險最高）

現況：串流橋接（`app.py` `generate_tts_audio_stream`）與 batch 路徑
（`_generate_tts_requests`）都用 `server_loop.run_until_complete(...)`
**從呼叫端執行緒驅動引擎的 loop**。兩個執行緒同時這樣做會直接崩潰
（"loop already running"）——這是併發的硬阻擋。

改造：

1. 引擎載入後，啟動**專屬 loop 執行緒**：`threading.Thread(target=
   server_loop.run_forever, daemon=True)`（在 `get_or_load_voxcpm`
   完成、`wait_for_ready` 之後——nano-vLLM 自己的初始化
   `run_until_complete` 都發生在那之前，不衝突）
2. 之後**所有**引擎互動一律
   `asyncio.run_coroutine_threadsafe(coro, server_loop)`：
   - 串流：`__anext__()`／`aclose()` 逐次提交，`.result(timeout)` 取回
   - batch：`collect_all()` 整包提交
3. `stop_voxcpm` 時 `loop.call_soon_threadsafe(loop.stop)` 收攤執行緒
4. **禁令**：改造後任何地方不得再呼叫 `run_until_complete`（會與
   run_forever 衝突）——加一個 debug assert 或 code comment 立牌

### 3.2 Gate 語意：互斥鎖 → 模型親和的容量制

`_gpu_lock`（mutex）改為「active model session」：

```
session = { model_id, refcount, capacity=VOXCPM_ENGINE_CONCURRENCY }
```

- **同模型**的工作：refcount < capacity 即可加入（refcount++），
  完成時 refcount--；串流計 1 單位，合批批次計 len(batch) 單位
  （上限夾在 nano-vLLM `max_num_seqs=16` 內）
- **不同模型**的工作：等 refcount 歸零 → `_switch_native_runtime`
  → 建立新 session（維持現行 drain-then-switch 語意）
- **Barbet**：視為 capacity=1 且要求 drain 的特殊 session
- admission／429／`VOXCPM_MAX_PENDING_SYNTHESIS`／queue timeout
  全部沿用，計數單位不變（一個請求一個 job）

### 3.3 與 coalescer 的關係

coalescer 保留：非串流請求仍先合批再提交（一次 submit、一次結果分發，
admission 效率好）。批次與串流是 session 內的平等公民，共享 capacity。
（v3 可評估：全部改 per-request 直接提交、讓引擎自然 batch，屆時
coalescer 可退役——本案不做，避免一次動兩個抽象。）

### 3.4 資源保護

- `VOXCPM_ENGINE_CONCURRENCY` 預設 **4**（保守上線值）；**營運目標
  配置為 16**——與引擎 `max_num_seqs=16` 對齊，成為批次與串流
  「共搶的 16 條通道」：合批批次占 `len(batch)` 單位、每路串流占
  1 單位，先到先搶、無保留席。屆時 `VOXCPM_INTERACTIVE_BATCH_MAX`
  作為批次單輪的切塊上限，受同一個總量池約束
- 客戶端斷線：該路的 `aclose()` 照舊（streaming 設計 v1 的語意），
  只釋放自己的 1 單位，不影響其他路
- 生成中途錯誤：per-request 隔離（引擎端本來就是獨立序列）

### 3.5 觀測性

- 回應 header 加 `X-Engine-Concurrency`：該請求開始執行時的
  session refcount（除錯／壓測用）
- log：`stage=session_join/session_leave model=... refcount=...`

## 4. 風險

| 風險 | 緩解 |
|---|---|
| loop 執行緒改造引入死鎖／飢餓 | `.result(timeout=...)` 全部帶逾時；§6 測試 8 專測 |
| 引擎對併發 async generate 的未知邊界 | batch 路徑已 gather 併發（既有事實）；壓測覆蓋 |
| 併發後單流 cadence 變慢導致播放斷續 | 驗收 §7.3 明定每路 cadence 下限；容量預設保守 |
| KV 池耗盡 | capacity 上限 + nano-vLLM 自身的 seq 上限雙保險 |
| 回滾 | `VOXCPM_ENGINE_CONCURRENCY=1` 即回現行為，不需回滾程式碼 |

## 5. 設計決策記錄

- **為何不直接每請求提交、廢掉 gate**：model switch 與 Barbet 互斥
  仍需 drain 語意；容量制是最小改動保留這些不變量的方案。
- **為何 loop 執行緒化而非多 loop**：nano-vLLM 的 pool 綁單一 loop；
  多 loop 需要改引擎，不值得。

## 6. 測試（fake async pool + TestClient）

1. 2 路併發串流：兩路都在 1 秒內收到首 chunk（fake 慢生成驗證
   TTFB 互不阻塞）；chunk 內容各自正確不混流
2. 串流 + 合批批次共存：同時進行、互不等待整段
3. capacity=1 時：行為與現行完全一致（回歸開關）
4. 容量滿時第 N+1 路排隊、429 語意不變
5. 模型切換：新模型請求等 refcount 歸零；期間不 starve（FIFO）
6. 一路斷線：其餘路不受影響、refcount 正確遞減、下一請求可加入
7. Barbet 請求：等 voxcpm2 session drain、期間 voxcpm2 新請求排隊
8. loop 執行緒：關閉服務時乾淨退出；無 run_until_complete 殘留
   （grep 斷言或 runtime assert）
9. 既有測試（133+）全過

## 7. 驗收（實機）

1. A4000：2 路併發串流，**兩路 TTFB 均 < 1s**（現況第二路 ~8s）
2. A4000：2 路併發總完成時間 ≤ 單路 × 1.6（引擎 batch 效益）
3. 每路串流 cadence ≥ 1.2× 即時（播放不追資料）；concurrency=2 下
   單路獨跑的延遲與現行差 < 5%
4. GB10：4 路併發，每路 cadence 與單路差 < 15%（M-flat 經濟學驗證）
5. 混合負載（4 非串流 + 2 串流）全部正確完成、`X-Engine-Concurrency`
   與 log 一致、`_inflight_jobs` 歸零

## 8. 後續（不在本案）

- per-request 直接提交、coalescer 退役評估（v3）
- 優先權（互動 > 批次）與搶佔
- 跨模型並存（多模型常駐，記憶體允許時）

# 逐請求完成（Per-Request Completion）— 併發串流設計增補

**狀態：Draft**
**日期：2026-08-31**
**性質：對 `2026-08-31-concurrent-streaming-design.md`（v2）的行為增補，即該文件 §8 的「v3」前半**
**動機（產品需求）**：客服情境下每位使用者是獨立請求，現行 coalescer
把同批請求鎖步到「同一瞬間一起回應」且**批內等最慢**——短句被同批長句
拖延。需求：**誰生成完誰先回**，回應時間彼此獨立。castvoice batch 端點
維持批語意不變。

## 1. 現況問題點

`_NativeCoalescer._drain`（api.py）把整批丟進單一
`_generate_native_batch_results`（thread）→ app.py `collect_all` 用
`asyncio.gather` 等**全部**序列完成 → 才一次分發所有 future。
引擎層每個序列其實各自完成（continuous batching 本來就支援早退），
被 gather 抹平。

## 2. 目標行為

1. 每個互動請求（非串流）的 HTTP 回應在**自己的音訊生成完成時**立即返回
2. 同時在引擎裡的請求仍共享 forward pass（吞吐不變或更好）
3. 容量單位逐請求釋放：一個請求完成即歸還 1 單位，佇列中下一個
   請求可立即遞補進引擎（不再等整批結束才放行）
4. castvoice batch 端點（單一呼叫者自帶 list）行為不變

## 3. 設計

### 3.1 drainer 改為「提交即走」

現行：撈批 → `await` 整批執行完 → 分發 → 下一輪。
改為：撈批（或單個）→ 對**每個 item**：

1. prep（`_prepare_tts_generation` 等價物）丟 thread pool（prep 內的
   engine 互動已走 `_call_engine_sync`，thread 安全）
2. prep 完成後把該 item 的 generate coroutine 以
   `run_coroutine_threadsafe` 提交至引擎 loop 執行緒
3. 為該 item 建立 completion task：await 自己的結果 →
   `item.future.set_result(...)` → **釋放自己的 1 容量單位** →
   log `stage=completed`（帶 per-request queue_wait/execution）
4. drainer **不等待**任何 item 完成，立刻回頭檢查佇列與剩餘容量，
   有空位就繼續放行下一個請求

如此「批」的概念退化為「一次撈多個一起提交」的最佳化；完成順序
由各序列實際長度決定。

### 3.2 容量語意

- session refcount 以 item 為單位：提交 +1、完成/失敗/斷線 -1
- `VOXCPM_INTERACTIVE_BATCH_MAX` 語意改為「單輪撈取上限」
  （提交節流，避免一瞬間塞爆 prep thread pool），不再影響完成時序
- `VOXCPM_ENGINE_CONCURRENCY` 仍是引擎在飛行序列的硬上限
  （串流 1 單位、互動請求 1 單位，共搶）

### 3.3 錯誤與斷線

- 單一 item 失敗只 fail 自己的 future（500），不影響其他 in-flight
- 客戶端斷線（非串流）：回應未送出前斷線，生成照跑完（CUDA 不可
  取消的既有語意）、結果丟棄、容量照釋放

### 3.4 timing headers

`X-Queue-Wait`／`X-GPU-Job-Time` 改為 per-request 真實值
（enqueue→submit、submit→own completion）。`X-Batch-Size` 更名或
保留為「同輪提交數」，語意在 README 註明。

## 4. 驗收（實機，A4000）

1. **長短混打**：同時發 1 長句（~40 字）＋ 3 短句（~6 字）：
   短句各自在 ~1–2s 內返回，**不等**長句；長句照常完成
2. 12 併發同長句：總時間 ≤ 現行 9.1s（吞吐不退步）
3. 完成時間戳分散（不再同一瞬間）：以回應到達時間驗證
4. 單發延遲不變（≤2.5s）
5. 容量遞補：13 個併發、`ENGINE_CONCURRENCY=12` 時，第 13 個在
   任一請求完成後立即進場（非等 12 個全完）
6. 串流／castvoice batch 行為回歸不變；全套測試綠

## 5. 風險

- prep 併發化後 reference 快取（latent_cache）的執行緒安全需檢查
- 完成順序不定 → 依賴「批次同返」的（若有）下游假設會被打破
  （已知沒有正式依賴；history 寫入本就 per-request）

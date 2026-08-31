# GB10（sm_121）CUDA 13 升級設計

**狀態：Draft**
**日期：2026-08-31**
**範圍：`Dockerfile`（base image 參數化）、`.env`（GB10 端）、flash-attn wheel 重編與發布**

## 1. 背景：為什麼要升

2026-08-31 調查定案（詳見專案記憶與當日對話）：GB10 推理比 A4000 慢
約 10 倍的根因是 **CUDA 12.8 的 cuBLAS 對 sm_121 缺少瘦長 GEMM 調優**：

- `(M,4096)×(4096,1024)` bf16：M=1 → 22.6µs（GEMV 路徑正常）；
  **M=2 → 253µs**，且 M=2~32 同價（掉到低平行度 generic kernel）
- DiT 因 CFG 恆為 batch=2，每個 AR step 約 635 次 matmul 全數踩懸崖
- 症狀特徵：GPU util 96% 但功耗僅 ~20W（絕大多數 SM 閒置）
- 已否證：flash-attn 版本（2.8.3 同速）、cuDNN conv、Triton、CPU 調度、
  CPU↔GPU 傳輸、CUDA graph 機制（合成 graph 0.9µs/節點）、PYTORCH_JIT

修法 = 換到對 sm_121 有完整函式庫調優的 CUDA 世代。

## 2. 目標組合（可用性已驗證）

| 元件 | 現況 | 目標 | 驗證 |
|---|---|---|---|
| base image | `nvidia/cuda:12.8.0-devel-ubuntu22.04` | `nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04` | Docker Hub tag 存在 ✅ |
| torch | 2.11.0+cu128 | **2.11.0+cu130**（cp310、aarch64/x86 皆有） | index 實查 ✅ |
| Python | 3.10（ubuntu22.04） | **3.10 不變**（特意選 ubuntu22.04 tag，避免 wheel ABI 連鎖） | — |
| flash-attn | 2.6.3＋手工 Blackwell patch | **2.8.3 官方版**（原生支援 sm_120；2.8.3 已在 GB10 編譯與運行驗證過，cu128 下效能中性） | 09:32 編過 ✅ |

## 3. Goals / Non-goals

**Goals**

1. GB10 的瘦長 GEMM 恢復正常（驗收見 §7），單發 RTF 從 ~0.4× 顯著回升
2. Dockerfile 保持單一檔案跨機通用：base image 與 torch index 以 ARG
   參數化，**x86 這輪不動**（cu128 在 A4000 上沒有問題）
3. flash-attn wheel 機制延續：cu130 版 wheel 編一次、發 Release、之後免編

**Non-goals**

- x86/A4000 遷移 cu130（無痛點，等 GB10 驗證穩定後另議）
- torch 版本升級（維持 2.11.0，只換 CUDA variant，變因最小化）
- nano-vllm-voxcpm 升級

## 4. 實作設計

### 4.1 Dockerfile 參數化（不破壞 x86 cache 的寫法）

```dockerfile
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu22.04
FROM ${CUDA_BASE_IMAGE} AS base
...
ARG TORCH_CUDA_VARIANT=cu128
# torch 安裝行改為:
#   --index-url https://download.pytorch.org/whl/${TORCH_CUDA_VARIANT}
```

注意：**FROM 行的 ARG 化與 torch 行的修改都會使 x86 既有編譯層 cache
失效**——但 x86 的 flash-attn wheel 已在 `wheels/`／GH Release，重建走
wheel 分支只需 ~15 分鐘，成本可接受（這正是 wheel 機制存在的理由）。

compose 對應加入：

```yaml
args:
  CUDA_BASE_IMAGE: ${CUDA_BASE_IMAGE:-nvidia/cuda:12.8.0-devel-ubuntu22.04}
  TORCH_CUDA_VARIANT: ${TORCH_CUDA_VARIANT:-cu128}
```

GB10 的 `.env` 設：

```bash
CUDA_BASE_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04
TORCH_CUDA_VARIANT=cu130
TORCH_ARCH_LIST=12.0
```

### 4.2 flash-attn：升 2.8.3、cu130 重編、wheel 隔離

1. 版本從 `v2.6.3` 升為 `v2.8.3`：官方原生支援 Blackwell，**移除**兩個
   手工 patch（`patch_flash_attention_arch.py` 呼叫與 `flash_api.cpp` 的
   sed）——2.8.3 的 setup 原生接受 `FLASH_ATTN_CUDA_ARCHS`。
   （2.6.3 的手工 patch 大概率無法在 nvcc 13 下直接編，升版順勢清債。）
2. **wheel 檔名衝突處理**：cu128 與 cu130 編出的 wheel 檔名相同
   （`flash_attn-2.8.3-cp310-cp310-linux_*.whl`）但 ABI 不相容。
   GH Release 改為**每個 CUDA variant 一個 tag**：
   - 既有 `flash-attn-wheels` → 視為 cu128（release notes 註明）
   - 新增 `flash-attn-wheels-cu130`
   Dockerfile 的自動下載 URL 改為
   `.../releases/download/flash-attn-wheels-${TORCH_CUDA_VARIANT#cu128?}/...`
   ——實作上直接用 `flash-attn-wheels-${TORCH_CUDA_VARIANT}` 全量命名
   （把既有 release 改名或重發一個 `flash-attn-wheels-cu128` 別名 tag，
   擇一，實作者決定後在 README 記錄）。
3. 本地 `wheels/` 目錄同理不可混放兩個 variant 的同名 wheel；
   `.env` 切 variant 時需清掉舊 wheel（在 README 註明此坑）。

### 4.3 PYTORCH_JIT=0 的再評估

該 workaround 的成因是 cu128 nvrtc 不認得 sm_121。CUDA 13 的 nvrtc
應原生支援——升級後**嘗試移除** `PYTORCH_JIT=0`（compose 對 GB10 的
env），讓 snake 等 op 恢復 JIT fusion。若 snake 的 dynamo recompile
警告（`audio_vae_v2.py` 的 `[0/8]` 上限）仍在，屬次要問題、另案處理。

### 4.4 GB10 驗證流程（沿用 08-31 已驗證的隔離手法，prod 零接觸）

1. `~/voxcpm-wheel-build` clone：`git pull` → `.env` 寫入 §4.1 的三行
2. wheel 編譯：`docker buildx build --target flash-wheel
   --build-arg CUDA_BASE_IMAGE=... --build-arg TORCH_CUDA_VARIANT=cu130
   --build-arg TORCH_ARCH_LIST=12.0 -o type=local,dest=wheels-cu130 .`
   （首次 ~20 分鐘；torch cu130 輪組若下載被爛 CDN 卡住，沿用
   「x86 代下＋rsync」流程，arm-pins 需以 cu130 index 重新解析）
3. 完整 image build（吃剛編的 wheel）→ tag `voxcpm360-app:cu130-test`
4. 測試容器起在 `127.0.0.1:18800`（記得快取掛載到
   `/home/voxcpm/.cache/huggingface`，08-31 踩過 `/root` 的坑）
5. 跑 §7 驗收 → 通過才動 prod：`docker compose build && up -d`
6. cu130 wheel 上傳對應 Release tag

## 5. 風險與回滾

| 風險 | 緩解 |
|---|---|
| torch 2.11+cu130 在 sm_121 仍未修瘦長 GEMM | §7 的微基準是第一道驗收，2 分鐘見真章，不過就停損（等更新 CUDA 或走 M=2→2×GEMV 的 nano-vllm patch 路線） |
| flash-attn 2.8.3 + nvcc 13 編譯失敗 | 官方 2.8.x 支援 CUDA 13；若仍失敗，試 2.8 系最新 patch 版 |
| driver 相容（GB10 host driver 需支援 CUDA 13 runtime） | 事前 `nvidia-smi` 查 driver 版本；DGX Spark 官方棧即為 CUDA 13 世代，預期相容 |
| nano-vllm / triton 3.6 與 cu130 相容性 | triton 隨 torch cu130 wheel 配對安裝；nano-vllm 為純 Python |
| 回滾 | `.env` 三行改回 cu128 組合＋既有 wheel → 重建即回舊棧；prod image 未刪即可直接 `up -d` 回滾 |

## 6. 不動 x86 的邊界確認

x86 `.env` 不設新變數 → compose 預設 cu128 組合，行為與現行完全一致；
唯 Dockerfile 修改導致一次 cache 失效重建（走 wheel 分支，~15 分鐘）。

## 7. 驗收

1. **微基準（第一道關卡）**：GB10 容器內
   `(2,4096)×(4096,1024)` bf16 ≤ **50µs**（現況 253µs、A4000 30µs）
2. 單發（43 字文本）：≤ **8s**（現況 25–27s），RTF ≥ 1.2×
3. 生成期間功耗顯著上升（>60W，現況 20W——SM 真的在做事的旁證）
4. 串流 TTFB ≤ 0.7s 維持；`ffmpeg` 解碼正常
5. 容器內 `pytest tests/` 全過（120+）
6. prod 切換後 catalog／history／castvoice 端點煙霧測試正常

## 8. 後續工作（不在本案）

- x86 遷移 cu130（統一 variant、Release 只留一套 wheel）
- `inference_timesteps` 被 voxcpm2 引擎靜默忽略的 API 誠實性修正
- snake dynamo recompile 上限（shape 動態性）優化
- 合批（interactive coalescing）在 GB10 的大 batch 驗證——M=2~32 同價
  意味著合批後多載 15 行近乎免費，GB10 批次吞吐會不成比例提升

# 各輪訓練詳細紀錄

總覽與操作手冊見 [TRAINING_LOG.md](TRAINING_LOG.md)。本檔只放完整設定、曲線與逐輪結論。

---

## 版本沿革

新版在上，僅記與前一版的差異。

### 全參數微調 (Full Fine-Tuning) 三對照組 — 2026-07-31

**首度擺脫 LoRA 容量上限，進行全參數微調對照實驗**：驗證全參數解鎖下，資料集來源（純 tai8 vs 純 naer vs 兩者混合）對語音合成與 Loss 瓶頸的真實影響。

- **`full_ft_tai8`（job `176449`，✅ 完成 3h07m）**：
  - **資料**：純 tai8（`tai8/manifests/voxcpm2_abs/`，22.2 萬句）
  - **設定**：無 `lora` 區塊，LR `1e-5`、`batch_size: 4`、`grad_accum: 8`（有效 batch 128）、`save_interval: 1000`
  - **儲存路徑**：`/app/checkpoints/full_ft_tai8`
- **`full_ft_naer`（job `176450`，✅ 完成 4h01m）**：
  - **資料**：純 naer（`naer/manifests/voxcpm2_abs/`，6.6 萬句朗讀語音）
  - **設定**：無 `lora` 區塊，LR `1e-5`、`max_batch_tokens: 16384`（因 naer 句長偏長）
  - **儲存路徑**：`/app/checkpoints/full_ft_naer`
- **`full_ft_mixed`（job `176451`，🏃 執行中）**：
  - **資料**：tai8 + naer 混合（`mixed/manifests/tai8_naer/`，28.8 萬句）
  - **設定**：無 `lora` 區塊，LR `1e-5`、`max_batch_tokens: 16384`
  - **儲存路徑**：`/app/checkpoints/full_ft_mixed`

> 📍 **結果**（2026-07-31 15:20）：
>
> | 對照組 | 最佳 val | 出現在 | 最終 (step 6999) | 狀態 |
> |---|---|---|---|---|
> | `full_ft_tai8` | 1.0954 | step 3,000 | 1.1964 | ✅ 完成 |
> | `full_ft_naer` | **0.9190** | step 1,500 | 0.9879 | ✅ 完成 |
> | `full_ft_mixed` | 0.9626↓ | step 1,500 | — | 🏃 21% |
>
> ⚠️ **三組 val 各用自己的 `val_seen.jsonl`，數字不可直接橫向比較** —— naer 的 0.9190
> 不代表「比 tai8 好」，只代表 naer 這個朗讀語料本身較易擬合（句長規整、錄音一致）。
> 要橫向比較必須用同一份 held-out 測試集跑 `eval_ckpt.py`。
>
> **共同形態**：三組都在極早期見頂後過擬合。tai8 在 step 3,000、naer 在 step 1,500，
> 之後單調上升到 7,000 步。全參微調的有效訓練窗口遠比 LoRA 短。
>
> ⚠️ **`save_interval: 1000` 太稀疏** —— run4 已將 LoRA 的 save_interval 收斂到 500，
> 但三組全參微調退回 1000。naer 最佳點在 step 1,500，實際只抓得到 1,000 與 2,000
> 兩個鄰點，真正的谷底取不到。**全參微調最佳點更早，反而更需要細的儲存間隔**，
> 下輪建議 250～500（單 ckpt 26 GB，需權衡磁碟：7,000 步 ÷ 500 ≈ 14 個 ≈ 364 GB）。

### run4 — 2026-07-31 01:23 起（✅ 完成 1h48m，job `176430`）

**首度嘗試 LoRA 擴容量（r=64）**：回歸純 tai8 資料集，唯一變更 LoRA 容量以驗證模型容量受限假說。

- **Changed** `lora.r` `32` → `64`、`lora.alpha` `32` → `64`（維持 alpha/r = 1）
  — 假說：run2 與 run3 在純 tai8 評測上實質等價（1.1098 vs 1.1122），且 loss/diff 皆卡在 0.995±0.002，形態指向容量飽和而非資料不足
- **Changed** `train_manifest`／`val_manifest` 回歸純 tai8（`tai8/manifests/voxcpm2_abs/`）
- **Changed** `max_steps` `34,800` → `7,000`（4.03 epoch），涵蓋 epoch 2~3.5 轉折點
- **Fixed** `save_interval` `1,000` → `500`（與 `valid_interval` 對齊，解決 run2 step 9,500 無 checkpoint 缺點）
- **Unchanged** `batch_size` 16、`grad_accum` 2、`max_batch_tokens` 8192、LR 5e-5

> 📍 **結果**：最佳 val **1.1143** @ step 6,999（全程 15 次驗證，單調下降未見過擬合）。
>
> ❌ **擴容量假說不成立**：r=64 的 1.1143 對比 run2 的 1.1098、run3 的 1.1122，
> 三者差距 < 0.005，落在先前確認的等價噪音範圍內。**加大 LoRA capacity 沒有突破
> 1.11 平台**，瓶頸不在 LoRA 秩的大小。此結論直接導向改採全參數微調。

### run3 — 2026-07-30 21:03 起（已完成，job `176247`）

**首次混合資料集**：tai8 + naer。**資料已偏離基準**，其餘超參沿用 run2 以隔離變因。

- **Changed** `train_manifest` → `mixed/manifests/tai8_naer/train.jsonl`
  — tai8 222,383 + naer 66,104 = **288,487 句**（naer 佔 22.9%）
- **Changed** `val_manifest` → 同上目錄 `val_seen.jsonl`（11,368 + 14,832 = 26,200）
- **Fixed** `max_batch_tokens` `8,192` → `16,384` — naer 句長 3.7 倍，混合後分布右移
  （p90 175→342、p99 222→467、max 630→784）。沿用 8,192 會靜默丟棄 973 筆
- **Changed** `max_steps` `34,800` → `22,600` — epoch 換算分母改為 288,487；
  取 10 epoch 而非 20（run2 在 epoch 3.5 即走平，後半段反而過擬合）
- **Unchanged** `batch_size` 16、`grad_accum` 2、有效 batch 128、LR 5e-5、LoRA r=32
  — **刻意只動資料一個變因**，才能與 run2 對照

> 📍 **進度**（2026-07-31 02:00）：step 15,200 / 22,600（67.3%），epoch 6.7。
>
> ✅ **公平評測已完成**（job `176338`）：換用純 tai8 驗證集後，
> run3 與 run2 **實質等價**（1.1122 vs 1.1098，全距 0.0034），
> 混合 val 的 0.9579 是驗證集變簡單造成的假象。
> **加入 22.9% 的 naer 未帶來可測量的助益** —— 詳見下方評測結論。

### run2 — 2026-07-30 12:35–18:45（過擬合，提早中止）

修正 run1 未收斂。

- **Changed** 有效 batch `8` → `128`（`batch_size` 1→16）— **主要修正**
- **Changed** `learning_rate` `1e-4` → `5e-5` — 大 batch 梯度較穩
- **Changed** `max_steps` `84,000` → `34,800` — epoch 3.02 → **20.03**
- **Changed** `warmup_steps` `1,000` → `500`、`num_workers` `4` → `8`
- **Changed** `valid_interval`／`save_interval` `2,000/8,000` → `500/1,000`
  — 69 個 val 點、34 個 checkpoint，便於挑最佳
- **Fixed** `max_batch_tokens` `4,096` → `8,192` — 必須隨 batch 同步調大，
  否則 `max_sample_len` 由 512 降為 256、靜默丟棄 599 筆
- **Known issue** epoch 3.5 後過擬合，val 由 1.1151 升至 1.1363 → 於 step 30,503 中止
- **Known issue** `save_interval` 1,000 ≠ `valid_interval` 500，
  導致最佳 val 點 step 9,500 **沒有對應 checkpoint**（見下方 run2 結論）

### run1 — 2026-07-30 02:00–08:52（未收斂）

首次全量訓練。

- **Changed** `max_steps` `1,280` → `84,000`（epoch 0.01 → 3.02）
- **Changed** `warmup_steps` `64` → `1,000`、`log_interval` `10` → `50`
- **Changed** `valid_interval`／`save_interval` `64/320` → `2,000/8,000`
- **Known issue** 有效 batch 仍為 `8` → 未收斂
- **Note** 此 yaml **至今未 commit**，卻是 12 個 checkpoint 的唯一來源設定

### trial-1280 — 2026-07-30 00:34（流程驗證通過）

從上游 Docker 設定遷移至 Taipei-1。

- **Changed** `pretrained_path`、`train_manifest`、`val_manifest` 改絕對路徑
- **Changed** manifest 從 `smoke_train.jsonl` → **全量 `train.jsonl`**（222,383 筆）
- **Known issue** 檔名 `20epochs` 雙重誤導（實際 0.01 epoch，且已非 trial 資料）
  → 建議改名 `trial_lora_1280steps.yaml`

### 環境層（非 config）— 2026-07-30

- **Changed** [`train.sh`](../train.sh) 移除 docker compose 包裝（叢集無 Docker／sudo）
- **Added** [`train.sh`](../train.sh) GPU 數自動偵測 → `torchrun`／`python` 分流

---

## trial-1280

Log：`~/.scripts_tmp/logs/train_test3.log`
有效 batch `1×2×4 = 8` · 1,280 步 = 0.0104 epoch · LR 1e-4

<details>
<summary>完整 config — <a href="../conf/voxcpm_v2/trial_lora_20epochs.yaml"><code>trial_lora_20epochs.yaml</code></a></summary>

```yaml
pretrained_path: /mnt/home/csl426-aicr-ae5f63/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/bffb3df5a29440629464e5e839f4d214c8714c3d
train_manifest: /mnt/home/csl426-aicr-ae5f63/dataset202607_1/tai8/manifests/voxcpm2_abs/train.jsonl
val_manifest: /mnt/home/csl426-aicr-ae5f63/dataset202607_1/tai8/manifests/voxcpm2_abs/val_seen.jsonl
sample_rate: 16000
out_sample_rate: 48000
batch_size: 1
grad_accum_steps: 2
num_workers: 2
num_iters: 1280
log_interval: 10
valid_interval: 64
save_interval: 320
learning_rate: 0.0001
weight_decay: 0.01
warmup_steps: 64
max_steps: 1280
max_batch_tokens: 4096
max_grad_norm: 1.0
save_path: /app/checkpoints/trial_lora_20epochs
tensorboard: /app/checkpoints/trial_lora_20epochs/logs
lambdas:
  loss/diff: 1.0
  loss/stop: 1.0
lora:
  enable_lm: true
  enable_dit: true
  enable_proj: false
  r: 32
  alpha: 32
  dropout: 0.0
```

</details>

| step | val total | diff | stop |
|---|---|---|---|
| 0 | 1.2919 | 1.0931 | 0.1988 |
| 192 | 1.0926 | 0.9698 | 0.1229 |
| **1152** | **1.0451** | 0.9481 | 0.0970 |
| 1279 | 1.1816 | 1.0312 | 0.1504 |

- ✅ 流程無誤：多 rank 初始化、AudioVAE 載入、checkpoint 存讀、音訊生成皆正常。
- 📌 **1.0451 至今仍是所有版本最佳**，而它只掃過 1% 資料 —— 見 run1 結論。

---

## run1

2026-07-30 02:00:47 → 08:52:11（6h51m）· 節點 `cnode2-021`
Log：`~/.scripts_tmp/logs/full_train_run1.log`
有效 batch `1×2×4 = 8` ← **問題根源** · 84,000 步 = 3.02 epoch · LR 1e-4

<details>
<summary>完整 config — <a href="../conf/voxcpm_v2/full_lora_run1.yaml"><code>full_lora_run1.yaml</code></a>（⚠️ 未 commit）</summary>

```yaml
pretrained_path: /mnt/home/csl426-aicr-ae5f63/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/bffb3df5a29440629464e5e839f4d214c8714c3d
train_manifest: /mnt/home/csl426-aicr-ae5f63/dataset202607_1/tai8/manifests/voxcpm2_abs/train.jsonl
val_manifest: /mnt/home/csl426-aicr-ae5f63/dataset202607_1/tai8/manifests/voxcpm2_abs/val_seen.jsonl
sample_rate: 16000
out_sample_rate: 48000
batch_size: 1
grad_accum_steps: 2
num_workers: 4
num_iters: 84000
log_interval: 50
valid_interval: 2000
save_interval: 8000
learning_rate: 0.0001
weight_decay: 0.01
warmup_steps: 1000
max_steps: 84000
max_batch_tokens: 4096
max_grad_norm: 1.0
save_path: /app/checkpoints/full_lora_run1
tensorboard: /app/checkpoints/full_lora_run1/logs
lambdas:
  loss/diff: 1.0
  loss/stop: 1.0
lora:
  enable_lm: true
  enable_dit: true
  enable_proj: false
  r: 32
  alpha: 32
  dropout: 0.0
```

</details>

**產出**：12 個 checkpoint + `latest`，LoRA 權重 72 MB（384 個張量）
**吞吐**：13.6s／50 步（前約 250 步 155→33s 為 DDN I/O 暖機，正常）

| step | epoch | val total | diff | stop |
|---|---|---|---|---|
| 0 | 0 | 1.3632 | 1.0884 | 0.2748 |
| 2,000 | 0.07 | 1.1493 | 1.0188 | 0.1305 |
| 20,000 | 0.72 | 1.1307 | 1.0084 | 0.1223 |
| **50,000** | 1.80 | **1.1129** | 0.9952 | 0.1178 |
| 78,000 | 2.80 | 1.1078 | 0.9833 | 0.1245 |
| 83,999 | 3.02 | 1.1516 | 1.0273 | 0.1243 |

**結論**

- ❌ **未收斂**。前 2,000 步降 0.21（幾乎全來自 `loss/stop` 0.275→0.130），
  之後 82,000 步在 1.11~1.15 震盪、尾段回升。`loss/diff` 六小時只從 1.088 到 1.027。
- ❌ **根因：有效 batch 僅 8**，梯度雜訊過大；LR 1e-4 對此 batch 偏大。
  **與資料量無關** —— 222,383 筆全數餵入、掃過 3 遍、零丟棄。
- 🔍 **決定性對照**：trial-1280 在 0.0104 epoch 就達 1.0451，優於 run1 全程。
- ⚠️ 最佳在 `step_0048000`（1.1129），`latest` 是 step_84000（1.1516）。
- ℹ️ 結束時 `destroy_process_group() was not called` 為無害警告。

---

## run2（執行中）

job `176103` · 2026-07-30 12:35 起 · 節點 `cnode2-021`
有效 batch `16×2×4 = 128` · 34,800 步 = 20.03 epoch · LR 5e-5

<details>
<summary>完整 config — <a href="../conf/voxcpm_v2/full_lora_run2.yaml"><code>full_lora_run2.yaml</code></a></summary>

```yaml
pretrained_path: /mnt/home/csl426-aicr-ae5f63/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/bffb3df5a29440629464e5e839f4d214c8714c3d
train_manifest: /mnt/home/csl426-aicr-ae5f63/dataset202607_1/tai8/manifests/voxcpm2_abs/train.jsonl
val_manifest: /mnt/home/csl426-aicr-ae5f63/dataset202607_1/tai8/manifests/voxcpm2_abs/val_seen.jsonl
sample_rate: 16000
out_sample_rate: 48000

# 有效 batch = batch_size × grad_accum_steps × world_size = 16 × 2 × 4 = 128
# （run1 為 1×2×4 = 8，梯度雜訊過大導致不收斂）
batch_size: 16
grad_accum_steps: 2
num_workers: 8

# max_sample_len = max_batch_tokens // batch_size = 8192 // 16 = 512
# 實測 train.jsonl 長度分布 p99=222、max=630，cap 512 只濾掉 4／222383 筆。
# 注意：調大 batch_size 時必須同步調大 max_batch_tokens，否則會靜默丟棄長樣本。
max_batch_tokens: 8192

# 222383 樣本 ÷ 有效 batch 128 ≈ 1738 步/epoch；20 epoch ≈ 34760 步
# 刻意跑長：LR 排程綁定 max_steps，事後從 34 個 checkpoint 挑收斂點即可
# （續訓會沿用舊 cosine 曲線、LR 已衰減到近 0，故不採「先短再延長」）
num_iters: 34800
max_steps: 34800
log_interval: 50
valid_interval: 500
save_interval: 1000

learning_rate: 0.00005
weight_decay: 0.01
warmup_steps: 500
max_grad_norm: 1.0

save_path: /app/checkpoints/full_lora_run2
tensorboard: /app/checkpoints/full_lora_run2/logs
lambdas:
  loss/diff: 1.0
  loss/stop: 1.0
lora:
  enable_lm: true
  enable_dit: true
  enable_proj: false
  r: 32
  alpha: 32
  dropout: 0.0
```

</details>

**開跑 6 項檢查全數通過**

| # | 檢查項 | 預期 | 實測 |
|---|---|---|---|
| 1 | `GPU 數：` | 4 | **4** ✅ |
| 2 | `Loading AudioVAE` | 4 次 | **4 次** ✅ |
| 3 | `Filtering N / 222383` | 不出現 | **0 次**（零丟棄）✅ |
| 4 | step 500 `epoch` | ≈0.29 | **0.2878** ✅ |
| 5 | step 500 val | <1.149 | **1.1388** ✅ |
| 6 | `log interval` | 穩定 | **33.5s／50 步** ✅ |

**val 走勢 —— 先降後升（過擬合）**

| step | epoch | val total | diff | stop | |
|---|---|---|---|---|---|
| 0 | 0 | 1.2742 | 1.0489 | 0.2253 | |
| 500 | 0.29 | 1.1388 | 1.0162 | 0.1226 | |
| 2,500 | 1.44 | 1.1238 | 1.0070 | 0.1168 | |
| 6,000 | 3.45 | 1.1151 | **1.0001** | 0.1150 | ← 可用最佳 |
| **9,500** | 5.47 | **1.1149** | — | — | ← 理論最佳（**無 checkpoint**） |
| 13,500 | 7.77 | 1.1183 | 0.9990 | 0.1193 | 開始回升 |
| 26,000 | 14.97 | 1.1281 | 0.9940 | 0.1341 | |
| 30,500 | 17.56 | **1.1363** | 0.9963 | 0.1399 | 中止點 |

epoch 3.5 後即走平，之後持續惡化：`loss/diff` 仍在降（訓練集學更好）但
`loss/stop` 由 0.115 升到 0.140、`grad_norm` 由 0.05 升到 0.35 —— 典型過擬合。
於 18:45 手動中止（step 30,503 / 87.6%），省下約 1.5 小時算力。

**與 run1 對比 —— 修正生效**

| 指標 | run1 | run2 | |
|---|---|---|---|
| `grad_norm` | 0.13~0.50 | **0.05~0.11**（前期） | 雜訊小 3~6 倍 |
| 達 val 1.11 | 50,000 步／4h | **6,000 步／1.5h** | 快 8 倍 |
| `loss/diff` 破 1.0 | 全程未達 | **step 6,000** | run1 最低 0.9833 |

吞吐 33.5s／50 步（run1 為 13.6s；batch 大 16 倍故單步更重）。

**已確認**：batch=16 無 OOM、`train.sh` 自動偵測在真實 Slurm 環境可用。

### ⚠️ 教訓：`save_interval` 必須整除 `valid_interval`

run2 設 `valid_interval: 500`、`save_interval: 1000`，導致**最佳 val 點
step 9,500 落在沒有存檔的位置**：

```
所有 val 最佳：1.114851 @9,500   ← 無 checkpoint
有 checkpoint 最佳：1.115083 @6,000
```

實際取用 `step_0006000`，只差 0.0002 尚可接受，但這是設定缺陷。
**下輪起應讓兩者相等，或 `save_interval` 為 `valid_interval` 的整數倍且對齊。**

**最終產出**：`step_0006000`（val 1.115083），已備份至
`docs/voxcpm360/runs/full_lora_run2/`（31 個 checkpoint + log + TensorBoard，2.2 GB）。

### 為何刻意跑 20 epoch，而非先短再延長

`get_cosine_schedule_with_warmup`（`train_voxcpm_finetune.py:209`）**把 `max_steps`
烘進 LR 曲線**，且續訓時 `scheduler.pth` 會還原該曲線（`load_checkpoint`，`:718`）。
因此「先跑 17,400 再加大 `max_steps` 續訓」的後果是：

- 舊排程在 17,400 步已把 LR 衰減到近 0，續訓等於 LR≈0 空轉；
- 刪掉 `scheduler.pth` 強行重置，則 LR 從峰值重新 warmup、破壞已收斂權重。

**一次跑到 20 epoch、事後挑最佳 checkpoint 才是正解。** 反正無 early stopping，
最佳點本來就得事後挑；多跑只增加候選，不損害既有結果。

### 實際結果：先降後升 → 過擬合

依上表判讀落在第三列，故取最低點 `step_0006000`，**非 `latest`**。
20 epoch 對純 tai8 明顯過量，**epoch 3.5 就該停**。

---

## run3 — tai8 + naer 混合（執行中）

job `176247` · 2026-07-30 21:03 起 · 節點 `cnode2-021`
有效 batch `16×2×4 = 128` · 22,600 步 = 10.03 epoch · LR 5e-5

<details>
<summary>完整 config — <a href="../conf/voxcpm_v2/full_lora_run3.yaml"><code>full_lora_run3.yaml</code></a></summary>

```yaml
pretrained_path: /mnt/home/csl426-aicr-ae5f63/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/bffb3df5a29440629464e5e839f4d214c8714c3d

# 資料已偏離基準：tai8 222,383 + naer 66,104 = 288,487 句（naer 佔 22.9%）
# naer = TAT-MOE 台語朗讀（16k mono，已驗證）；tai8 = 台八戲劇對白
# dataset_id 保留來源：tai8=1、naer=0
train_manifest: /mnt/home/csl426-aicr-ae5f63/dataset202607_1/mixed/manifests/tai8_naer/train.jsonl
val_manifest: /mnt/home/csl426-aicr-ae5f63/dataset202607_1/mixed/manifests/tai8_naer/val_seen.jsonl
sample_rate: 16000
out_sample_rate: 48000

# 與 run2 相同（有效 batch 128），只動資料這一個變因以便對照
batch_size: 16
grad_accum_steps: 2
num_workers: 8

# ⚠️ 必須從 run2 的 8192 調高：naer 句子長 3.7 倍，混合後長度分布右移
# （p90 175→342、p99 222→467、max 630→784）
# max_sample_len = 16384 // 16 = 1024 → 零丟棄
# 若沿用 8192 則 msl=512，會靜默丟棄 973 筆
max_batch_tokens: 16384

# 288,487 樣本 ÷ 有效 batch 128 ≈ 2254 步/epoch；10 epoch ≈ 22540 步
# 取 10 epoch 而非 20：run2 在 epoch 3.5 即走平，後半段無改善
num_iters: 22600
max_steps: 22600
log_interval: 50
valid_interval: 500
save_interval: 1000

learning_rate: 0.00005
weight_decay: 0.01
warmup_steps: 500
max_grad_norm: 1.0

save_path: /app/checkpoints/full_lora_run3
tensorboard: /app/checkpoints/full_lora_run3/logs
lambdas:
  loss/diff: 1.0
  loss/stop: 1.0
lora:
  enable_lm: true
  enable_dit: true
  enable_proj: false
  r: 32
  alpha: 32
  dropout: 0.0
```

</details>

### 資料（**已偏離基準**）

| 來源 | train | val_seen | `dataset_id` |
|---|---|---|---|
| tai8（台八戲劇對白） | 222,383 | 11,368 | `1` |
| naer（TAT-MOE 台語朗讀） | **66,104** | **14,832** | `0` |
| **合計** | **288,487** | **26,200** | |

naer 佔 train 的 22.9%。實測 16 kHz 單聲道 16-bit，與 tai8 規格一致，**不需重新取樣**。
合併後 manifest 位於 `dataset202607_1/mixed/manifests/tai8_naer/`。

**長度分布右移**（naer 平均 6.53s，tai8 僅 1.75s）：

| | tai8 | tai8+naer |
|---|---|---|
| p50 | 138 | 147 |
| p90 | 175 | **342** |
| p99 | 222 | **467** |
| max | 630 | **784** |

故 `max_batch_tokens` 必須由 8,192 提高到 **16,384**
（`max_sample_len = 16384//16 = 1024` → 零丟棄；沿用 8,192 會丟 973 筆）。

### 開跑檢查

| # | 檢查項 | 預期 | 實測 |
|---|---|---|---|
| 1 | `GPU 數：` | 4 | **4** ✅ |
| 2 | `Filtering N / 288487` | 不出現 | **0 次**（零丟棄）✅ |
| 3 | step 500 `epoch` | ≈0.222 | **0.221847** ✅ |
| 4 | `log interval` | 穩定 | **49~54s／50 步** |

吞吐比 run2 慢約 50%（33.5s → 51s），因 naer 句長 3.7 倍且 token 上限加倍。
22,600 步預估 **6~7 小時**，約 2026-07-31 04:00–05:00 完成。

### val 走勢（進行中）

截至 step 11,500（2026-07-31 01:00，訓練時間 ~4:00，進度 50.9%）：

| step | epoch | val total | diff | stop |
|---|---|---|---|---|
| 0 | 0.00 | 1.1733 | 0.9471 | 0.2262 |
| 500 | 0.22 | 0.9914 | 0.9312 | 0.0602 |
| 1,000 | 0.44 | 0.9755 | 0.9242 | 0.0513 |
| 1,500 | 0.67 | 0.9685 | 0.9200 | 0.0486 |
| 2,000 | 0.89 | 0.9687 | 0.9216 | 0.0472 |
| 2,500 | 1.11 | 0.9711 | 0.9250 | 0.0461 |
| 3,000 | 1.33 | 0.9700 | 0.9251 | 0.0448 |
| 3,500 | 1.55 | 0.9654 | 0.9210 | 0.0444 |
| 4,000 | 1.77 | 0.9635 | 0.9189 | 0.0446 |
| 4,500 | 2.00 | 0.9622 | 0.9183 | 0.0439 |
| 5,000 | 2.22 | 0.9631 | 0.9186 | 0.0445 |
| 5,500 | 2.44 | 0.9588 | 0.9140 | 0.0448 |
| 6,000 | 2.66 | 0.9620 | 0.9185 | 0.0436 |
| 6,500 | 2.88 | 0.9643 | 0.9202 | 0.0441 |
| 7,000 | 3.10 | 0.9631 | 0.9190 | 0.0441 |
| 7,500 | 3.33 | 0.9629 | 0.9195 | 0.0434 |
| 8,000 | 3.55 | 0.9620 | 0.9188 | 0.0431 |
| 8,500 | 3.77 | 0.9609 | 0.9180 | 0.0429 |
| 9,000 | 3.99 | **0.9579** | 0.9141 | 0.0437 |
| 9,500 | 4.21 | 0.9612 | 0.9180 | 0.0433 |
| 10,000 | 4.44 | 0.9596 | 0.9172 | 0.0424 |
| 10,500 | 4.66 | 0.9609 | 0.9174 | 0.0435 |
| 11,000 | 4.88 | 0.9581 | 0.9154 | 0.0427 |
| 11,500 | 5.10 | 0.9593 | 0.9157 | 0.0435 |
| 13,000 | 5.77 | 0.9576 | 0.9140 | 0.0436 |
| 14,000 | 6.21 | 0.9606 | 0.9164 | 0.0442 |
| 14,500 | 6.43 | 0.9592 | 0.9158 | 0.0433 |
| 15,000 | 6.65 | 0.9597 | 0.9161 | 0.0435 |

最低點為 **step 9,000（0.9579）**，次低 step 11,000（0.9581）——兩者僅差
**0.0002**，實務上等價。曲線自 step 1,500 起就在 **0.958–0.971 窄帶內震盪**，
一萬步下來未再出現有意義的下降；`loss/stop` 收斂於 0.043 附近不動，
殘餘變化幾乎全由 `loss/diff` 貢獻。

⚠️ **上一版設定的「epoch 4 判斷點」已驗證**：當時預告若 epoch 4（step ≈9,000）
仍無新低就考慮停止。實際結果是 step 9,000 確實出現新低 0.9579，
**但只比前低（step 5,500 的 0.9588）低 0.0009**，落在震盪雜訊內。
嚴格說「有新低」，實質是**持續走平** —— 與 run2 在 epoch 2~3.5 走平的形態一致。

**但走平不等於該停**：run3 的 val 用的是混合驗證集（含 14,832 筆 naer），
**與 run1／run2 不可比**。在下方「待辦評測」完成前，無法判斷 run3 是否真的更好，
因此也無從設計 run4 的變因。已決定**讓 run3 跑完**（剩約 4 小時），
評測可用單卡推論並行進行，不需佔用訓練資源。

### 已取回的 checkpoint（本機唯讀備份）

`docs/voxcpm360/runs/run3/`，兩者皆 384 tensors、位元組數與叢集端一致：

| checkpoint | val total | 備註 |
|---|---|---|
| `step_0009000` | **0.9579** | 目前最低 |
| `step_0011000` | 0.9581 | 次低，與最低差 0.0002 |

> 取回前需先跑 `fix_perms.sh` —— 訓練程式寫出的 `lora_weights.safetensors`
> 是 `-rw-------`，本輪修正了 8 個項目。

### ✅ 評測結論：混合資料對 tai8 **沒有助益**（2026-07-31 01:47，job `176338`）

用**同一份純 tai8 `val_seen.jsonl`**（11,368 筆，711 batch，完整跑完）
重新評測三個 checkpoint，這才是可以並排比較的數字：

| checkpoint | total | diff | stop |
|---|---|---|---|
| **run2 `step_0006000`** | **1.1098** | 0.9948 | 0.1150 |
| run3 `step_0009000` | 1.1122 | 0.9961 | 0.1161 |
| run3 `step_0011000` | 1.1132 | 0.9971 | 0.1161 |

**全距僅 0.0034 —— 三者實質等價**，run3 甚至微輸 0.0024。

> 📌 **run3 混合 val 的 0.9579 是假象**。它與 run2 的 1.1151 相差 0.157，
> 看似大勝，實際完全來自驗證集裡 22.9% 的 naer 朗讀語音好預測。
> 換回同一把尺後優勢歸零。**naer 佔了近四分之一訓練資料，換來持平。**

### ⚠️ `max_val_batches = 10` —— 訓練期 val 不可作為選 checkpoint 的依據

`train_voxcpm_finetune.py:390` 的 `validate()` 寫死只取 **10 個 batch（160 筆）**，
並非整份驗證集。這正是 run3 曲線在 ±0.003 間震盪的機制成因 ——
**波動來自每次抽到不同樣本，不是模型在變**。

本次評測意外提供了強力佐證：先用 3 batch（48 筆）煙霧測試，
排名與完整 711 batch **完全顛倒**：

| checkpoint | 3 batch（48 筆） | 711 batch（11,368 筆） |
|---|---|---|
| run2 `step_0006000` | 1.1493（最差） | **1.1098（最佳）** |
| run3 `step_0009000` | 1.0938 | 1.1122 |
| run3 `step_0011000` | 1.0642（最佳） | 1.1132（最差） |

小樣本量出的「run3 大勝 0.085」純屬雜訊。
**日後挑最佳 checkpoint 一律用 `scripts/eval_ckpt.py` 跑完整驗證集**，
不可直接讀訓練 log 的 val 欄位。

### 評測工具：`scripts/eval_ckpt.py`

固定驗證集、跑完整資料、對多個 checkpoint 重複同一套 forward
（刻意複製 `validate()` 的加權方式，差別只在不限 batch 數、不生成音檔）。

```bash
sbatch ~/scripts/eval_run2_vs_run3.sh          # 完整評測（單卡，約 20 分鐘）
MAX_BATCHES=3 sbatch ~/scripts/eval_run2_vs_run3.sh   # 煙霧測試（僅驗證管線）
```

單卡即可，不必等訓練結束，也不會搶 run3 的 GPU。

> ⚠️ **純 naer 訓練不是這個問題的對照組** —— 它同時換掉訓練資料與驗證集，
> 會產生第三把尺，只能證明「naer 好學」（從 `loss/stop` 一開始就掉到 0.06
> 已可看出）。純 naer 僅在「確實需要台語朗讀合成模型」這個獨立產品需求下才值得跑。

### run4 方向（依評測結果修正）

混合資料這條路已驗證無效，**不應再往「提高 naer 比例／加更多朗讀語料」走**。
三輪的 val 都在 epoch 2~3 走平，瓶頸更可能在容量或最佳化，而非資料量：

1. **LoRA 容量** —— r=32 可能已飽和，試 r=64，或開 `enable_proj`
2. **LR** —— 三輪都是 5e-5 且都早早走平，可能偏低導致卡在局部
3. **資料品質** —— tai8 有 92.9% 來自 drama1，語者／風格分布極不均，
   清洗或重採樣可能比加量有效

（尚未選定，待決。）

⚠️ **這個數字不可與 run1／run2 直接比較** —— run3 的 `val_seen` 含 14,832 筆 naer，
朗讀語音規整、停頓明確，本來就比戲劇對白易預測。低 val 有一部分來自
**驗證集變簡單**，不等於模型更強。

**要公平比較，必須用同一份驗證集**（例如純 tai8 的 `val_seen.jsonl`）
對 run2 與 run3 的最佳 checkpoint 各跑一次推論評測。

### ⚠️ 首次送出失敗：`episode` 欄位型別衝突

job `176205` 開跑 67 秒即失敗：

```
pyarrow.lib.ArrowInvalid: JSON parse error:
Column(/episode) changed from number to string in row 37
```

| 資料集 | `episode` 範例 | 型別 |
|---|---|---|
| tai8 | `11` | **int** |
| naer | `"A025-1"` | **str** |

`datasets.load_dataset` 用 pyarrow 建 schema，**同名欄位型別不一致即整批失敗**。
log 中 rank 2/3 的 `SIGTERM` 只是 rank 1 死後的連帶清理，非根因。

**修正**：合併時統一型別 —— 識別用欄位（`episode`／`speaker_id`／`utterance_id`）
一律轉 `str`，`duration`／`ref_duration` 轉 `float`，`dataset_id` 轉 `int`。

**日後合併任何資料集都要先驗證**：

```bash
# ① 掃全檔確認同名欄位型別一致
# ② 實際跑 load_dataset 驗證（型別一致不保證 pyarrow 能建 schema）
srun -p p06 --gres=gpu:h100:1 --container-image=<sqsh> --container-mount-home \
  python3 -c "from datasets import load_dataset; \
    print(len(load_dataset('json',data_files='<merged>.jsonl',split='train')))"
```

**尚未產生 manifest**，以下為調查結果。

### naer 現況（實測）

| 項目 | 值 |
|---|---|
| 位置 | `/mnt/home/csl426-aicr-ae5f63/dataset202607_1/naer/` |
| 語料 | TAT-MOE（台語朗讀，非戲劇） |
| VoxCPM2 manifest | ✅ 已存在 `manifests/voxcpm2/` |
| `train.jsonl` | **66,104 句 / 119.99 h** |
| val_seen／val_unseen／test_unseen | 14,832／13,269／13,087 句 |
| **取樣率** | ✅ **16 kHz 單聲道 16-bit**（抽樣 400 檔全相符、0 缺失） |
| `dataset_id` | `0`（tai8 為 `1`，可區分來源） |

✅ **不需重新取樣** —— 與 tai8 的 `profiles/16k_mono/` 規格一致。

### 混合前必解的兩件事

**1. 路徑是相對路徑**

```
audio: ../../segments/TAT-MOE-train/lavalier/IU_IUF1001/A025-1.1.wav
```

tai8 當初為此另建 `voxcpm2_abs/`。naer 需同樣處理，否則從 `/app` 解析不到。
可沿用 `~/.scripts_tmp/fix_manifest_paths.py`。

**2. 句長差 3.7 倍 → `max_batch_tokens` 必須重算**

| | tai8 | naer |
|---|---|---|
| 平均句長 | 1.75 s | **6.53 s** |
| 最長 | 630 tokens | **23.12 s** |

run2 的 `max_sample_len = 512` 對 tai8 只丟 4 筆，但對 naer 長句**會大量丟棄**。
須重掃混合後長度分布，實測各組合保留率再定參數。

### 建議順序

1. **先讓 run2 跑完** —— 需要純 tai8 基準，才知道加 naer 是變好還是變壞。
2. 產生 naer 絕對路徑 manifest。
3. 合併兩份 `train.jsonl`，用 `dataset_id` 保留來源標記。
4. 重掃長度分布，重算 `max_batch_tokens`／`batch_size`。
5. 建立 `full_lora_run3.yaml`，並**標示資料已偏離基準**。

⚠️ 混合後樣本數 **288,487**，epoch 換算分母改變：
20 epoch @有效 batch 128 = **45,076 步**（不再是 34,800）。

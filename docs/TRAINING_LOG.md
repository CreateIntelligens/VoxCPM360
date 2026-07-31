# VoxCPM2 微調訓練總覽

Taipei-1 · `p06` · 4× H100（torchrun/NCCL）· 模型 `openbmb/VoxCPM2` @ `bffb3df5`
細節見 [RUNS.md](RUNS.md)（各輪完整設定與曲線）· 名詞見 [GLOSSARY.md](GLOSSARY.md)

---

## 現況（2026-07-31 15:20）

| | |
|---|---|
| **正在跑** | `full_ft_mixed` · Slurm job `176451` · 節點 `cnode2-021` |
| **進度** | step 1,500 / 7,000 (21.4%) · val 0.9626 且仍在降 |
| **已完成** | run4 (`176430`) · `full_ft_tai8` (`176449`) · `full_ft_naer` (`176450`) |
| **跑完要做** | ① 跑 `eval_ckpt.py` 評測 ② `bash ~/scripts/fix_perms.sh` ③ 取回本機 |

> ⚠️ **全參微調的最佳點極早**：naer 在 step 1,500 見頂 (0.9190)，之後一路過擬合到
> 7,000 步的 0.9879。`save_interval: 1000` 對這種曲線**太稀疏**，真正最佳點可能落在
> 1,000～2,000 之間而抓不到。下輪建議改 250～500。

---

## 版本

| 版本 | Config | 有效 batch | 步數 | Epoch | 最佳 val | 結論 |
|---|---|---|---|---|---|---|
| trial-1280 | [`trial_lora_20epochs.yaml`](../conf/voxcpm_v2/trial_lora_20epochs.yaml) | 8 | 1,280 | 0.01 | **1.0451** | 流程驗證通過 |
| run1 | [`full_lora_run1.yaml`](../conf/voxcpm_v2/full_lora_run1.yaml) | 8 | 84,000 | 3.02 | 1.1129 | ❌ 未收斂（batch 太小） |
| run2 | [`full_lora_run2.yaml`](../conf/voxcpm_v2/full_lora_run2.yaml) | 128 | 34,800 | 20.03 | 1.1098 | 🔄 epoch 3.5 後過擬合 |
| run3 | [`full_lora_run3.yaml`](../conf/voxcpm_v2/full_lora_run3.yaml) | 128 | 22,600 | 10.03 | 1.1122 | ✅ naer 混入對 tai8 無加分 |
| run4 | [`full_lora_run4.yaml`](../conf/voxcpm_v2/full_lora_run4.yaml) | 128 | 7,000 | 4.03 | 1.1143 | ❌ r=64 未突破平台 |
| full_ft_tai8 | [`full_ft_tai8.yaml`](../conf/voxcpm_v2/full_ft_tai8.yaml) | 128 | 7,000 | 4.03 | 1.0954 | 全參 tai8：勝 LoRA 但幅度小 |
| full_ft_naer | [`full_ft_naer.yaml`](../conf/voxcpm_v2/full_ft_naer.yaml) | 128 | 7,000 | 11.81 | **0.9190** | ✅ 最佳（step 1,500 見頂後過擬合）|
| full_ft_mixed | [`full_ft_mixed.yaml`](../conf/voxcpm_v2/full_ft_mixed.yaml) | 128 | 7,000 | 4.03 | 🏃 0.9626↓ | 執行中（step 1,500）|

前四輪（trial～run4）資料相同（tai8 `train.jsonl`），差異只在超參；
後三輪 `full_ft_*` 則是**固定超參、變動資料**的對照組。
⚠️ 三組 val 用各自的 `val_seen.jsonl`，**資料集不同時數字不可直接橫向比較**。

---

## 核心結論

**run1 失敗不是資料不足，是有效 batch 太小。**

trial-1280 只掃 **1% 資料**（0.01 epoch）就達 1.0451，勝過 run1 掃 **3 遍**（84,000 步）的 1.1129。
瓶頸在最佳化設定，不在資料預算。

run2 把有效 batch 從 8 提到 128，結果立刻驗證：

| | run1 | run2 |
|---|---|---|
| `grad_norm`（梯度雜訊） | 0.13~0.50 | **0.05~0.11** |
| 達到 val 1.11 | 50,000 步 / 4h | **6,000 步 / 1.5h** |

---

## 三個必記陷阱

**1. `max_batch_tokens` 與 `batch_size` 是除法關係**

```
max_sample_len = max_batch_tokens // batch_size    # 超過者靜默丟棄
```

調大 batch 不同步調大 tokens 會無聲丟資料。`batch_size: 32` + `4096` → **只剩 32.79%**。

**2. `latest` ≠ 最佳**

無 early stopping。run1 最佳在 `step_0048000`，`latest` 卻是較差的 84,000。
**每輪都要自己從 log 挑。**

**3. 共享目錄權限不會自動給 group**

訓練寫的 `lora_weights.safetensors` 是 `-rw-------`，p06 團隊**連讀都不行**。
每輪結束跑 `bash ~/scripts/fix_perms.sh`。

---

## 資料

`tai8` 台八戲劇 · `/mnt/home/<USER>/dataset202607_1/tai8/manifests/voxcpm2_abs/`

| Manifest | 規模 | 用途 |
|---|---|---|
| `train.jsonl` | **222,383 句 / 108.19 h** | 訓練 |
| `val_seen.jsonl` | 11,368 句 | 驗證（說話人**全部**與 train 重疊） |
| `val_unseen.jsonl` | 10,666 句 | ⚠️ **從未使用** |
| `test_unseen.jsonl` | 18,820 句 | ⚠️ **從未使用** |

16 kHz 單聲道 · 平均句長 1.75 s · 2,928 位說話人 · drama1 92.9% / drama2 7.1%

⚠️ **`val_seen` 看不出過擬合**（說話人全重疊）。要判斷泛化須另跑 `val_unseen` 推論評測。
⚠️ **drama1 佔 92.9%**，模型會偏向願望的音色，鳥來伯品質預期較弱。

**Epoch 換算**（`train_voxcpm_finetune.py:332`）：

```
epoch = step × grad_accum × batch_size × world_size / 222383
```

規劃步數必用此式反推 —— run1 的 84,000 步只是 3.02 epoch。

---

## 操作手冊

### 送出訓練

```bash
ssh taipei-1
sbatch ~/scripts/submit_vox_batch.sh conf/voxcpm_v2/full_lora_run2.yaml
```

**送出後 30 秒**確認多卡生效，否則白跑數小時：

```bash
head -3 ~/vox_train_<JOBID>.log     # 要看到「GPU 數：4」
```

> `train.sh` 自動偵測 GPU 數 → 多卡走 `torchrun`、單卡走 `python`。
> **裸 `python` 會使 `WORLD_SIZE=1` 只用單卡**（有效 batch 剩 32、epoch 剩 5.0）。
> 單卡除錯用 `NPROC_PER_NODE=1`。

### 監看

```bash
tail -f ~/vox_train_<JOBID>.log              # 完整輸出（等同 docker logs）
grep '^\[val\]' ~/vox_train_<JOBID>.log      # 只看 val 走勢
ml slurm && squeue -u $USER                  # 作業狀態
```

| log 來源 | 內容 |
|---|---|
| `~/vox_train_<JOBID>.log` | 完整，含啟動訊息 |
| `checkpoints/<run>/train.log` | 只有 `[train]`／`[val]` |
| `checkpoints/<run>/logs/events.*` | TensorBoard 曲線與 mel 圖 |

進容器（需帶與訓練相同的 container 參數）：

```bash
srun --jobid=<JOBID> --overlap --pty \
  --container-image=/mnt/home/csl426-aicr-ae5f63/containers/voxcpm360-app.sqsh \
  --container-mounts=/mnt/shared/p06/VoxCPM360:/app \
  --container-mount-home --container-writable /bin/bash
```

省略 `--container-*` 只會進到運算節點（非容器）。
`groups: cannot find name for group ID 1023` 是無害警告。
⚠️ 依 CLAUDE.md §5.2，此通道**嚴禁執行額外計算**。

### 中止

```bash
scancel <JOBID>
```

⚠️ 訓練在互動式 shell 裡時，`scancel` 會連 shell 一起殺。
⚠️ 重跑前若 `checkpoints/<run>/latest/` 存在，**程式會自動續訓**，要從頭須先移除。

### 跑完後（四步，缺一不可）

```bash
# ① 挑最佳 checkpoint
ssh taipei-1 "grep '^\[val\]' /mnt/shared/p06/VoxCPM360/checkpoints/<run>/train.log \
  | sed -E 's/.*step ([0-9]+): loss\/total: ([0-9.]+).*/\2 \1/' | sort -n | head -5"

# ② 修權限（訓練會產生 group 不可讀的權重檔）
ssh taipei-1 "bash ~/scripts/fix_perms.sh"

# ③ 取回本機（預設只拉推論權重；也可跑 fetch_all_ckpt.sh，支援 run3 簡寫與 --with-pth 強制拉取 optimizer.pth）
bash scripts/fetch_run.sh <run> [best_step]
# 或拉全量 checkpoint：
bash scripts/fetch_all_ckpt.sh <run> [--with-pth]

# ④ 同步文件備份
scp taipei-1:/mnt/shared/p06/VoxCPM360/docs/*.md docs/voxcpm360/
```

**⑤ 評估泛化（強烈建議）**：訓練程式只吃單一 `val_manifest`，
故須另拿從未使用的 `val_unseen.jsonl`（163 位全新說話人）對最佳 checkpoint 跑推論評測。

---

## 維護規範

- 每輪一份**獨立 yaml**，檔名須反映真實設定（見 [RUNS.md](RUNS.md) 的 trial-1280 命名教訓）。
- **yaml 必須 commit**，並在本檔版本表加一列、[RUNS.md](RUNS.md) 加一節。
- 資料若偏離上方「資料」節（換 manifest／子集／過濾），**須在該輪明確標示**。
- 一次性腳本與 log 放 `~/.scripts_tmp/`（CLAUDE.md §4.2）。
- checkpoint 路徑 = yaml 的 `save_path`。
- 叢集端為工作副本，本機 `docs/voxcpm360/` 為唯讀備份，**改完即同步**。

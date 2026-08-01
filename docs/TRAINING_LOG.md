# VoxCPM2 微調訓練總覽

Taipei-1 · `p06` · 4× H100（torchrun/NCCL）· 模型 `openbmb/VoxCPM2` @ `bffb3df5`
細節見 [RUNS.md](RUNS.md)（各輪完整設定與曲線）· 名詞見 [GLOSSARY.md](GLOSSARY.md)

---

## 現況（2026-08-01 13:30）

### 一句話

VoxCPM2 全參微調已可用（最佳 `full_ft_tai8_step3000`，val 1.0954，已部署 GB10）；
目前在跑 **epoch 計量的 6 組 VoxCPM2 對照** 與 **4 組 BlueMagpie/Barbet 換腦實驗**。

### 執行中與排隊（Slurm，`p06` · `cnode2-021`）

| Job | 設定 | 說明 |
|---|---|---|
| 176753 | `e_tai8_base` | 🏃 基準，3 epoch |
| 176756 | `e_tai8_bs256` | 🏃 有效 batch 256 |
| 176786 | `e_tai8_lr2e5` | ⏳ LR ×2（曾因 YAML bug 失敗，已修並重送）|
| 176787 | `e_tai8_lr5e6` | ⏳ LR ÷2（同上）|
| 176757 | `e_mixed_base` | ⏳ 混合資料 |
| 176758 | `e_naer_base` | ⏳ naer 資料 |
| 176778 | `bm_bridge_first` | ⏳ **首次跑 BlueMagpie**，只訓 bridge，1 epoch |
| 176779 | `bm_tslm_base` | ⏳ 解凍 bridge+Barbet+spk_proj，聲學凍結 |
| 176780 | `bm_tslm_lr5e5` | ⏳ 同上，LR 5e-5 |
| 176781 | `bm_full` | ⏳ 除 AudioVAE 全解凍 |

> ⚠️ **BlueMagpie 四組是第一次跑，尚未驗證可行**。`bm_bridge_first` 刻意排最前面
> （參數最少、1 epoch），它若失敗其餘三組必然同樣失敗。

### 已部署到 GB10（`http://10.9.0.37:8800/`）

`models/native/` 放 LoRA、`checkpoints/` 放全參模型，按「重新掃描」即現身。

| 模型 | 最佳 val | val 集 |
|---|---|---|
| `full_ft_tai8_step3000` | 1.0954 | tai8 |
| `full_ft_mixed_step4000` | 0.9547 | mixed |
| `full_ft_naer_step1000` | 0.9347※ | naer |
| `lora-run1/2/3/4` | 1.1129 / 1.1098 / 1.0957 / 1.1143 | tai8 |
| `tai8-barbet-merge-v0` | — | 會出聲但**不講台語** |

※ naer 真正最佳在 step 1,500（0.9190），但 `save_interval: 1000` 沒存到。

---

## 今天學到的四件事（新 session 必讀）

### 1. val loss 只能同 val 集內比較

各輪用各自的 `val_seen.jsonl`。naer 的 0.93 看似勝過 tai8 的 1.09，
但那只反映朗讀語料好擬合，**不代表台語品質更好**。跨組比較必須用
`eval_ckpt.py` 跑同一份測試集。

### 2. 改用 epoch 計量後，過擬合點的規律才浮現

step 數跨資料集沒有可比性。換算後三輪的最佳點全落在 **epoch 1.7~2.5**：

| 輪次 | 總 epoch | 最佳點 |
|---|---|---|
| tai8 | 4.03 | epoch 1.73（step 3,000）|
| naer | 11.81 | epoch 2.53（step 1,500）|
| mixed | 4.03 | epoch 2.30（step 4,000）|

naer 看似「見頂特別早」只是資料量小、同樣步數跑了近 3 倍 epoch。
新設定一律 `num_epochs: 3.0`，超過必然過擬合。

### 3. Barbet 換腦：merge 可行但台語不會轉移

`tai8-barbet-merge-v0` 由 `merge_voxcpm_acoustic.py` 拼接而成（**非訓練產物**），
把 full_ft_tai8 的聲學 323 個 tensor 換進 BlueMagpie 骨架，保留 410 個。
實測**會出聲但完全不講台語**。

逐 tensor 比對確認原因：全參微調動了 577 個 tensor 的**全部**，其中
**254 個是 `base_lm.*`（MiniCPM4 文字腦）**，而那些在架構上無法搬進 Barbet
（1,536 hidden、28 層混合 Mamba2，形狀對不上）。**台語發音知識在文字腦，
merge 只搬得動聲學。**

tokenizer 實測：Barbet 少用 25% token，`恁`／`佗` 等台語專用字能整字編碼
（VoxCPM2 要拆 3 個 byte fallback）。**但 tai8 訓練文本全是華語漢字**
（「你為什麼不說實話」而非「你是按怎毋講實話」），優勢發揮不出來。

### 4. YAML 科學記號的地雷

`learning_rate: 2e-05` 會被 YAML 1.1 解析成**字串**（規範要求帶小數點與正負號，
如 `2.0e-05`），一路傳到 AdamW 才炸在
`TypeError: '<=' not supported between instances of 'float' and 'str'`，
且已浪費 20 分鐘 GPU（176754／176755 即因此 FAILED）。

已在 `train()` 進入點加防呆轉型。**寫設定時一律用 `0.00002` 這種十進位寫法。**


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

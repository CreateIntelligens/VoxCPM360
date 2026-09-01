# VoxCPM360 訓練文件導覽

**要幹什麼 → 看哪份**

| 我想… | 開這份 |
|---|---|
| 第一次做微調，要從環境準備開始 | [TRAINING_GUIDE_zh-TW.md](TRAINING_GUIDE_zh-TW.md) |
| 知道現在跑到哪、跑完要做什麼 | [TRAINING_LOG.md](TRAINING_LOG.md) 開頭「現況」 |
| 送出一輪訓練／監看／中止／取回結果 | [TRAINING_LOG.md](TRAINING_LOG.md) 的「操作手冊」 |
| 查某一輪用了什麼設定、val 曲線長怎樣 | [RUNS.md](RUNS.md) |
| 看不懂某個參數或名詞 | [GLOSSARY.md](GLOSSARY.md) |
| 規劃下一輪（tai8 + naer 混合） | [RUNS.md](RUNS.md) 的「run3 規劃」 |

---

## 三份文件的分工

刻意拆開，避免單一檔案變成流水帳：

| 檔案 | 行數 | 性質 | 更新時機 |
|---|---|---|---|
| **TRAINING_GUIDE_zh-TW.md** | ~370 | **入門指南**：環境準備、資料格式、LoRA／全參數微調步驟 | 流程或設定介面變動時 |
| **TRAINING_LOG.md** | ~182 | **總覽**：現況、版本表、核心結論、陷阱、操作手冊 | 每輪開跑／結束、狀態變化 |
| **RUNS.md** | ~334 | **逐輪細節**：changelog 沿革、完整 config、val 曲線、逐輪結論 | 新增一輪、跑完填結果 |
| **GLOSSARY.md** | ~300 | **名詞解釋**：訓練參數、分散式、Slurm/Enroot、診斷 | 出現新概念時 |

**分界原則**

- 「**現在**該做什麼」→ TRAINING_LOG（讀者是要動手的人）
- 「**當時**做了什麼、結果如何」→ RUNS（讀者是要回溯的人）
- 「這個詞**是什麼意思**」→ GLOSSARY（隨時查閱）

同一件事只在一處講清楚，其餘用連結指過去。例如 `max_batch_tokens` 陷阱只在
TRAINING_LOG 的「三個必記陷阱」寫，RUNS 提到時只引用。

---

## 目錄結構

```
docs/voxcpm360/                    ← 本機唯讀備份
├── README.md                      ← 本檔
├── TRAINING_LOG.md                總覽 + 操作手冊
├── RUNS.md                        逐輪細節
├── GLOSSARY.md                    名詞解釋
├── conf/                          7 份訓練設定（yaml）
│   ├── full_lora_run1.yaml        run1（已跑，未收斂）
│   ├── full_lora_run2.yaml        run2（過擬合，提早中止）
│   ├── full_lora_run3.yaml        run3（tai8+naer 混合）
│   ├── trial_lora_20epochs.yaml   流程驗證（檔名誤導，見 RUNS.md）
│   └── （其餘 4 份為上游 Docker 設定，未在 Taipei-1 執行）
├── train.sh                       訓練啟動腳本（含 GPU 自動偵測）
├── eval_ckpt.py                   跨 run 比較 checkpoint（固定驗證集、跑完整資料）
├── eval_run2_vs_run3.sh           上述評測的 sbatch 包裝（單卡）
└── logs/
    ├── full_train_run1.log        run1 完整原始輸出（363 KB）
    └── run1_lora_config.json      run1 checkpoint 的 LoRA 設定
```

`logs/` 只放**不可重建**的東西。run1 的訓練歷程只存在那個 log 裡 ——
checkpoint 本身不含曲線，一旦遺失無法重建。

---

## 本機 vs 叢集端

| | 路徑 | 角色 |
|---|---|---|
| **叢集端** | `/mnt/shared/p06/VoxCPM360/` | **工作副本**，實際被執行的版本 |
| **本機** | `F:\Taipei1\loginnode\docs\voxcpm360\` | **唯讀備份** |

⚠️ **改動一律在叢集端進行，再同步下來**，避免兩邊分歧。

```bash
cd F:/Taipei1/loginnode
scp taipei-1:/mnt/shared/p06/VoxCPM360/docs/*.md docs/voxcpm360/
scp "taipei-1:/mnt/shared/p06/VoxCPM360/conf/voxcpm_v2/*.yaml" docs/voxcpm360/conf/
scp taipei-1:/mnt/shared/p06/VoxCPM360/train.sh docs/voxcpm360/
```

**未備份**（留在叢集端）：checkpoint 權重（約 2.6 GB／輪）、TensorBoard 事件檔、
資料集（`tai8` 108 h + `naer` 120 h）。需要時用 `scripts/fetch_run.sh` 選擇性取回。

---

## 相關腳本

腳本分兩類，**執行位置不同**：

### 在叢集上跑（`~/scripts/`，叢集端）

| 腳本 | 用途 |
|---|---|
| `submit_vox_batch.sh` | 送出 sbatch 訓練作業（推薦入口） |
| `run_vox_interactive.sh` | 申請 4×H100 互動式 shell + 進容器 |
| `cancel_jobs.sh` | 互動式選取並中止作業 |
| `ssh_current_node.sh` | 連到當前作業的運算節點 |
| `fix_perms.sh` | **修 p06 共享權限**（每輪訓練後必跑） |
| `eval_run2_vs_run3.sh` | 跨 run 公平比較 checkpoint（單卡，不必等訓練結束） |

> ⚠️ **挑最佳 checkpoint 不可直接讀訓練 log 的 val** —— `validate()` 只取
> 10 個 batch（160 筆），排名會被抽樣雜訊翻轉。一律用 `eval_ckpt.py`
> 跑完整驗證集，詳見 [RUNS.md](RUNS.md) 的評測章節。

### 在本機跑（`F:\Taipei1\loginnode\scripts\`）

| 腳本 | 用途 |
|---|---|
| `fetch_run.sh` | 訓練後取回 log／權重／TensorBoard 到 `docs/voxcpm360/runs/` |
| `check_sync_status.sh` | 檢查資料集同步狀態 |

> `fix_perms.sh` 本機也留一份（供編輯／版控），但**要在叢集端執行**。

---

## 為什麼需要 fix_perms.sh

`/mnt/shared/p06` 是團隊共享目錄，但新檔案預設不給 group 寫入：

| 來源 | 預設權限 | 問題 |
|---|---|---|
| `scp` 上傳 | `-rw-r-----` | umask 0022 剝掉 group write |
| 訓練寫的權重檔 | `-rw-------` | **p06 成員連讀都不行** |

目錄的 SGID 只保證 group 歸屬正確，**不會補 write 權限**。
訓練進行中會持續產生新的權重檔，所以**每輪結束都要再跑一次**。

---

## 專案其他文件

| 檔案 | 內容 |
|---|---|
| [../OPERATIONS.md](../OPERATIONS.md) | Taipei-1 實戰操作、容器轉換、資料集傳輸 |
| [../TAI8_SYNC_REPORT.md](../TAI8_SYNC_REPORT.md) | `tai8` 同步、權限 SOP、容量統計 |
| `../../CLAUDE.md` | 叢集操作規範（登入節點限制、Slurm、Enroot） |

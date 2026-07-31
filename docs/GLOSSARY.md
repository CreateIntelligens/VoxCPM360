# 名詞對照表（VoxCPM2 LoRA 微調 @ Taipei-1）

> 本檔整理訓練設定、叢集環境與診斷過程中出現的名詞。
> 相關文件：[TRAINING_LOG.md](TRAINING_LOG.md)（各版本結果紀錄）

---

## 1. 訓練規模：步數與資料量

這組名詞最容易混淆，也是 run1 未收斂的根源。

| 名詞 | 含義 | run1 | run2 |
|---|---|---|---|
| `batch_size` | **單張 GPU 單次** forward 的樣本數 | 1 | 16 |
| `grad_accum_steps` | 累積幾次梯度才更新一次權重 | 2 | 2 |
| `world_size` | 參與訓練的 GPU 總數（由 torchrun 注入） | 4 | 4 |
| **有效 batch** | 一次權重更新真正看到的樣本數<br>= `batch_size × grad_accum × world_size` | **8** | **128** |
| `max_steps` / `num_iters` | 權重更新的總次數 | 84000 | 17400 |
| **epoch** | 完整掃過訓練集幾遍 | **3.02** | **10.02** |

### 為什麼「步數多」不等於「資料看得多」

epoch 由程式在 `scripts/train_voxcpm_finetune.py:332` 計算：

```
epoch = step × grad_accum_steps × batch_size × world_size / num_train_samples
```

run1 的 84000 步 × 有效 batch 8 = 672,000 樣本 ÷ 222,383 ≈ **3.02 epoch**。
run2 只用 17400 步，但 × 有效 batch 128 = 2,227,200 ÷ 222,383 ≈ **10.02 epoch**。

**步數少 4.8 倍，資料量反而多 3.3 倍。** 規劃訓練時要看有效 batch 和 epoch，不能只看 `max_steps`。

### 有效 batch 為何影響收斂

有效 batch 太小 → 每次更新的梯度是少數樣本的估計 → **梯度雜訊大**，模型在雜訊中震盪而非穩定下降。run1 有效 batch 僅 8，val loss 在 1.11~1.15 間來回 82000 步就是這個現象。

---

## 2. `max_batch_tokens` 與樣本過濾（重要陷阱）

| 名詞 | 含義 |
|---|---|
| `max_batch_tokens` | 一個 batch 的 token 預算上限，用來防 OOM |
| `max_sample_len` | **單筆樣本**允許的最大 token 數（衍生值，非直接設定） |
| token 長度 | 有 ref_audio 時 = `文字長度 + 音訊幀數 + 參考音訊幀數 + 4` |

### 陷阱：兩者是除法關係

`scripts/train_voxcpm_finetune.py:147`：

```python
max_sample_len = max_batch_tokens // batch_size   # 超過此長度的樣本會被「靜默丟棄」
```

**放大 `batch_size` 會反過來壓縮單筆樣本長度上限**。若只調 batch 而不同步調大 `max_batch_tokens`，資料會無聲消失——log 只印一行 `Filtering N / 222383 ...`，很容易漏看。

實測本資料集長度分布：p50=138、p90=175、p99=222、max=630。

| batch_size | max_batch_tokens | max_sample_len | 保留率 |
|---|---|---|---|
| 1（run1） | 4096 | 4096 | 100% |
| 16 | 4096 | 256 | 99.73% |
| **16（run2）** | **8192** | **512** | **100.00%**（僅丟 4 筆） |
| 32 | 4096 | 128 | **32.79%** ⚠️ 丟掉 149,454 筆 |

若當初照「batch 開到 32」而沒動 `max_batch_tokens`，會靜默丟掉 **67% 的訓練資料**。

---

## 3. 損失函數（Loss）

| 名詞 | 含義 | run1 起→迄 |
|---|---|---|
| `loss/diff` | Diffusion 損失，負責**音訊內容與音質**，是主要學習目標 | 1.088 → 1.027 |
| `loss/stop` | 停止預測損失，決定**何時結束發話**，通常最先收斂 | 0.275 → 0.124 |
| `loss/total` | 加權總和，權重由 config 的 `lambdas` 指定（目前皆 1.0） | 1.363 → 1.152 |
| `grad_norm` | 梯度範數，反映更新幅度；配合 `max_grad_norm: 1.0` 做裁剪防爆炸 | 約 0.13~0.50 |

run1 的問題正是 `loss/stop` 在前 2000 步就收斂了（貢獻了幾乎全部的下降），而真正重要的 `loss/diff` 六小時只進步 0.06。

---

## 4. 學習率排程

| 名詞 | 含義 |
|---|---|
| `learning_rate` | 學習率峰值（**非全程固定值**） |
| `warmup_steps` | 從 0 線性爬升到峰值的步數，避免初期大梯度破壞預訓練權重 |
| cosine decay | warmup 後依餘弦曲線衰減，尾段趨近 0 |
| `weight_decay` | 權重衰減（L2 正則化），抑制過擬合 |

run1 log 尾段顯示 `lr: 0.000000` 是 **6 位小數的四捨五入顯示**，不是排程壞掉——cosine decay 到末期本就趨近 0。

批量與學習率需一起調：run2 有效 batch 放大 16 倍，LR 反而由 1e-4 降到 5e-5，因為大 batch 的梯度已較穩定，過大的 LR 會跳過極小值。

---

## 5. LoRA 相關

| 名詞 | 含義 |
|---|---|
| LoRA | Low-Rank Adaptation，凍結原模型、只訓練插入的低秩矩陣，產出檔案僅 72 MB |
| `lora_A` / `lora_B` | 低秩分解的兩個矩陣，log 中顯示 `True`（可訓練）；原權重顯示 `False`（凍結） |
| `r` | 秩（rank），控制 LoRA 容量，目前 32 |
| `alpha` | 縮放係數，實際縮放為 `alpha / r`；目前 32/32 = 1.0 |
| `dropout` | LoRA 層的 dropout，目前 0.0 |
| `enable_lm` | 對 `base_lm` + `residual_lm` 套用 LoRA（`src/voxcpm/model/voxcpm2.py:135`） |
| `enable_dit` | 對 `VoxCPMLocDiT` 套用 LoRA |
| `enable_proj` | 對 projection Linear 層套用 LoRA（目前關閉） |

---

## 6. 資料集與 Manifest

| 名詞 | 含義 | 樣本數 |
|---|---|---|
| manifest | JSONL 格式的資料清單，每行一筆樣本的路徑與中介資料 | — |
| `train.jsonl` | 訓練集 | 222,383 |
| `val_seen.jsonl` | 驗證集，**說話人已在訓練集出現**（測擬合程度） | 11,368 |
| `val_unseen.jsonl` | 驗證集，**說話人未見過**（測泛化能力） | 10,666 |
| `test_unseen.jsonl` | 最終測試集 | 18,820 |

manifest 欄位：`audio`、`duration`、`ref_audio`、`ref_duration`、`text`、`speaker_id`、`episode`、`utterance_id`、`dataset_id`。

| 名詞 | 含義 |
|---|---|
| `ref_audio` | 參考音訊，提供音色資訊供模型複製（zero-shot TTS 的關鍵輸入） |
| `voxcpm2_abs` | 此 manifest 目錄名，`abs` 指路徑已改為**絕對路徑** |

run1 只用 `val_seen` 評估，**無法判斷泛化能力**——這是 run2 建議加入 `val_unseen` 的原因。

---

## 7. 分散式訓練與啟動方式

| 名詞 | 含義 |
|---|---|
| `torchrun` | PyTorch 官方多進程啟動器，**負責注入** `WORLD_SIZE`/`RANK`/`LOCAL_RANK` |
| `--nproc_per_node` | 每節點啟動幾個進程（= 用幾張 GPU） |
| `rank` | 全域進程編號（4 卡為 0~3）；`local_rank` 是節點內編號 |
| NCCL | NVIDIA 的 GPU 間集體通訊庫，多卡梯度同步用 |
| DDP | Distributed Data Parallel，各卡跑不同資料、同步梯度 |
| `DistributedSampler` | 確保各 rank 拿到不重複的資料切片；每 epoch 需 `set_epoch()` 重洗 |
| AMP | Automatic Mixed Precision，本專案實際以 bfloat16 執行 |

### 為什麼「裸 python」只會用單卡

`src/voxcpm/training/accelerator.py:24`：

```python
self.world_size = int(os.getenv("WORLD_SIZE", "1"))
```

`WORLD_SIZE` 由 torchrun 注入。裸 `python` 執行 → 取到預設值 `1` → 走單卡路徑，**即使 Slurm 已配 4 張卡也只用 1 張**。

專案**支援多卡**（run1 就是實跑 4 卡成功），差別純粹在啟動方式。修改後的 `train.sh` 會自動偵測 GPU 數並選擇 `torchrun` 或 `python`，也可用 `NPROC_PER_NODE` 覆寫。

---

## 8. Checkpoint

| 名詞 | 含義 |
|---|---|
| `save_interval` | 每幾步存一次 checkpoint（run1: 8000，run2: 1000） |
| `valid_interval` | 每幾步跑一次驗證（run1: 2000，run2: 500） |
| `lora_weights.safetensors` | LoRA 權重本體（72 MB），推論時搭配原模型載入 |
| `optimizer.pth` | 優化器狀態（145 MB），**續訓才需要**，單純推論可忽略 |
| `scheduler.pth` | 學習率排程器狀態，續訓用 |
| `training_state.json` | 記錄當前 step，如 `{"step": 84000}` |
| `latest/` | 指向**最後一次**存檔，**不是最佳存檔** |

⚠️ **`latest` ≠ best**。run1 最佳在 step_48000（val 1.1129），而 `latest` 是 step_84000（val 1.1516，較差）。本專案**無 best-checkpoint 追蹤與 early stopping**，只能事後從 log 挑。

---

## 9. Taipei-1 叢集環境

| 名詞 | 含義 |
|---|---|
| Login Node | 共享純 CPU 節點，僅供編輯/傳檔/提交作業；**禁止重度運算** |
| `cnodeXXX` / `cnode2-021` | 運算節點，僅在作業執行期間可連線（run1 跑在 `cnode2-021`） |
| Slurm | 叢集作業排程系統 |
| `sbatch` | 提交背景批次作業 |
| `srun` | 申請資源並執行（可搭 `--pty` 開互動 shell） |
| `squeue` | 查詢佇列狀態 |
| partition (`-p p06`) | 資源分區；`p06` 為團隊專屬（免排隊），`defq` 全叢集共享（常 PENDING） |
| `--gres=gpu:h100:4` | 申請 4 張 H100 |
| Enroot | NVIDIA 的無 root 容器執行環境（叢集無 Docker、無 sudo） |
| `.sqsh` | SquashFS 容器映像檔，Enroot 使用的格式 |
| `--container-mounts` | 掛載主機路徑進容器，如 `/mnt/shared/p06/VoxCPM360:/app` |
| `--container-mount-home` | 自動掛載家目錄 |
| `--container-writable` | 允許容器內寫入 |
| DDN | 叢集的高效能平行檔案系統（`/mnt/home` 與 `/mnt/shared` 皆在其上） |

### 路徑對照

| 路徑 | 用途 |
|---|---|
| `/mnt/home/<USER>/` | 個人家目錄，配額 5 TB |
| `/mnt/shared/p06/` | 團隊共享目錄（分區名即目錄名） |
| `/app` | 容器內掛載點，對應 `/mnt/shared/p06/VoxCPM360` |
| `~/.scripts_tmp/` | 一次性腳本與 log 存放處（依專案規範，勿散落家目錄頂層） |

---

## 10. 診斷相關

| 名詞 | 含義 |
|---|---|
| `log interval` | 兩次 log 之間的耗時，用來判斷吞吐是否穩定 |
| I/O 暖機 | run1 前約 250 步耗時 155s→33s，之後穩定 13.6s；DDN 首次讀取需建快取，**非 bug** |
| TensorBoard | 訓練曲線視覺化，事件檔在 `checkpoints/<run>/logs/events.out.tfevents.*` |
| `destroy_process_group() was not called` | 程式結束未清理進程組的**無害警告** |
| `D` state | Linux 不可中斷睡眠；真正的 I/O 卡死徵兆（`kill -9` 無效） |

---

## 11. 命名教訓

`conf/voxcpm_v2/trial_lora_20epochs.yaml` 的檔名**同時錯在兩處**：

1. 不是 20 epochs — 實際 `max_steps: 1280`，僅 **0.0115 epoch**
2. 不再是 trial 資料 — manifest 已從 `smoke_train.jsonl` 改成全量 `train.jsonl`（222,383 筆）

建議改名 `trial_lora_1280steps.yaml`。**檔名應描述真實設定**，尤其步數與資料集這類事後難以回溯的資訊。

# NVIDIA Taipei-1 DGX Cluster 專案知識與操作規範 (Project Rules)

> 本文件整合 **NVIDIA Taipei-1 官方 Onboarding 指南**、**david888 wiki** 及 **Mason Wu 使用者手冊**。任何 AI 助手或開發人員在此專案工作區執行指令、撰寫腳本或部署作業時，**必須嚴格遵守以下條款與限制**。
>
> 📌 **實戰操作、容器轉換與資料集傳輸紀錄**：請參閱 [docs/OPERATIONS.md](file:///F:/Taipei1/loginnode/docs/OPERATIONS.md)。
>
> 📌 **`tai8` 資料集同步、權限 SOP 與容量統計報告**：請參閱 [docs/TAI8_SYNC_REPORT.md](file:///F:/Taipei1/loginnode/docs/TAI8_SYNC_REPORT.md)。
>
> 📌 **VoxCPM360 LoRA 訓練**：先看 [docs/voxcpm360/README.md](file:///F:/Taipei1/loginnode/docs/voxcpm360/README.md)（文件導覽），內含 `TRAINING_LOG.md`（現況＋操作手冊）、`RUNS.md`（逐輪細節）、`GLOSSARY.md`（名詞）的分工說明。叢集端來源 `/mnt/shared/p06/VoxCPM360/`，本機為唯讀備份。

---

## 1. ⚠️ 登入節點 (Login Node) 嚴格限制與禁忌 (Critical Rules)

Taipei-1 的登入節點為所有人共享的純 CPU 伺服器，**僅用於程式編輯、檔案管理與 Slurm 作業提交**。

- ❌ **嚴禁執行重度運算**：禁止在 Login Node 上執行任何計算密集或記憶體密集的任務，必須透過 Slurm 提交至 DGX H100 運算節點。
- ❌ **嚴禁在 Login Node 執行 VS Code Server**：直接透過 VS Code Remote 連接 Login Node 會迅速耗盡 Process 上限，導致整個環境鎖死。請改在分派到的運算節點上使用 [VS Code Remote Tunnels](https://code.visualstudio.com/docs/remote/tunnels)。
- ⚠️ **資源配額上限**：
  - **Process 上限**：每個工作階段最多 **1024 個 processes**（避免開過多 `tmux` 視窗或背景程序）。
  - **SSH 連線上限**：每位使用者最多 **20 個同時 SSH 登入**。
  - **記憶體上限**：每個工作階段最多 **8 GB RAM**。
- 💡 **tmux 使用建議**：避免在背景留下多個閒置的 tmux，並建議為 tmux 設定非登入 shell (non-login shell)。

---

## 2. 🔐 連線、驗證與 SSH 憑證管理

### 憑證與帳號
- **憑證認證平台**：[NVIDIA Frontier](https://frontier.nvidia.com/)
- **帳號格式**：Email 前綴與網域 + `-{chars}` (例如 `username-nvid-xxxxxx`)。
- **SSH 憑證有效期**：**建議設定 2d (48 小時)**（`-cert.pub` 憑證失效時需重新至 Frontier 生成，系統不會自動通知）。
- **檢查憑證效期指令**：
  ```bash
  ssh-keygen -L -f ~/.ssh/id_ed25519-cert.pub
  ```
- **清除過期 Known Hosts**（當出現 `REMOTE HOST IDENTIFICATION HAS CHANGED` 警告時）：
  ```bash
  sed -i '/jb1\.frontier\.nvidia\.com/d;/\.tun\.tp1-cluster\.nvidia\.com/d' ~/.ssh/known_hosts
  ```

### SSH 連線設定檔範例 (`~/.ssh/config`)
> ✅ **已驗證**（2026-07-29）：`taipei-1` 為登入節點的固定入口（`hostname -f` 顯示為 `MOEA.cm.cluster`），與第 5 節的 `cnodeXXX` 運算節點是不同層級——`cnodeXXX` 只在 Slurm 作業（`srun`/`sbatch`）實際執行期間才會被分配、才能連線，作業結束即失效。設好本節設定檔後，日常操作只需 `ssh taipei-1`，不需預先知道任何運算節點名稱。
```sshconfig
Host jb1
    HostName jb1.frontier.nvidia.com
    User <YOUR_USER_NAME>
    Port 2222
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host taipei-1
    HostName <TARGET_IP>
    User <YOUR_USER_NAME>
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    Port 2222
    ProxyJump jb1
```
> 設定後只需執行：`ssh taipei-1`

### 內網其他主機（非 Taipei-1，用於推論／前端部署）

Taipei-1 只負責訓練（無 sudo、無 docker）。**訓練產出的權重要實際跑起來，
必須送到下列內網機器**——它們才有 docker/compose 環境。

| 主機 | 連線 | 規格 | 用途 | 狀態 |
|---|---|---|---|---|
| **GB10** | `ssh altos@10.9.0.37` | `gn100-d091` · NVIDIA GB10 · Docker 29.2.1 · 剩 **2.4 TB** | 跑 VoxCPM360 推論服務（docker compose） | ✅ 金鑰已布署（2026-07-31） |
| **A4000** | `ssh human@10.9.0.32` | `gpu-a4000` · RTX A4000 · Docker 29.1.3 · 剩 993 GB | 中繼站／次要推論 | ✅ 金鑰已布署（2026-07-31） |

> ⚠️ 主機名 `gn100-d091` **無 DNS 記錄**，必須用 IP `10.9.0.37`。
>
> ⚠️ **GB10 是 Grace-Blackwell，回報 sm_121**，但 torch cu128 的 nvrtc 只認到
> sm_120，runtime JIT fusion（如 `@torch.jit.script` 的 snake op）會以
> `invalid value for --gpu-architecture` 失敗。**必須設 `PYTORCH_JIT=0`**
> 讓這些走 eager——`docker-compose.yml` 的 `app` 服務已內建此環境變數。

**權重搬運路線**：Taipei-1 `/mnt/shared/p06/VoxCPM360/checkpoints/`
→ 本機 `docs/voxcpm360/runs/`（`scripts/fetch_run.sh`）→ A4000／GB10。

> 💡 **推論只需 `model.safetensors`**（全參微調約 9.2 GB），
> **不需 `optimizer.pth`**（18 GB，僅續訓用）。單個全參 ckpt 完整為 26 GB，
> 只取權重可省掉約七成傳輸量。

---

## 3. 📂 儲存空間與資料傳輸指南

- **個人主目錄 (Home)**：`/mnt/home/<USERNAME>`
  - 預設配額 **5 TB**，屬 DDN 高效能平行檔案系統。
  - 訓練資料與執行腳本應存放於此。
- **團隊共享目錄 (Shared)**：`/mnt/shared/<ORG_NAME>`
  - 用於團隊成員間共用資料（名稱通常為 Slurm partition 名稱）。

### 資料傳輸範例 (sftp / scp / rsync)
```bash
# RSYNC 傳輸 (最推薦，支援斷點續傳)
rsync -azv --progress --partial -e \
  "ssh -i ~/.ssh/id_ed25519 -p 2222 -o 'ProxyCommand=ssh -i ~/.ssh/id_ed25519 -W %h:%p <USER>@jb1.frontier.nvidia.com -p 2222'" \
  <LOCAL_FILE_OR_DIR> <USER>@<TARGET_IP>:/mnt/home/<USER>/

# SFTP 傳輸
sftp -i ~/.ssh/id_ed25519 -P 2222 -o "ProxyCommand=ssh -i ~/.ssh/id_ed25519 -W %h:%p <USER>@jb1.frontier.nvidia.com -p 2222" <USER>@<TARGET_IP>
```

---

## 4. 🚀 Slurm 排程與 Enroot 容器化開發

### 4.1 載入 Slurm 模組
登入 Taipei-1 後，請先載入 Slurm 模組環境：
```bash
ml slurm   # 或 module load slurm
```
> ⚠️ **注意**：`ml`/`module` 是定義在互動式 login shell（`.bashrc`）裡的 shell function。若透過 `ssh taipei-1 "command"` 這種非互動方式直接下指令，會出現 `command not found`；需改用 `ssh taipei-1 "bash -lc 'ml slurm; sinfo'"`，或先 `ssh taipei-1` 進入互動 shell 後再執行。

### 4.2 常用 Slurm 指令
- **檢視叢集狀態**：`sinfo`
- **檢視佇列作業**：`squeue -u <USERNAME>`
- **提交 Batch 作業**：`sbatch <script.sh>`
- **互動式申請運算節點**：`srun -N 1 -p p06 --mpi=pmix --gres=gpu:h100:1 --ntasks-per-node 1 --pty /bin/bash`
- **取消作業**：`scancel <JOB_ID>`
- 💡 **Partition 分區選擇重點**：預設 `-p defq` 為全叢集共享，常常排隊很久（`PENDING`）；提交 Slurm 作業時**強烈建議優先指定團隊專屬分區 `-p p06`**，可免去漫長排隊立刻取得資源執行！
- 💡 **一次性/暫存腳本存放規範**：所有自動生成或一次性的作業腳本，請統一建立並存放於 `~/.scripts_tmp/` 目錄中，**嚴禁散落放在 Home 家目錄頂層 (`~/`)**，維持主目錄整潔。

### 4.3 Enroot 自訂映像檔 (6 步驟)
Taipei-1 無法直接執行 `docker build` 且無 `sudo` 權限，自訂環境需透過 Enroot：
1. **登入運算節點**申請臨時開發環境 (`srun ...`)
2. **建立基礎容器**：`enroot create --name my_custom_env base_image.sqsh`
3. **以 Root 模式啟動**：`enroot start --root -w my_custom_env /bin/bash`
4. **安裝套件/修改代碼**：`pip install <package>` 或 `apt install <package>` 後輸入 `exit`
5. **導出為新映像檔**：`enroot export --output my_custom_env.sqsh my_custom_env`
6. **刪除臨時容器**：`enroot remove my_custom_env`

### 4.4 容器運行常用參數
- `--container-writable`：啟用容器內檔案系統寫入。
- `--container-remap-root`：容器內映射為 root 權限。
- `--container-mount-home`：自動將主機家目錄掛載至容器內。

---

## 5. 📊 GPU 使用率監控與日誌

### 5.1 批次作業中加入背景 GPU 日誌紀錄
在 `sbatch` 腳本中加入 `nvidia-smi dmon`：
```bash
srun --mpi=pmix --container-image $CONT --container-writable --container-mount-home \
  bash -c \
  "nohup nvidia-smi dmon -s puc -o DT -f $HOME/gpu_usage-\$(hostname).log > /dev/null 2>&1 & \
   exec python /workspace/your_model/train.py"
```

### 5.2 SSH 直連運算節點監控
作業執行中可開另一個終端分頁直連運算節點監控：
```bash
ssh <node-name>   # 例如 ssh cnode004
nvidia-smi        # 查看基本 GPU 與 PID
nvidia-smi dmon   # 查看動態利用率
```
> ⚠️ **注意**：SSH 存取僅在作業執行期間啟用，且**嚴禁在此通道執行額外計算**。

---

## 6. 🔒 共享目錄 `/mnt/shared/p06` 權限（上傳／訓練後**每次都要做**）

⚠️ **任何東西傳上 `/mnt/shared/p06` 之後，都要檢查「群組」與「權限」兩件事。**

### 兩種各自獨立的問題

| 問題 | 症狀 | 成因 |
|---|---|---|
| **① 群組錯成 `1023`** | `drwx------ 1023` | `rsync -a`／`cp -a` 保留來源家目錄的群組，**覆蓋掉 SGID 的 p06 繼承** |
| **② group 無寫入** | `-rw-r-----`（scp）<br>`-rw-------`（訓練權重） | scp 受 umask 0022；訓練程式自己寫出 600 |
| **③ 腳本 group 不可執行** | `-rw-rw----` 的 `.sh` | scp 不保留執行位元，團隊成員無法執行 |

**① 才是最容易漏的** —— 光補權限沒有用：`g+rw` 給的是「`1023` 群組可讀寫」，
p06 成員依然存取不到。**SGID 不保證群組正確**，rsync 會直接蓋掉它。

**③ 也不能只靠 `chmod -R g+rwX`** —— 大寫 `X` 只對「目錄」或「已有任一 x 位元
的檔案」生效，**對 scp 上來完全沒有 x 的 `.sh` 無效**，必須明確補 `g+x`。

### 修正

```bash
bash ~/scripts/fix_perms.sh                       # 預設 VoxCPM360
bash ~/scripts/fix_perms.sh /mnt/shared/p06/<其他路徑>
```

腳本會做四件事：`chgrp -R p06` → `chmod -R g+rwX` →
`.sh`／帶 shebang 的檔案補 `g+x` → 目錄補 SGID。

大目錄（如 `tai8` 有 1.1M 檔案）需數分鐘，建議背景執行：

```bash
nohup bash ~/scripts/fix_perms.sh /mnt/shared/p06/dataset202607_1/tai8 \
  > ~/.scripts_tmp/logs/fixperm.log 2>&1 &
```

### 檢查

```bash
find <路徑> ! -group p06 -print | head                    # 群組錯誤者
find <路徑> ! -perm -g=w -print | head                    # group 不可寫者
find <路徑> -name "*.sh" ! -perm -g=x -print | head       # 腳本 group 不可執行
```

> ⚠️ **不是修過一次就一勞永逸** —— 訓練進行中會持續產生新的 `-rw-------`
> 權重檔，**每輪訓練結束後都要再跑一次**。
>
> 📌 **前車之鑑**：`dataset202607_1/tai8/` 用 rsync 從家目錄搬過去後，
> 整個目錄樹是 `1023` 群組 + `700`，p06 團隊完全無法存取，
> 而同目錄下較早建立的 `naer/` 卻是正確的 `p06`。

---

## 7. 🧠 VoxCPM360 訓練：已知地雷與現況

### 7.1 文件與備份
- **入口**：[docs/voxcpm360/README.md](file:///F:/Taipei1/loginnode/docs/voxcpm360/README.md)
  （導覽）→ `TRAINING_LOG.md`（現況＋操作手冊）→ `RUNS.md`（逐輪細節）。
- **叢集端 `/mnt/shared/p06/VoxCPM360/` 是工作副本**，本機 `docs/voxcpm360/` 為唯讀備份。
  改動一律在叢集端進行後同步下來，**改完即同步**。

> ⚠️ **本機 `AGENTS.md`（= `CLAUDE.md` 符號連結）才是新 session 讀的檔案。**
> 只更新叢集端文件，下次對話會讀到舊資訊。重要結論兩邊都要寫。

**三台機器的分工**（詳見第 2 節主機表）：

| 機器 | 角色 | GitHub 寫入 | 可連 Taipei-1 |
|---|---|---|---|
| Taipei-1 | 訓練（Slurm + Enroot） | ❌ 無憑證、無 `gh`、**無對外網路** | — |
| GB10 `10.9.0.37` | 推論服務、前端、主要 git 操作 | ✅ `gh` 已登入 `ac-Spark` | ❌ |
| A4000 `10.9.0.32` | 備援、可直接與叢集同步 | ✅ `gh` + `credential.helper store` | ✅ |

Taipei-1 無 docker、無 sudo。GB10 與 A4000 **不同網段，兩者無法直連**。

**權重搬運**：Taipei-1 → 本機 → GB10（GB10 連不到叢集，必須中轉）。
**Taipei-1 的 commit 要推 GitHub**：兩條路
- 經 GB10：`git bundle` 搬過去再推（一批 commit 通常只有幾十 KB）
- 經 A4000：它連得到叢集，可直接 `git pull` 再 push，少一次中轉

> ⚠️ **程式碼改動一律在 GB10 做，Taipei-1 只 pull。**
> 在 Taipei-1 直接編輯看似省一輪往返，但遠端一有整合（別人或 codex 推的
> merge commit）就會分歧，得手工解衝突 —— 2026-08-03 即因此在
> `AGENTS.md`／`TRAINING_LOG.md`／`train_voxcpm_finetune.py` 撞上三處衝突。
> 唯一該破例的情況是叢集上的訓練正在失敗、需要立刻修；**破例後要馬上
> 同步回 GB10 並推送**，不要累積。

> ⚠️ **在叢集端絕不要用 `git add -A`** —— 它會遞迴 stat 800 GB 以上的
> `checkpoints/`（雖被 ignore 仍要掃），在 Lustre 上會卡數分鐘並鎖住 index。
> 一律用明確路徑。

**SSH 憑證 48 小時到期**，過期徵狀是 `Permission denied (publickey)`。
到 [Frontier](https://frontier.nvidia.com/) 重簽後，本機與 A4000 都要更新
（`~/.ssh/id_ed25519-cert.pub`）。
- **本機 `scripts/fetch_run.sh` / `scripts/fetch_all_ckpt.sh`** 在本機執行，訓練後取回 log／權重／TensorBoard。支援簡寫名稱（如 `run3` / `3` 自動擴充為 `full_lora_run3`）與 `--with-pth` 強制下載續訓 `.pth` 檔。

### 7.2 訓練設定的地雷
1. **`max_sample_len = max_batch_tokens // batch_size`** —— 兩者是除法關係。
   調大 `batch_size` 不同步調大 `max_batch_tokens` 會**靜默丟棄**超長樣本
   （`batch_size: 32` + `4096` 只剩 32.79% 資料）。
2. **`latest` ≠ 最佳 checkpoint** —— run1 最佳在 step_0048000，`latest` 是較差的
   84000。**2026-08-01 起已自動追蹤 `best/`**（每次驗證比對 val，一有新低就另存，
   並寫 `best_metric.json` 記 step 與 val），不必再人工從 log 挑。
3. **LR 排程綁定 `max_steps`** —— `get_cosine_schedule_with_warmup` 把 `max_steps`
   烘進曲線，且續訓會還原 `scheduler.pth`。**不可「先跑短再加大 max_steps 續訓」**
   （等於 LR≈0 空轉）；要跑更長只能重跑。
4. **YAML 科學記號會被讀成字串** —— `learning_rate: 2e-05` 依 YAML 1.1 規範必須
   帶小數點與正負號（`2.0e-05`）才算數字，否則解析為**字串**，一路傳到 AdamW
   才炸在 `TypeError: '<=' not supported between float and str`，且已浪費模型載入
   與 warmup 的時間（job 176754／176755 因此 FAILED）。**一律用 `0.00002` 這種
   十進位寫法。** `train()` 進入點已加防呆轉型。
5. **用 epoch 計量，不要用 step** —— step 跨資料集沒有可比性：tai8 的 7,000 步是
   4.03 epoch，naer 同樣 7,000 步卻是 11.81 epoch。換算後三輪全參微調的最佳點
   全落在 **epoch 1.7~2.5**，規律才浮現（naer 看似見頂特別早，只是資料量小、
   同樣步數跑了近 3 倍 epoch）。設定用 `num_epochs` / `valid_per_epoch` /
   `save_per_epoch`，腳本於資料集載入後換算並印出對照。
6. **val loss 只能同 val 集內比較** —— 各輪用各自的 `val_seen.jsonl`。naer 的 0.93
   看似勝過 tai8 的 1.09，但那只反映朗讀語料好擬合。⚠️ **mixed 的 val 有 56%
   來自 naer**（naer val 14,832 句 > tai8 的 11,473），其「優勢」多半來自測試集
   組成而非模型。跨組比較**必須**用 `eval_ckpt.py` 跑同一份測試集。

### 7.2.1 資料集（三種）

| 資料集 | train | val | 內容 |
|---|---|---|---|
| `tai8` | 228,961 | 11,473 | 純台語劇集對白 |
| `naer` | 66,104 | 14,832 | 純朗讀語音 |
| `mixed` | 288,487 | 26,200 | 前兩者**聯集**（差 2.2% 應為去重／過濾）|

路徑 `/mnt/home/<USER>/dataset202607_1/<名稱>/manifests/*/`。
⚠️ **文本一律是華語漢字**（「你為什麼不說實話」而非台語漢字「你是按怎毋講實話」）。

### 7.2.2 Barbet／BlueMagpie 換腦的結論

`models/barbet/tai8-barbet-merge-v0` 由 `merge_voxcpm_acoustic.py` **拼接**而成
（**非訓練產物**）：BlueMagpie 骨架保留 410 個 tensor（Barbet + 已訓練的 bridge），
聲學 323 個換成 `full_ft_tai8_step3000` 的權重。

**實測會出聲但完全不講台語。** 逐 tensor 比對確認原因：全參微調動了 577 個
tensor 的**全部**，其中 **254 個是 `base_lm.*`（MiniCPM4 文字腦）**，而那些
**在架構上無法搬進 Barbet**（1,536 hidden、28 層混合 Mamba2，形狀對不上）。
**台語發音知識在文字腦，merge 只搬得動聲學。**

tokenizer 實測：Barbet 少用 25% token，`恁`／`佗` 等台語專用字能整字編碼
（VoxCPM2 要拆 3 個 byte fallback）。**但訓練文本全是華語漢字**，優勢發揮不出來。

**訓練流程**（`conf/voxcpm_v2/bm_*.yaml`）：merge 產物已含台語聲學，
直接拿 tai8 資料續訓，由 `bluemagpie.loading.set_training_stage` 決定解凍範圍：

| stage | 解凍 | 可訓練參數 | 實測 val |
|---|---|---|---|
| `bridge` | 只有 `enc_to_tslm_proj` + `tslm_adapter` | 29.9M / 2030.8M (1.5%) | 1.1350 |
| `tslm` | bridge + Barbet + speaker_projector，**聲學凍結** | 大部分 | 1.1035 (LR 1e-5) |
| `tslm` | 同上，**LR 5e-5** | 大部分 | **1.0935** ⭐ |
| `full` | 除 AudioVAE 外全部 | 全部 | **CUDA OOM**（4 卡放不下）|

⭐ **`tslm` + LR 5e-5 與最佳 VoxCPM2（1.0935）完全打平。**
LR 5e-5 勝過 1e-5 達 0.010（超出噪音 0.005）—— Barbet 學的是全新映射
而非微調，需要較大 LR。聲學凍結是刻意的：保住 merge 帶入的台語聲學，
只讓 Barbet 學「文字 → 語意」那一段。

> ⚠️ **DDP 地雷**：`tslm`／`full` stage 會解凍 `speaker_projector`，但訓練資料
> 沒有 speaker centroid，該模組拿不到梯度，DDP 會報
> `Expected to have finished reduction in the prior iteration`。
> 訓練腳本已明確凍結六個不參與 forward 的模組（`speaker_projector`、
> `duration_head`、`stop_context_*`、`spk_*_gate`）。**不要改用
> `find_unused_parameters=True`** —— 那會每步掃描計算圖且掩蓋真正的問題。
`bluemagpie` 與 `barbet` 是純 Python 套件但未裝進容器，且**運算節點無對外網路**
無法 pip install，故放在 `/app/vendor` 由 `train.sh` 以 `PYTHONPATH` 載入
（該目錄已 gitignore，不屬於本 repo）。

### 7.3 多卡啟動
`train.sh` 已支援 GPU 數自動偵測（多卡 `torchrun`／單卡 `python`）。
**裸 `python` 會使 `WORLD_SIZE=1` 只用單卡**，即使 Slurm 配了 4 張卡。
送出後 30 秒務必確認 log 首行 `GPU 數：4`。

### 7.3.1 模型命名規則（2026-08-03 起）

```
<訓練法>-<資料>-<關鍵超參>-e<最佳epoch>-<MMDD>
```

| 欄位 | 取值 |
|---|---|
| 訓練法 | `ft` 全參微調／`lora`／`bm` BlueMagpie(Barbet) |
| 資料 | `tai8`／`naer`／`mixed` |
| 關鍵超參 | 該輪唯一變動者，如 `lr2e5`、`bs256`、`r64` |
| 最佳 epoch | **一位小數**，如 `e1.7` |
| 日期 | `MMDD`，區分同設定重跑 |

範例：`ft-tai8-lr2e5-e1.6-0801`、`lora-r32-mixed-e7.6-0731`、
`bm-tslm-tai8-lr5e5-e1.9-0803`。

> ⚠️ **epoch 不可取整** —— 全參微調的最佳點全落在 1.5~2.5，取整後會全部
> 變成 `e2` 而失去區分能力。一位小數是資訊所在：`e1.5` 與 `e2.3` 的差別是
> 「訓練不到一半就見頂」對「快到三分之二才見頂」。
>
> ⚠️ **不要用 step** —— step 跨資料集沒有可比性（見 7.2 第 5 點）。

### 7.4 命名教訓
`conf/voxcpm_v2/trial_lora_20epochs.yaml` 檔名**雙重誤導**：實際只 0.01 epoch，
且資料早已從 smoke 子集改為全量。**檔名務必反映真實設定。**

### 7.5 現況與交接（2026-08-03 10:00）

> ⚠️ 這節會過期。**開工先跑 `ssh taipei-1 "bash -lc 'ml slurm; squeue -u \$USER'"`
> 對照實況**，並看叢集端 `docs/TRAINING_LOG.md`（本機為備份，可能較舊）。

**已完成 11 輪訓練**，同一份 tai8 val 集上的排名：

| 模型 | val | 備註 |
|---|---|---|
| `ft-tai8-lr2e5-e1.6-0801` | **1.0935** | VoxCPM2 最佳 |
| `bm-tslm-tai8-lr5e5-e1.9-0803` | **1.0935** | **BlueMagpie 打平** |
| `ft-mixed-lr1e5-e2.0-0803` | 1.0951 | 混合資料 |
| `ft-tai8-lr1e5-e1.7-0731` | 1.0954 | |
| `lora-r32-mixed-e7.6-0731` | 1.0957 | LoRA，僅 72 MB |

**三個關鍵結論**：

1. **超參已無調整空間** —— LR ×2、÷2、batch ×2、max_steps 縮短，七輪全部落在
   1.09~1.10。這是配置上限，不是還沒調到。**別再掃超參。**
2. **混合資料對 tai8 沒有幫助** —— `ft-mixed-lr1e5-e2.0` 用 mixed 訓練但以
   tai8 val 評估得 1.0951，夾在純 tai8 兩輪之間。先前 mixed 看似的優勢
   （0.9550）純粹來自其 val 集有 56% 是好擬合的 naer。
3. **Barbet 換腦可行且已打平** —— `bm-tslm-tai8-lr5e5` 達 1.0935，與最佳
   VoxCPM2 完全相同。LR 5e-5 明顯勝過 1e-5（1.1035，差 0.010 超出噪音），
   符合「Barbet 學的是全新映射而非微調」的預期。

**⚠️ 訓練資料的文字全是華語漢字**（見 7.2.1）。VoxCPM2 屬 zero-shot TTS，
語言跟隨 reference —— 沒有台語 reference 時會唸成華語。已在
`assets/default_reference/` 放台語預設參考音，`api.py` 於未上傳時自動採用。

**取回並部署**：`bash scripts/fetch_best.sh <ckpt 目錄名>`（沿用原名）或
`scripts/deploy_named.sh`（改為正規化名稱，見 7.3.1）。完成後於
`http://10.9.0.37:8800/` 按「重新掃描」即可試聽。

**已知失敗**：`bm_full`（full stage）為 **CUDA OOM** —— 全解凍 2B 參數加
optimizer state 在 4 卡放不下，需 8 卡或梯度檢查點。

**下一步**：val loss 已分辨不出東西（前五名差距 0.002，噪音是 0.005）。
**必須實聽才能決定方向**，尤其 `bm-tslm-tai8-lr5e5`（Barbet）與
`ft-tai8-lr2e5`（MiniCPM4）數值相同但架構完全不同；以及 LoRA 只有 72 MB
卻只差 0.002。若聽感無別，選小的。

### 7.6 GB10 推論服務與前端

**入口** `http://10.9.0.37:8800/`（nginx 為唯一對外 port，見 `nginx.template`）

```
nginx ─┬─ /        → web 服務（React 靜態檔，multi-stage build 烤進 nginx image）
       ├─ /api/    → app:8000（api.py，FastAPI gateway）
       └─ /legacy/ → app:8000（舊 Gradio，過渡期對照）
```

| 位置 | 放什麼 |
|---|---|
| `checkpoints/<名稱>/` | 全參微調與 BlueMagpie（完整模型，約 8~9 GB）|
| `models/native/<名稱>/` | LoRA adapter（`lora_config.json` + `lora_weights.safetensors`）|
| `models/barbet/<名稱>/` | BlueMagpie 完整 checkpoint |
| `assets/default_reference/` | **預設台語參考音**，未上傳時自動採用 |
| `docs/model_registry.json` | 模型比較資料（17 筆，含 val_set 分組）|

放好後在前端按「重新掃描」即現身，不必重建 image。

**預設參考音的必要性**：**沒有任何 reference 時模型會唸成華語**，加了就講台語
——這是「模型好像只講中文」的成因，**不是訓練失敗**。
`api.py` 於未上傳時取 `assets/default_reference/` 中字典序最前的 wav；
`VOXCPM_DEFAULT_REFERENCE` 可覆寫路徑，設為空字串則停用。

> 📌 **修正（2026-08-03）**：先前記載為「輸出語言跟隨 reference，故需台語
> reference」，**該解釋有誤** —— 實測用**華語** reference 也會輸出台語。
> 關鍵在「有沒有 reference」而非「reference 是什麼語言」：無 reference 時
> 條件路徑缺失（log 不會出現 `[Voice Control] reference_wav only`），
> 模型退回依文字表面發音。故預設參考音仍屬必要，但**不必堅持用台語音檔**。

> ⚠️ 實作細節：預設路徑**不可**寫進 `temp_path`，那個變數在 `finally` 會被
> `os.unlink` —— 會刪掉預設檔本身。故另用 `active_reference` 傳給推論。

**改善音質的參數**（`/api/v1/synthesize`）：`inference_timesteps` 預設 10 偏低，
提到 25~30 明顯較穩；`prompt_text` 填參考音的逐字稿可讓音色與內容解耦；
`cfg_value` 2.0→2.5 更貼合 reference。

**生成佇列與長尾防護（2026-08-17）**：VoxCPM2／Barbet 共用同一張 GPU，
`api.py` 已改為共用單一 GPU gate，不可再拆成兩把 engine lock。預設僅 1 個任務
執行、最多 2 個等待；滿載回 429，等待超過 120 秒回 503。client 中斷時必須繼續
持有 gate，直到已進入 thread/CUDA 的工作結束，否則下一筆會和殘留工作同時進 GPU。
每筆 response 帶 `X-Request-ID`／`X-Queue-Wait`／`X-GPU-Job-Time`／`X-Total-Time`，
log 以 `received → queued → started → completed/rejected` 追蹤。相關環境變數：
`VOXCPM_MAX_PENDING_SYNTHESIS`、`VOXCPM_QUEUE_TIMEOUT_SECONDS`。

VoxCPM2 的 nano-vLLM 原本只設固定 `max_generate_length=2000`，missed EOS 時實測會
占 GPU 9~11 分鐘；現已比照上游 runtime，把實際上限設為
`min(VOXCPM_MAX_GENERATE_LENGTH, 文字長度 × VOXCPM_MAX_AUDIO_TEXT_RATIO + 10)`，
ratio 預設 6.0。`application/x-www-form-urlencoded` 與 `multipart/form-data` 都支援；
8/15 的 499/504 也出現在瀏覽器 multipart，故**表單編碼不是根因**。

**重建**：`docker compose up -d --build web`（前端）／
`docker compose restart app`（只改 Python）／若改 compose 環境變數則用
`docker compose up -d --no-deps --force-recreate app`。

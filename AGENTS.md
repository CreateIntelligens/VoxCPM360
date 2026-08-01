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
- **本機 `scripts/fetch_run.sh` / `scripts/fetch_all_ckpt.sh`** 在本機執行，訓練後取回 log／權重／TensorBoard。支援簡寫名稱（如 `run3` / `3` 自動擴充為 `full_lora_run3`）與 `--with-pth` 強制下載續訓 `.pth` 檔。

### 7.2 訓練設定的地雷
1. **`max_sample_len = max_batch_tokens // batch_size`** —— 兩者是除法關係。
   調大 `batch_size` 不同步調大 `max_batch_tokens` 會**靜默丟棄**超長樣本
   （`batch_size: 32` + `4096` 只剩 32.79% 資料）。
2. **`latest` ≠ 最佳 checkpoint** —— 無 early stopping、無 best 追蹤，
   每輪都要自己從 log 挑（run1 最佳在 step_0048000，`latest` 是較差的 84000）。
3. **LR 排程綁定 `max_steps`** —— `get_cosine_schedule_with_warmup` 把 `max_steps`
   烘進曲線，且續訓會還原 `scheduler.pth`。**不可「先跑短再加大 max_steps 續訓」**
   （等於 LR≈0 空轉）；要跑更長只能重跑。

4. **YAML 科學記號會被讀成字串** —— `learning_rate: 2e-05` 依 YAML 1.1
   規範必須帶小數點與正負號（`2.0e-05`）才算數字，否則解析為**字串**，
   一路傳到 AdamW 才炸在 `TypeError: '<=' not supported between instances
   of 'float' and 'str'`，且已浪費模型載入與 warmup 的時間。
   **一律用 `0.00002` 這種十進位寫法。** `train()` 進入點已加防呆轉型。

5. **改用 epoch 計量，不要用 step** —— step 數跨資料集沒有可比性：
   tai8 跑 7,000 步是 4.03 epoch，naer 同樣 7,000 步卻是 11.81 epoch。
   換算後三輪全參微調的最佳點全落在 **epoch 1.7~2.5**，規律才浮現。
   設定用 `num_epochs` / `valid_per_epoch` / `save_per_epoch`，
   腳本於資料集載入後換算並印出對照。

6. **`best/` 已自動追蹤** —— 每次驗證後比對 val loss，一有新低就另存
   `best/` 與 `best_metric.json`（記 step 與 val）。不必再人工從 log 挑，
   谷底也不會落在兩個 `save_interval` 之間而漏掉。

### 7.2.1 Barbet／BlueMagpie 換腦的結論

`tai8-barbet-merge-v0` 由 `merge_voxcpm_acoustic.py` **拼接**而成（非訓練產物）：
BlueMagpie 骨架保留 410 個 tensor（Barbet + 已訓練的 bridge），
聲學 323 個換成 `full_ft_tai8_step3000` 的權重。

**實測會出聲但完全不講台語。** 逐 tensor 比對確認原因：全參微調動了
577 個 tensor 的**全部**，其中 **254 個是 `base_lm.*`（MiniCPM4 文字腦）**，
而那些**在架構上無法搬進 Barbet**（1,536 hidden、28 層混合 Mamba2，形狀對不上）。
**台語發音知識在文字腦，merge 只搬得動聲學。**

tokenizer 實測：Barbet 少用 25% token，`恁`／`佗` 等台語專用字能整字編碼
（VoxCPM2 要拆 3 個 byte fallback）。**但 tai8 訓練文本全是華語漢字**
（「你為什麼不說實話」而非「你是按怎毋講實話」），這個優勢發揮不出來。

要用 Barbet 必須訓練（`bm_*` 四組設定）。`bluemagpie` 與 `barbet` 是純 Python
套件但未裝進容器，且**運算節點無對外網路**無法 pip install，
故放在 `/app/vendor` 由 `train.sh` 以 `PYTHONPATH` 載入。

### 7.3 多卡啟動
`train.sh` 已支援 GPU 數自動偵測（多卡 `torchrun`／單卡 `python`）。
**裸 `python` 會使 `WORLD_SIZE=1` 只用單卡**，即使 Slurm 配了 4 張卡。
送出後 30 秒務必確認 log 首行 `GPU 數：4`。

### 7.4 命名教訓
`conf/voxcpm_v2/trial_lora_20epochs.yaml` 檔名**雙重誤導**：實際只 0.01 epoch，
且資料早已從 smoke 子集改為全量。**檔名務必反映真實設定。**


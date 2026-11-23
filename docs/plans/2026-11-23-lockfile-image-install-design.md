# 映像依 uv.lock 安裝應用層依賴

**狀態：Draft**
**日期：2026-11-23**
**範圍：`Dockerfile`（應用層依賴那一層）、`pyproject.toml`、`uv.lock`、CI、`AGENTS.md`**

## 1. 背景

`Dockerfile` 的應用層依賴目前這樣裝：

```dockerfile
COPY pyproject.toml uv.lock ./
RUN uv pip install -r pyproject.toml && \
    uv pip install nano-vllm-voxcpm==2.0.4 && \
    uv pip install -r pyproject.toml --extra test
```

`uv pip install -r pyproject.toml` 每次都重新解析，**完全不讀 `uv.lock`**（只 COPY
進去當快取鍵）。實際後果，都在 2026-11-23 發生：

- **CVE 修不到**：Dependabot 只改 `uv.lock`（PR #11，opentelemetry-api 1.45.1→1.45.2），
  合併了也不會進映像；最後是在 `pyproject.toml` 手動加下限（PR #13）才修到。
- **每次重建都偷升版**：PR #13 只改兩行下限，重建卻有 50 多個套件跟著升
  （transformers 5.21→5.24、python-dateutil 3→4、fastapi 0.162→0.165 等）。
  同一份 commit 在不同日子建，會得到不同的映像。
- **CUDA 函式庫混版**：目前正式映像裡的 `nvidia-nvjitlink` 是 13.4.92，
  其他 `nvidia-*` 都還是 13.0.x。⚠️ 實作時查明這**不是**應用層造成的：編譯層安裝
  torch 時，`cuda-toolkit[nvjitlink]` 只要求 `>=13.0.88`，uv 就挑了最新版。
  本設計修不到，見第 7 節。

## 2. 目標

1. 應用層依賴的版本只由 `uv.lock` 決定。同一份 commit 重建兩次，`pip freeze` 必須相同。
2. 改依賴的唯一流程是「改 `pyproject.toml`（或什麼都不改）→ `uv lock` → commit
   兩個檔案」。Dependabot 改 `uv.lock` 的 PR 合併後，重建就會生效。
3. torch／torchaudio／flash-attn 仍照現在的方式裝：由 build args
   （`TORCH_VERSION`、`TORCH_CUDA_VARIANT`、`FLASH_ATTN_VERSION`）決定，從
   PyTorch 的 CUDA index 安裝，或使用預編好的 flash-attn wheel。cu128 退回路線
   （`.env.example` 那組六個變數）必須照常可用。
4. 改 `uv.lock` 或 `pyproject.toml` 時，仍然只重跑應用層那一層，不重編 flash-attn。
5. 同一份 lock 在 GB10（aarch64）和 A4000（x86_64）都能建。`uv.lock` 本身是
   跨平台的，不要改成單一平台的 lock。

## 3. 非目標

- 不改 torch、flash-attn、CUDA base image 的版本。
- 不改任何執行期程式碼或 API 行為。
- 不追求「映像和現在完全相同」。改成照 lock 安裝後版本一定會跟現在不同，
  要做的是**列出差異並驗證**，不是把差異消掉。

## 4. 已知的難點（實作前先確認）

1. **lock 裡的 torch 跟映像裡的不一樣**：`uv.lock` 從 PyPI 解出 torch 2.14.1 和一整串
   `nvidia-*`、`triton`、`cuda-bindings`；映像裡的則是 cu130 index 的 torch 2.13.0
   （`+cu130`）和它自帶的 CUDA 函式庫。照 lock 安裝時不能讓 PyPI 的 torch 和 CUDA
   函式庫蓋掉編譯層裝好的那一套。
2. **跟 torch 綁版本的套件**：`torchcodec`、`torchaudio` 這類套件和 torch 有 ABI
   相依。lock 是對著 torch 2.14.1 解的，挑出來的版本不一定能跟 2.13.0 一起用。
   可以考慮用 `[tool.uv]` 的 `override-dependencies` 或 `constraint-dependencies`，
   讓 lock 對著與映像相同的 torch 版本來解；選哪種、為什麼，寫進 PR 說明。
3. **nano-vllm-voxcpm 不在 lock 裡**：它現在在 Dockerfile 單獨 `uv pip install ...==2.0.4`，
   它的依賴也就不受 lock 管。它只在 Linux 加 CUDA 才能用，`pyproject.toml` 又是上游
   OpenBMB 的套件定義，不要放進一般依賴，可考慮放進新的 optional extra（例如
   `serve`），讓它和它的依賴也進 lock。
4. **`uv pip sync` 會移除 lock 以外的東西**：直接 sync 會把編譯層裝的 torch 和
   flash-attn 刪掉。可以用 `uv export --frozen ...` 產出 requirements 後
   `uv pip install --no-deps -r ...`，但匯出時要排除 torch 那一整串（見第 1 點）；
   `--no-emit-package` 只排除指定的套件本身，**不會**排除它的依賴，排除清單要自己算清楚。
   用別的做法也可以，但要滿足第 5 節的驗收條件。
5. 應用層那一層要用 `--frozen` 或同等設定：lock 跟 `pyproject.toml` 不一致時，
   建置要直接失敗，不能靜默重新解析。

## 5. 驗收條件

- **重現性**：同一份 commit 從頭建兩次應用層（第二次用 `--no-cache-filter` 或
  改快取鍵逼它重跑），兩次的 `pip freeze` 完全相同。
- **lock 生效**：在 `uv.lock` 單獨改一個套件的版本（不動 `pyproject.toml`）後重建，
  映像裡的版本跟著變。驗完把這個改動還原。
- **torch 這一串不被動到**：`torch`、`torchaudio`、`flash_attn`、所有 `nvidia-*`、
  `triton`、`cuda-*` 的版本，跟編譯層裝完時相同，不會被 PyPI 的版本蓋掉。
- **差異清單**：新舊映像（舊的是 `voxcpm360-app:latest`，目前 `369a4cbf396c`）
  的 `pip freeze` 差異完整列在 PR 說明裡。
- **測試**：新映像內全套 pytest 通過（目前 282 passed、5 skipped；
  `test_concurrent_streaming_and_batch_coexist` 偶爾會失敗，單獨重跑會過）。
- **GPU 實測**：用新映像在 GB10 上載入正式台語模型
  `/app/checkpoints/ft-mixed-lr2e5-avgE-e12run-0820`。
  - `/api/v1/synthesize`、`/api/v1/synthesize/stream`、`/api/v1/tts/synthesize`
    與 LoRA `lora::vh387-v6-ep20` 都要回 200，log 不能有 Traceback。
  - 同一句話、同一個 seed，音長要跟正式環境接近。
- **防漂移**：加一個 CI 檢查（例如在 `.github/workflows/` 新增只跑
  `uv lock --check` 的 workflow），`pyproject.toml` 改了卻沒更新 lock 時要失敗。
  光寫在文件裡不算數。
- **文件**：
  - 更新 `Dockerfile` 那一層的註解。
  - 改掉 `pyproject.toml` 安全下限上方「不讀 uv.lock」的說明。
  - `AGENTS.md` 加一段「改依賴的流程」。

## 6. 實作與驗證的限制

- 在 worktree `/home/altos/VoxCPM360/.worktree/uv-lock-install`（分支
  `build/uv-lock-install`）裡做。
- **不要碰正式環境**：
  - 不重建或重啟 `voxcpm_app`、`voxcpm_app_zh`、`voxcpm_nginx`。
  - 不改 `voxcpm360-app:latest` 這個標籤：建置一律用 `docker build -t voxcpm360-app:uvlock-test`，
    不要用 `docker compose build`，那會覆蓋 `latest`。
  - 不改 `/home/altos/VoxCPM360` 主目錄。
- **一次只建一個**，不要平行建置，也不要 `--no-cache` 整份重建：那會重編 flash-attn，
  而且這台曾因 I/O 塞滿當機。build args 要跟正式環境一致，才吃得到已有的編譯層快取：
  `TORCH_ARCH_LIST=12.0`、`TORCH_CUDA_VARIANT=cu130`、`TORCH_VERSION=2.13.0`、
  `TORCHAUDIO_VERSION=2.11.0`、`FLASH_ATTN_VERSION=2.8.3`、
  `CUDA_BASE_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04`、
  `CUDA_RUNTIME_IMAGE=nvidia/cuda:13.0.3-cudnn-runtime-ubuntu22.04`、
  `APP_UID=1000`、`APP_GID=1000`。
- GPU 實測用臨時容器，例如在 worktree 執行
  `docker compose -p uvlocktest run -d --rm --name vox-uvlock-smoke --no-deps -e VOXCPM_HISTORY_DIR=/tmp/h -e VOXCPM_CASTVOICE_BATCH_DIR=/tmp/b app`，
  映像改成 `voxcpm360-app:uvlock-test` 的方法是另外加一份 override：用
  `-f docker-compose.yml -f /tmp/<你的目錄>/override.yml`，override 裡寫
  `services: {app: {image: voxcpm360-app:uvlock-test, build: !reset null}}`。
  worktree 裡要先有 `.env`：從主目錄複製一份，這個檔案已被 gitignore。
  測完就停掉，GB10 的記憶體很緊。
- 在 worktree 跑測試時，要把主 repo 的 `.git` 掛進去，否則 CastVoice 的 model_version
  測試會因為拿不到 git hash 而失敗：
  `-v /home/altos/VoxCPM360/.git:/home/altos/VoxCPM360/.git:ro` 加上
  `GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'`。
- 不要 commit、push 或開 PR，由主 session 審過後處理。

## 7. 實作後的發現與後續

- **nvjitlink 混版來自編譯層**：只看編譯層的映像，`nvidia-nvjitlink` 就已經是 13.4.92。
  driver 支援到 CUDA 13.0；正式環境從 10/28 起就是 13.3 混 13.0，近四週沒有出錯。
  要修得在編譯層安裝 torch 時一起釘 `nvidia-nvjitlink`，但這會讓編譯層快取失效、
  重新下載 torch。另案處理。
- **builder 的 uv 版本沒釘**（`ghcr.io/astral-sh/uv:latest`）：改這一行同樣會讓
  編譯層快取失效，所以沒在這次一起改。另案處理。
- **lock 用 `uv lock --upgrade` 重新產生**：舊 lock 早已過期，連 `pyproject.toml` 的
  安全下限都不滿足。重新產生後跟原本的正式映像只差 filelock 4.2.4→4.3.0、
  markupsafe 3.0.3→3.0.4。
- **沒測到的**：cu128 退回路線，以及 A4000（x86_64）的建置。

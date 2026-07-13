# VoxCPM2 大型多說話者訓練指南（繁體中文）

本文件針對大量、多說話者語音資料，說明如何建立可透過參考音訊切換音色的 VoxCPM2 模型，並使用本專案進行 LoRA 或全參數微調。

## 1. 環境準備

建議從專案根目錄執行所有命令：

```bash
cd /home/altos/VoxCPM360
uv sync
```

訓練需要 NVIDIA GPU。實際顯存需求會受到模型版本、音訊長度、`batch_size` 和 `max_batch_tokens` 影響。顯存不足時，先降低 `batch_size`。

### 下載 VoxCPM2

訓練設定中的 `pretrained_path` 必須指向完整的本機模型目錄。可使用 ModelScope 下載：

```bash
uv run python -c "from modelscope import snapshot_download; snapshot_download('OpenBMB/VoxCPM2', local_dir='./pretrained_models/VoxCPM2')"
```

下載後應可看到：

```text
pretrained_models/VoxCPM2/
├── config.json
├── tokenizer.json
├── audiovae.safetensors（或 audiovae.pth）
└── ...
```

## 2. 準備訓練資料

訓練程式先讀取 JSONL 清單，再依照每筆資料的 `audio` 與 `ref_audio` 路徑載入目標和參考音檔。建議的基本目錄結構如下：

```text
dataset/
├── audio/
│   ├── speaker_000001/
│   │   ├── 000001.wav
│   │   └── 000002.wav
│   └── speaker_000002/
├── metadata.parquet
└── manifests/
    ├── train.jsonl
    ├── val_seen.jsonl
    ├── val_unseen.jsonl
    └── test_unseen.jsonl
```

### 音訊建議

- 一個音檔對應一句或一小段連續語音。
- 建議每段約 2 至 15 秒；超過 30 秒可能造成顯存不足。
- 使用單聲道 WAV，避免背景音樂、明顯混響、爆音和多人同時說話。
- 去除過長的頭尾靜音，但不要切掉語音內容。
- VoxCPM2  使用 16000 Hz 輸入音訊。
- 同一份資料應維持一致的音量、收音環境和文字標註規則。

### 逐字稿建議

- `text` 必須與音訊實際說出的內容一致，不可漏字或多字。
- 使用自然標點，並統一繁簡體、數字及英文大小寫的寫法。
- 不要加入音訊中沒有念出的說話者名稱、時間碼或註解。
- 空白文字不可用於訓練。

## 3. 建立 JSONL 清單

JSONL 是「每行一個 JSON 物件」，不是把所有資料放進一個 JSON 陣列。大型多說話者條件式訓練的必要欄位為：

- `audio`：模型要學習生成的目標音檔。
- `text`：目標音檔的準確逐字稿。
- `ref_audio`：與目標音檔相同說話者的另一段參考音訊。

`dataset/manifests/train.jsonl` 範例：

```jsonl
{"audio": "/data/audio/speaker_000001/000002.wav", "text": "大家好，歡迎來到今天的節目。", "ref_audio": "/data/audio/speaker_000001/000015.wav"}
{"audio": "/data/audio/speaker_000002/000008.wav", "text": "接下來介紹語音模型的使用方式。", "ref_audio": "/data/audio/speaker_000002/000021.wav"}
{"audio": "/data/audio/speaker_000003/000004.wav", "text": "這是一段用於訓練的語音資料。", "ref_audio": "/data/audio/speaker_000003/000011.wav"}
```

建議使用絕對路徑，避免從不同目錄啟動訓練時找不到檔案。音檔本身不會存入 JSONL，JSONL 只保存路徑和標註。每一列的 `audio` 與 `ref_audio` 必須屬於相同說話者，但不可是同一個檔案。

以下欄位為選填：

```jsonl
{"audio": "/data/alice/a002.wav", "text": "第一句。", "ref_audio": "/data/alice/a015.wav", "duration": 3.52, "ref_duration": 5.18, "dataset_id": 0}
```

- `duration`：音訊秒數，可避免篩選長度時再次讀取音檔。
- `ref_duration`：參考音訊秒數，可減少長度估算時的音訊讀取。
- `dataset_id`：混合多個資料來源時使用；單一資料集可省略，程式會設為 `0`。它不是說話者 ID，也不能用來在推論時選擇音色。

可參考以下

```
{"audio": "examples/example.wav", "text": "This is an example audio transcript for training."}
{"audio": "/absolute/path/to/audio1.wav", "text": "You can use absolute paths for audio files."}
{"audio": "relative/path/to/audio2.wav", "text": "Or relative paths from the working directory."}
{"audio": "data/audio3.wav", "text": "Each line is a JSON object with audio path and text.", "duration": 3.5}
{"audio": "data/audio4.wav", "text": "Optional: add duration field to skip audio loading during filtering.", "duration": 2.8}
{"audio": "data/audio5.wav", "text": "Optional: add dataset_id for multi-dataset training.", "dataset_id": 1}
```

### 切分訓練集與驗證集

大量多說話者資料必須依 speaker 和錄音 session 切分，不能只隨機切資料列。完整的 `train`、`val_seen`、`val_unseen` 與 `test_unseen` 規劃請依下一節執行。

## 4. 大型多說話者資料規劃

本指南的目標是訓練一個可透過參考音訊切換音色的 VoxCPM2 模型。不要只把所有音檔隨機混進單一 JSONL，也不要把 `dataset_id` 當作 speaker ID。

### 保留主資料索引

訓練 JSONL 只保存模型當次需要的欄位。完整資料應另外保留一份 CSV、JSONL 或 Parquet 主索引，以便清理、統計、重新配對和切分：

```text
dataset/
├── audio/
│   ├── speaker_000001/
│   │   ├── session_01/
│   │   │   ├── 000001.wav
│   │   │   └── 000002.wav
│   │   └── session_02/
│   └── speaker_000002/
├── metadata.parquet
├── manifests/
│   ├── train.jsonl
│   ├── val_seen.jsonl
│   ├── val_unseen.jsonl
│   └── test_unseen.jsonl
└── reports/
```

主索引建議包含：

| 欄位 | 用途 |
| --- | --- |
| `audio_path` | 音檔的唯一位置 |
| `speaker_id` | 穩定且唯一的說話者識別碼 |
| `text` | 準確逐字稿 |
| `duration` | 音訊秒數 |
| `language` | 語言或方言 |
| `session_id` | 錄音場次，用於避免資料洩漏 |
| `source` | 資料來源或資料集名稱 |
| `sample_rate` | 原始取樣率 |
| `quality_score` | 自訂的品質分數或篩選結果 |

### 建立 target/reference 配對

每個目標音檔應配對一段相同說話者的 `ref_audio`：

```jsonl
{"audio": "/data/alice/a002.wav", "text": "今天的天氣很好。", "ref_audio": "/data/alice/a015.wav"}
{"audio": "/data/bob/b008.wav", "text": "歡迎使用語音模型。", "ref_audio": "/data/bob/b021.wav"}
```

配對規則：

- `audio` 與 `ref_audio` 必須來自同一位說話者。
- 兩者不可指向同一個檔案，並優先使用不同句子。
- 有多個錄音場次時，可安排部分跨 session 配對，提升對收音差異的穩健性。
- `text` 只對應目標 `audio`；目前格式不需要 `ref_audio` 的逐字稿。
- `ref_audio` 建議約 3 至 10 秒，內容應清楚且只有一位說話者。
- 每位說話者至少要有兩段音訊；實務上建議保留五段以上，才有足夠的配對與驗證空間。
- 每個 target 配 1 至 2 個 reference 即可，不要產生所有排列組合。
- 配對程式應固定 random seed，讓相同資料版本可以重現。

程式會將每筆資料組合為「參考音色 + 目標文字 -> 目標音訊」，並只對目標音訊計算訓練 loss。由於 reference 也會占用序列長度，過長的 target/reference 組合可能導致 OOM。

### 依說話者切分資料

多說話者模型應同時評估看過與未看過的音色：

- `val_seen.jsonl`：說話者出現在 train，但驗證音檔、句子不可出現在 train。
- `val_unseen.jsonl`：整位說話者完全不放入 train，用來評估陌生音色複製。
- `test_unseen.jsonl`：另一批完全未見過的說話者，保留作最終測試。

可先採用以下 speaker-level 比例：

```text
train speakers:       90%
validation speakers:   5%
test speakers:         5%
```

再從 train speakers 各保留少量音檔形成 `val_seen`。同一段長錄音切出的相鄰片段必須放在同一個 split；同一場錄音也建議保持在同一側，避免背景與設備特徵洩漏。

訓練腳本一次只接受一個 `val_manifest`，正式訓練時可使用 `val_seen.jsonl`，並在固定 checkpoint 上另外評估 `val_unseen.jsonl` 與 `test_unseen.jsonl`。

### 平衡說話者與資料來源

目前訓練 dataloader 會按 JSONL 資料列隨機抽樣，不會自動平衡說話者。若某位說話者有十萬段、另一位只有一百段，前者將主導訓練。

產生 `train.jsonl` 時應：

- 以每位說話者的總秒數進行抽樣或設定上限，而不只是比較檔案數。
- 避免少量說話者或單一資料來源占據大部分訓練資料。
- 同時檢查語言、方言、錄音設備、性別與音質的分布。
- 不要大量複製小樣本說話者，否則容易記憶音檔內容。
- 保存每個資料版本的統計報告及 random seed。

### 建議處理順序

1. 建立穩定且唯一的 `speaker_id`。
2. 去除損壞、空白、過短、過長及多人重疊的音檔。
3. 使用 VAD 切段，並保留自然完整的句子。
4. 轉為單聲道、16000 Hz WAV（VoxCPM2）。
5. 檢查逐字稿與語音是否一致。
6. 移除重複音檔、近似重複切段與資料洩漏。
7. 依 speaker 和 session 建立 train/validation/test split。
8. 建立同說話者的 target/reference 配對。
9. 平衡說話者、語言和資料來源後輸出 JSONL。
10. 執行資料驗證並保存統計報告。

### 分階段訓練

大量資料不要直接開始長時間全參數訓練，建議依序進行：

1. 使用 100 至 500 筆資料跑 20 至 50 steps，確認資料、loss 和 checkpoint 流程正常。
2. 使用數千至數萬筆高品質資料建立 LoRA baseline。
3. 擴充至完整資料，檢查 LoRA 是否仍有足夠容量。
4. 若目標包含大量新語言、新方言或新領域，而且 LoRA 明顯欠擬合，再使用較低 learning rate 進行全參數微調。

## 5. 驗證資料

開始訓練前，先檢查 JSONL 格式、必填欄位、音檔是否存在、取樣率以及異常長度：

```bash
uv run voxcpm validate \
  --manifest dataset/manifests/train.jsonl \
  --sample-rate 16000
```

VoxCPM1.5 請將取樣率改為 `44100`：

```bash
uv run voxcpm validate \
  --manifest dataset/manifests/train.jsonl \
  --sample-rate 44100
```

驗證出現 error 時應先修正；過短或超過 30 秒的 warning 則建議回頭檢查或重新切段。

## 6. 設定 VoxCPM2 LoRA 訓練

先複製設定範本，保留原始檔案方便比較：

```bash
cp conf/voxcpm_v2/voxcpm_finetune_lora.yaml \
  conf/voxcpm_v2/my_finetune_lora.yaml
```

編輯 `conf/voxcpm_v2/my_finetune_lora.yaml`：

```yaml
pretrained_path: /home/altos/VoxCPM360/pretrained_models/VoxCPM2
train_manifest: /home/altos/VoxCPM360/dataset/manifests/train.jsonl
val_manifest: /home/altos/VoxCPM360/dataset/manifests/val_seen.jsonl

sample_rate: 16000
out_sample_rate: 48000

batch_size: 2
grad_accum_steps: 8
num_workers: 8

num_iters: 1000
max_steps: 1000
log_interval: 10
valid_interval: 500
save_interval: 500

learning_rate: 0.0001
weight_decay: 0.01
warmup_steps: 100
max_batch_tokens: 8192
max_grad_norm: 1.0

save_path: /home/altos/VoxCPM360/checkpoints/my_lora
tensorboard: /home/altos/VoxCPM360/checkpoints/my_lora/logs

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

重要參數：

| 參數 | 說明 |
| --- | --- |
| `pretrained_path` | 完整的本機基礎模型目錄 |
| `train_manifest` | 訓練 JSONL 路徑 |
| `val_manifest` | 驗證 JSONL 路徑，沒有時設為 `null` |
| `batch_size` | 每次載入的樣本數；顯存不足時優先降低 |
| `grad_accum_steps` | 梯度累積次數；有效 batch 約為兩者相乘 |
| `max_steps` | 訓練步數；大於 0 時作為排程總步數 |
| `max_batch_tokens` | 過長樣本的篩選上限；可降低長音訊造成的 OOM 風險 |
| `save_interval` | 每隔多少步儲存 checkpoint |
| `save_path` | checkpoint 與訓練記錄輸出位置 |

## 7. 啟動訓練

### LoRA 微調（建議先使用）

```bash
uv run python scripts/train_voxcpm_finetune.py \
  --config_path conf/voxcpm_v2/my_finetune_lora.yaml
```

### 全參數微調

先修改 [`conf/voxcpm_v2/voxcpm_finetune_all.yaml`](conf/voxcpm_v2/voxcpm_finetune_all.yaml) 中的路徑，再執行：

```bash
uv run python scripts/train_voxcpm_finetune.py \
  --config_path conf/voxcpm_v2/voxcpm_finetune_all.yaml
```

全參數微調需要更多顯存與儲存空間。自訂說話者或少量資料通常先使用 LoRA。

### WebUI

也可以使用專案提供的 WebUI 設定及啟動訓練：

```bash
uv run python lora_ft_webui.py
```

啟動後開啟 `http://localhost:7860`。

## 8. Checkpoint 與續訓

假設 `save_path` 是 `checkpoints/my_lora`，輸出大致如下：

```text
checkpoints/my_lora/
├── step_0000500/
├── step_0001000/
├── latest/
├── logs/
└── train.log
```

- `step_XXXXXXX/`：各次儲存的 checkpoint。
- `latest/`：最近一次 checkpoint 的副本。
- `train.log`：文字訓練記錄。
- `logs/`：TensorBoard 記錄。

使用相同 `save_path` 再次執行時，訓練腳本會嘗試從 `latest/` 恢復模型、optimizer、scheduler 與步數。若要開始全新的實驗，請改用新的 `save_path`。

## 9. 常見問題

### 找不到音檔

確認 JSONL 中的 `audio` 路徑存在。建議改用絕對路徑，並再次執行 `voxcpm validate`。

### Sample rate mismatch

訓練 YAML 的 `sample_rate` 必須符合模型 `config.json` 中 AudioVAE 的輸入取樣率：VoxCPM2 通常是 16000，VoxCPM1.5 是 44100。音檔也建議預先轉為相同取樣率。

### CUDA out of memory

依序嘗試：

1. 降低 `batch_size`，例如從 `2` 改為 `1`。
2. 使用 `grad_accum_steps` 維持有效 batch 大小。
3. 將過長音訊重新切段。
4. 降低 `max_batch_tokens`。
5. 使用 LoRA，而不是全參數微調。

### 訓練結果聲音不穩定

優先檢查逐字稿是否準確、音量是否一致、是否混入其他說話者或背景音樂，以及資料是否包含過多靜音。資料品質通常比單純增加訓練步數更重要。

## 10. 快速檢查清單

- [ ] 每個音檔只包含一段清楚且連續的語音。
- [ ] 音檔與逐字稿完全一致。
- [ ] JSONL 每行都是合法 JSON，並包含 `audio`、`text` 和 `ref_audio`。
- [ ] `audio` 使用可讀取的路徑，建議為絕對路徑。
- [ ] 音訊取樣率符合所選模型。
- [ ] 多說話者資料具有穩定的 `speaker_id`，且 `ref_audio` 與目標為同一人。
- [ ] 未見說話者及相同錄音 session 沒有洩漏到其他 split。
- [ ] 已檢查每位說話者、語言和資料來源的時數分布。
- [ ] `voxcpm validate` 沒有 error。
- [ ] YAML 的模型、資料與輸出路徑都已修改。
- [ ] 首次測試使用 LoRA、較小 `batch_size` 和較少步數。

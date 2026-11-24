# 華語參考音（華語服務專用）

`VOXCPM_LANG=zh`（compose 的 `app-zh`）時，內建聲音改讀這個目錄，
台語服務仍讀 `assets/default_reference/`。兩邊的聲音 id 與檔名相同，
前端與 CastVoice 不必區分。

**來源**：taigi-cosyvoice 的 `models/prompts/`（清單在它的 `data/voices.json`），
即 CosyVoice 各聲線的原始錄音。台語那組 `cosy-*` 正是 CosyVoice 用這些錄音
轉念台語產生的，所以兩組是同一批音色。轉成與台語那組相同的規格：
24 kHz、單聲道、32 kbps mp3。

**為什麼不能沿用台語那組**：華語服務拿台語錄音加逐字稿做接續克隆時，原版
模型會順著念成台語；只用音色也會帶台語腔（2026-11-24 實聽）。

**逐字稿**：見 `gateway/presets.py` 的 `_COSY_ZH_PROMPT_TEXT`。15 支經 SenseVoice
判定為 `zh`，並與 Breeze-ASR 交叉核對。`cosy-young-female-02`（Hayley）與
`cosy-child-female-01` 開頭多一聲「嗨，」，兩套 ASR 都聽到；`cosy-teen-male-02`
開頭的「哎呀」只有 SenseVoice 聽到，視為雜音不列入。

Gemini 的 6 個聲音沒有華語錄音，華語服務不提供。

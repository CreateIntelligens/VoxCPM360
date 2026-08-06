# 預設參考音檔

前端會將本目錄內的三個白名單音檔顯示為下拉選單。未上傳 reference audio
時，`api.py` 會採用使用者選取的內建音檔；未指定時預設為「中年女聲」。
使用者自行上傳音檔時，上傳檔案優先於下拉選擇。

舊版用戶端未傳送選項時，可用環境變數 `VOXCPM_DEFAULT_REFERENCE` 覆寫預設
路徑；設為空字串則停用預設參考音訊。

來源為 tai8 資料集（台語劇集），逐字稿是華語文字但**音檔為台語發音** ——
這正是 tai8 的資料形式。VoxCPM2 屬 zero-shot TTS，輸出的語言與音色會跟隨
reference，因此拿華語 reference 生成必然得到華語；要輸出台語就需要台語
reference。這也是「模型好像只講中文」的成因。

| 檔案 | 下拉選單顯示 | 逐字稿 | 長度 |
|---|---|---|---|
| `tai8_drama1_005.wav` | 中年女聲 | 碧玉拿給我看他先生的相片好像不是長這樣 | 5.2s |
| `tai8_female_drama1_002.wav` | 中年男聲 | 只要一套比基尼這樣就夠了 | 2.7s |
| `hayley_happy_opening.mp3` | Hayley 開心說開場白 | Hi，我是創造智能的 AI 代言人愛卡，想知道你的 MBTI 是哪一型嗎？還是對我們的 AI 服務好奇，我都可以告訴你，快來跟我聊聊吧！ | 12.9s |

下拉項目與檔名的白名單定義於 `api.py`；新增檔案後仍須在該處加入項目。

## ⚠️ 逐字稿是必要的，不是選填

`api.py` 的 `_REFERENCE_AUDIO_PRESETS` 每筆都帶 `prompt_text`，未上傳自訂
音檔時會**自動帶入**。原因見 AGENTS.md 7.6.1：`barbet_runtime.py:148` 寫成
`prompt_wav_path=reference_path if prompt_text else ""`，逐字稿為空會讓
參考音整個被丟掉、聲音克隆完全失效（實測輸出 peak 0.973 vs 參考音 0.259）。

**新增內建參考音時，務必同時在 `api.py` 補上 `prompt_text`。**
使用者自行上傳的音檔不適用（我們不知道其逐字稿），需由使用者自行填寫。

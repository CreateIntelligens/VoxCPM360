"""華語服務要用華語錄的參考音，不能沿用台語那組。

台語錄音加上逐字稿做接續克隆，原版模型會整句念成台語；只用音色也會帶台語腔。
"""

import os
import subprocess
import sys
from pathlib import Path

from gateway import presets

ROOT = Path(__file__).resolve().parents[1]


def test_mandarin_presets_are_mandarin_recordings_of_the_same_voices():
    tw = presets.reference_presets_for("tw")
    zh = presets.reference_presets_for("zh")
    zh_dir = presets.reference_audio_dir_for("zh")

    assert {p["id"] for p in zh} == {p["id"] for p in tw if p["id"].startswith("cosy-")}
    assert len(zh) == 15
    for preset in zh:
        assert (zh_dir / preset["filename"]).is_file(), preset["filename"]
        assert preset["language"] == "zh-TW"
        assert "臺灣台語" not in preset["label"]
        assert "AI代言人艾卡" in preset["prompt_text"]
    assert zh_dir != presets.reference_audio_dir_for("tw")


def test_taiwanese_presets_are_unchanged():
    tw = presets.reference_presets_for("tw")
    assert any(p["id"].startswith("gemini-") for p in tw)
    assert all(p["language"] == "nan-TW" for p in tw)
    assert presets.reference_audio_dir_for("tw").name == "default_reference"


def test_service_language_selects_the_preset_set_at_startup():
    # 其他模組在 import 時就拿走這兩個名字，所以要在全新的程序裡驗。
    code = (
        "from gateway import presets as p;"
        "print(p._REFERENCE_AUDIO_DIR.name, len(p._REFERENCE_AUDIO_PRESETS),"
        " p._REFERENCE_AUDIO_PRESETS[0]['language'])"
    )
    for lang, expected in (("zh", "default_reference_zh 15 zh-TW"), ("tw", "default_reference 21 nan-TW")):
        env = {**os.environ, "VOXCPM_LANG": lang}
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, env=env,
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[-1]
        assert out == expected, (lang, out)

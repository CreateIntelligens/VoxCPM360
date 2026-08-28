from __future__ import annotations

import api

_REQUIRED_FIELDS = (
    "id",
    "label",
    "filename",
    "description",
    "prompt_text",
    "gender",
    "language",
)


def test_reference_preset_assets_exist_and_nonempty():
    for preset in api._REFERENCE_AUDIO_PRESETS:
        path = api._REFERENCE_AUDIO_DIR / preset["filename"]
        assert path.is_file(), f"{preset['id']} 缺少音檔：{path}"
        assert path.stat().st_size > 0, f"{preset['id']} 音檔為空：{path}"


def test_reference_preset_ids_unique():
    ids = [preset["id"] for preset in api._REFERENCE_AUDIO_PRESETS]
    assert len(ids) == len(set(ids))


def test_reference_presets_have_required_fields():
    for preset in api._REFERENCE_AUDIO_PRESETS:
        for key in _REQUIRED_FIELDS:
            assert preset.get(key), f"{preset['id']} 缺少欄位 {key}"
        assert preset["language"] in (api._LANG_NAN_TW, api._LANG_ZH_TW)
        assert preset["gender"] in ("female", "male")


def test_default_reference_preset_id_exists():
    assert any(
        preset["id"] == api._DEFAULT_REFERENCE_PRESET_ID
        for preset in api._REFERENCE_AUDIO_PRESETS
    )


def test_removed_reference_audio_contracts_do_not_return():
    assert all(
        not preset["filename"].startswith("tai8_")
        for preset in api._REFERENCE_AUDIO_PRESETS
    )
    assert all(
        "tai8" not in definition["voice_id"]
        for definition in api._CASTVOICE_DEFINITIONS
    )
    assert all(
        "hayley" not in preset["id"].lower()
        and "hayley" not in preset["label"].lower()
        for preset in api._REFERENCE_AUDIO_PRESETS
    )
    assert all(
        "hayley" not in definition["voice_id"].lower()
        and "hayley" not in definition["label"].lower()
        for definition in api._CASTVOICE_DEFINITIONS
    )

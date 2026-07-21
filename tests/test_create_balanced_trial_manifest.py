from __future__ import annotations

import json

from scripts.create_balanced_trial_manifest import build_manifests


def _write_source_data(metadata_path, processed_root, speaker_count=4, rows_per_speaker=8):
    rows = []
    for speaker_index in range(speaker_count):
        speaker_id = f"drama1_{speaker_index:03d}"
        for row_index in range(rows_per_speaker):
            episode = row_index + 1
            filename = f"{speaker_index:03d}_{episode:03d}_{row_index:06d}.wav"
            audio = processed_root / speaker_id / f"{episode:03d}" / filename
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b"wav")
            rows.append(
                {
                    "audio": f"segments/drama1/{speaker_index:03d}/{episode:03d}/{filename}",
                    "duration": 2.0 + row_index / 10,
                    "episode": episode,
                    "speaker_id": speaker_id,
                    "text": f"測試文字 {speaker_index} {row_index}",
                    "utterance_id": f"{speaker_id}_{row_index}",
                }
            )

    metadata_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_builds_balanced_disjoint_manifests(tmp_path):
    metadata_path = tmp_path / "metadata.jsonl"
    processed_root = tmp_path / "16k_mono"
    _write_source_data(metadata_path, processed_root)

    train_rows, val_rows, speakers = build_manifests(
        metadata_path=metadata_path,
        processed_root=processed_root,
        speaker_count=2,
        train_per_speaker=3,
        train_reference_count=2,
        seed=42,
    )

    assert len(train_rows) == 6
    assert len(val_rows) == 2
    assert len(speakers) == 2
    assert {row["speaker_id"] for row in train_rows} == set(speakers)
    assert {row["speaker_id"] for row in val_rows} == set(speakers)

    train_audio = {row["audio"] for row in train_rows}
    train_references = {row["ref_audio"] for row in train_rows}
    val_audio = {row["audio"] for row in val_rows}
    val_references = {row["ref_audio"] for row in val_rows}

    assert train_audio.isdisjoint(val_audio)
    assert train_audio.isdisjoint(val_references)
    assert train_references.isdisjoint(val_audio)
    assert train_references.isdisjoint(val_references)
    assert val_audio.isdisjoint(val_references)
    assert all(row["audio"] != row["ref_audio"] for row in train_rows + val_rows)


from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any


def _is_valid_source_row(row: dict[str, Any]) -> bool:
    duration = float(row.get("duration") or 0)
    return bool(
        row.get("audio")
        and row.get("speaker_id")
        and str(row.get("text") or "").strip()
        and 1.0 <= duration <= 15.0
    )


def _processed_audio_path(row: dict[str, Any], processed_root: Path) -> Path:
    episode = int(row["episode"])
    return (
        processed_root
        / str(row["speaker_id"])
        / f"{episode:03d}"
        / Path(str(row["audio"])).name
    )


def _manifest_row(
    target: dict[str, Any],
    reference: dict[str, Any],
    processed_root: Path,
) -> dict[str, Any]:
    return {
        "audio": str(_processed_audio_path(target, processed_root)),
        "duration": float(target["duration"]),
        "episode": int(target["episode"]),
        "ref_audio": str(_processed_audio_path(reference, processed_root)),
        "ref_duration": float(reference["duration"]),
        "speaker_id": str(target["speaker_id"]),
        "text": str(target["text"]).strip(),
        "utterance_id": str(target["utterance_id"]),
    }


def build_manifests(
    metadata_path: Path,
    processed_root: Path,
    speaker_count: int,
    train_per_speaker: int,
    train_reference_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    metadata_path = Path(metadata_path)
    processed_root = Path(processed_root)
    required_per_speaker = train_per_speaker + train_reference_count + 2

    # Parse metadata once to group valid rows by speaker
    rows_by_speaker: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if _is_valid_source_row(row):
                speaker_id = str(row["speaker_id"])
                rows_by_speaker[speaker_id].append(row)

    randomizer = random.Random(seed)
    candidates = [
        speaker_id
        for speaker_id, rows in rows_by_speaker.items()
        if len(rows) >= required_per_speaker
    ]
    randomizer.shuffle(candidates)
    candidate_limit = min(len(candidates), max(speaker_count * 4, speaker_count))
    candidate_speakers = candidates[:candidate_limit]

    selected: dict[str, list[dict[str, Any]]] = {}
    for speaker_id in candidate_speakers:
        speaker_rows = rows_by_speaker[speaker_id]
        randomizer.shuffle(speaker_rows)
        available_rows = [
            row
            for row in speaker_rows
            if _processed_audio_path(row, processed_root).is_file()
        ]
        if len(available_rows) >= required_per_speaker:
            selected[speaker_id] = available_rows[:required_per_speaker]
        if len(selected) == speaker_count:
            break

    if len(selected) < speaker_count:
        raise ValueError(
            f"Needed {speaker_count} speakers with processed audio, found {len(selected)}"
        )

    train_rows = []
    val_rows = []
    for speaker_rows in selected.values():
        train_targets = speaker_rows[:train_per_speaker]
        train_references = speaker_rows[
            train_per_speaker : train_per_speaker + train_reference_count
        ]
        val_target = speaker_rows[train_per_speaker + train_reference_count]
        val_reference = speaker_rows[train_per_speaker + train_reference_count + 1]

        for index, target in enumerate(train_targets):
            ref = train_references[index % len(train_references)]
            train_rows.append(_manifest_row(target, ref, processed_root))

        val_rows.append(_manifest_row(val_target, val_reference, processed_root))

    randomizer.shuffle(train_rows)
    randomizer.shuffle(val_rows)
    return train_rows, val_rows, list(selected)


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build balanced train/val dataset manifests for VoxCPM."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--speakers", type=int, default=64)
    parser.add_argument("--train-per-speaker", type=int, default=8)
    parser.add_argument("--train-reference-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    train_rows, val_rows, speakers = build_manifests(
        metadata_path=args.metadata,
        processed_root=args.processed_root,
        speaker_count=args.speakers,
        train_per_speaker=args.train_per_speaker,
        train_reference_count=args.train_reference_count,
        seed=args.seed,
    )
    write_manifest(args.train_output, train_rows)
    write_manifest(args.val_output, val_rows)
    print(
        json.dumps(
            {
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "speakers": len(speakers),
                "seed": args.seed,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

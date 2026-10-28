"""PyTorch IterableDataset and clean-window Parquet/Tensor shard exporter.

Slices 200 Hz validated telemetry into contiguous 64-step windows ([64, 11]
float32 tensors) and hard-excises any window containing at least one corrupted
sample. Never interpolates across corrupted physics.

Provides:
  - ``CleanRoboticsShardDataset`` — streaming ``torch.utils.data.IterableDataset``.
  - ``export_clean_shards`` — persists clean windows as ``.npy`` tensors, per-shard
    ``.parquet`` files, and a ``qa_manifest.json`` to disk.
"""

import json
import os
from typing import Dict, Iterator, List

import numpy as np
import polars as pl
import torch
from torch.utils.data import IterableDataset

FEATURE_COLUMNS: List[str] = [
    "pos_x", "pos_y", "pos_z",
    "acc_x", "acc_y", "acc_z",
    "force_thumb", "force_index", "force_middle", "force_ring", "force_pinky",
]

WINDOW_SIZE: int = 64


class CleanRoboticsShardDataset(IterableDataset):
    """Streaming dataset of clean 64-step telemetry windows.

    Each yielded tensor has shape ``(64, 11)`` and dtype ``torch.float32``.
    Windows overlapping any ``is_corrupted == True`` sample are silently
    dropped (hard excision).
    """

    def __init__(self, validated_df: pl.DataFrame) -> None:
        super().__init__()
        self._data = (
            validated_df
            .select(FEATURE_COLUMNS)
            .to_numpy()
            .astype(np.float32)
        )
        self._corrupted = validated_df["is_corrupted"].to_numpy().astype(bool)
        self._n_windows = len(self._data) // WINDOW_SIZE

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in range(self._n_windows):
            start = i * WINDOW_SIZE
            end = start + WINDOW_SIZE
            if not np.any(self._corrupted[start:end]):
                yield torch.from_numpy(self._data[start:end].copy())

    def __len__(self) -> int:
        """Return the number of *clean* windows (post-excision)."""
        count = 0
        for i in range(self._n_windows):
            start = i * WINDOW_SIZE
            end = start + WINDOW_SIZE
            if not np.any(self._corrupted[start:end]):
                count += 1
        return count


def export_clean_shards(
    validated_df: pl.DataFrame,
    export_dir: str = "./artifacts/shards",
) -> Dict:
    """Persist clean 64-step windows to disk as tensors and Parquet files.

    Directory layout::

        export_dir/
        ├── shard_0000.npy
        ├── shard_0000.parquet
        ├── ...
        └── qa_manifest.json

    Args:
        validated_df: Validated Polars DataFrame with ``is_corrupted`` column.
        export_dir: Target directory (created if absent).

    Returns:
        qa_manifest: Dictionary summarising the shard export.
    """
    os.makedirs(export_dir, exist_ok=True)

    data = (
        validated_df
        .select(FEATURE_COLUMNS)
        .to_numpy()
        .astype(np.float32)
    )
    corrupted = validated_df["is_corrupted"].to_numpy().astype(bool)
    n_windows = len(data) // WINDOW_SIZE

    clean_indices: List[int] = []

    for i in range(n_windows):
        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE
        if np.any(corrupted[start:end]):
            continue

        clean_indices.append(i)
        window = data[start:end]
        shard_name = f"shard_{i:04d}"

        np.save(os.path.join(export_dir, f"{shard_name}.npy"), window)

        window_df = pl.DataFrame(
            {col: window[:, j] for j, col in enumerate(FEATURE_COLUMNS)},
        )
        window_df.write_parquet(os.path.join(export_dir, f"{shard_name}.parquet"))

    manifest: Dict = {
        "total_windows": n_windows,
        "clean_windows": len(clean_indices),
        "excised_windows": n_windows - len(clean_indices),
        "window_shape": [WINDOW_SIZE, len(FEATURE_COLUMNS)],
        "feature_columns": FEATURE_COLUMNS,
        "clean_shard_indices": clean_indices,
    }

    with open(os.path.join(export_dir, "qa_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest

"""Vectorized Polars QA engine with OpenCV optical-flow cross-modal sync gate.

Aligns 30 Hz video optical-flow onto the 200 Hz sensor clock via
``pl.DataFrame.join_asof`` and evaluates three deterministic quality gates:

1. **flag_sensor_flatline** — rolling σ (window=20) on any finger force < 1e-4 N.
2. **flag_kinematic_spike** — L2‖(acc_x, acc_y, acc_z)‖ > 45 m/s².
3. **flag_cross_modal_desync** — 2-D wrist speed > 0.15 m/s while aligned
   optical_flow < 0.01 (after t > 0.5 s).

Returns ``(validated_df, qa_report_dict)`` with exact anomaly counts, processing
latency in milliseconds, rejected percentage, and release status.
"""

import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import polars as pl

SENSOR_HZ: int = 200

FORCE_COLUMNS: List[str] = [
    "force_thumb",
    "force_index",
    "force_middle",
    "force_ring",
    "force_pinky",
]


def _compute_optical_flow(video_frames: List[np.ndarray]) -> List[float]:
    """Mean absolute grayscale pixel difference between consecutive frames."""
    flow: List[float] = [0.0]
    prev_gray: np.ndarray | None = None
    for frame in video_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            flow.append(float(np.mean(diff)))
        prev_gray = gray
    return flow


def run_quality_gate(
    sensor_df: pl.DataFrame,
    video_frames: List[np.ndarray],
    video_timestamps: np.ndarray,
) -> Tuple[pl.DataFrame, Dict]:
    """Run vectorized QA checks on a factory episode.

    Args:
        sensor_df: 200 Hz Polars DataFrame from ``generate_factory_episode``.
        video_frames: 30 Hz BGR video frames.
        video_timestamps: 1-D array of video-frame timestamps (seconds).

    Returns:
        validated_df: ``sensor_df`` augmented with boolean flag columns and
            ``is_corrupted``.
        qa_report: Dictionary of QA metrics.
    """
    t_start = time.perf_counter()

    optical_flow = _compute_optical_flow(video_frames)
    video_df = pl.DataFrame({
        "timestamp_s": video_timestamps.tolist(),
        "optical_flow": optical_flow,
    }).sort("timestamp_s")

    df = sensor_df.sort("timestamp_s")
    df = df.join_asof(video_df, on="timestamp_s", strategy="backward")

    df = df.with_columns(
        pl.col("optical_flow").fill_null(999.0).alias("optical_flow"),
    )

    rstd_exprs = [
        pl.col(col)
        .rolling_std(window_size=20)
        .fill_null(999.0)
        .alias(f"_rstd_{col}")
        for col in FORCE_COLUMNS
    ]
    df = df.with_columns(rstd_exprs)

    flatline_expr = pl.lit(False)
    for col in FORCE_COLUMNS:
        flatline_expr = flatline_expr | (pl.col(f"_rstd_{col}") < 1e-4)
    df = df.with_columns(flatline_expr.alias("flag_sensor_flatline"))

    df = df.with_columns(
        (
            pl.col("acc_x").pow(2)
            + pl.col("acc_y").pow(2)
            + pl.col("acc_z").pow(2)
        ).sqrt().alias("acc_norm"),
    )
    df = df.with_columns(
        (pl.col("acc_norm") > 45.0).alias("flag_kinematic_spike"),
    )

    dt = 1.0 / SENSOR_HZ
    df = df.with_columns([
        (pl.col("pos_x").diff() / dt).fill_null(0.0).alias("_vx"),
        (pl.col("pos_y").diff() / dt).fill_null(0.0).alias("_vy"),
    ])
    df = df.with_columns(
        (pl.col("_vx").pow(2) + pl.col("_vy").pow(2)).sqrt().alias("speed_2d"),
    )
    df = df.with_columns(
        (
            (pl.col("speed_2d") > 0.15)
            & (pl.col("optical_flow") < 0.01)
            & (pl.col("timestamp_s") > 0.5)
        ).alias("flag_cross_modal_desync"),
    )

    df = df.with_columns(
        (
            pl.col("flag_sensor_flatline")
            | pl.col("flag_kinematic_spike")
            | pl.col("flag_cross_modal_desync")
        ).alias("is_corrupted"),
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    total = len(df)
    flatline_count = int(df["flag_sensor_flatline"].sum())
    spike_count = int(df["flag_kinematic_spike"].sum())
    desync_count = int(df["flag_cross_modal_desync"].sum())
    corrupted_count = int(df["is_corrupted"].sum())
    rejected_pct = round(corrupted_count / total * 100.0, 2) if total else 0.0
    release_status = (
        "SHARDED_WITH_EXCISION" if rejected_pct < 30.0 else "QUARANTINED"
    )

    qa_report: Dict = {
        "episode_id": "pune_rig_04_ep_0089",
        "processing_time_ms": round(elapsed_ms, 2),
        "total_samples": total,
        "flag_sensor_flatline_count": flatline_count,
        "flag_kinematic_spike_count": spike_count,
        "flag_cross_modal_desync_count": desync_count,
        "total_corrupted_count": corrupted_count,
        "rejected_pct": rejected_pct,
        "release_status": release_status,
    }

    drop_cols = [c for c in df.columns if c.startswith("_rstd_") or c.startswith("_v")]
    validated_df = df.drop(drop_cols)

    return validated_df, qa_report

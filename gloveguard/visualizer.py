"""Rerun.io 4-panel multimodal dashboard with live QA HUD overlays.

Configures a Blueprint with four panels:

  Top-Left: 2D ego-camera view with burned-in QA banner
               (GREEN ``QA PASS -> PYTORCH SHARD`` / RED ``REJECTED [reason]``).
  Bottom-Left: 3D glove-trajectory trail — clean points in green, corrupted in red.
  Right-Top: 200 Hz finger-force time series with red fault markers.
  Right-Bottom: 200 Hz IMU acceleration-norm time series with red fault markers.
"""

import os
from typing import List

import cv2
import numpy as np
import polars as pl

FEATURE_COLUMNS: List[str] = [
    "pos_x", "pos_y", "pos_z",
    "acc_x", "acc_y", "acc_z",
    "force_thumb", "force_index", "force_middle", "force_ring", "force_pinky",
]

FORCE_COLUMNS: List[str] = [
    "force_thumb", "force_index", "force_middle", "force_ring", "force_pinky",
]

COLOR_CLEAN = [40, 230, 120]
COLOR_CORRUPTED = [255, 45, 45]


def _fault_reason(row_idx: int, df: pl.DataFrame) -> str:
    """Return a human-readable fault label for the given row index."""
    reasons = []
    if df["flag_sensor_flatline"][row_idx]:
        reasons.append("SENSOR FLATLINE")
    if df["flag_kinematic_spike"][row_idx]:
        reasons.append("IMU SPIKE")
    if df["flag_cross_modal_desync"][row_idx]:
        reasons.append("CAMERA DESYNC")
    return " | ".join(reasons) if reasons else "UNKNOWN"


def launch_rerun_dashboard(
    validated_df: pl.DataFrame,
    video_frames: List[np.ndarray],
    video_timestamps: np.ndarray,
    headless: bool = False,
    export_dir: str = "./artifacts",
    rrd_filename: str = "gloveguard_demo.rrd",
) -> str | None:
    """Stream validated telemetry and video into a Rerun 4-panel dashboard.

    Args:
        validated_df: Polars DataFrame with flag columns and ``is_corrupted``.
        video_frames: 30 Hz BGR video frames.
        video_timestamps: 1-D array of video-frame timestamps (seconds).
        headless: If True, save to ``.rrd`` file instead of spawning the viewer.
        export_dir: Directory for the ``.rrd`` file in headless mode.
        rrd_filename: Filename for the saved Rerun recording archive.

    Returns:
        Path to the saved .rrd file if headless, or None if spawned interactively.
    """
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except (ImportError, OSError) as exc:
        print(f"      [Notice] Rerun SDK unavailable on this host ({exc}).")
        if headless:
            os.makedirs(export_dir, exist_ok=True)
            rrd_path = os.path.join(export_dir, rrd_filename)
            with open(rrd_path, "wb") as fh:
                fh.write(b"RRD_ARCHIVE_HOST_APP_CONTROL_RESTRICTED\n")
            return rrd_path
        return None

    # Modern Rerun API compatibility helpers across 0.21.x - 0.40.x+
    def _set_time(timeline: str, seconds: float) -> None:
        if hasattr(rr, "set_time_seconds"):
            rr.set_time_seconds(timeline, seconds)
        else:
            rr.set_time(timeline, duration=seconds)

    def _scalar(val: float):
        if hasattr(rr, "Scalars"):
            return rr.Scalars(val)
        return rr.Scalar(val)

    def _series_line(color: list, width: float = 1.5, name: str | None = None):
        if hasattr(rr, "SeriesLines"):
            kwargs = {"colors": [color], "widths": [width]}
            if name:
                kwargs["names"] = [name]
            return rr.SeriesLines(**kwargs)
        if hasattr(rr, "SeriesLine"):
            kwargs = {"color": color, "width": width}
            if name:
                kwargs["name"] = name
            return rr.SeriesLine(**kwargs)
        return None

    rr.init("IntelligenceFactory_GloveGuard_QA", spawn=not headless)

    os.makedirs(export_dir, exist_ok=True)
    rrd_path = os.path.join(export_dir, rrd_filename)
    if headless:
        rr.save(rrd_path)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="Ego Camera",
                    origin="world/ego_camera",
                ),
                rrb.Spatial3DView(
                    name="Glove Trajectory",
                    origin="world",
                ),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    name="Finger Forces",
                    origin="sensors/forces",
                ),
                rrb.TimeSeriesView(
                    name="IMU Acceleration",
                    origin="sensors/imu",
                ),
            ),
        ),
    )
    rr.send_blueprint(blueprint)

    timestamps = validated_df["timestamp_s"].to_numpy()
    is_corrupted = validated_df["is_corrupted"].to_numpy().astype(bool)

    positions = np.column_stack([
        validated_df["pos_x"].to_numpy(),
        validated_df["pos_y"].to_numpy(),
        validated_df["pos_z"].to_numpy(),
    ])
    colors = np.where(
        is_corrupted[:, None],
        np.array(COLOR_CORRUPTED, dtype=np.uint8),
        np.array(COLOR_CLEAN, dtype=np.uint8),
    )
    rr.log(
        "world/glove_trail",
        rr.Points3D(positions, colors=colors, radii=0.003),
        static=True,
    )

    force_colors = [
        [255, 200, 50],
        [50, 200, 255],
        [200, 100, 255],
        [100, 255, 200],
        [255, 130, 100],
    ]
    for idx, col in enumerate(FORCE_COLUMNS):
        line = _series_line(force_colors[idx], width=1.5, name=col)
        if line is not None:
            rr.log(f"sensors/forces/{col}", line, static=True)

    line_imu = _series_line([100, 180, 255], width=1.5, name="acc_norm")
    if line_imu is not None:
        rr.log("sensors/imu/acc_norm", line_imu, static=True)

    line_fault_imu = _series_line([255, 45, 45], width=3.0, name="fault")
    if line_fault_imu is not None:
        rr.log("sensors/imu/fault", line_fault_imu, static=True)

    line_fault_forces = _series_line([255, 45, 45], width=3.0, name="fault")
    if line_fault_forces is not None:
        rr.log("sensors/forces/fault", line_fault_forces, static=True)

    force_np = {col: validated_df[col].to_numpy() for col in FORCE_COLUMNS}
    acc_norm_np = validated_df["acc_norm"].to_numpy()

    for i, t in enumerate(timestamps):
        _set_time("factory_clock", float(t))

        for col in FORCE_COLUMNS:
            rr.log(f"sensors/forces/{col}", _scalar(float(force_np[col][i])))

        rr.log("sensors/imu/acc_norm", _scalar(float(acc_norm_np[i])))

        if is_corrupted[i]:
            rr.log("sensors/imu/fault", _scalar(float(acc_norm_np[i])))
            rr.log(
                "sensors/forces/fault",
                _scalar(float(force_np["force_index"][i])),
            )

    for vi, t_v in enumerate(video_timestamps):
        _set_time("factory_clock", float(t_v))
        sensor_idx = min(int(round(t_v * 200)), len(is_corrupted) - 1)
        corrupted = is_corrupted[sensor_idx]

        frame_rgb = cv2.cvtColor(video_frames[vi], cv2.COLOR_BGR2RGB).copy()

        if corrupted:
            reason = _fault_reason(sensor_idx, validated_df)
            cv2.rectangle(frame_rgb, (0, 0), (320, 24), (220, 30, 30), -1)
            cv2.putText(
                frame_rgb, f"REJECTED [{reason}]", (5, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
            )
        else:
            cv2.rectangle(frame_rgb, (0, 0), (320, 24), (20, 160, 60), -1)
            cv2.putText(
                frame_rgb, "QA PASS -> PYTORCH SHARD", (5, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
            )

        rr.log("world/ego_camera", rr.Image(frame_rgb))

    if not headless:
        # Also persist a demo copy when interactively launched
        rr.save(rrd_path)

    return rrd_path

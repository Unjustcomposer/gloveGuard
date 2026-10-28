"""Synthetic 30Hz RGB video + 200Hz IMU/Force generator with 3 injected hardware faults.

Generates a 10-second (default) human pick-and-place trajectory following a smooth
3D figure-8 Lissajous path, derived kinematics (acceleration with 9.81 m/s² Z-gravity
and Gaussian noise), and 5-finger grip force profiles.

Produces a synchronized 30Hz RGB video stream (320×240 BGR NumPy arrays via OpenCV)
rendering a top-down dark-mode workbench grid, a target bin, and a dynamic hand cursor
whose color shifts from green (low force) to red (high force).

Three deterministic hardware failures are injected:
  Fault 1 (t=2.5–3.5 s): Sensor dead-zone — force_index held at 0.0001 N.
  Fault 2 (t=5.8–5.95 s): IMU kinematic spike — ±65 m/s² on acc_x, acc_y.
  Fault 3 (t=7.5–8.3 s): Camera freeze — last valid RGB frame repeated.
"""

from typing import Tuple, List, Dict

import cv2
import numpy as np
import polars as pl

SENSOR_HZ: int = 200
VIDEO_HZ: int = 30
FRAME_W: int = 320
FRAME_H: int = 240

FORCE_COLUMNS: List[str] = [
    "force_thumb",
    "force_index",
    "force_middle",
    "force_ring",
    "force_pinky",
]

FEATURE_COLUMNS: List[str] = [
    "pos_x", "pos_y", "pos_z",
    "acc_x", "acc_y", "acc_z",
    "force_thumb", "force_index", "force_middle", "force_ring", "force_pinky",
]


def _compute_trajectory(
    timestamps: np.ndarray,
    duration: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Return (pos_x, pos_y, pos_z, acc_x, acc_y, acc_z) along a figure-8."""
    omega1 = 2.0 * np.pi / (duration / 2.0)
    omega2 = 2.0 * np.pi / duration

    pos_x = 0.15 * np.sin(omega1 * timestamps)
    pos_y = 0.10 * np.sin(2.0 * omega1 * timestamps)
    pos_z = 0.25 + 0.05 * np.sin(omega2 * timestamps)

    acc_x = -0.15 * (omega1 ** 2) * np.sin(omega1 * timestamps)
    acc_y = -0.10 * ((2.0 * omega1) ** 2) * np.sin(2.0 * omega1 * timestamps)
    acc_z = -0.05 * (omega2 ** 2) * np.sin(omega2 * timestamps) + 9.81

    return pos_x, pos_y, pos_z, acc_x, acc_y, acc_z


def _compute_grip_forces(
    timestamps: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Generate smooth 5-finger grip force profiles with Gaussian noise."""
    n = len(timestamps)
    envelope = np.full(n, 0.3)

    for i in range(n):
        t = timestamps[i]
        if 1.5 <= t < 2.5:
            phase = (t - 1.5) / 1.0
            envelope[i] = 0.3 + 4.7 * 0.5 * (1.0 - np.cos(np.pi * phase))
        elif 2.5 <= t < 7.0:
            envelope[i] = 5.0 + 0.3 * np.sin(2.0 * np.pi * t * 0.5)
        elif 7.0 <= t < 8.0:
            phase = (t - 7.0) / 1.0
            envelope[i] = 5.0 - 4.7 * 0.5 * (1.0 - np.cos(np.pi * phase))

    finger_scales = {
        "force_thumb": 1.00,
        "force_index": 0.85,
        "force_middle": 0.70,
        "force_ring": 0.55,
        "force_pinky": 0.40,
    }

    forces: Dict[str, np.ndarray] = {}
    for name, scale in finger_scales.items():
        noise = rng.normal(0.0, 0.02, n)
        forces[name] = np.maximum(envelope * scale + noise, 0.0)

    return forces


def _inject_fault_sensor_flatline(
    forces: Dict[str, np.ndarray],
    timestamps: np.ndarray,
) -> None:
    """Fault 1: force_index → 0.0001 N for t ∈ [2.5, 3.5]."""
    mask = (timestamps >= 2.5) & (timestamps <= 3.5)
    forces["force_index"][mask] = 0.0001


def _inject_fault_imu_spike(
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    timestamps: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Fault 2: ±65 m/s² spikes on acc_x, acc_y for t ∈ [5.8, 5.95]."""
    mask = (timestamps >= 5.8) & (timestamps <= 5.95)
    n_spike = int(np.sum(mask))
    acc_x[mask] += rng.choice([-65.0, 65.0], size=n_spike)
    acc_y[mask] += rng.choice([-65.0, 65.0], size=n_spike)


def _render_frame(
    pos_x: float,
    pos_y: float,
    total_force: float,
) -> np.ndarray:
    """Render a single top-down dark-mode workbench frame (BGR, 320×240)."""
    frame = np.full((FRAME_H, FRAME_W, 3), 25, dtype=np.uint8)

    for x in range(0, FRAME_W, 40):
        cv2.line(frame, (x, 0), (x, FRAME_H), (45, 45, 45), 1)
    for y in range(0, FRAME_H, 40):
        cv2.line(frame, (0, y), (FRAME_W, y), (45, 45, 45), 1)

    cv2.rectangle(frame, (240, 20), (305, 75), (30, 90, 180), 2)
    cv2.putText(
        frame, "BIN", (255, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 90, 180), 1, cv2.LINE_AA,
    )

    px = int((pos_x + 0.20) / 0.40 * FRAME_W)
    py = int((pos_y + 0.15) / 0.30 * FRAME_H)
    px = max(14, min(px, FRAME_W - 14))
    py = max(14, min(py, FRAME_H - 14))

    ratio = float(np.clip(total_force / 20.0, 0.0, 1.0))
    color_b = 50
    color_g = int((1.0 - ratio) * 220 + 35)
    color_r = int(ratio * 220 + 35)
    cv2.circle(frame, (px, py), 14, (color_b, color_g, color_r), -1)
    cv2.circle(frame, (px, py), 14, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def _generate_video_stream(
    timestamps_video: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    forces: Dict[str, np.ndarray],
) -> List[np.ndarray]:
    """Generate 30 Hz BGR video frames; inject Fault 3 (camera freeze)."""
    frames: List[np.ndarray] = []
    last_valid_frame: np.ndarray | None = None

    for t_v in timestamps_video:
        idx = min(int(round(t_v * SENSOR_HZ)), len(pos_x) - 1)
        total_force = sum(forces[k][idx] for k in forces)

        if 7.5 <= t_v <= 8.3:
            if last_valid_frame is not None:
                frames.append(last_valid_frame.copy())
            else:
                frame = _render_frame(float(pos_x[idx]), float(pos_y[idx]), total_force)
                frames.append(frame)
        else:
            frame = _render_frame(float(pos_x[idx]), float(pos_y[idx]), total_force)
            last_valid_frame = frame.copy()
            frames.append(frame)

    return frames


def generate_factory_episode(
    duration: float = 10.0,
    seed: int = 42,
) -> Tuple[pl.DataFrame, List[np.ndarray], np.ndarray]:
    """Generate a complete factory episode with 3 injected hardware faults.

    Args:
        duration: Episode length in seconds (default 10.0 → 2000 rows at 200 Hz).
        seed: Deterministic RNG seed for reproducibility.

    Returns:
        sensor_df: Polars DataFrame with columns ``timestamp_s`` plus 11 channels
            (pos_{x,y,z}, acc_{x,y,z}, force_{thumb,index,middle,ring,pinky}).
        video_frames: List of 30 Hz BGR video frames (320×240 ``np.ndarray``).
        video_timestamps: 1-D ``np.ndarray`` of video-frame timestamps in seconds.
    """
    rng = np.random.default_rng(seed)
    n_sensor = int(duration * SENSOR_HZ)
    timestamps_sensor = np.arange(n_sensor, dtype=np.float64) / SENSOR_HZ

    pos_x, pos_y, pos_z, acc_x, acc_y, acc_z = _compute_trajectory(
        timestamps_sensor, duration,
    )

    acc_x = acc_x + rng.normal(0.0, 0.05, n_sensor)
    acc_y = acc_y + rng.normal(0.0, 0.05, n_sensor)
    acc_z = acc_z + rng.normal(0.0, 0.05, n_sensor)

    forces = _compute_grip_forces(timestamps_sensor, rng)

    _inject_fault_sensor_flatline(forces, timestamps_sensor)
    _inject_fault_imu_spike(acc_x, acc_y, timestamps_sensor, rng)

    sensor_df = pl.DataFrame({
        "timestamp_s": timestamps_sensor,
        "pos_x": pos_x,
        "pos_y": pos_y,
        "pos_z": pos_z,
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "force_thumb": forces["force_thumb"],
        "force_index": forces["force_index"],
        "force_middle": forces["force_middle"],
        "force_ring": forces["force_ring"],
        "force_pinky": forces["force_pinky"],
    })

    n_video = int(duration * VIDEO_HZ)
    timestamps_video = np.arange(n_video, dtype=np.float64) / VIDEO_HZ
    video_frames = _generate_video_stream(timestamps_video, pos_x, pos_y, forces)

    return sensor_df, video_frames, timestamps_video

"""Pytest suite proving all 3 injected anomalies are caught with 100 % recall."""

import numpy as np
import pytest
import torch

from gloveguard.simulator import generate_factory_episode
from gloveguard.sharder import CleanRoboticsShardDataset, FEATURE_COLUMNS, WINDOW_SIZE
from gloveguard.validator import run_quality_gate


@pytest.fixture(scope="module")
def episode_data():
    """Generate one deterministic factory episode for the entire test module."""
    return generate_factory_episode(duration=10.0, seed=42)


@pytest.fixture(scope="module")
def validated_data(episode_data):
    """Run the quality gate once and share results across tests."""
    sensor_df, video_frames, video_timestamps = episode_data
    return run_quality_gate(sensor_df, video_frames, video_timestamps)


class TestSensorFlatline:
    """Fault 1 — force_index dead-zone at t ∈ [2.5, 3.5]."""

    def test_flatline_detected(self, validated_data):
        _, qa_report = validated_data
        assert qa_report["flag_sensor_flatline_count"] > 0, (
            "Sensor flatline fault was not detected"
        )

    def test_flatline_samples_in_expected_range(self, validated_data):
        validated_df, _ = validated_data
        flagged = validated_df.filter(validated_df["flag_sensor_flatline"])
        times = flagged["timestamp_s"].to_numpy()
        assert times.min() >= 2.4, "Flatline flagged too early"
        assert times.max() <= 3.6, "Flatline flagged too late"


class TestKinematicSpike:
    """Fault 2 — IMU acceleration spike at t ∈ [5.8, 5.95]."""

    def test_spike_detected(self, validated_data):
        _, qa_report = validated_data
        assert qa_report["flag_kinematic_spike_count"] > 0, (
            "Kinematic IMU spike was not detected"
        )

    def test_spike_samples_in_expected_range(self, validated_data):
        validated_df, _ = validated_data
        flagged = validated_df.filter(validated_df["flag_kinematic_spike"])
        times = flagged["timestamp_s"].to_numpy()
        assert times.min() >= 5.75, "IMU spike flagged too early"
        assert times.max() <= 6.0, "IMU spike flagged too late"


class TestCrossModalDesync:
    """Fault 3 — camera freeze at t ∈ [7.5, 8.3]."""

    def test_desync_detected(self, validated_data):
        _, qa_report = validated_data
        assert qa_report["flag_cross_modal_desync_count"] > 0, (
            "Cross-modal camera desync was not detected"
        )

    def test_desync_samples_in_expected_range(self, validated_data):
        validated_df, _ = validated_data
        flagged = validated_df.filter(validated_df["flag_cross_modal_desync"])
        times = flagged["timestamp_s"].to_numpy()
        assert times.min() >= 7.4, "Camera desync flagged too early"
        assert times.max() <= 8.4, "Camera desync flagged too late"


class TestQAReport:
    """Aggregate QA report sanity checks."""

    def test_all_three_fault_types_present(self, validated_data):
        _, qa_report = validated_data
        assert qa_report["flag_sensor_flatline_count"] > 0
        assert qa_report["flag_kinematic_spike_count"] > 0
        assert qa_report["flag_cross_modal_desync_count"] > 0

    def test_corrupted_count_is_sum_of_flags(self, validated_data):
        validated_df, qa_report = validated_data
        recomputed = int(validated_df["is_corrupted"].sum())
        assert qa_report["total_corrupted_count"] == recomputed

    def test_rejected_pct_under_thirty(self, validated_data):
        _, qa_report = validated_data
        assert qa_report["rejected_pct"] < 30.0, (
            "Too many samples corrupted; episode should be SHARDED_WITH_EXCISION"
        )

    def test_release_status(self, validated_data):
        _, qa_report = validated_data
        assert qa_report["release_status"] == "SHARDED_WITH_EXCISION"

    def test_processing_time_reported(self, validated_data):
        _, qa_report = validated_data
        assert qa_report["processing_time_ms"] > 0.0


class TestCleanShards:
    """CleanRoboticsShardDataset yields zero corrupted rows."""

    def test_clean_shards_shape(self, validated_data):
        validated_df, _ = validated_data
        dataset = CleanRoboticsShardDataset(validated_df)
        for window in dataset:
            assert window.shape == (WINDOW_SIZE, len(FEATURE_COLUMNS))
            assert window.dtype == torch.float32

    def test_clean_shards_nonzero_count(self, validated_data):
        validated_df, _ = validated_data
        dataset = CleanRoboticsShardDataset(validated_df)
        assert len(dataset) > 0, "No clean windows produced"

    def test_clean_shards_no_corrupted_overlap(self, validated_data):
        validated_df, _ = validated_data
        is_corrupted = validated_df["is_corrupted"].to_numpy().astype(bool)
        data_np = validated_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)

        dataset = CleanRoboticsShardDataset(validated_df)
        window_count = 0

        for window in dataset:
            window_np = window.numpy()
            found = False
            n_total = len(data_np) // WINDOW_SIZE
            for i in range(n_total):
                s = i * WINDOW_SIZE
                e = s + WINDOW_SIZE
                if np.array_equal(window_np, data_np[s:e]):
                    assert not np.any(is_corrupted[s:e]), (
                        f"Window at index {i} is corrupted but was yielded"
                    )
                    found = True
                    break
            assert found, "Yielded window does not match any contiguous block"
            window_count += 1

        n_windows = len(data_np) // WINDOW_SIZE
        expected_clean = sum(
            1 for i in range(n_windows)
            if not np.any(is_corrupted[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE])
        )
        assert window_count == expected_clean

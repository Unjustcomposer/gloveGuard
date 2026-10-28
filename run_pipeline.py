#!/usr/bin/env python
"""GloveGuard — end-to-end CLI pipeline.

Usage::

    python run_pipeline.py                      # full run with Rerun viewer
    python run_pipeline.py --headless           # skip viewer, save .rrd
    python run_pipeline.py --duration 5 --seed 0
    python run_pipeline.py --export-dir ./out
"""

import argparse
import json
import os
import sys

from gloveguard.simulator import generate_factory_episode
from gloveguard.validator import run_quality_gate
from gloveguard.sharder import export_clean_shards


def _print_table(qa: dict) -> None:
    """Print a formatted CLI table from the QA report dictionary."""
    w = 52
    print("+" + "-" * w + "+")
    print("|{:^{w}}|".format("GloveGuard QA Report", w=w))
    print("+" + "-" * w + "+")
    rows = [
        ("Total Samples", f"{qa['total_samples']}"),
        ("Sensor Flatline (Fault 1)", f"{qa['flag_sensor_flatline_count']}"),
        ("IMU Spike (Fault 2)", f"{qa['flag_kinematic_spike_count']}"),
        ("Camera Desync (Fault 3)", f"{qa['flag_cross_modal_desync_count']}"),
        ("Total Corrupted", f"{qa['total_corrupted_count']}"),
        ("Rejected %", f"{qa['rejected_pct']:.2f}%"),
        ("Processing Time", f"{qa['processing_time_ms']:.2f} ms"),
        ("Release Status", qa["release_status"]),
    ]
    for label, value in rows:
        print(f"| {label:<30} | {value:>17} |")
    print("+" + "-" * w + "+")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GloveGuard end-to-end pipeline",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Episode duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for deterministic generation (default: 42)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Skip Rerun viewer; save .rrd to export dir instead",
    )
    parser.add_argument(
        "--export-dir", type=str, default="./artifacts",
        help="Root directory for shard and dashboard exports (default: ./artifacts)",
    )
    args = parser.parse_args()

    print("\n[1/4] Generating factory episode "
          f"({args.duration}s, seed={args.seed}) ...")
    sensor_df, video_frames, video_timestamps = generate_factory_episode(
        duration=args.duration, seed=args.seed,
    )
    print(f"      Sensor rows : {len(sensor_df)}")
    print(f"      Video frames: {len(video_frames)}")

    print("\n[2/4] Running vectorized QA gate ...")
    validated_df, qa_report = run_quality_gate(
        sensor_df, video_frames, video_timestamps,
    )

    print()
    _print_table(qa_report)

    qa_json_path = os.path.join(args.export_dir, "qa_report.json")
    os.makedirs(args.export_dir, exist_ok=True)
    with open(qa_json_path, "w") as fh:
        json.dump(qa_report, fh, indent=2)
    print(f"\n      QA JSON saved to {qa_json_path}")

    print("\n[3/4] Exporting clean shards ...")
    shard_dir = os.path.join(args.export_dir, "shards")
    manifest = export_clean_shards(validated_df, export_dir=shard_dir)
    print(f"      Total windows : {manifest['total_windows']}")
    print(f"      Clean windows : {manifest['clean_windows']}")
    print(f"      Excised       : {manifest['excised_windows']}")

    print("\n[4/4] Launching Rerun dashboard "
          f"({'headless' if args.headless else 'interactive'}) ...")
    from gloveguard.visualizer import launch_rerun_dashboard
    launch_rerun_dashboard(
        validated_df,
        video_frames,
        video_timestamps,
        headless=args.headless,
        export_dir=args.export_dir,
    )
    print("      Done.\n")


if __name__ == "__main__":
    main()

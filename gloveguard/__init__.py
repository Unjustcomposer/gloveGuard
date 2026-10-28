"""GloveGuard — Multimodal Telemetry QA & PyTorch Ingestion Engine."""

from gloveguard.simulator import generate_factory_episode
from gloveguard.validator import run_quality_gate
from gloveguard.sharder import CleanRoboticsShardDataset, export_clean_shards

__all__ = [
    "generate_factory_episode",
    "run_quality_gate",
    "CleanRoboticsShardDataset",
    "export_clean_shards",
]

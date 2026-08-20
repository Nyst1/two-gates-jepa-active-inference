from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import (
    CHECKPOINT_MANIFEST_NAME,
    CHECKPOINT_NAME,
    DEFAULT_BETA,
    DEFAULT_CUE_RELIABILITY,
    DEFAULT_PRIOR,
    DEFAULT_SEED,
    SCHEMA_VERSION,
)
from .model import CheckpointModel
from .planner import EpisodeConfig, run_episode_trace


def export_replays(project_root: Path) -> list[dict[str, Any]]:
    checkpoint_dir = project_root / "artifacts" / "checkpoints"
    model = CheckpointModel(
        checkpoint_dir / CHECKPOINT_NAME,
        checkpoint_dir / CHECKPOINT_MANIFEST_NAME,
    )
    destinations = [
        project_root / "artifacts" / "replays",
        project_root / "frontend" / "public" / "replays",
    ]
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)

    index: list[dict[str, Any]] = []
    for agent in ("balanced", "pragmatic", "information"):
        replay_id = f"two-gates-{agent}-seed-{DEFAULT_SEED}"
        config = EpisodeConfig(
            agent_type=agent,
            seed=DEFAULT_SEED,
            beta=DEFAULT_BETA,
            prior=DEFAULT_PRIOR,
            cue_reliability=DEFAULT_CUE_RELIABILITY,
            source="replay",
        )
        trace = run_episode_trace(
            config,
            checkpoint_hash=model.checkpoint_hash,
            latent_provider=model.latent_prediction if model.available else None,
        )
        trace["id"] = replay_id
        trace["modelMode"] = "checkpoint" if model.available else "analytic fallback"
        file_name = f"{replay_id}.json"
        for destination in destinations:
            (destination / file_name).write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
        index.append(
            {
                "id": replay_id,
                "agentType": agent,
                "seed": DEFAULT_SEED,
                "schemaVersion": SCHEMA_VERSION,
                "checkpointHash": model.checkpoint_hash,
                "modelMode": trace["modelMode"],
                "summary": trace["summary"],
                "file": file_name,
            }
        )
    for destination in destinations:
        (destination / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def read_replays(project_root: Path) -> list[dict[str, Any]]:
    index_path = project_root / "artifacts" / "replays" / "index.json"
    if not index_path.exists():
        return []
    return json.loads(index_path.read_text(encoding="utf-8"))

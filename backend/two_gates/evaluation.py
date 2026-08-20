from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any

from .planner import EpisodeConfig, run_episode_trace


def evaluate_behavior(episodes: int = 200) -> dict[str, Any]:
    results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed in range(episodes):
        for agent in ("pragmatic", "balanced", "information"):
            trace = run_episode_trace(EpisodeConfig(agent_type=agent, seed=seed, source="replay"))
            results[agent].append(trace["summary"])

    summary: dict[str, Any] = {}
    for agent, rows in results.items():
        summary[agent] = {
            "episodes": len(rows),
            "successRate": sum(int(row["success"]) for row in rows) / len(rows),
            "medianSteps": median(row["steps"] for row in rows),
            "meanWrongGateCommitments": sum(row["wrongGateCommitments"] for row in rows) / len(rows),
            "meanGateTests": sum(row["gateTests"] for row in rows) / len(rows),
            "meanCollisions": sum(row["collisions"] for row in rows) / len(rows),
        }

    pragmatic = summary["pragmatic"]
    balanced = summary["balanced"]
    wrong_reduction = (
        1.0 - balanced["meanWrongGateCommitments"] / pragmatic["meanWrongGateCommitments"]
        if pragmatic["meanWrongGateCommitments"] > 0
        else 1.0
    )
    summary["acceptance"] = {
        "balancedSuccessNotLower": balanced["successRate"] >= pragmatic["successRate"],
        "wrongGateReduction": wrong_reduction,
        "wrongGateReductionPass": wrong_reduction >= 0.30,
        "medianPathRatio": balanced["medianSteps"] / max(1.0, pragmatic["medianSteps"]),
        "medianPathRatioPass": balanced["medianSteps"] <= pragmatic["medianSteps"] * 1.20,
    }
    summary["passed"] = all(
        (
            summary["acceptance"]["balancedSuccessNotLower"],
            summary["acceptance"]["wrongGateReductionPass"],
            summary["acceptance"]["medianPathRatioPass"],
        )
    )
    return summary

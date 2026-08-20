from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from statistics import median
import time
from typing import Any

import numpy as np

from .belief import posterior_after_cue
from .constants import (
    ACTION_INDEX,
    ACTIONS,
    CHECKPOINT_MANIFEST_NAME,
    CHECKPOINT_NAME,
    DATASET_VERSION,
    GOAL,
    SCHEMA_VERSION,
)
from .environment import TwoGatesEnv
from .model import TORCH_AVAILABLE, TinyLatentWorldModel

if TORCH_AVAILABLE:
    import torch
    from torch import nn
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, Dataset, random_split


@dataclass(slots=True)
class TrainingConfig:
    transitions: int = 50_000
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 3e-4
    seeds: tuple[int, ...] = (11, 29, 47)
    dataset_seed: int = 20260818
    latent_dim: int = 8
    diffusion_steps: int = 8


if TORCH_AVAILABLE:

    class TransitionDataset(Dataset[dict[str, torch.Tensor]]):
        def __init__(self, records: dict[str, np.ndarray]) -> None:
            self.records = records

        def __len__(self) -> int:
            return int(self.records["actions"].shape[0])

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {
                "observation": torch.from_numpy(self.records["observations"][index]).float() / 255.0,
                "nextObservation": torch.from_numpy(self.records["next_observations"][index]).float() / 255.0,
                "action": torch.tensor(self.records["actions"][index], dtype=torch.long),
                "belief": torch.tensor([self.records["beliefs"][index]], dtype=torch.float32),
                "position": torch.from_numpy(self.records["positions"][index]).float(),
                "nextPosition": torch.from_numpy(self.records["next_positions"][index]).float(),
                "gateBelief": torch.tensor([self.records["beliefs"][index]], dtype=torch.float32),
                "goal": torch.tensor([self.records["goals"][index]], dtype=torch.float32),
            }


def _choose_collection_action(env: TwoGatesEnv, rng: random.Random) -> str:
    if rng.random() < 0.55:
        return rng.choice(list(ACTIONS))
    target = GOAL if env.sensor_visited or rng.random() < 0.55 else env.sensor
    dx = target[0] - env.position[0]
    dy = target[1] - env.position[1]
    candidates: list[str] = []
    if dx > 0:
        candidates.append("right")
    elif dx < 0:
        candidates.append("left")
    if dy > 0:
        candidates.append("down")
    elif dy < 0:
        candidates.append("up")
    return rng.choice(candidates or list(ACTIONS))


def generate_transition_dataset(count: int, seed: int) -> dict[str, np.ndarray]:
    rng = random.Random(seed)
    observations = np.empty((count, 3, 32, 32), dtype=np.uint8)
    next_observations = np.empty_like(observations)
    actions = np.empty((count,), dtype=np.int64)
    beliefs = np.empty((count,), dtype=np.float32)
    positions = np.empty((count, 2), dtype=np.float32)
    next_positions = np.empty((count, 2), dtype=np.float32)
    goals = np.empty((count,), dtype=np.float32)

    index = 0
    episode_seed = seed
    while index < count:
        env = TwoGatesEnv(seed=episode_seed, prior_upper=0.5, cue_reliability=0.9)
        belief = 0.5
        episode_seed += 1
        for _ in range(36):
            if index >= count:
                break
            observation = env.render_observation(belief)
            position = env.position
            action = _choose_collection_action(env, rng)
            transition = env.step(action)
            next_belief = belief
            if transition.cue:
                next_belief = posterior_after_cue(belief, transition.cue, env.cue_reliability)
            if transition.revealed_gate:
                next_belief = 1.0 if transition.revealed_gate == "upper" else 0.0
            next_observation = env.render_observation(next_belief)

            observations[index] = np.transpose((observation * 255.0).round().astype(np.uint8), (2, 0, 1))
            next_observations[index] = np.transpose(
                (next_observation * 255.0).round().astype(np.uint8), (2, 0, 1)
            )
            actions[index] = ACTION_INDEX[action]
            beliefs[index] = belief
            positions[index] = (position[0] / 12.0, position[1] / 8.0)
            next_positions[index] = (env.position[0] / 12.0, env.position[1] / 8.0)
            goals[index] = float(position == GOAL)
            belief = next_belief
            index += 1
            if env.done:
                break
    return {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "beliefs": beliefs,
        "positions": positions,
        "next_positions": next_positions,
        "goals": goals,
    }


def _variance_covariance_loss(latent: Any) -> tuple[Any, Any]:
    centered = latent - latent.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0) + 1e-4)
    variance_loss = torch.relu(1.0 - std).mean()
    covariance = centered.T @ centered / max(1, latent.shape[0] - 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    covariance_loss = off_diagonal.pow(2).sum() / latent.shape[1]
    return variance_loss, covariance_loss


def _batch_loss(model: Any, batch: dict[str, Any], device: str, *, train: bool) -> tuple[Any, dict[str, float]]:
    observation = batch["observation"].to(device)
    next_observation = batch["nextObservation"].to(device)
    action = batch["action"].to(device)
    belief = batch["belief"].to(device)
    position = batch["position"].to(device)
    next_position = batch["nextPosition"].to(device)
    gate_belief = batch["gateBelief"].to(device)
    goal = batch["goal"].to(device)

    current = model.encode(observation)
    next_online = model.encode(next_observation)
    with torch.no_grad():
        target = model.target_encode(next_observation)

    deterministic = model.deterministic_next(current, action, belief)
    deterministic_loss = functional.mse_loss(deterministic, target)
    diffusion_index = torch.randint(0, model.diffusion_steps, (target.shape[0],), device=device)
    noise = torch.randn_like(target)
    alpha_bar = model.alpha_bars[diffusion_index].unsqueeze(-1)
    noisy_target = torch.sqrt(alpha_bar) * target + torch.sqrt(1.0 - alpha_bar) * noise
    predicted_noise = model.denoise(noisy_target, current, action, belief, diffusion_index)
    diffusion_loss = functional.mse_loss(predicted_noise, noise)

    variance_loss, covariance_loss = _variance_covariance_loss(current)
    inverse_loss = functional.cross_entropy(model.inverse_head(torch.cat((current, next_online), dim=-1)), action)
    position_loss = functional.mse_loss(model.position_head(current), position)
    action_position_loss = functional.mse_loss(model.position_head(deterministic), next_position)
    gate_loss = functional.binary_cross_entropy_with_logits(model.gate_head(current), gate_belief)
    goal_loss = functional.binary_cross_entropy_with_logits(model.goal_head(current), goal)

    total = (
        diffusion_loss
        + 1.00 * deterministic_loss
        + 0.20 * variance_loss
        + 0.05 * covariance_loss
        + 0.50 * inverse_loss
        + 5.00 * position_loss
        + 5.00 * action_position_loss
        + 0.35 * gate_loss
        + 0.10 * goal_loss
    )
    metrics = {
        "total": float(total.detach().cpu()),
        "diffusion": float(diffusion_loss.detach().cpu()),
        "deterministic": float(deterministic_loss.detach().cpu()),
        "variance": float(variance_loss.detach().cpu()),
        "covariance": float(covariance_loss.detach().cpu()),
        "inverse": float(inverse_loss.detach().cpu()),
        "position": float(position_loss.detach().cpu()),
        "actionPosition": float(action_position_loss.detach().cpu()),
    }
    return total, metrics


def _evaluate_model(model: Any, loader: Any, device: str) -> dict[str, float]:
    model.eval()
    collected: dict[str, list[float]] = {}
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            _, metrics = _batch_loss(model, batch, device, train=False)
            for key, value in metrics.items():
                collected.setdefault(key, []).append(value)
            if batch_index >= 20:
                break
    return {key: float(np.mean(values)) for key, values in collected.items()}


def _fit_seed(
    seed: int,
    dataset: Any,
    config: TrainingConfig,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_size = int(len(dataset) * 0.8)
    validation_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(config.dataset_seed)
    train_set, validation_set = random_split(dataset, [train_size, validation_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = TinyLatentWorldModel(config.latent_dim, config.diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        epoch_metrics: dict[str, list[float]] = {}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _batch_loss(model, batch, device, train=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            model.update_target_encoder(momentum=0.99)
            for key, value in metrics.items():
                epoch_metrics.setdefault(key, []).append(value)
        validation = _evaluate_model(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train": {key: float(np.mean(values)) for key, values in epoch_metrics.items()},
                "validation": validation,
            }
        )
    elapsed = time.perf_counter() - started
    result = {
        "seed": seed,
        "elapsedSeconds": elapsed,
        "history": history,
        "validationLoss": history[-1]["validation"]["total"],
    }
    return model, result


def _compute_pca(model: Any, records: dict[str, np.ndarray], device: str) -> dict[str, Any]:
    sample = torch.from_numpy(records["observations"][:2048]).float().to(device) / 255.0
    with torch.inference_mode():
        latent = model.encode(sample).cpu().numpy()
    center = latent.mean(axis=0)
    _, _, right = np.linalg.svd(latent - center, full_matrices=False)
    return {
        "center": center.astype(float).tolist(),
        "components": right[:2].astype(float).tolist(),
    }


def _model_quality(model: Any, records: dict[str, np.ndarray], device: str) -> dict[str, Any]:
    observations = torch.from_numpy(records["observations"][:4096]).float().to(device) / 255.0
    next_observations = torch.from_numpy(records["next_observations"][:4096]).float().to(device) / 255.0
    actions = torch.from_numpy(records["actions"][:4096]).long().to(device)
    beliefs = torch.from_numpy(records["beliefs"][:4096]).float().unsqueeze(1).to(device)
    positions = torch.from_numpy(records["positions"][:4096]).float().to(device)
    next_positions = torch.from_numpy(records["next_positions"][:4096]).float().to(device)
    with torch.inference_mode():
        current = model.encode(observations)
        target = model.target_encode(next_observations)
        correct_prediction = model.deterministic_next(current, actions, beliefs)
        shuffled_prediction = model.deterministic_next(current, actions.roll(1), beliefs)
        correct = functional.mse_loss(correct_prediction, target)
        shuffled = functional.mse_loss(shuffled_prediction, target)
        correct_position = functional.mse_loss(model.position_head(correct_prediction), next_positions)
        shuffled_position = functional.mse_loss(model.position_head(shuffled_prediction), next_positions)
        position_mse = functional.mse_loss(model.position_head(current), positions)
        gate_brier = functional.mse_loss(torch.sigmoid(model.gate_head(current)), beliefs)
    latent_ratio = float((shuffled / correct).cpu())
    position_ratio = float((shuffled_position / correct_position).cpu())
    ratio = latent_ratio
    minimum_std = float(current.std(dim=0).min().cpu())
    position_value = float(position_mse.cpu())
    gate_value = float(gate_brier.cpu())
    checks = {
        "actionShuffleRatioAtLeast1_2": ratio >= 1.2,
        "minimumLatentStdAtLeast0_25": minimum_std >= 0.25,
        "positionMseAtMost0_03": position_value <= 0.03,
        "gateBeliefBrierAtMost0_08": gate_value <= 0.08,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "actionConditioning": {
            "test": "deterministic next-latent prediction with action shuffling",
            "correctMse": float(correct.cpu()),
            "shuffledMse": float(shuffled.cpu()),
            "shuffledToCorrectRatio": ratio,
            "nextPositionProbe": {
                "correctMse": float(correct_position.cpu()),
                "shuffledMse": float(shuffled_position.cpu()),
                "shuffledToCorrectRatio": position_ratio,
            },
        },
        "latent": {"minimumDimensionStd": minimum_std},
        "positionHeadMse": position_value,
        "gateBeliefBrier": gate_value,
    }


def train_production_checkpoint(
    output_dir: Path,
    *,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to train the live model.")
    config = config or TrainingConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = generate_transition_dataset(config.transitions, config.dataset_seed)
    dataset = TransitionDataset(records)

    trained: list[tuple[Any, dict[str, Any]]] = []
    for seed in config.seeds:
        trained.append(_fit_seed(seed, dataset, config, device))
    ordered = sorted(trained, key=lambda item: item[1]["validationLoss"])
    selected_model, selected_result = ordered[len(ordered) // 2]
    selected_model.eval()

    checkpoint_path = output_dir / CHECKPOINT_NAME
    torch.save(
        {
            "model": selected_model.state_dict(),
            "trainingConfig": asdict(config),
            "selectedSeed": selected_result["seed"],
        },
        checkpoint_path,
    )
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "datasetVersion": DATASET_VERSION,
        "checkpointHash": checkpoint_hash,
        "checkpointFile": CHECKPOINT_NAME,
        "deviceUsedForTraining": device,
        "selection": "median validation loss across seeds",
        "selectedSeed": selected_result["seed"],
        "model": {
            "type": "minimal JEPA-style conditional latent diffusion",
            "latentDim": config.latent_dim,
            "diffusionSteps": config.diffusion_steps,
            "futureSamples": 16,
            "planningHorizon": 5,
        },
        "training": {
            "transitions": config.transitions,
            "epochs": config.epochs,
            "batchSize": config.batch_size,
            "seeds": list(config.seeds),
            "runs": [result for _, result in trained],
        },
        "pca": _compute_pca(selected_model, records, device),
        "qualityAssurance": _model_quality(selected_model, records, device),
    }
    manifest_path = output_dir / CHECKPOINT_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_production_checkpoint(output_dir: Path, *, samples: int = 5_000) -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to validate the live model.")
    checkpoint_path = output_dir / CHECKPOINT_NAME
    manifest_path = output_dir / CHECKPOINT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyLatentWorldModel(
        int(manifest["model"]["latentDim"]),
        int(manifest["model"]["diffusionSteps"]),
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"])
    model.eval()
    records = generate_transition_dataset(samples, 20260819)
    manifest["qualityAssurance"] = _model_quality(model, records, device)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest["qualityAssurance"]

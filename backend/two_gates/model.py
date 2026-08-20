from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .constants import ACTION_INDEX, FUTURE_SAMPLES, WIDTH, HEIGHT

try:
    import torch
    from torch import nn
    import torch.nn.functional as functional

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the replay-only runtime
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class TinyEncoder(nn.Module):
        def __init__(self, latent_dim: int = 8) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 48, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Flatten(),
                nn.Linear(48 * 4 * 4, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Linear(64, latent_dim),
            )

        def forward(self, observation: torch.Tensor) -> torch.Tensor:
            return self.network(observation)


    class TinyLatentWorldModel(nn.Module):
        """Small JEPA-style encoder plus an 8-step conditional latent denoiser."""

        def __init__(self, latent_dim: int = 8, diffusion_steps: int = 8) -> None:
            super().__init__()
            self.latent_dim = latent_dim
            self.diffusion_steps = diffusion_steps
            self.encoder = TinyEncoder(latent_dim)
            self.target_encoder = TinyEncoder(latent_dim)
            self.target_encoder.load_state_dict(self.encoder.state_dict())
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)

            condition_dim = latent_dim * 2 + 4 + 1 + 1
            self.denoiser = nn.Sequential(
                nn.Linear(condition_dim, 96),
                nn.SiLU(),
                nn.Linear(96, 96),
                nn.SiLU(),
                nn.Linear(96, latent_dim),
            )
            self.deterministic_predictor = nn.Sequential(
                nn.Linear(latent_dim + 4 + 1, 64),
                nn.SiLU(),
                nn.Linear(64, latent_dim),
            )
            self.inverse_head = nn.Sequential(
                nn.Linear(latent_dim * 2, 48),
                nn.SiLU(),
                nn.Linear(48, 4),
            )
            self.position_head = nn.Sequential(nn.Linear(latent_dim, 32), nn.SiLU(), nn.Linear(32, 2), nn.Sigmoid())
            self.gate_head = nn.Sequential(nn.Linear(latent_dim, 16), nn.SiLU(), nn.Linear(16, 1))
            self.goal_head = nn.Sequential(nn.Linear(latent_dim, 16), nn.SiLU(), nn.Linear(16, 1))

            betas = torch.linspace(0.025, 0.18, diffusion_steps)
            alphas = 1.0 - betas
            self.register_buffer("betas", betas)
            self.register_buffer("alphas", alphas)
            self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

        def encode(self, observation: torch.Tensor) -> torch.Tensor:
            return self.encoder(observation)

        def target_encode(self, observation: torch.Tensor) -> torch.Tensor:
            return self.target_encoder(observation)

        @staticmethod
        def _one_hot_action(action_index: torch.Tensor) -> torch.Tensor:
            return functional.one_hot(action_index.long(), num_classes=4).float()

        def deterministic_next(
            self,
            current_latent: torch.Tensor,
            action_index: torch.Tensor,
            belief_upper: torch.Tensor,
        ) -> torch.Tensor:
            delta = self.deterministic_predictor(
                torch.cat((current_latent, self._one_hot_action(action_index), belief_upper), dim=-1)
            )
            return current_latent + delta

        def denoise(
            self,
            noisy_target: torch.Tensor,
            current_latent: torch.Tensor,
            action_index: torch.Tensor,
            belief_upper: torch.Tensor,
            diffusion_index: torch.Tensor,
        ) -> torch.Tensor:
            normalized_t = diffusion_index.float().unsqueeze(-1) / max(1, self.diffusion_steps - 1)
            condition = torch.cat(
                (
                    noisy_target,
                    current_latent,
                    self._one_hot_action(action_index),
                    belief_upper,
                    normalized_t,
                ),
                dim=-1,
            )
            return self.denoiser(condition)

        @torch.no_grad()
        def update_target_encoder(self, momentum: float = 0.99) -> None:
            for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters(), strict=True):
                target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)

        @torch.no_grad()
        def sample_next(
            self,
            observation: torch.Tensor,
            action_index: torch.Tensor,
            belief_upper: torch.Tensor,
            *,
            samples: int = FUTURE_SAMPLES,
            generator: torch.Generator | None = None,
        ) -> torch.Tensor:
            current = self.encode(observation)
            current = current.repeat_interleave(samples, dim=0)
            actions = action_index.repeat_interleave(samples, dim=0)
            beliefs = belief_upper.repeat_interleave(samples, dim=0)
            latent = torch.randn(
                (current.shape[0], self.latent_dim),
                device=current.device,
                generator=generator,
            )
            for step in reversed(range(self.diffusion_steps)):
                t = torch.full((latent.shape[0],), step, device=latent.device, dtype=torch.long)
                predicted_noise = self.denoise(latent, current, actions, beliefs, t)
                alpha = self.alphas[step]
                alpha_bar = self.alpha_bars[step]
                beta = self.betas[step]
                latent = (latent - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)) * predicted_noise) / torch.sqrt(alpha)
                if step > 0:
                    latent = latent + torch.sqrt(beta) * torch.randn(
                        latent.shape,
                        device=latent.device,
                        generator=generator,
                    )
            return latent


else:

    class TinyLatentWorldModel:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("PyTorch is not installed; use verified replay mode.")


class CheckpointModel:
    """Loads one checkpoint and exposes projected samples to the planner."""

    def __init__(self, checkpoint_path: Path, manifest_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.manifest_path = manifest_path
        self.available = False
        self.reason = "checkpoint unavailable"
        self.checkpoint_hash = "analytic-replay-v1"
        self.device = "cpu"
        self.model: Any | None = None
        self.pca_center = np.zeros(8, dtype=np.float32)
        self.pca_components = np.eye(8, dtype=np.float32)[:2]

        if not TORCH_AVAILABLE:
            self.reason = "PyTorch is not installed"
            return
        if not checkpoint_path.exists() or not manifest_path.exists():
            self.reason = "trained checkpoint is missing"
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            quality = manifest.get("qualityAssurance")
            if quality is not None and not quality.get("passed", False):
                raise ValueError("checkpoint failed declared quality gates")
            self.checkpoint_hash = str(manifest["checkpointHash"])
            self.pca_center = np.asarray(manifest["pca"]["center"], dtype=np.float32)
            self.pca_components = np.asarray(manifest["pca"]["components"], dtype=np.float32)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model = TinyLatentWorldModel(
                latent_dim=int(manifest["model"]["latentDim"]),
                diffusion_steps=int(manifest["model"]["diffusionSteps"]),
            )
            payload = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(payload["model"])
            model.to(self.device).eval()
            self.model = model
            self.available = True
            self.reason = "ready"
        except Exception as error:  # pragma: no cover - defensive runtime fallback
            self.reason = f"checkpoint load failed: {error}"
            self.model = None

    def project(self, latent: np.ndarray) -> np.ndarray:
        return (latent - self.pca_center) @ self.pca_components.T

    @staticmethod
    def _rounded(values: Any) -> list[float]:
        if torch is not None and isinstance(values, torch.Tensor):
            values = values.detach().cpu().flatten().tolist()
        return [round(float(value), 5) for value in values]

    @staticmethod
    def _top_contribution_edges(
        source: Any,
        weight: Any,
        *,
        source_layer: str,
        target_layer: str,
        limit: int = 32,
    ) -> list[dict[str, Any]]:
        if torch is None:
            return []
        source_vector = source.detach().flatten()
        matrix = weight.detach()
        contributions = matrix * source_vector.unsqueeze(0)
        count = min(limit, contributions.numel())
        indices = torch.topk(contributions.abs().flatten(), k=count).indices
        edges: list[dict[str, Any]] = []
        source_width = contributions.shape[1]
        for flat_index in indices.tolist():
            target_index = flat_index // source_width
            source_index = flat_index % source_width
            edges.append(
                {
                    "sourceLayer": source_layer,
                    "source": source_index,
                    "targetLayer": target_layer,
                    "target": target_index,
                    "weight": round(float(matrix[target_index, source_index].cpu()), 5),
                    "contribution": round(float(contributions[target_index, source_index].cpu()), 5),
                }
            )
        return edges

    def _encoder_inspection(self, observation: Any) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
        if self.model is None or torch is None:
            raise RuntimeError("Model is not available")
        x = observation
        layers: list[dict[str, Any]] = [
            {
                "id": "rgb",
                "label": "RGB input",
                "kind": "input channels",
                "values": self._rounded(x[0].mean(dim=(1, 2))),
                "totalUnits": 3,
            }
        ]
        dense_features = None
        for index, module in enumerate(self.model.encoder.network):
            x = module(x)
            if index in (1, 3, 5):
                layer_id = {1: "conv1", 3: "conv2", 5: "conv3"}[index]
                layers.append(
                    {
                        "id": layer_id,
                        "label": {1: "Conv features 1", 3: "Conv features 2", 5: "Conv features 3"}[index],
                        "kind": "spatial channel mean",
                        "values": self._rounded(x[0].mean(dim=(1, 2))),
                        "totalUnits": int(x.shape[1]),
                    }
                )
            elif index == 9:
                dense_features = x[0]
                layers.append(
                    {
                        "id": "dense64",
                        "label": "Dense features",
                        "kind": "individual units",
                        "values": self._rounded(dense_features),
                        "totalUnits": 64,
                    }
                )
            elif index == 10:
                layers.append(
                    {
                        "id": "latent8",
                        "label": "Current latent",
                        "kind": "individual units",
                        "values": self._rounded(x[0]),
                        "totalUnits": int(x.shape[1]),
                    }
                )
        edges = []
        if dense_features is not None:
            edges = self._top_contribution_edges(
                dense_features,
                self.model.encoder.network[10].weight,
                source_layer="dense64",
                target_layer="latent8",
            )
        return x, layers, edges

    def _sample_with_inspection(
        self,
        observation: Any,
        action_tensor: Any,
        belief_tensor: Any,
        *,
        generator: Any,
    ) -> tuple[Any, dict[str, Any]]:
        if self.model is None or torch is None or functional is None:
            raise RuntimeError("Model is not available")
        current_single, encoder_layers, encoder_edges = self._encoder_inspection(observation)
        samples = FUTURE_SAMPLES
        current = current_single.repeat_interleave(samples, dim=0)
        actions = action_tensor.repeat_interleave(samples, dim=0)
        beliefs = belief_tensor.repeat_interleave(samples, dim=0)
        latent = torch.randn(
            (samples, self.model.latent_dim),
            device=self.device,
            generator=generator,
        )
        diffusion_trajectory: list[dict[str, Any]] = []
        denoiser_layers: list[dict[str, Any]] = []
        denoiser_edges: list[dict[str, Any]] = []
        for step in reversed(range(self.model.diffusion_steps)):
            t = torch.full((samples,), step, device=self.device, dtype=torch.long)
            predicted_noise = self.model.denoise(latent, current, actions, beliefs, t)
            diffusion_trajectory.append(
                {
                    "step": step,
                    "latentNorm": round(float(latent[0].norm().cpu()), 5),
                    "predictedNoiseNorm": round(float(predicted_noise[0].norm().cpu()), 5),
                }
            )
            if step == 0:
                normalized_t = torch.zeros((1, 1), device=self.device)
                condition = torch.cat(
                    (
                        latent[:1],
                        current[:1],
                        self.model._one_hot_action(actions[:1]),
                        beliefs[:1],
                        normalized_t,
                    ),
                    dim=-1,
                )[0]
                hidden1_pre = self.model.denoiser[0](condition)
                hidden1 = self.model.denoiser[1](hidden1_pre)
                hidden2_pre = self.model.denoiser[2](hidden1)
                hidden2 = self.model.denoiser[3](hidden2_pre)
                output = self.model.denoiser[4](hidden2)
                denoiser_layers = [
                    {
                        "id": "condition",
                        "label": "Condition vector",
                        "kind": "noise + latent + action + belief + t",
                        "values": self._rounded(condition),
                        "totalUnits": int(condition.shape[0]),
                    },
                    {
                        "id": "denoise1",
                        "label": "Denoiser hidden 1",
                        "kind": "individual units",
                        "values": self._rounded(hidden1),
                        "totalUnits": int(hidden1.shape[0]),
                    },
                    {
                        "id": "denoise2",
                        "label": "Denoiser hidden 2",
                        "kind": "individual units",
                        "values": self._rounded(hidden2),
                        "totalUnits": int(hidden2.shape[0]),
                    },
                    {
                        "id": "noise8",
                        "label": "Predicted noise",
                        "kind": "individual units",
                        "values": self._rounded(output),
                        "totalUnits": int(output.shape[0]),
                    },
                ]
                denoiser_edges = (
                    self._top_contribution_edges(
                        condition,
                        self.model.denoiser[0].weight,
                        source_layer="condition",
                        target_layer="denoise1",
                    )
                    + self._top_contribution_edges(
                        hidden1,
                        self.model.denoiser[2].weight,
                        source_layer="denoise1",
                        target_layer="denoise2",
                    )
                    + self._top_contribution_edges(
                        hidden2,
                        self.model.denoiser[4].weight,
                        source_layer="denoise2",
                        target_layer="noise8",
                    )
                )
            alpha = self.model.alphas[step]
            alpha_bar = self.model.alpha_bars[step]
            beta = self.model.betas[step]
            latent = (
                latent
                - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)) * predicted_noise
            ) / torch.sqrt(alpha)
            if step > 0:
                latent = latent + torch.sqrt(beta) * torch.randn(
                    latent.shape,
                    device=self.device,
                    generator=generator,
                )

        current_latent = current_single[0]
        future_latent = latent.mean(dim=0)
        thumbnail = functional.adaptive_avg_pool2d(observation, (8, 8))[0].permute(1, 2, 0)

        def readouts(value: Any) -> dict[str, Any]:
            batch = value.unsqueeze(0)
            return {
                "position": self._rounded(self.model.position_head(batch)[0]),
                "upperGateProbe": round(float(torch.sigmoid(self.model.gate_head(batch))[0, 0].cpu()), 5),
                "goalProbability": round(float(torch.sigmoid(self.model.goal_head(batch))[0, 0].cpu()), 5),
            }

        inspection = {
            "observation": {
                "shape": [3, 32, 32],
                "thumbnail": [
                    [[round(float(channel), 4) for channel in pixel] for pixel in row]
                    for row in thumbnail.detach().cpu().tolist()
                ],
            },
            "conditioning": {
                "beliefUpper": round(float(belief_tensor[0, 0].cpu()), 5),
                "actionOneHot": self._rounded(self.model._one_hot_action(action_tensor)[0]),
                "diffusionStepShown": 0,
            },
            "encoder": {"layers": encoder_layers, "edges": encoder_edges},
            "denoiser": {"layers": denoiser_layers, "edges": denoiser_edges},
            "diffusionTrajectory": diffusion_trajectory,
            "outputs": {
                "current": readouts(current_latent),
                "meanFuture": readouts(future_latent),
                "futureLatentMean": self._rounded(future_latent),
            },
            "explanation": (
                "CNN nodes summarize real spatial channel activations. Dense nodes are individual units; "
                "shown edges are the largest signed activation-times-weight contributions."
            ),
        }
        return latent, inspection

    def latent_prediction(
        self,
        env: Any,
        belief_upper: float,
        action: str,
        *,
        capture: bool = False,
    ) -> dict[str, Any] | None:
        if not self.available or self.model is None or torch is None:
            return None
        observation = env.render_observation(belief_upper)
        tensor = torch.from_numpy(observation).permute(2, 0, 1).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor([ACTION_INDEX[action]], device=self.device)
        belief_tensor = torch.tensor([[belief_upper]], dtype=torch.float32, device=self.device)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(env.seed * 10_000 + env.step_count * 101 + ACTION_INDEX[action])
        with torch.inference_mode():
            if capture:
                latent, inspection = self._sample_with_inspection(
                    tensor,
                    action_tensor,
                    belief_tensor,
                    generator=generator,
                )
            else:
                latent = self.model.sample_next(
                    tensor,
                    action_tensor,
                    belief_tensor,
                    samples=FUTURE_SAMPLES,
                    generator=generator,
                )
                inspection = None
        projected = self.project(latent.detach().cpu().numpy())
        scale = max(1.0, float(np.abs(projected).max()))
        projected = projected / scale
        return {
            "samples": projected.astype(float).tolist(),
            "inspection": inspection,
        }

    def latent_samples(self, env: Any, belief_upper: float, action: str) -> list[list[float]] | None:
        prediction = self.latent_prediction(env, belief_upper, action)
        return None if prediction is None else prediction["samples"]

    def meta(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "device": self.device,
            "checkpointHash": self.checkpoint_hash,
            "torchAvailable": TORCH_AVAILABLE,
        }

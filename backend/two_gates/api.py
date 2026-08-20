from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .constants import (
    CHECKPOINT_MANIFEST_NAME,
    CHECKPOINT_NAME,
    DEFAULT_BETA,
    DEFAULT_CUE_RELIABILITY,
    DEFAULT_PRIOR,
    DEFAULT_SEED,
    FUTURE_SAMPLES,
    PLANNING_HORIZON,
    SCHEMA_VERSION,
)
from .model import CheckpointModel
from .planner import ActiveInferenceEpisode, EpisodeConfig
from .replay import export_replays, read_replays

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
REPLAY_DIR = PROJECT_ROOT / "artifacts" / "replays"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
PAPER_DIR = PROJECT_ROOT / "jepa_active_inference_starter" / "jepa_active_inference_starter" / "papers"
PAPERS = {
    "jedi": "01_JEDI_2026.pdf",
    "eb-jepa": "02_EB-JEPA_2026.pdf",
    "fep-deep-learning": "03_FEP_Deep_Learning_Perspective_2022.pdf",
    "pymdp": "04_pymdp_Active_Inference_2022.pdf",
    "value-guided-jepa": "05_Value_Guided_JEPA_Planning_2026.pdf",
    "i-jepa": "06_I-JEPA_2023.pdf",
    "d-jepa": "07_D-JEPA_2024.pdf",
}


class EpisodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agentType: Literal["balanced", "pragmatic", "information"] = "balanced"
    seed: int = DEFAULT_SEED
    beta: float = Field(DEFAULT_BETA, ge=0.0, le=8.0)
    prior: float = Field(DEFAULT_PRIOR, gt=0.0, lt=1.0)
    cueReliability: float = Field(DEFAULT_CUE_RELIABILITY, ge=0.5, le=1.0)
    gateTesting: Literal["allowed", "prohibited"] = "allowed"
    source: Literal["live", "replay"] = "live"


model_service = CheckpointModel(
    CHECKPOINT_DIR / CHECKPOINT_NAME,
    CHECKPOINT_DIR / CHECKPOINT_MANIFEST_NAME,
)
sessions: dict[str, ActiveInferenceEpisode] = {}

app = FastAPI(
    title="Two Gates: JEPA + Active Inference",
    version="0.1.0",
    description="A local, inspectable teaching demo with exact discrete belief updates.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/meta")
def get_meta() -> dict:
    replays = read_replays(PROJECT_ROOT)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "appName": "Two Gates",
        "liveModel": model_service.meta(),
        "replayAvailable": bool(replays),
        "futureSamples": FUTURE_SAMPLES,
        "planningHorizon": PLANNING_HORIZON,
        "worldVariation": {
            "seedControls": ["hidden gate", "upper gate row", "lower gate row", "sensor position", "cue draw"],
            "layoutRange": {
                "upperGateRows": [1, 2, 3],
                "lowerGateRows": [5, 6, 7],
                "sensorColumns": [2, 3, 4, 5],
                "sensorRows": [1, 2, 3, 4, 5, 6, 7],
            },
        },
        "gateTestingModes": {
            "allowed": "Diagnostic gate contact gives exact evidence, but a closed gate still blocks movement and counts as a collision.",
            "prohibited": "The agent must visit the sensor before contacting an unresolved gate, unless prior confidence is at least 98%.",
        },
        "modelArchitecture": {
            "observation": [3, 32, 32],
            "encoderChannels": [16, 32, 48],
            "denseUnits": 64,
            "latentUnits": 8,
            "denoiserUnits": [96, 96, 8],
            "diffusionSteps": 8,
        },
        "scientificLabels": [
            "minimal JEPA-style latent world model",
            "conditional latent diffusion",
            "exact Bayesian information gain in a toy POMDP",
            "operational EFE approximation",
        ],
    }


@app.post("/api/episodes")
def create_episode(request: EpisodeRequest) -> dict:
    if request.source == "live" and not model_service.available:
        raise HTTPException(status_code=503, detail=model_service.reason)
    config = EpisodeConfig(
        agent_type=request.agentType,
        seed=request.seed,
        beta=request.beta,
        prior=request.prior,
        cue_reliability=request.cueReliability,
        gate_testing=request.gateTesting,
        source=request.source,
    )
    episode = ActiveInferenceEpisode(
        config,
        checkpoint_hash=model_service.checkpoint_hash,
        latent_provider=model_service.latent_prediction if model_service.available else None,
    )
    episode_id = str(uuid4())
    sessions[episode_id] = episode
    return {"episodeId": episode_id, "frame": episode.initial_frame()}


@app.post("/api/episodes/{episode_id}/step")
def step_episode(episode_id: str) -> dict:
    episode = sessions.get(episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode.step()


@app.get("/api/replays")
def get_replays() -> list[dict]:
    replays = read_replays(PROJECT_ROOT)
    if not replays:
        replays = export_replays(PROJECT_ROOT)
    return replays


@app.get("/api/replays/{replay_id}")
def get_replay(replay_id: str) -> FileResponse:
    path = REPLAY_DIR / f"{replay_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Replay not found")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="Replay is invalid") from error
    if payload.get("schemaVersion") != SCHEMA_VERSION:
        raise HTTPException(status_code=409, detail="Replay schema version mismatch")
    return FileResponse(path, media_type="application/json")


@app.get("/api/papers/{paper_id}")
def get_paper(paper_id: str) -> FileResponse:
    file_name = PAPERS.get(paper_id)
    if file_name is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    path = PAPER_DIR / file_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Bundled paper is missing")
    return FileResponse(path, media_type="application/pdf")


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

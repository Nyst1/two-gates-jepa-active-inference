from fastapi.testclient import TestClient

from two_gates.api import app


client = TestClient(app)


def test_meta_exposes_scientific_scope() -> None:
    response = client.get("/api/meta")
    assert response.status_code == 200
    payload = response.json()
    assert payload["schemaVersion"] == "1.4"
    assert "operational EFE approximation" in payload["scientificLabels"]
    assert payload["worldVariation"]["seedControls"]
    assert payload["modelArchitecture"]["latentUnits"] == 8
    assert "allowed" in payload["gateTestingModes"]


def test_replay_episode_contract_and_step() -> None:
    response = client.post(
        "/api/episodes",
        json={
            "agentType": "balanced",
            "seed": 0,
            "beta": 1.0,
            "prior": 0.5,
            "cueReliability": 0.9,
            "source": "replay",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    frame = payload["frame"]
    assert len(frame["candidates"]) == 4
    assert sum(int(candidate["selected"]) for candidate in frame["candidates"]) == 1
    assert frame["world"]["layoutId"].startswith("g")
    assert frame["modelInspection"]["encoder"]["layers"][-1]["totalUnits"] == 8
    assert frame["modelInspection"]["denoiser"]["edges"]
    assert frame["world"]["gateTesting"] == "allowed"
    assert frame["world"]["cueReliability"] == 0.9
    assert all("informationSource" in candidate for candidate in frame["candidates"])
    assert all("informationSequence" in candidate for candidate in frame["candidates"])
    assert all("sensorRawInformationGain" in candidate for candidate in frame["candidates"])
    assert all("scoredSensorInformationGain" in candidate for candidate in frame["candidates"])
    assert all("scoredGateInformationGain" in candidate for candidate in frame["candidates"])
    stepped = client.post(f"/api/episodes/{payload['episodeId']}/step")
    assert stepped.status_code == 200
    assert stepped.json()["step"] == 1


def test_episode_can_enable_diagnostic_gate_testing() -> None:
    response = client.post(
        "/api/episodes",
        json={
            "agentType": "balanced",
            "seed": 5,
            "beta": 1.0,
            "prior": 0.5,
            "cueReliability": 0.9,
            "gateTesting": "allowed",
            "source": "replay",
        },
    )
    assert response.status_code == 200
    frame = response.json()["frame"]
    assert frame["world"]["gateTesting"] == "allowed"
    assert any(candidate["gateInformationGain"] > 0 for candidate in frame["candidates"])


def test_missing_episode_returns_404() -> None:
    assert client.post("/api/episodes/not-found/step").status_code == 404


def test_bundled_paper_endpoint() -> None:
    response = client.get("/api/papers/jedi")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"

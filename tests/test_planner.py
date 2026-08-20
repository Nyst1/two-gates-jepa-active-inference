import math

import pytest

from two_gates.belief import expected_posterior_entropy, state_information_gain
from two_gates.evaluation import evaluate_behavior
from two_gates.planner import ActiveInferenceEpisode, EpisodeConfig, run_episode_trace


def actions(trace: dict) -> list[str]:
    return [frame["selectedAction"] for frame in trace["frames"]]


def test_default_story_exposes_epistemic_advantage() -> None:
    pragmatic = run_episode_trace(EpisodeConfig(agent_type="pragmatic", seed=0, source="replay"))
    balanced = run_episode_trace(EpisodeConfig(agent_type="balanced", seed=0, source="replay"))
    assert pragmatic["summary"]["success"] is True
    assert balanced["summary"]["success"] is True
    assert pragmatic["summary"]["wrongGateCommitments"] == 1
    assert balanced["summary"]["wrongGateCommitments"] == 0
    assert balanced["summary"]["steps"] <= pragmatic["summary"]["steps"]
    assert balanced["summary"]["gateTests"] >= 1


def test_beta_zero_reproduces_pragmatic_policy() -> None:
    pragmatic = run_episode_trace(EpisodeConfig(agent_type="pragmatic", seed=0, source="replay"))
    balanced = run_episode_trace(EpisodeConfig(agent_type="balanced", seed=0, beta=0.0, source="replay"))
    assert actions(balanced) == actions(pragmatic)


def test_confident_balanced_agent_does_not_take_sensor_detour() -> None:
    pragmatic = run_episode_trace(EpisodeConfig(agent_type="pragmatic", seed=4, prior=0.99, source="replay"))
    balanced = run_episode_trace(EpisodeConfig(agent_type="balanced", seed=4, prior=0.99, source="replay"))
    assert actions(balanced)[:8] == actions(pragmatic)[:8]


def test_locked_behavioral_evaluation_passes() -> None:
    report = evaluate_behavior(80)
    assert report["passed"] is True


def test_gate_testing_mode_changes_information_and_admissibility() -> None:
    allowed = ActiveInferenceEpisode(EpisodeConfig(gate_testing="allowed"))
    prohibited = ActiveInferenceEpisode(EpisodeConfig(gate_testing="prohibited"))
    approach = (allowed.env.layout.upper_gate[0] - 1, allowed.env.layout.upper_gate[1])
    allowed.env.position = approach
    prohibited.env.position = approach

    allowed_right = next(item for item in allowed.candidates() if item["action"] == "right")
    prohibited_right = next(item for item in prohibited.candidates() if item["action"] == "right")

    assert allowed_right["gateInformationGain"] > allowed_right["sensorInformationGain"]
    assert allowed_right["admissible"] is True
    assert prohibited_right["gateInformationGain"] == 0.0
    assert prohibited_right["admissible"] is False
    assert prohibited_right["preferenceCost"] > allowed_right["preferenceCost"]


def test_gate_scoring_and_visual_plan_share_the_same_target() -> None:
    episode = ActiveInferenceEpisode(
        EpisodeConfig(
            seed=5,
            beta=1.0,
            cue_reliability=0.5,
            gate_testing="allowed",
        )
    )
    frame = episode.initial_frame()
    for _ in range(4):
        frame = episode.step()

    selected = next(item for item in frame["candidates"] if item["selected"])
    assert selected["action"] == "down"
    assert selected["informationSource"] == "gate"
    assert selected["planObjective"] == "gate"
    assert selected["informationTarget"] == list(episode.env.layout.lower_gate)
    assert selected["sensorRawInformationGain"] == 0.0
    assert selected["gateRawInformationGain"] == pytest.approx(math.log(2.0), abs=1e-5)
    assert selected["gateReachFactor"] == pytest.approx(5.0 / 6.0, abs=1e-5)
    assert selected["gateInformationGain"] == pytest.approx(math.log(2.0) * 5.0 / 6.0, abs=1e-5)
    assert all(path["objective"] == "gate" for path in selected["sampledPaths"])
    assert all(list(episode.env.sensor) not in path["points"] for path in selected["sampledPaths"])


def test_earlier_sensor_cue_breaks_equal_cost_tie_without_double_counting() -> None:
    episode = ActiveInferenceEpisode(
        EpisodeConfig(
            seed=3,
            beta=1.0,
            prior=0.5,
            cue_reliability=0.6,
            gate_testing="allowed",
        )
    )
    episode.initial_frame()
    for _ in range(4):
        frame = episode.step()

    candidates = {item["action"]: item for item in frame["candidates"]}
    up = candidates["up"]
    down = candidates["down"]
    cue_gain = state_information_gain(0.5, 0.6)
    entropy_after_cue = expected_posterior_entropy(0.5, 0.6)

    assert up["preferenceCost"] == down["preferenceCost"]
    assert up["informationSequence"] == ["gate"]
    assert down["informationSequence"] == ["sensor", "gate"]
    assert down["scoredSensorInformationGain"] == pytest.approx(cue_gain * 4.0 / 6.0, abs=1e-5)
    assert down["scoredGateInformationGain"] == pytest.approx(entropy_after_cue * 3.0 / 6.0, abs=1e-5)
    assert down["informationGain"] == pytest.approx(
        down["scoredSensorInformationGain"] + down["scoredGateInformationGain"],
        abs=1e-5,
    )
    assert down["informationGain"] <= math.log(2.0)
    assert down["efeScore"] < up["efeScore"]
    assert frame["selectedAction"] == "down"

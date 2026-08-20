from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Literal

from .belief import (
    binary_entropy,
    expected_posterior_entropy,
    posterior_after_cue,
    state_information_gain,
)
from .constants import (
    ACTIONS,
    DEFAULT_BETA,
    DEFAULT_CUE_RELIABILITY,
    DEFAULT_PRIOR,
    DEFAULT_SEED,
    FUTURE_SAMPLES,
    GOAL,
    HEIGHT,
    PLANNING_HORIZON,
    SCHEMA_VERSION,
    START,
    WIDTH,
)
from .environment import Position, Transition, TwoGatesEnv

AgentType = Literal["balanced", "pragmatic", "information"]
GateTestingMode = Literal["allowed", "prohibited"]
InformationSource = Literal["sensor", "gate", "sensor_then_gate", "none"]
PlanObjective = Literal["sensor", "gate", "goal"]


@dataclass(slots=True)
class EpisodeConfig:
    agent_type: AgentType = "balanced"
    seed: int = DEFAULT_SEED
    beta: float = DEFAULT_BETA
    prior: float = DEFAULT_PRIOR
    cue_reliability: float = DEFAULT_CUE_RELIABILITY
    gate_testing: GateTestingMode = "allowed"
    source: Literal["live", "replay"] = "live"

    def normalized(self) -> "EpisodeConfig":
        return EpisodeConfig(
            agent_type=self.agent_type,
            seed=int(self.seed),
            beta=max(0.0, min(8.0, float(self.beta))),
            prior=max(0.01, min(0.99, float(self.prior))),
            cue_reliability=max(0.5, min(1.0, float(self.cue_reliability))),
            gate_testing=self.gate_testing,
            source=self.source,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "agentType": self.agent_type,
            "seed": self.seed,
            "beta": self.beta,
            "prior": self.prior,
            "cueReliability": self.cue_reliability,
            "gateTesting": self.gate_testing,
            "source": self.source,
        }


def _next_position(env: TwoGatesEnv, action: str, hidden_upper: bool) -> tuple[Position, bool]:
    dx, dy = ACTIONS[action]
    proposed = (env.position[0] + dx, env.position[1] + dy)
    blocked = env.is_blocked(proposed, hidden_upper=hidden_upper)
    return (env.position if blocked else proposed), blocked


def _path_via_sensor(env: TwoGatesEnv, action: str, hidden_upper: bool) -> list[Position]:
    first_leg = env.shortest_path(
        env.position,
        env.sensor,
        hidden_upper=hidden_upper,
        first_action=action,
    )
    sensor_reached = first_leg[-1] == env.sensor
    if not sensor_reached:
        return env.shortest_path(
            env.position,
            GOAL,
            hidden_upper=hidden_upper,
            first_action=action,
        )
    second_leg = env.shortest_path(env.sensor, GOAL, hidden_upper=hidden_upper)
    return first_leg + second_leg[1:]


def _path_via_gate_test(
    env: TwoGatesEnv,
    action: str,
    hidden_upper: bool,
    target_gate: Position,
) -> list[Position]:
    """Return physical positions for a policy that deliberately tests target_gate.

    A blocked contact repeats the approach position for one step. The UI marks the
    intended gate separately, so the trace does not pretend that the agent entered
    a closed cell.
    """

    dx, dy = ACTIONS[action]
    proposed = (env.position[0] + dx, env.position[1] + dy)
    next_pos, _ = _next_position(env, action, hidden_upper)
    prefix = [env.position, next_pos]
    if proposed == target_gate:
        continuation = env.shortest_path(next_pos, GOAL, hidden_upper=hidden_upper)
        return prefix + continuation[1:]

    approach = (target_gate[0] - 1, target_gate[1])
    approach_path = env.shortest_path(next_pos, approach, hidden_upper=hidden_upper)
    if approach_path[-1] != approach:
        return env.shortest_path(
            env.position,
            GOAL,
            hidden_upper=hidden_upper,
            first_action=action,
        )
    prefix = prefix[:-1] + approach_path
    testing_upper = target_gate == env.layout.upper_gate
    gate_open = hidden_upper if testing_upper else not hidden_upper
    if gate_open:
        continuation = env.shortest_path(target_gate, GOAL, hidden_upper=hidden_upper)
        return prefix + continuation

    # A closed gate consumes an action without changing physical position.
    continuation = env.shortest_path(approach, GOAL, hidden_upper=hidden_upper)
    return prefix + [approach] + continuation[1:]


def _path_to_goal(env: TwoGatesEnv, action: str, hidden_upper: bool) -> list[Position]:
    return env.shortest_path(
        env.position,
        GOAL,
        hidden_upper=hidden_upper,
        first_action=action,
    )


def _gate_status(env: TwoGatesEnv, belief_upper: float, upper: bool) -> str:
    if env.revealed_upper is not None:
        is_open = env.revealed_upper if upper else not env.revealed_upper
        return "open" if is_open else "closed"
    confidence = belief_upper if upper else 1.0 - belief_upper
    if confidence >= 0.995:
        return "open"
    if confidence <= 0.005:
        return "closed"
    return "unknown"


def _phase_for_transition(transition: Transition | None) -> str:
    if transition is None:
        return "observe"
    if transition.cue or transition.revealed_gate:
        return "update"
    if transition.reached_goal:
        return "act"
    return "evaluate"


class ActiveInferenceEpisode:
    def __init__(
        self,
        config: EpisodeConfig,
        *,
        checkpoint_hash: str = "analytic-replay-v1",
        latent_provider: Any | None = None,
    ) -> None:
        self.config = config.normalized()
        self.env = TwoGatesEnv(
            seed=self.config.seed,
            prior_upper=self.config.prior,
            cue_reliability=self.config.cue_reliability,
        )
        self.belief_upper = self.config.prior
        self.checkpoint_hash = checkpoint_hash
        self.latent_provider = latent_provider
        self.frames: list[dict[str, Any]] = []
        self.wrong_gate_commitments = 0
        self.gate_tests = 0
        self._last_transition: Transition | None = None
        self._last_model_inspection: dict[str, Any] | None = None

    @property
    def done(self) -> bool:
        return self.env.done or self.env.step_count >= 48

    def _gate_testing_restricted(self) -> bool:
        return (
            self.config.gate_testing == "prohibited"
            and not self.env.sensor_visited
            and self.env.revealed_upper is None
            and max(self.belief_upper, 1.0 - self.belief_upper) < 0.98
        )

    def _is_prohibited_gate_contact(self, action: str) -> bool:
        if not self._gate_testing_restricted():
            return False
        dx, dy = ACTIONS[action]
        proposed = (self.env.position[0] + dx, self.env.position[1] + dy)
        return proposed in {self.env.layout.upper_gate, self.env.layout.lower_gate}

    def _expected_preference_cost(self, action: str) -> tuple[float, float]:
        weighted_distance = 0.0
        collision_probability = 0.0
        for hidden_upper, probability in ((True, self.belief_upper), (False, 1.0 - self.belief_upper)):
            next_pos, collision = _next_position(self.env, action, hidden_upper)
            if self._gate_testing_restricted():
                path = _path_via_sensor(self.env, action, hidden_upper)
            else:
                path = self.env.shortest_path(next_pos, GOAL, hidden_upper=hidden_upper)
            distance = max(0, len(path) - 1)
            weighted_distance += probability * distance
            collision_probability += probability * float(collision)
        dx, dy = ACTIONS[action]
        proposed = (self.env.position[0] + dx, self.env.position[1] + dy)
        known_wall = proposed in (self.env.boundary_walls() | self.env.divider_walls())
        normalized_distance = weighted_distance / float(WIDTH + HEIGHT)
        return (
            normalized_distance
            + (0.8 if known_wall else 0.0)
            + (4.0 if self._is_prohibited_gate_contact(action) else 0.0),
            collision_probability,
        )

    def _sensor_evidence(self, action: str) -> dict[str, Any]:
        raw_gain = (
            0.0
            if self.env.sensor_visited or binary_entropy(self.belief_upper) <= 1e-9
            else state_information_gain(self.belief_upper, self.config.cue_reliability)
        )
        next_upper, _ = _next_position(self.env, action, True)
        next_lower, _ = _next_position(self.env, action, False)
        upper_path = self.env.shortest_path(next_upper, self.env.sensor, hidden_upper=True)
        lower_path = self.env.shortest_path(next_lower, self.env.sensor, hidden_upper=False)
        expected_steps_after_action = (
            self.belief_upper * max(0, len(upper_path) - 1)
            + (1.0 - self.belief_upper) * max(0, len(lower_path) - 1)
        )
        reach_factor = max(
            0.0,
            1.0 - expected_steps_after_action / float(PLANNING_HORIZON + 1),
        )
        return {
            "raw": raw_gain,
            "reach": reach_factor,
            "value": raw_gain * reach_factor,
            "target": self.env.sensor,
            "stepsAfterAction": expected_steps_after_action,
        }

    def _gate_test_target(self, action: str) -> tuple[Position | None, int]:
        dx, dy = ACTIONS[action]
        proposed = (self.env.position[0] + dx, self.env.position[1] + dy)
        gates = (self.env.layout.upper_gate, self.env.layout.lower_gate)
        if proposed in gates:
            return proposed, 0

        next_position, _ = _next_position(self.env, action, True)
        options: list[tuple[int, Position]] = []
        for gate in gates:
            approach = (gate[0] - 1, gate[1])
            path = self.env.shortest_path(next_position, approach, hidden_upper=True)
            if path[-1] == approach:
                options.append((max(0, len(path) - 1) + 1, gate))
        if not options:
            return None, PLANNING_HORIZON + 1
        steps, target = min(options, key=lambda item: (item[0], item[1][1]))
        return target, steps

    def _gate_evidence(self, action: str) -> dict[str, Any]:
        if (
            self._gate_testing_restricted()
            or self.env.revealed_upper is not None
            or self.env.position[0] >= self.env.layout.upper_gate[0]
            or binary_entropy(self.belief_upper) <= 1e-9
        ):
            return {
                "raw": 0.0,
                "reach": 0.0,
                "value": 0.0,
                "target": None,
                "stepsAfterAction": None,
            }
        target, steps_after_action = self._gate_test_target(action)
        raw_gain = binary_entropy(self.belief_upper)
        reach_factor = max(
            0.0,
            1.0 - steps_after_action / float(PLANNING_HORIZON + 1),
        )
        return {
            "raw": raw_gain,
            "reach": reach_factor,
            "value": raw_gain * reach_factor,
            "target": target,
            "stepsAfterAction": steps_after_action,
        }

    @staticmethod
    def _reach_factor(steps_after_action: float) -> float:
        return max(
            0.0,
            1.0 - steps_after_action / float(PLANNING_HORIZON + 1),
        )

    def _sensor_steps_before_gate(
        self,
        action: str,
        target_gate: Position,
    ) -> int | None:
        """Return cue timing when the planned gate-test route crosses the sensor."""

        if self.env.sensor_visited:
            return None
        dx, dy = ACTIONS[action]
        proposed = (self.env.position[0] + dx, self.env.position[1] + dy)
        if proposed == target_gate:
            return None

        next_position, _ = _next_position(self.env, action, True)
        approach = (target_gate[0] - 1, target_gate[1])
        approach_path = self.env.shortest_path(next_position, approach, hidden_upper=True)
        if approach_path[-1] != approach or self.env.sensor not in approach_path:
            return None
        return approach_path.index(self.env.sensor)

    def _policy_evidence(
        self,
        action: str,
        sensor: dict[str, Any],
        gate: dict[str, Any],
    ) -> dict[str, Any]:
        """Score incremental evidence in the order a candidate policy reveals it.

        If a route reaches the noisy sensor before an exact gate test, the cue
        removes part of the current entropy and gate contact can only remove the
        expected entropy that remains. The discounted increments are added, so
        earlier evidence is useful without counting the same uncertainty twice.
        """

        sensor_only = {
            "source": "sensor",
            "sequence": ["sensor"],
            "target": sensor["target"],
            "sensorRaw": sensor["raw"],
            "sensorReach": sensor["reach"],
            "sensorValue": sensor["value"],
            "gateRaw": 0.0,
            "gateReach": 0.0,
            "gateValue": 0.0,
            "value": sensor["value"],
        }

        gate_plan = {
            "source": "gate",
            "sequence": ["gate"],
            "target": gate["target"],
            "sensorRaw": 0.0,
            "sensorReach": 0.0,
            "sensorValue": 0.0,
            "gateRaw": gate["raw"],
            "gateReach": gate["reach"],
            "gateValue": gate["value"],
            "value": gate["value"],
        }

        if gate["target"] is not None and gate["value"] > 1e-9 and sensor["raw"] > 1e-9:
            sensor_steps = self._sensor_steps_before_gate(action, gate["target"])
            if sensor_steps is not None:
                sensor_reach = self._reach_factor(sensor_steps)
                sensor_value = sensor["raw"] * sensor_reach
                remaining_entropy = expected_posterior_entropy(
                    self.belief_upper,
                    self.config.cue_reliability,
                )
                gate_value = remaining_entropy * gate["reach"]
                gate_plan = {
                    "source": "sensor_then_gate",
                    "sequence": ["sensor", "gate"],
                    "target": gate["target"],
                    "sensorRaw": sensor["raw"],
                    "sensorReach": sensor_reach,
                    "sensorValue": sensor_value,
                    "gateRaw": remaining_entropy,
                    "gateReach": gate["reach"],
                    "gateValue": gate_value,
                    "value": sensor_value + gate_value,
                }

        best = gate_plan if gate_plan["value"] >= sensor_only["value"] else sensor_only
        if best["value"] > 1e-9:
            return best
        return {
            "source": "none",
            "sequence": [],
            "target": None,
            "sensorRaw": 0.0,
            "sensorReach": 0.0,
            "sensorValue": 0.0,
            "gateRaw": 0.0,
            "gateReach": 0.0,
            "gateValue": 0.0,
            "value": 0.0,
        }

    def _plan_objective(self, source: InformationSource) -> PlanObjective:
        if self._gate_testing_restricted():
            return "sensor"
        epistemic_term_active = self.config.agent_type == "information" or (
            self.config.agent_type == "balanced" and self.config.beta > 0.0
        )
        if epistemic_term_active and source == "sensor":
            return "sensor"
        if epistemic_term_active and source in {"gate", "sensor_then_gate"}:
            return "gate"
        return "goal"

    def _sampled_paths(
        self,
        action: str,
        objective: PlanObjective,
        target: Position | None,
    ) -> list[dict[str, Any]]:
        seed = self.config.seed * 10_000 + self.env.step_count * 101 + list(ACTIONS).index(action)
        rng = random.Random(seed)
        samples: list[dict[str, Any]] = []
        for _ in range(FUTURE_SAMPLES):
            hidden_upper = rng.random() < self.belief_upper
            if objective == "sensor":
                path = _path_via_sensor(self.env, action, hidden_upper)
            elif objective == "gate" and target is not None:
                path = _path_via_gate_test(self.env, action, hidden_upper, target)
            else:
                path = _path_to_goal(self.env, action, hidden_upper)
            visible_path = path[: PLANNING_HORIZON + 1]
            samples.append(
                {
                    "hypothesis": "upper" if hidden_upper else "lower",
                    "points": [[x, y] for x, y in visible_path],
                    "objective": objective,
                }
            )
        return samples

    def _analytic_latent_samples(self, action: str) -> list[list[float]]:
        seed = self.config.seed * 1_000 + self.env.step_count * 17 + list(ACTIONS).index(action)
        rng = random.Random(seed)
        samples: list[list[float]] = []
        for _ in range(FUTURE_SAMPLES):
            hidden_upper = rng.random() < self.belief_upper
            next_pos, collision = _next_position(self.env, action, hidden_upper)
            branch = 0.28 if hidden_upper else -0.28
            uncertainty = binary_entropy(self.belief_upper) / math.log(2.0)
            samples.append(
                [
                    (next_pos[0] / (WIDTH - 1) - 0.5) * 2.0 + branch * uncertainty + rng.gauss(0, 0.025),
                    (next_pos[1] / (HEIGHT - 1) - 0.5) * 2.0
                    + (0.08 if collision else 0.0)
                    + rng.gauss(0, 0.025),
                ]
            )
        return samples

    def candidates(self) -> list[dict[str, Any]]:
        raw: list[dict[str, Any]] = []
        for action in ACTIONS:
            preference_cost, collision_probability = self._expected_preference_cost(action)
            sensor_evidence = self._sensor_evidence(action)
            gate_evidence = self._gate_evidence(action)
            policy_evidence = self._policy_evidence(action, sensor_evidence, gate_evidence)
            information_source = policy_evidence["source"]
            information_gain = policy_evidence["value"]
            plan_objective = self._plan_objective(information_source)
            information_target = (
                policy_evidence["target"]
                if plan_objective == "gate"
                else self.env.sensor if plan_objective == "sensor" else None
            )
            if self.config.agent_type == "pragmatic":
                efe_score = preference_cost
            elif self.config.agent_type == "information":
                efe_score = -information_gain + 0.03 * preference_cost
            else:
                efe_score = preference_cost - self.config.beta * information_gain
            raw.append(
                {
                    "action": action,
                    "preferenceCost": round(preference_cost, 5),
                    "informationGain": round(information_gain, 5),
                    "sensorInformationGain": round(sensor_evidence["value"], 5),
                    "sensorRawInformationGain": round(sensor_evidence["raw"], 5),
                    "sensorReachFactor": round(sensor_evidence["reach"], 5),
                    "gateInformationGain": round(gate_evidence["value"], 5),
                    "gateRawInformationGain": round(gate_evidence["raw"], 5),
                    "gateReachFactor": round(gate_evidence["reach"], 5),
                    "scoredSensorInformationGain": round(policy_evidence["sensorValue"], 5),
                    "scoredSensorRawInformationGain": round(policy_evidence["sensorRaw"], 5),
                    "scoredSensorReachFactor": round(policy_evidence["sensorReach"], 5),
                    "scoredGateInformationGain": round(policy_evidence["gateValue"], 5),
                    "scoredGateRawInformationGain": round(policy_evidence["gateRaw"], 5),
                    "scoredGateReachFactor": round(policy_evidence["gateReach"], 5),
                    "informationSource": information_source,
                    "informationSequence": policy_evidence["sequence"],
                    "planObjective": plan_objective,
                    "informationTarget": (
                        list(information_target) if information_target is not None else None
                    ),
                    "efeScore": round(efe_score, 5),
                    "collisionProbability": round(collision_probability, 5),
                    "sampledPaths": self._sampled_paths(
                        action,
                        plan_objective,
                        information_target,
                    ),
                    "latentSamples": [
                        [round(x, 5), round(y, 5)]
                        for x, y in self._analytic_latent_samples(action)
                    ],
                    "selected": False,
                    "admissible": not self._is_prohibited_gate_contact(action),
                    "constraintReason": (
                        "sensor evidence required before gate contact"
                        if self._is_prohibited_gate_contact(action)
                        else None
                    ),
                }
            )
        selected = min(raw, key=lambda item: (item["efeScore"], list(ACTIONS).index(item["action"])))
        selected["selected"] = True
        self._last_model_inspection = None
        if self.latent_provider is not None:
            for candidate in raw:
                supplied = self.latent_provider(
                    self.env,
                    self.belief_upper,
                    candidate["action"],
                    capture=candidate is selected,
                )
                if not supplied:
                    continue
                samples = supplied.get("samples") if isinstance(supplied, dict) else supplied
                if samples:
                    candidate["latentSamples"] = [
                        [round(float(x), 5), round(float(y), 5)] for x, y in samples
                    ]
                if candidate is selected and isinstance(supplied, dict):
                    inspection = supplied.get("inspection")
                    if inspection:
                        inspection["mode"] = (
                            "live checkpoint inference"
                            if self.config.source == "live"
                            else "recorded checkpoint inference"
                        )
                        inspection["selectedAction"] = candidate["action"]
                        self._last_model_inspection = inspection
        return raw

    def _world(self, position_before: Position | None = None) -> dict[str, Any]:
        return {
            "width": WIDTH,
            "height": HEIGHT,
            "agent": list(self.env.position),
            "previousAgent": list(position_before or self.env.position),
            "start": list(START),
            "goal": list(GOAL),
            "sensor": list(self.env.sensor),
            "layoutId": self.env.layout.layout_id,
            "upperGate": {
                "position": list(self.env.layout.upper_gate),
                "status": _gate_status(self.env, self.belief_upper, True),
            },
            "lowerGate": {
                "position": list(self.env.layout.lower_gate),
                "status": _gate_status(self.env, self.belief_upper, False),
            },
            "walls": self.env.public_walls(),
            "sensorVisited": self.env.sensor_visited,
            "cue": self.env.last_cue,
            "cueReliability": self.config.cue_reliability,
            "gateTesting": self.config.gate_testing,
            "gateTestingRestricted": self._gate_testing_restricted(),
            "reachedGoal": self.env.done,
        }

    def _metrics(self) -> dict[str, Any]:
        return {
            "steps": self.env.step_count,
            "collisions": self.env.collisions,
            "wrongGateCommitments": self.wrong_gate_commitments,
            "gateTests": self.gate_tests,
            "pathLength": max(0, len(self.env.path) - 1),
            "success": self.env.done,
        }

    def initial_frame(self) -> dict[str, Any]:
        candidates = self.candidates()
        frame = {
            "schemaVersion": SCHEMA_VERSION,
            "checkpointHash": self.checkpoint_hash,
            "source": self.config.source,
            "step": 0,
            "phase": "observe",
            "world": self._world(),
            "belief": {
                "upper": self.belief_upper,
                "lower": 1.0 - self.belief_upper,
                "entropyBefore": binary_entropy(self.belief_upper),
                "entropyAfter": binary_entropy(self.belief_upper),
                "updateReason": "prior",
            },
            "candidates": candidates,
            "selectedAction": next(item["action"] for item in candidates if item["selected"]),
            "modelInspection": self._last_model_inspection,
            "transition": None,
            "metrics": self._metrics(),
            "done": self.done,
        }
        self.frames.append(frame)
        return frame

    def step(self) -> dict[str, Any]:
        if self.done:
            return self.frames[-1]
        candidates = self.candidates()
        selected = next(item for item in candidates if item["selected"])
        belief_before = self.belief_upper
        deliberate_gate_test = (
            self.config.gate_testing == "allowed"
            and selected["planObjective"] == "gate"
            and (
                self.config.agent_type == "information"
                or (self.config.agent_type == "balanced" and self.config.beta > 0.0)
            )
        )
        transition = self.env.step(selected["action"])
        update_reason = "prediction only"
        if transition.cue:
            self.belief_upper = posterior_after_cue(
                belief_before,
                transition.cue,
                self.config.cue_reliability,
            )
            update_reason = f"{transition.cue} sensor cue"
        if transition.revealed_gate:
            if deliberate_gate_test:
                self.gate_tests += 1
            was_wrong = transition.collision
            self.wrong_gate_commitments += int(was_wrong and not deliberate_gate_test)
            self.belief_upper = 1.0 if transition.revealed_gate == "upper" else 0.0
            update_reason = (
                f"{transition.revealed_gate} gate learned by direct test"
                if deliberate_gate_test
                else f"{transition.revealed_gate} gate revealed"
            )
        belief_after = self.belief_upper
        self._last_transition = transition
        displayed_candidates = candidates if self.done else self.candidates()
        displayed_action = (
            selected["action"]
            if self.done
            else next(item["action"] for item in displayed_candidates if item["selected"])
        )
        frame = {
            "schemaVersion": SCHEMA_VERSION,
            "checkpointHash": self.checkpoint_hash,
            "source": self.config.source,
            "step": self.env.step_count,
            "phase": _phase_for_transition(transition),
            "world": self._world(transition.position_before),
            "belief": {
                "upper": belief_after,
                "lower": 1.0 - belief_after,
                "entropyBefore": binary_entropy(belief_before),
                "entropyAfter": binary_entropy(belief_after),
                "updateReason": update_reason,
            },
            "candidates": displayed_candidates,
            "selectedAction": displayed_action,
            "modelInspection": self._last_model_inspection,
            "transition": {
                "action": transition.action,
                "moved": transition.moved,
                "collision": transition.collision,
                "cue": transition.cue,
                "revealedGate": transition.revealed_gate,
                "reachedGoal": transition.reached_goal,
            },
            "metrics": self._metrics(),
            "done": self.done,
        }
        self.frames.append(frame)
        return frame

    def trace(self, *, max_steps: int = 48) -> dict[str, Any]:
        if not self.frames:
            self.initial_frame()
        while not self.done and self.env.step_count < max_steps:
            self.step()
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "checkpointHash": self.checkpoint_hash,
            "config": self.config.to_json(),
            "frames": self.frames,
            "summary": self._metrics(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["traceHash"] = hashlib.sha256(canonical).hexdigest()
        return payload


def run_episode_trace(
    config: EpisodeConfig,
    *,
    checkpoint_hash: str = "analytic-replay-v1",
    latent_provider: Any | None = None,
) -> dict[str, Any]:
    return ActiveInferenceEpisode(
        config,
        checkpoint_hash=checkpoint_hash,
        latent_provider=latent_provider,
    ).trace()

from __future__ import annotations

import math


def clamp_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def binary_entropy(probability: float) -> float:
    """Binary entropy in nats, with exact behavior at the boundaries."""

    p = clamp_probability(probability)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def cue_probability_upper(prior_upper: float, reliability: float) -> float:
    p = clamp_probability(prior_upper)
    r = clamp_probability(reliability)
    return p * r + (1.0 - p) * (1.0 - r)


def posterior_after_cue(prior_upper: float, cue: str, reliability: float) -> float:
    """P(upper gate open | cue) for a symmetric binary cue model."""

    if cue not in {"upper", "lower"}:
        raise ValueError(f"Unsupported cue: {cue}")
    p = clamp_probability(prior_upper)
    r = clamp_probability(reliability)
    likelihood_if_upper = r if cue == "upper" else 1.0 - r
    likelihood_if_lower = 1.0 - r if cue == "upper" else r
    evidence = p * likelihood_if_upper + (1.0 - p) * likelihood_if_lower
    if evidence <= 0.0:
        return p
    return clamp_probability((p * likelihood_if_upper) / evidence)


def expected_posterior_entropy(prior_upper: float, reliability: float) -> float:
    p_cue_upper = cue_probability_upper(prior_upper, reliability)
    post_upper = posterior_after_cue(prior_upper, "upper", reliability)
    post_lower = posterior_after_cue(prior_upper, "lower", reliability)
    return p_cue_upper * binary_entropy(post_upper) + (1.0 - p_cue_upper) * binary_entropy(post_lower)


def state_information_gain(prior_upper: float, reliability: float) -> float:
    """Exact expected reduction in hidden-state entropy from visiting the sensor."""

    return max(0.0, binary_entropy(prior_upper) - expected_posterior_entropy(prior_upper, reliability))


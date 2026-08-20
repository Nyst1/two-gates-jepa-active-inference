import math

import pytest

from two_gates.belief import (
    binary_entropy,
    posterior_after_cue,
    state_information_gain,
)


def test_binary_entropy_has_expected_boundaries_and_maximum() -> None:
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert binary_entropy(0.5) == pytest.approx(math.log(2.0))


def test_symmetric_reliable_cue_updates_belief_exactly() -> None:
    assert posterior_after_cue(0.5, "upper", 0.9) == pytest.approx(0.9)
    assert posterior_after_cue(0.5, "lower", 0.9) == pytest.approx(0.1)


def test_less_reliable_cue_has_lower_expected_information_gain() -> None:
    assert state_information_gain(0.5, 0.9) > state_information_gain(0.5, 0.7)
    assert state_information_gain(0.5, 0.5) == pytest.approx(0.0)


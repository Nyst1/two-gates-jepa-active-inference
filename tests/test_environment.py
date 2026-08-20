import numpy as np

from two_gates.constants import LOWER_GATE, UPPER_GATE
from two_gates.environment import TwoGatesEnv, layout_from_seed


def test_observation_does_not_leak_hidden_gate_before_evidence() -> None:
    upper = TwoGatesEnv(seed=1, hidden_upper=True)
    lower = TwoGatesEnv(seed=1, hidden_upper=False)
    assert np.array_equal(upper.render_observation(0.5), lower.render_observation(0.5))


def test_closed_gate_collision_reveals_open_gate() -> None:
    env = TwoGatesEnv(seed=0, hidden_upper=False)
    env.position = (UPPER_GATE[0] - 1, UPPER_GATE[1])
    transition = env.step("right")
    assert transition.collision is True
    assert transition.revealed_gate == "lower"
    assert env.position != UPPER_GATE


def test_open_gate_can_be_crossed() -> None:
    env = TwoGatesEnv(seed=0, hidden_upper=False)
    env.position = (LOWER_GATE[0] - 1, LOWER_GATE[1])
    transition = env.step("right")
    assert transition.collision is False
    assert transition.revealed_gate == "lower"
    assert env.position == LOWER_GATE


def test_world_seed_varies_layout_but_is_deterministic() -> None:
    layouts = [layout_from_seed(seed) for seed in range(1, 20)]
    assert len({layout.layout_id for layout in layouts}) >= 10
    assert layout_from_seed(7) == layout_from_seed(7)
    for layout in layouts:
        assert layout.upper_gate[0] == layout.lower_gate[0] == 6
        assert layout.upper_gate[1] in {1, 2, 3}
        assert layout.lower_gate[1] in {5, 6, 7}
        assert 2 <= layout.sensor[0] <= 5
        assert 1 <= layout.sensor[1] <= 7


def test_varied_layouts_remain_navigable_under_both_hidden_states() -> None:
    for seed in range(20):
        env = TwoGatesEnv(seed=seed)
        for hidden_upper in (True, False):
            path = env.shortest_path(env.position, hidden_upper=hidden_upper)
            assert path[-1] == (11, 4)

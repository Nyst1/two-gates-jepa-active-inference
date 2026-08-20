from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final[str] = "1.4"
DATASET_VERSION: Final[str] = "two-gates-varied-layout-v2"
CHECKPOINT_NAME: Final[str] = "two-gates-median.pt"
CHECKPOINT_MANIFEST_NAME: Final[str] = "two-gates-median.manifest.json"

WIDTH: Final[int] = 13
HEIGHT: Final[int] = 9
START: Final[tuple[int, int]] = (1, 4)
GOAL: Final[tuple[int, int]] = (11, 4)
SENSOR: Final[tuple[int, int]] = (3, 3)
UPPER_GATE: Final[tuple[int, int]] = (6, 2)
LOWER_GATE: Final[tuple[int, int]] = (6, 6)
WALL_X: Final[int] = 6

ACTIONS: Final[dict[str, tuple[int, int]]] = {
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
}
ACTION_INDEX: Final[dict[str, int]] = {name: index for index, name in enumerate(ACTIONS)}

DEFAULT_SEED: Final[int] = 0
DEFAULT_BETA: Final[float] = 1.0
DEFAULT_PRIOR: Final[float] = 0.5
DEFAULT_CUE_RELIABILITY: Final[float] = 0.9
PLANNING_HORIZON: Final[int] = 5
FUTURE_SAMPLES: Final[int] = 16

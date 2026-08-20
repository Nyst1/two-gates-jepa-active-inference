from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import ClassVar, Iterable

import numpy as np

from .constants import (
    ACTIONS,
    GOAL,
    HEIGHT,
    LOWER_GATE,
    SENSOR,
    START,
    UPPER_GATE,
    WALL_X,
    WIDTH,
)

Position = tuple[int, int]


@dataclass(frozen=True, slots=True)
class WorldLayout:
    upper_gate: Position
    lower_gate: Position
    sensor: Position

    @property
    def layout_id(self) -> str:
        return (
            f"g{self.upper_gate[1]}-{self.lower_gate[1]}"
            f"-s{self.sensor[0]}-{self.sensor[1]}"
        )


def layout_from_seed(seed: int) -> WorldLayout:
    """Generate stable geometry independently from hidden-state and cue draws."""

    if int(seed) == 0:
        return WorldLayout(upper_gate=UPPER_GATE, lower_gate=LOWER_GATE, sensor=SENSOR)
    rng = random.Random(int(seed) ^ 0x5EED)
    upper_gate = (WALL_X, rng.choice((1, 2, 3)))
    lower_gate = (WALL_X, rng.choice((5, 6, 7)))
    sensor_candidates = [
        (x, y)
        for x in range(2, WALL_X)
        for y in range(1, HEIGHT - 1)
        if (x, y) != START
    ]
    sensor = rng.choice(sensor_candidates)
    return WorldLayout(upper_gate=upper_gate, lower_gate=lower_gate, sensor=sensor)


@dataclass(slots=True)
class Transition:
    action: str
    position_before: Position
    position_after: Position
    moved: bool
    collision: bool
    cue: str | None
    revealed_gate: str | None
    reached_goal: bool


class TwoGatesEnv:
    """A tiny POMDP where only one of two visually ambiguous gates is open."""

    _observation_bases: ClassVar[dict[tuple[Position, Position], np.ndarray]] = {}

    def __init__(
        self,
        *,
        seed: int = 0,
        prior_upper: float = 0.5,
        cue_reliability: float = 0.9,
        hidden_upper: bool | None = None,
        layout: WorldLayout | None = None,
    ) -> None:
        self.seed = int(seed)
        self.prior_upper = float(prior_upper)
        self.cue_reliability = float(cue_reliability)
        self.layout = layout or layout_from_seed(self.seed)
        hidden_rng = random.Random(self.seed ^ 0xA11CE)
        self._cue_rng = random.Random(self.seed ^ 0xC0DE)
        self.hidden_upper = (
            hidden_rng.random() < self.prior_upper if hidden_upper is None else bool(hidden_upper)
        )
        self.position: Position = START
        self.step_count = 0
        self.sensor_visited = False
        self.last_cue: str | None = None
        self.revealed_upper: bool | None = None
        self.collisions = 0
        self.path: list[Position] = [START]

    @property
    def done(self) -> bool:
        return self.position == GOAL

    @property
    def open_gate(self) -> Position:
        return self.layout.upper_gate if self.hidden_upper else self.layout.lower_gate

    @property
    def closed_gate(self) -> Position:
        return self.layout.lower_gate if self.hidden_upper else self.layout.upper_gate

    @property
    def sensor(self) -> Position:
        return self.layout.sensor

    @staticmethod
    def boundary_walls() -> set[Position]:
        walls: set[Position] = set()
        for x in range(WIDTH):
            walls.add((x, 0))
            walls.add((x, HEIGHT - 1))
        for y in range(HEIGHT):
            walls.add((0, y))
            walls.add((WIDTH - 1, y))
        return walls

    def divider_walls(self) -> set[Position]:
        return {
            (WALL_X, y)
            for y in range(1, HEIGHT - 1)
            if (WALL_X, y) not in {self.layout.upper_gate, self.layout.lower_gate}
        }

    def walls(self, *, hidden_upper: bool | None = None) -> set[Position]:
        upper_open = self.hidden_upper if hidden_upper is None else bool(hidden_upper)
        closed_gate = self.layout.lower_gate if upper_open else self.layout.upper_gate
        return self.boundary_walls() | self.divider_walls() | {closed_gate}

    def is_blocked(self, position: Position, *, hidden_upper: bool | None = None) -> bool:
        return position in self.walls(hidden_upper=hidden_upper)

    def step(self, action: str) -> Transition:
        if action not in ACTIONS:
            raise ValueError(f"Unsupported action: {action}")
        before = self.position
        dx, dy = ACTIONS[action]
        proposed = (before[0] + dx, before[1] + dy)
        collision = self.is_blocked(proposed)
        if collision:
            after = before
            self.collisions += 1
        else:
            after = proposed
            self.position = after

        cue: str | None = None
        revealed_gate: str | None = None
        if proposed in {self.layout.upper_gate, self.layout.lower_gate}:
            observed_upper = proposed == self.layout.upper_gate
            if collision:
                self.revealed_upper = not observed_upper
            else:
                self.revealed_upper = observed_upper
            revealed_gate = "upper" if self.revealed_upper else "lower"

        if after == self.sensor and not self.sensor_visited:
            self.sensor_visited = True
            truthful_cue = "upper" if self.hidden_upper else "lower"
            if self._cue_rng.random() <= self.cue_reliability:
                cue = truthful_cue
            else:
                cue = "lower" if truthful_cue == "upper" else "upper"
            self.last_cue = cue

        self.step_count += 1
        self.path.append(after)
        return Transition(
            action=action,
            position_before=before,
            position_after=after,
            moved=after != before,
            collision=collision,
            cue=cue,
            revealed_gate=revealed_gate,
            reached_goal=after == GOAL,
        )

    def available_positions(self, *, hidden_upper: bool | None = None) -> list[Position]:
        walls = self.walls(hidden_upper=hidden_upper)
        return [
            (x, y)
            for y in range(1, HEIGHT - 1)
            for x in range(1, WIDTH - 1)
            if (x, y) not in walls
        ]

    def shortest_path(
        self,
        start: Position,
        goal: Position = GOAL,
        *,
        hidden_upper: bool | None = None,
        first_action: str | None = None,
    ) -> list[Position]:
        walls = self.walls(hidden_upper=hidden_upper)
        initial = start
        prefix = [start]
        if first_action is not None:
            dx, dy = ACTIONS[first_action]
            proposed = (start[0] + dx, start[1] + dy)
            initial = start if proposed in walls else proposed
            prefix.append(initial)
        if initial == goal:
            return prefix

        queue: deque[Position] = deque([initial])
        parent: dict[Position, Position | None] = {initial: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for dx, dy in ACTIONS.values():
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in walls or neighbor in parent:
                    continue
                parent[neighbor] = current
                queue.append(neighbor)
        if goal not in parent:
            return prefix
        suffix: list[Position] = []
        cursor: Position | None = goal
        while cursor is not None:
            suffix.append(cursor)
            cursor = parent[cursor]
        suffix.reverse()
        return prefix[:-1] + suffix

    def public_walls(self) -> list[list[int]]:
        walls = self.boundary_walls() | self.divider_walls()
        return [[x, y] for x, y in sorted(walls)]

    def render_observation(self, belief_upper: float = 0.5) -> np.ndarray:
        """Render the agent's observation without leaking the hidden gate state."""

        scale = 32
        image = np.empty((scale, scale, 3), dtype=np.float32)

        def paint(cell: Position, color: Iterable[float], inset: int = 0) -> None:
            x, y = cell
            x0 = round(x * scale / WIDTH) + inset
            x1 = round((x + 1) * scale / WIDTH) - inset
            y0 = round(y * scale / HEIGHT) + inset
            y1 = round((y + 1) * scale / HEIGHT) - inset
            image[max(0, y0) : min(scale, y1), max(0, x0) : min(scale, x1)] = color

        cache_key = (self.layout.upper_gate, self.layout.lower_gate)
        cached = self._observation_bases.get(cache_key)
        if cached is None:
            image[:] = np.array([0.035, 0.045, 0.075], dtype=np.float32)
            for wall in self.boundary_walls() | self.divider_walls():
                paint(wall, (0.22, 0.24, 0.34))
            paint(self.layout.upper_gate, (0.15, 0.19, 0.28))
            paint(self.layout.lower_gate, (0.15, 0.19, 0.28))
            paint(GOAL, (0.23, 0.82, 0.48), inset=1)
            type(self)._observation_bases[cache_key] = image.copy()
        else:
            image[:] = cached
        sensor_color = (0.95, 0.62, 0.21) if self.sensor_visited else (0.42, 0.31, 0.18)
        paint(self.sensor, sensor_color, inset=1)
        if self.last_cue == "upper":
            paint(self.sensor, (0.21, 0.78, 0.94), inset=1)
        elif self.last_cue == "lower":
            paint(self.sensor, (0.78, 0.48, 0.94), inset=1)
        paint(self.position, (0.96, 0.93, 0.79), inset=1)
        # The world image stays partially observable, while this narrow band carries
        # the agent's explicit belief memory into the learned encoder.
        image[0:2, :, 0] = float(belief_upper)
        image[0:2, :, 2] = 1.0 - float(belief_upper)
        return image

    def clone_for_hidden(self, hidden_upper: bool) -> "TwoGatesEnv":
        clone = TwoGatesEnv(
            seed=self.seed,
            prior_upper=self.prior_upper,
            cue_reliability=self.cue_reliability,
            hidden_upper=hidden_upper,
            layout=self.layout,
        )
        clone.position = self.position
        clone.step_count = self.step_count
        clone.sensor_visited = self.sensor_visited
        clone.last_cue = self.last_cue
        clone.revealed_upper = self.revealed_upper
        clone.collisions = self.collisions
        clone.path = list(self.path)
        return clone

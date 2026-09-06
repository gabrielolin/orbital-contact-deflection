"""Mean projectile spacetime geometry; tau is seconds after reference_time."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class WorkspaceIntersection(Protocol):
    def line_intervals(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> list[tuple[float, float]]: ...


@dataclass(frozen=True)
class SpacetimeLine:
    position_W0: np.ndarray
    velocity_W: np.ndarray
    reference_time: float

    def __post_init__(self) -> None:
        for name in ("position_W0", "velocity_W"):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 3-vector")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.isfinite(self.reference_time):
            raise ValueError("reference_time must be finite")

    def evaluate(self, tau: float) -> tuple[np.ndarray, float]:
        if not np.isfinite(tau):
            raise ValueError("tau must be finite")
        return self.position_W0 + tau * self.velocity_W, self.reference_time + tau

    def tangent_xyz(self) -> np.ndarray:
        speed = np.linalg.norm(self.velocity_W)
        return self.velocity_W / speed if speed > 1e-12 else np.zeros(3)


@dataclass(frozen=True)
class InterceptCorridor:
    line: SpacetimeLine
    intervals: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        previous = -np.inf
        for lower, upper in self.intervals:
            if not (0 <= lower <= upper and np.isfinite(upper) and lower >= previous):
                raise ValueError("corridor requires ordered finite future intervals")
            previous = upper
        if not self.intervals:
            raise ValueError("use NoReachableWorkspaceIntersection for empty corridors")


@dataclass(frozen=True)
class NoReachableWorkspaceIntersection:
    line: SpacetimeLine
    reason: str


def intersect_workspace(
    line: SpacetimeLine,
    workspace: WorkspaceIntersection,
    *,
    horizon: float = 5.0,
) -> InterceptCorridor | NoReachableWorkspaceIntersection:
    """Intersect over a configured future observation horizon, not time feasibility."""
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be finite and positive")
    intervals = tuple(
        (max(0.0, lower), min(horizon, upper))
        for lower, upper in workspace.line_intervals(line.position_W0, line.velocity_W)
        if max(0.0, lower) <= min(horizon, upper)
    )
    if not intervals:
        return NoReachableWorkspaceIntersection(
            line, "mean_path_outside_workspace_in_prediction_horizon"
        )
    return InterceptCorridor(line, intervals)

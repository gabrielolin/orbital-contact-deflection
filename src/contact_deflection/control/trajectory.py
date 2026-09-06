"""Joint trajectory contracts for best-effort realization of contact intent.

Residuals use requested minus achieved joint state. A missing trajectory has
unknown terminal residuals (NaN), never zero residuals or a claim of feasibility.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


def _vector(value: npt.ArrayLike, name: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite joint vector")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class JointKinematicState:
    """Arm joint position, velocity and optionally known acceleration, in SI units."""

    q: npt.ArrayLike
    qd: npt.ArrayLike
    qdd: npt.ArrayLike | None = None

    def __post_init__(self) -> None:
        q = _vector(self.q, "q")
        qd = _vector(self.qd, "qd")
        qdd = None if self.qdd is None else _vector(self.qdd, "qdd")
        if q.shape != qd.shape or (qdd is not None and qdd.shape != q.shape):
            raise ValueError("q, qd and qdd must have equal shapes")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "qd", qd)
        object.__setattr__(self, "qdd", qdd)


@dataclass(frozen=True)
class JointTrajectoryLimits:
    """Hard trajectory bounds; derivative bounds are symmetric positive magnitudes.

    Position bounds should come from the robot model. Acceleration and jerk
    bounds must be supplied deliberately from controller configuration.
    """

    position_lower: npt.ArrayLike
    position_upper: npt.ArrayLike
    velocity: npt.ArrayLike
    acceleration: npt.ArrayLike
    jerk: npt.ArrayLike

    def __post_init__(self) -> None:
        names = ("position_lower", "position_upper", "velocity", "acceleration", "jerk")
        arrays = [_vector(getattr(self, name), name) for name in names]
        if any(array.shape != arrays[0].shape for array in arrays[1:]):
            raise ValueError("joint limit arrays must have equal shapes")
        if np.any(arrays[0] >= arrays[1]):
            raise ValueError("position_lower must be strictly below position_upper")
        if any(np.any(array <= 0) for array in arrays[2:]):
            raise ValueError("velocity, acceleration and jerk limits must be positive")
        for name, array in zip(names, arrays, strict=True):
            object.__setattr__(self, name, array)


class JointTrajectory(Protocol):
    """A kinematic trajectory indexed by elapsed time from its initial state."""

    def sample(self, time: float) -> JointKinematicState:
        """Evaluate position, velocity and acceleration at elapsed seconds."""
        ...


@dataclass(frozen=True)
class TrajectorySolveResult:
    """Keep request and realization distinct, including when no solver is available.

    ``success`` reports backend success, not exact interception. Consumers must
    also inspect terminal residuals and achieved duration. NaN residual entries
    mean unknown, as when ``trajectory`` is None.
    """

    trajectory: JointTrajectory | None
    requested_duration: float
    achieved_duration: float | None
    terminal_position_residual: np.ndarray
    terminal_velocity_residual: np.ndarray
    success: bool
    status: str


class JointTrajectoryGenerator(Protocol):
    """Best-effort kinematic planning; infeasible requests return diagnostics."""

    def solve(
        self,
        initial_state: JointKinematicState,
        target_state: JointKinematicState,
        duration: float,
    ) -> TrajectorySolveResult:
        """Try to realize a terminal state after the requested elapsed seconds.

        A finite nonnegative duration, including an immediate zero-time request,
        is valid input. Dynamic infeasibility is a result, not an input error.
        """
        ...


class PlaceholderTrajectoryGenerator:
    """Explicitly unavailable backend that preserves requests and physical limits.

    TODO: minimize integrated squared jerk over spline/B-spline or polynomial
    coefficients, with fixed requested terminal time when feasible, soft terminal
    position/velocity residuals and hard position/velocity/acceleration/jerk
    bounds. A future QP/SQP backend must expose best-effort residuals and duration
    changes instead of selecting a different contact event. An initially
    out-of-bounds state requires an explicit infeasibility/recovery result.

    No interpolation is returned: even a polynomial matching endpoint states
    would not establish that physical bounds hold between endpoints.
    """

    def __init__(self, limits: JointTrajectoryLimits) -> None:
        self.limits = limits

    def solve(
        self,
        initial_state: JointKinematicState,
        target_state: JointKinematicState,
        duration: float,
    ) -> TrajectorySolveResult:
        expected = np.asarray(self.limits.position_lower).shape
        if any(
            np.asarray(state.q).shape != expected
            for state in (initial_state, target_state)
        ):
            raise ValueError("state and trajectory limit joint counts must agree")
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and nonnegative")
        return TrajectorySolveResult(
            trajectory=None,
            requested_duration=float(duration),
            achieved_duration=None,
            terminal_position_residual=np.full(expected, np.nan),
            terminal_velocity_residual=np.full(expected, np.nan),
            success=False,
            status="not_implemented",
        )

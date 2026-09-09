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


@dataclass(frozen=True)
class QuinticJointTrajectory:
    """Endpoint-interpolating quintic trajectory used by the executable baseline.

    It is deliberately a kinematic reference generator, not the eventual
    constrained minimum-jerk QP/SQP backend.  Bound checks are reported by the
    generator and are never hidden through clipping.
    """

    coefficients: np.ndarray
    duration: float

    def sample(self, time: float) -> JointKinematicState:
        clipped_time = float(np.clip(time, 0.0, self.duration))
        powers = np.array([1.0, clipped_time, clipped_time**2, clipped_time**3,
                           clipped_time**4, clipped_time**5])
        velocity_powers = np.array(
            [0.0, 1.0, 2.0 * clipped_time, 3.0 * clipped_time**2,
             4.0 * clipped_time**3, 5.0 * clipped_time**4]
        )
        acceleration_powers = np.array(
            [0.0, 0.0, 2.0, 6.0 * clipped_time, 12.0 * clipped_time**2,
             20.0 * clipped_time**3]
        )
        return JointKinematicState(
            powers @ self.coefficients,
            velocity_powers @ self.coefficients,
            acceleration_powers @ self.coefficients,
        )


class QuinticTrajectoryGenerator:
    """Generate an endpoint quintic and explicitly diagnose limit violations.

    This provides a real control reference for the RL integration without
    claiming dynamic feasibility.  A later optimization backend can implement
    the same ``JointTrajectoryGenerator`` protocol.
    """

    def __init__(self, limits: JointTrajectoryLimits) -> None:
        self.limits = limits

    def solve(
        self,
        initial_state: JointKinematicState,
        target_state: JointKinematicState,
        duration: float,
    ) -> TrajectorySolveResult:
        lower = np.asarray(self.limits.position_lower, dtype=float)
        expected = lower.shape
        initial_q = np.asarray(initial_state.q, dtype=float)
        initial_qd = np.asarray(initial_state.qd, dtype=float)
        target_q = np.asarray(target_state.q, dtype=float)
        target_qd = np.asarray(target_state.qd, dtype=float)
        if initial_q.shape != expected or target_q.shape != expected:
            raise ValueError("state and trajectory limit joint counts must agree")
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and nonnegative")
        if duration <= 1e-6:
            return TrajectorySolveResult(
                trajectory=None,
                requested_duration=float(duration),
                achieved_duration=None,
                terminal_position_residual=np.full(expected, np.nan),
                terminal_velocity_residual=np.full(expected, np.nan),
                success=False,
                status="duration_too_short",
            )

        qdd0 = (
            np.zeros_like(initial_q)
            if initial_state.qdd is None
            else np.asarray(initial_state.qdd, dtype=float)
        )
        qdd1 = (
            np.zeros_like(target_q)
            if target_state.qdd is None
            else np.asarray(target_state.qdd, dtype=float)
        )
        t = float(duration)
        boundary = np.array(
            [
                initial_q,
                initial_qd,
                qdd0,
                target_q,
                target_qd,
                qdd1,
            ]
        )
        system = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, t, t**2, t**3, t**4, t**5],
                [0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4],
                [0, 0, 2, 6 * t, 12 * t**2, 20 * t**3],
            ],
            dtype=float,
        )
        trajectory = QuinticJointTrajectory(np.linalg.solve(system, boundary), t)
        sample_times = np.linspace(0.0, t, 101)
        time_powers = np.column_stack(
            [sample_times**power for power in range(6)]
        )
        velocity_powers = np.column_stack(
            (
                np.zeros_like(sample_times),
                np.ones_like(sample_times),
                2.0 * sample_times,
                3.0 * sample_times**2,
                4.0 * sample_times**3,
                5.0 * sample_times**4,
            )
        )
        acceleration_powers = np.column_stack(
            (
                np.zeros_like(sample_times),
                np.zeros_like(sample_times),
                2.0 * np.ones_like(sample_times),
                6.0 * sample_times,
                12.0 * sample_times**2,
                20.0 * sample_times**3,
            )
        )
        positions = time_powers @ trajectory.coefficients
        velocities = velocity_powers @ trajectory.coefficients
        accelerations = acceleration_powers @ trajectory.coefficients
        jerks = (
            6.0 * trajectory.coefficients[3]
            + 24.0 * sample_times[:, None] * trajectory.coefficients[4]
            + 60.0 * sample_times[:, None] ** 2 * trajectory.coefficients[5]
        )
        within_limits = bool(
            np.all(positions >= self.limits.position_lower)
            and np.all(positions <= self.limits.position_upper)
            and np.all(np.abs(velocities) <= self.limits.velocity)
            and np.all(np.abs(accelerations) <= self.limits.acceleration)
            and np.all(np.abs(jerks) <= self.limits.jerk)
        )
        terminal = trajectory.sample(t)
        return TrajectorySolveResult(
            trajectory=trajectory,
            requested_duration=t,
            achieved_duration=t,
            terminal_position_residual=target_q - np.asarray(terminal.q, dtype=float),
            terminal_velocity_residual=target_qd - np.asarray(terminal.qd, dtype=float),
            success=within_limits,
            status="success" if within_limits else "limit_violation",
        )

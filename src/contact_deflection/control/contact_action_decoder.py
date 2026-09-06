"""Structured Box(8) contact intent and best-effort kinematic realization.

Coordinates: time-on-line, tangent1/2 normal tilt, normal/tangent1/2 linear
velocity, tangent1/2 angular velocity. Contact frame columns are [n,t1,t2];
the model's shield normal is its local +z. Angular scales default to zero.
All velocities are world velocities of the shield-center site. IK and twist
mapping freeze the current base; they do not predict base reaction dynamics.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from contact_deflection.control.trajectory import (
    JointKinematicState,
    JointTrajectoryGenerator,
    TrajectorySolveResult,
)
from contact_deflection.control.twist_mapping import map_terminal_twist
from contact_deflection.estimation.spacetime_line import (
    InterceptCorridor,
    NoReachableWorkspaceIntersection,
)
from contact_deflection.geometry.contact_frame import contact_frame
from contact_deflection.kinematics.mink_ik import IKSolution, MinkIK


@dataclass(frozen=True)
class DecoderConfig:
    orientation_scales: tuple[float, float] = (0.35, 0.35)
    linear_velocity_scales: tuple[float, float, float] = (0.5, 0.25, 0.25)
    linear_velocity_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_scales: tuple[float, float] = (0.0, 0.0)
    interval_selection: str = "continuity"

    def __post_init__(self) -> None:
        for name, length in (
            ("orientation_scales", 2),
            ("linear_velocity_scales", 3),
            ("linear_velocity_center", 3),
            ("angular_velocity_scales", 2),
        ):
            values = np.asarray(getattr(self, name))
            if values.shape != (length,) or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid {name}")
            if name != "linear_velocity_center" and np.any(values < 0):
                raise ValueError(f"{name} must be nonnegative")
        if np.linalg.norm(self.orientation_scales) >= np.pi:
            raise ValueError("tilt chart must stay below pi")
        if self.interval_selection not in ("continuity", "earliest"):
            raise ValueError("interval_selection must be continuity or earliest")


@dataclass(frozen=True)
class ContactGoal:
    position_W: np.ndarray
    contact_time: float
    orientation_W: np.ndarray
    linear_velocity_W: np.ndarray
    angular_velocity_W: np.ndarray
    line_parameter: float

    @property
    def shield_normal_W(self) -> np.ndarray:
        return self.orientation_W[:, 2].copy()


@dataclass(frozen=True)
class DecoderState:
    full_qpos: np.ndarray
    joints: JointKinematicState
    time: float


@dataclass(frozen=True)
class DecodedAction:
    latent_action: np.ndarray
    requested_contact_goal: ContactGoal
    ik_solution: IKSolution
    requested_joint_velocity: np.ndarray
    realized_contact_twist: np.ndarray
    twist_residual: np.ndarray
    trajectory_result: TrajectorySolveResult
    contact_frame_W: np.ndarray
    twist_success: bool
    twist_status: str


class ContactActionDecoder:
    def __init__(
        self,
        ik: MinkIK,
        trajectory_generator: JointTrajectoryGenerator,
        config: DecoderConfig | None = None,
    ) -> None:
        self.ik = ik
        self.trajectory_generator = trajectory_generator
        self.config = config or DecoderConfig()
        self.reset()

    def reset(self) -> None:
        """Clear chart/IK continuity state at episode reset."""
        self._previous_frame: np.ndarray | None = None
        self._previous_time: float | None = None
        self._previous_q: np.ndarray | None = None

    def contact_goal(
        self, corridor: InterceptCorridor, action: np.ndarray
    ) -> tuple[ContactGoal, np.ndarray]:
        action = np.asarray(action, dtype=float)
        if (
            action.shape != (8,)
            or not np.all(np.isfinite(action))
            or np.any(abs(action) > 1)
        ):
            raise ValueError("action must be finite Box(8) coordinates in [-1,1]")
        index = 0
        if (
            self.config.interval_selection == "continuity"
            and self._previous_time is not None
        ):
            # Compare absolute times; tau origins change at each KF update.
            previous_time = self._previous_time
            index = min(
                range(len(corridor.intervals)),
                key=lambda i: max(
                    corridor.line.reference_time
                    + corridor.intervals[i][0]
                    - previous_time,
                    previous_time
                    - corridor.line.reference_time
                    - corridor.intervals[i][1],
                    0.0,
                ),
            )
        lower, upper = corridor.intervals[index]
        tau = lower + 0.5 * (action[0] + 1.0) * (upper - lower)
        position, time = corridor.line.evaluate(tau)
        frame = contact_frame(corridor.line.velocity_W, self._previous_frame)
        nominal = frame[:, [1, 2, 0]]  # shield +z = frame normal
        rotation_vector_W = frame[:, 1:] @ (
            np.asarray(self.config.orientation_scales) * action[1:3]
        )
        orientation = Rotation.from_rotvec(rotation_vector_W).as_matrix() @ nominal
        linear = frame @ (
            np.asarray(self.config.linear_velocity_center)
            + np.asarray(self.config.linear_velocity_scales) * action[3:6]
        )
        angular = frame[:, 1:] @ (
            np.asarray(self.config.angular_velocity_scales) * action[6:8]
        )
        goal = ContactGoal(position, time, orientation, linear, angular, float(tau))
        self._previous_frame, self._previous_time = frame.copy(), time
        return goal, frame

    def decode(
        self,
        state: DecoderState,
        intercept_corridor: InterceptCorridor | NoReachableWorkspaceIntersection,
        action: np.ndarray,
    ) -> DecodedAction | NoReachableWorkspaceIntersection:
        if isinstance(intercept_corridor, NoReachableWorkspaceIntersection):
            return intercept_corridor
        if not np.isclose(
            state.time, intercept_corridor.line.reference_time, atol=1e-8, rtol=0
        ):
            raise ValueError("propagate belief to current state time before decoding")
        goal, frame = self.contact_goal(intercept_corridor, action)
        solution = self.ik.solve(
            state.full_qpos, goal.position_W, goal.orientation_W, seed=self._previous_q
        )
        self._previous_q = solution.q_target.copy()
        jacobian = self.ik.jacobian(state.full_qpos, solution.q_target)
        twist = map_terminal_twist(
            jacobian,
            np.r_[goal.linear_velocity_W, goal.angular_velocity_W],
            self.ik.velocity_limits,
        )
        trajectory = self.trajectory_generator.solve(
            state.joints,
            JointKinematicState(
                solution.q_target, twist.qdot_target, np.zeros_like(twist.qdot_target)
            ),
            goal.contact_time - state.time,
        )
        return DecodedAction(
            np.asarray(action).copy(),
            goal,
            solution,
            twist.qdot_target,
            twist.realized_twist,
            twist.twist_residual,
            trajectory,
            frame,
            twist.success,
            twist.status,
        )

"""Gymnasium task exposing the structured contact-intent action space."""

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from contact_deflection.control import PDControllerConfig, TorqueController
from contact_deflection.control.contact_action_decoder import (
    ContactActionDecoder,
    DecodedAction,
    DecoderState,
)
from contact_deflection.control.decoder_config import StructuredDecoderConfig
from contact_deflection.control.trajectory import (
    JointKinematicState,
    QuinticTrajectoryGenerator,
)
from contact_deflection.envs.space_robot_env import SpaceRobotConfig, SpaceRobotEnv
from contact_deflection.estimation import ProjectileKalmanFilter
from contact_deflection.estimation.spacetime_line import (
    InterceptCorridor,
    NoReachableWorkspaceIntersection,
)
from contact_deflection.kinematics.mink_ik import MinkIK
from contact_deflection.kinematics.reachable_workspace import ReachableWorkspace


@dataclass(frozen=True)
class ContactDeflectionConfig:
    robot: SpaceRobotConfig = field(default_factory=SpaceRobotConfig)
    decoder: StructuredDecoderConfig = field(default_factory=StructuredDecoderConfig)
    sensor_dt: float = 0.01
    measurement_noise_std: float = 0.01
    acceleration_noise_std: float = 0.05
    episode_duration: float = 2.0
    projectile_speed: float = 0.6
    projectile_lead_time: float = 1.0


class ContactDeflectionEnv(SpaceRobotEnv):
    """Free-floating contact task driven by Box(8) structured contact intent.

    ``workspace`` is a base-frame cache generated offline with constrained IK.
    It is placed at the measured base pose each control step and is never
    regenerated during reset.
    """

    def __init__(
        self,
        workspace: ReachableWorkspace,
        config: ContactDeflectionConfig | None = None,
        *,
        render_mode: str | None = None,
    ) -> None:
        self.task_config = config if config is not None else ContactDeflectionConfig()
        super().__init__(self.task_config.robot, render_mode=render_mode)
        if workspace.points.size == 0:
            raise ValueError("workspace must contain at least one IK-accepted point")
        self.workspace_B = workspace
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.finfo(np.float32).max, np.finfo(np.float32).max, (49,), np.float32
        )
        self.controller = TorqueController(
            PDControllerConfig(
                kp=[35, 35, 30, 20, 15, 10],
                kd=[5, 5, 4.5, 3, 2.5, 2],
                torque_limits=[150, 150, 150, 28, 28, 28],
            )
        )
        self.ik = MinkIK(self.model, self.task_config.decoder.ik)
        self.decoder = ContactActionDecoder(
            self.ik,
            QuinticTrajectoryGenerator(self.task_config.decoder.trajectory.limits(self.ik)),
            self.task_config.decoder.decoder,
        )
        self._previous_action = np.zeros(8, dtype=float)
        self._filter: ProjectileKalmanFilter
        self._next_sensor_time = 0.0
        self._initial_base_position = np.zeros(3)
        self._last_corridor: (
            InterceptCorridor | NoReachableWorkspaceIntersection | None
        ) = None
        self._last_decoded: DecodedAction | None = None

    def _spacecraft_measurement(self) -> np.ndarray:
        relative_true = self.spacecraft_rotation_world.T @ (
            self.projectile_position_world - self.spacecraft_position_world
        )
        return relative_true + self.np_random.normal(
            0.0, self.task_config.measurement_noise_std, size=3
        )

    def _update_sensor(self) -> None:
        while self.data.time + 1e-12 >= self._next_sensor_time:
            self._filter.update(
                self._spacecraft_measurement(),
                self.spacecraft_position_world,
                self.spacecraft_rotation_world,
            )
            self._next_sensor_time += self.task_config.sensor_dt

    def _world_workspace(self) -> ReachableWorkspace:
        return self.workspace_B.placed(
            self.spacecraft_position_world, self.spacecraft_rotation_world
        )

    def projectile_intercept_corridor(
        self, *, horizon: float | None = None
    ) -> InterceptCorridor | NoReachableWorkspaceIntersection:
        """Return the belief corridor against the workspace at the current base pose."""
        return self._filter.intercept_corridor(
            self._world_workspace(),
            horizon=(
                self.task_config.decoder.prediction_horizon
                if horizon is None
                else horizon
            ),
        )

    @staticmethod
    def _corridor_features(
        corridor: InterceptCorridor | NoReachableWorkspaceIntersection | None,
    ) -> np.ndarray:
        if not isinstance(corridor, InterceptCorridor):
            return np.zeros(3)
        lower, upper = corridor.intervals[0]
        return np.array([1.0, lower, upper])

    def _observation(self) -> np.ndarray:
        belief = self._filter.relative_belief(
            self.spacecraft_position_world,
            self.spacecraft_velocity_world,
            self.spacecraft_rotation_world,
        )
        progress = min(1.0, self.data.time / self.task_config.episode_duration)
        observation = np.concatenate(
            (
                belief.mean,
                np.diag(belief.covariance),
                self.arm_q,
                self.arm_qd,
                self.spacecraft_position_world,
                self.spacecraft_quaternion_world,
                self.spacecraft_velocity_world,
                self.spacecraft_angular_velocity,
                self._previous_action,
                self._corridor_features(self._last_corridor),
                [progress],
            )
        )
        return observation.astype(np.float32)

    def _set_episode_projectile(self) -> None:
        """Place a constant-velocity projectile through the initial shield center."""
        shield = self.data.site("shield_center")
        normal_W = shield.xmat.reshape(3, 3)[:, 2]
        velocity = self.task_config.projectile_speed * normal_W
        qpos = self.model.jnt_qposadr[self._projectile_joint_id]
        dof = self.model.jnt_dofadr[self._projectile_joint_id]
        self.data.qpos[qpos : qpos + 3] = (
            shield.xpos - self.task_config.projectile_lead_time * velocity
        )
        self.data.qpos[qpos + 3 : qpos + 7] = [1, 0, 0, 0]
        self.data.qvel[dof : dof + 3] = velocity
        self.data.qvel[dof + 3 : dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed, options=options)
        self._reset_physics()
        self._set_episode_projectile()
        self._filter = ProjectileKalmanFilter(
            np.r_[self.projectile_position_world, self.projectile_velocity_world],
            0.05 * np.eye(6),
            acceleration_noise_std=self.task_config.acceleration_noise_std,
            measurement_noise_std=self.task_config.measurement_noise_std,
        )
        self.decoder.reset()
        self._previous_action.fill(0.0)
        self._next_sensor_time = 0.0
        self._initial_base_position = self.spacecraft_position_world
        self._update_sensor()
        self._last_corridor = self.projectile_intercept_corridor()
        self._last_decoded = None
        return self._observation(), {"true_state": self._true_diagnostics()}

    def _true_diagnostics(self) -> dict[str, np.ndarray]:
        return {
            "projectile_position_world": self.projectile_position_world,
            "projectile_velocity_world": self.projectile_velocity_world,
            "spacecraft_position_world": self.spacecraft_position_world,
        }

    def _decode(
        self, action: np.ndarray
    ) -> DecodedAction | NoReachableWorkspaceIntersection:
        corridor = self.projectile_intercept_corridor()
        self._last_corridor = corridor
        return self.decoder.decode(
            DecoderState(
                self.data.qpos.copy(),
                JointKinematicState(
                    self.arm_q, self.arm_qd, self.data.qacc[self._arm_dof_indices]
                ),
                self.data.time,
            ),
            corridor,
            action,
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (8,) or not np.all(np.isfinite(action_array)):
            raise ValueError("action must be finite with shape (8,)")
        if np.any(np.abs(action_array) > 1.0 + 1e-6):
            raise ValueError("action must lie within Box(8) bounds")
        action_array = np.clip(action_array, -1.0, 1.0)
        decoded = self._decode(action_array)
        self._last_decoded = decoded if isinstance(decoded, DecodedAction) else None
        contact = False
        for step_index in range(self.config.physics_steps_per_action):
            if (
                isinstance(decoded, DecodedAction)
                and decoded.trajectory_result.trajectory
            ):
                reference = decoded.trajectory_result.trajectory.sample(
                    (step_index + 1) * self.config.sim_dt
                )
                q_reference, qd_reference = reference.q, reference.qd
            else:
                q_reference, qd_reference = self.arm_q, np.zeros(6)
            torque = self.controller.compute(
                self.arm_q, self.arm_qd, q_reference, qd_reference
            )
            self._step_physics(torque)
            self._filter.predict(self.config.sim_dt)
            self._update_sensor()
            contact = contact or self.shield_projectile_contact()
            if contact:
                break
        self._previous_action = action_array.copy()
        distance = float(
            np.linalg.norm(
                self.projectile_position_world - self.data.site("shield_center").xpos
            )
        )
        position_residual = (
            float(self._last_decoded.ik_solution.position_residual)
            if self._last_decoded is not None
            else 1.0
        )
        reward = 10.0 if contact else -distance - 0.1 * position_residual
        terminated = contact
        truncated = self.data.time >= self.task_config.episode_duration
        trajectory_status = (
            self._last_decoded.trajectory_result.status
            if self._last_decoded is not None
            else "no_reachable_workspace_intersection"
        )
        info: dict[str, Any] = {
            "contact_success": float(contact),
            "projectile_shield_distance": distance,
            "base_displacement": float(
                np.linalg.norm(
                    self.spacecraft_position_world - self._initial_base_position
                )
            ),
            "ik_position_residual": position_residual,
            "twist_residual_norm": (
                float(np.linalg.norm(self._last_decoded.twist_residual))
                if self._last_decoded is not None
                else float("nan")
            ),
            "trajectory_feasible": (
                float(self._last_decoded.trajectory_result.success)
                if self._last_decoded is not None
                else 0.0
            ),
            "trajectory_status": trajectory_status,
            "true_state": self._true_diagnostics(),
        }
        return self._observation(), float(reward), terminated, truncated, info

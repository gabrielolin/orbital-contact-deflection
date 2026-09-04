"""Sequential contact-deflection task built on the SpaceRobotEnv model."""

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from contact_deflection.control import PDControllerConfig, TorqueController
from contact_deflection.envs.space_robot_env import SpaceRobotConfig, SpaceRobotEnv
from contact_deflection.estimation import ProjectileKalmanFilter


@dataclass(frozen=True)
class ContactDeflectionConfig:
    robot: SpaceRobotConfig = field(default_factory=SpaceRobotConfig)
    sensor_dt: float = 0.01
    measurement_noise_std: float = 0.01
    acceleration_noise_std: float = 0.05
    episode_duration: float = 2.0


class ContactDeflectionEnv(SpaceRobotEnv):
    """Gymnasium environment with joint targets and a filtered projectile belief."""

    def __init__(
        self,
        config: ContactDeflectionConfig | None = None,
        *,
        render_mode: str | None = None,
    ) -> None:
        self.task_config = config if config is not None else ContactDeflectionConfig()
        super().__init__(self.task_config.robot, render_mode=render_mode)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        observation_limit = np.finfo(np.float64).max
        self.observation_space = gym.spaces.Box(
            -observation_limit,
            observation_limit,
            shape=(44,),
            dtype=np.float64,
        )
        self.controller = TorqueController(
            PDControllerConfig(
                kp=[35, 35, 30, 20, 15, 10],
                kd=[5, 5, 4.5, 3, 2.5, 2],
                torque_limits=[150, 150, 150, 28, 28, 28],
            )
        )
        self._joint_target_scale = np.array([2.0942] * 3 + [np.pi] * 3)
        self._previous_action = np.zeros(6)
        self._filter: ProjectileKalmanFilter
        self._next_sensor_time = 0.0
        self._initial_base_position = np.zeros(3)

    def _spacecraft_measurement(self) -> np.ndarray:
        relative_true = self.spacecraft_rotation_world.T @ (
            self.projectile_position_world - self.spacecraft_position_world
        )
        noise = self.np_random.normal(
            0.0, self.task_config.measurement_noise_std, size=3
        )
        return relative_true + noise

    def _update_sensor(self) -> None:
        while self.data.time + 1e-12 >= self._next_sensor_time:
            self._filter.update(
                self._spacecraft_measurement(),
                self.spacecraft_position_world,
                self.spacecraft_rotation_world,
            )
            self._next_sensor_time += self.task_config.sensor_dt

    def _observation(self) -> np.ndarray:
        belief = self._filter.relative_belief(
            self.spacecraft_position_world,
            self.spacecraft_velocity_world,
            self.spacecraft_rotation_world,
        )
        progress = min(1.0, self.data.time / self.task_config.episode_duration)
        return np.concatenate(
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
                [progress],
            )
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed, options=options)
        self._reset_physics()
        initial_mean = np.concatenate(
            (self.projectile_position_world, self.projectile_velocity_world)
        )
        self._filter = ProjectileKalmanFilter(
            initial_mean,
            0.05 * np.eye(6),
            acceleration_noise_std=self.task_config.acceleration_noise_std,
            measurement_noise_std=self.task_config.measurement_noise_std,
        )
        self._previous_action.fill(0.0)
        self._next_sensor_time = 0.0
        self._initial_base_position = self.spacecraft_position_world
        self._update_sensor()
        return self._observation(), {"true_state": self._true_diagnostics()}

    def _true_diagnostics(self) -> dict[str, np.ndarray]:
        return {
            "projectile_position_world": self.projectile_position_world,
            "projectile_velocity_world": self.projectile_velocity_world,
            "spacecraft_position_world": self.spacecraft_position_world,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (6,) or not np.all(np.isfinite(action_array)):
            raise ValueError("action must be finite with shape (6,)")
        clipped_action = np.clip(action_array, -1.0, 1.0)
        q_reference = clipped_action * self._joint_target_scale
        contact = False
        for _ in range(self.config.physics_steps_per_action):
            torque = self.controller.compute(
                self.arm_q, self.arm_qd, q_reference, np.zeros(6)
            )
            self._step_physics(torque)
            self._filter.predict(self.config.sim_dt)
            self._update_sensor()
            contact = contact or self.shield_projectile_contact()
            if contact:
                break
        self._previous_action = clipped_action
        relative_position = self.spacecraft_rotation_world.T @ (
            self.projectile_position_world - self.data.site("shield_center").xpos
        )
        distance = float(np.linalg.norm(relative_position))
        reward = 10.0 if contact else -distance
        terminated = contact
        truncated = self.data.time >= self.task_config.episode_duration
        info: dict[str, Any] = {
            "contact_success": contact,
            "projectile_shield_distance": distance,
            "base_displacement": float(
                np.linalg.norm(
                    self.spacecraft_position_world - self._initial_base_position
                )
            ),
            "true_state": self._true_diagnostics(),
        }
        return self._observation(), reward, terminated, truncated, info

"""Gymnasium task exposing the structured contact-intent action space."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import yaml

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
from contact_deflection.geometry.contact_frame import contact_frame
from contact_deflection.kinematics.mink_ik import MinkIK
from contact_deflection.kinematics.reachable_workspace import EllipsoidalWorkspace


@dataclass(frozen=True)
class DesiredVelocityGoalConfig:
    """Reachable-inspired Gaussian goal distribution in the contact frame."""

    normal_mean: float = 1.45
    normal_std: float = 0.25
    tangent_std: float = 0.40
    minimum_speed: float = 0.75
    maximum_speed: float = 2.50

    def __post_init__(self) -> None:
        if (
            self.normal_mean <= 0
            or self.normal_std < 0
            or self.tangent_std < 0
            or self.minimum_speed <= 0
            or self.maximum_speed <= self.minimum_speed
        ):
            raise ValueError("invalid desired-velocity goal distribution")


@dataclass(frozen=True)
class TerminalRewardConfig:
    """Terminal evaluation and dimensionless reward scales."""

    velocity_tolerance: float = 0.50
    miss_penalty: float = 1.0
    miss_distance_scale: float = 0.25
    huber_delta: float = 1.0
    angular_momentum_scale: float = 10.0
    angular_momentum_weight: float = 0.05
    follow_through_duration: float = 0.050
    stopping_duration: float = 0.100
    outgoing_velocity_average_duration: float = 0.040

    def __post_init__(self) -> None:
        positive = (
            self.velocity_tolerance,
            self.miss_distance_scale,
            self.huber_delta,
            self.angular_momentum_scale,
            self.follow_through_duration,
            self.stopping_duration,
            self.outgoing_velocity_average_duration,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(
                "terminal evaluation durations and scales must be positive"
            )
        if self.miss_penalty < 0 or self.angular_momentum_weight < 0:
            raise ValueError("terminal penalty weights must be nonnegative")


@dataclass(frozen=True)
class EpisodeRandomizationConfig:
    """Gaussian initial-state and dynamics distribution sampled at reset."""

    contact_solver_time_constant_mean: float = 0.010
    contact_solver_time_constant_std: float = 0.0005
    contact_solver_time_constant_minimum: float = 0.0085
    contact_damping_ratio_mean: float = 0.100
    contact_damping_ratio_std: float = 0.005
    contact_damping_ratio_minimum: float = 0.080
    projectile_mass_mean: float = 1.50
    projectile_mass_std: float = 0.15
    projectile_mass_minimum: float = 0.25
    shield_mass_mean: float = 4.50
    shield_mass_std: float = 0.30
    shield_mass_minimum: float = 0.50
    projectile_position_mean_shield: tuple[float, float, float] = (0.0, 0.0, 3.0)
    projectile_position_std: tuple[float, float, float] = (0.05, 0.05, 0.15)
    projectile_velocity_component_std: float = 0.10
    arm_position_std: float = 0.02
    spacecraft_linear_velocity_std: float = 0.005
    spacecraft_angular_velocity_std: float = 0.002

    def __post_init__(self) -> None:
        nonnegative = (
            self.contact_solver_time_constant_std,
            self.contact_damping_ratio_std,
            self.projectile_mass_std,
            self.shield_mass_std,
            self.projectile_velocity_component_std,
            self.arm_position_std,
            self.spacecraft_linear_velocity_std,
            self.spacecraft_angular_velocity_std,
        )
        positive = (
            self.contact_solver_time_constant_minimum,
            self.contact_solver_time_constant_mean,
            self.contact_damping_ratio_minimum,
            self.contact_damping_ratio_mean,
            self.projectile_mass_mean,
            self.projectile_mass_minimum,
            self.shield_mass_mean,
            self.shield_mass_minimum,
        )
        if any(value < 0 for value in nonnegative) or any(
            value <= 0 for value in positive
        ):
            raise ValueError("episode randomization scales must be physical")
        if (
            self.contact_solver_time_constant_mean
            <= self.contact_solver_time_constant_minimum
            or self.contact_damping_ratio_mean
            <= self.contact_damping_ratio_minimum
            or self.projectile_mass_mean <= self.projectile_mass_minimum
            or self.shield_mass_mean <= self.shield_mass_minimum
        ):
            raise ValueError("Gaussian means must exceed their rejection floors")
        position_mean = np.asarray(self.projectile_position_mean_shield, dtype=float)
        position_std = np.asarray(self.projectile_position_std, dtype=float)
        if (
            position_mean.shape != (3,)
            or position_std.shape != (3,)
            or not np.all(np.isfinite(position_mean))
            or not np.all(np.isfinite(position_std))
            or np.any(position_std < 0)
        ):
            raise ValueError(
                "projectile position distribution must be finite 3-vectors"
            )


@dataclass(frozen=True)
class ContactDeflectionConfig:
    robot: SpaceRobotConfig = field(default_factory=SpaceRobotConfig)
    decoder: StructuredDecoderConfig = field(default_factory=StructuredDecoderConfig)
    desired_velocity_goal: DesiredVelocityGoalConfig = field(
        default_factory=DesiredVelocityGoalConfig
    )
    terminal_reward: TerminalRewardConfig = field(default_factory=TerminalRewardConfig)
    episode_randomization: EpisodeRandomizationConfig = field(
        default_factory=EpisodeRandomizationConfig
    )
    controller: PDControllerConfig = field(
        default_factory=lambda: PDControllerConfig(
            kp=[35, 35, 30, 20, 15, 10],
            kd=[5, 5, 4.5, 3, 2.5, 2],
            torque_limits=[150, 150, 150, 28, 28, 28],
        )
    )
    sensor_dt: float = 0.01
    measurement_noise_std: float = 0.01
    acceleration_noise_std: float = 0.05
    episode_duration: float = 2.0
    projectile_speed: float = 3.0

    def __post_init__(self) -> None:
        if self.sensor_dt <= 0 or self.episode_duration <= 0:
            raise ValueError("sensor period and episode duration must be positive")
        if self.measurement_noise_std <= 0 or self.acceleration_noise_std < 0:
            raise ValueError("estimator noise standard deviations must be valid")
        if self.projectile_speed <= 0:
            raise ValueError("projectile speed must be positive")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        decoder: StructuredDecoderConfig | None = None,
    ) -> "ContactDeflectionConfig":
        """Load the benchmark configuration used by experiment entry points."""
        with Path(path).open() as stream:
            values = yaml.safe_load(stream)
        simulation = values.get("simulation", {})
        robot_keys = {"sim_dt", "control_dt", "policy_dt", "initial_arm_q"}
        robot = SpaceRobotConfig(
            **{key: value for key, value in simulation.items() if key in robot_keys}
        )
        return cls(
            robot=robot,
            decoder=decoder or StructuredDecoderConfig(),
            desired_velocity_goal=DesiredVelocityGoalConfig(
                **values.get("desired_velocity_goal", {})
            ),
            terminal_reward=TerminalRewardConfig(**values.get("terminal_reward", {})),
            episode_randomization=EpisodeRandomizationConfig(
                **values.get("episode_randomization", {})
            ),
            controller=PDControllerConfig(**values.get("controller", {})),
            sensor_dt=simulation.get("sensor_dt", 0.01),
            measurement_noise_std=values.get("estimation", {}).get(
                "measurement_noise_std", 0.01
            ),
            acceleration_noise_std=values.get("estimation", {}).get(
                "acceleration_noise_std", 0.05
            ),
            episode_duration=simulation.get("episode_duration", 2.0),
            projectile_speed=simulation.get("projectile_speed", 3.0),
        )


class ContactDeflectionEnv(SpaceRobotEnv):
    """Free-floating contact task driven by Box(8) structured contact intent.

    A smooth spacecraft-frame candidate envelope defines the contact-time
    chart. Every selected terminal pose is validated by online constrained IK.
    """

    def __init__(
        self,
        config: ContactDeflectionConfig | None = None,
        *,
        workspace: EllipsoidalWorkspace | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.task_config = config if config is not None else ContactDeflectionConfig()
        super().__init__(self.task_config.robot, render_mode=render_mode)
        self.workspace_B = workspace or self.task_config.decoder.workspace.build()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        vector_space = gym.spaces.Box(
            -np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            (49,),
            np.float32,
        )
        goal_space = gym.spaces.Box(
            -np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            (3,),
            np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {"observation": vector_space, "desired_goal": goal_space}
        )
        self.controller = TorqueController(self.task_config.controller)
        self.ik = MinkIK(self.model, self.task_config.decoder.ik)
        self.decoder = ContactActionDecoder(
            self.ik,
            QuinticTrajectoryGenerator(
                self.task_config.decoder.trajectory.limits(self.ik)
            ),
            self.task_config.decoder.decoder,
        )
        self._previous_action = np.zeros(8, dtype=float)
        self._filter: ProjectileKalmanFilter
        self._next_sensor_time = 0.0
        self._initial_base_position = np.zeros(3)
        self._initial_spacecraft_angular_momentum = np.zeros(3)
        self._desired_velocity_world = np.zeros(3)
        self._minimum_projectile_shield_distance = float("inf")
        self._last_outgoing_velocity = np.zeros(3)
        self._last_reward_components: dict[str, float] = {}
        self._episode_parameters: dict[str, Any] = {}
        shield_body_id = self.model.body("shield").id
        self._randomized_body_ids = {
            "projectile": self._projectile_body_id,
            "shield": shield_body_id,
        }
        self._body_inertia_per_mass = {
            name: self.model.body_inertia[body_id].copy()
            / self.model.body_mass[body_id]
            for name, body_id in self._randomized_body_ids.items()
        }
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

    def _world_workspace(self) -> EllipsoidalWorkspace:
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

    def _observation(self) -> dict[str, np.ndarray]:
        belief = self._filter.relative_belief(
            self.spacecraft_position_world,
            self.spacecraft_velocity_world,
            self.spacecraft_rotation_world,
        )
        progress = min(1.0, self.data.time / self.task_config.episode_duration)
        state = np.concatenate(
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
        desired_goal = (
            self.spacecraft_rotation_world.T @ self._desired_velocity_world
        )
        return {
            "observation": state.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def _sample_desired_velocity(self) -> np.ndarray:
        config = self.task_config.desired_velocity_goal
        frame = contact_frame(self.projectile_velocity_world)
        for _ in range(10_000):
            components = np.array(
                [
                    self.np_random.normal(config.normal_mean, config.normal_std),
                    self.np_random.normal(0.0, config.tangent_std),
                    self.np_random.normal(0.0, config.tangent_std),
                ]
            )
            speed = float(np.linalg.norm(components))
            if (
                components[0] > 0
                and config.minimum_speed <= speed <= config.maximum_speed
            ):
                return frame @ components
        raise RuntimeError("could not sample a valid desired outgoing velocity")

    def _sample_positive_gaussian(
        self, mean: float, std: float, minimum: float
    ) -> float:
        for _ in range(10_000):
            value = float(self.np_random.normal(mean, std))
            if value > minimum:
                return value
        raise RuntimeError("could not sample a physical episode parameter")

    def _randomize_episode_dynamics(self) -> None:
        config = self.task_config.episode_randomization
        time_constant = self._sample_positive_gaussian(
            config.contact_solver_time_constant_mean,
            config.contact_solver_time_constant_std,
            config.contact_solver_time_constant_minimum,
        )
        damping_ratio = self._sample_positive_gaussian(
            config.contact_damping_ratio_mean,
            config.contact_damping_ratio_std,
            config.contact_damping_ratio_minimum,
        )
        projectile_mass = self._sample_positive_gaussian(
            config.projectile_mass_mean,
            config.projectile_mass_std,
            config.projectile_mass_minimum,
        )
        shield_mass = self._sample_positive_gaussian(
            config.shield_mass_mean,
            config.shield_mass_std,
            config.shield_mass_minimum,
        )
        for name, mass in (
            ("projectile", projectile_mass),
            ("shield", shield_mass),
        ):
            body_id = self._randomized_body_ids[name]
            self.model.body_mass[body_id] = mass
            self.model.body_inertia[body_id] = self._body_inertia_per_mass[name] * mass
        contact_geoms = np.array([self._shield_geom_id, self._projectile_geom_id])
        self.model.geom_solref[contact_geoms] = [time_constant, damping_ratio]
        mujoco.mj_setConst(self.model, self.data)
        self._episode_parameters = {
            "contact_solver_time_constant": time_constant,
            "contact_damping_ratio": damping_ratio,
            "projectile_mass": projectile_mass,
            "shield_mass": shield_mass,
        }

    def _randomize_robot_initial_state(self) -> None:
        config = self.task_config.episode_randomization
        self.data.qpos[self._arm_qpos_indices] += self.np_random.normal(
            0.0, config.arm_position_std, size=6
        )
        self.data.qvel[self._arm_dof_indices] = 0.0
        spacecraft_dof = self.model.jnt_dofadr[self._spacecraft_joint_id]
        self.data.qvel[spacecraft_dof : spacecraft_dof + 3] = self.np_random.normal(
            0.0, config.spacecraft_linear_velocity_std, size=3
        )
        self.data.qvel[spacecraft_dof + 3 : spacecraft_dof + 6] = (
            self.np_random.normal(
                0.0, config.spacecraft_angular_velocity_std, size=3
            )
        )
        mujoco.mj_forward(self.model, self.data)

    def _set_episode_projectile(self) -> None:
        """Sample an incoming line aimed near the initial shield center."""
        shield = self.data.site("shield_center")
        shield_rotation = shield.xmat.reshape(3, 3)
        normal_W = shield_rotation[:, 2]
        randomization = self.task_config.episode_randomization
        # Approach the outward-facing shield from free space in front of it.
        velocity = (
            -self.task_config.projectile_speed * normal_W
            + self.np_random.normal(
                0.0, randomization.projectile_velocity_component_std, size=3
            )
        )
        if velocity @ normal_W >= 0:
            raise RuntimeError("sampled projectile does not approach the shield")
        position_shield = self.np_random.normal(
            np.asarray(randomization.projectile_position_mean_shield, dtype=float),
            np.asarray(randomization.projectile_position_std, dtype=float),
        )
        position_world = shield.xpos + shield_rotation @ position_shield
        qpos = self.model.jnt_qposadr[self._projectile_joint_id]
        dof = self.model.jnt_dofadr[self._projectile_joint_id]
        self.data.qpos[qpos : qpos + 3] = position_world
        self.data.qpos[qpos + 3 : qpos + 7] = [1, 0, 0, 0]
        self.data.qvel[dof : dof + 3] = velocity
        self.data.qvel[dof + 3 : dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._episode_parameters.update(
            {
                "projectile_velocity_world": velocity.copy(),
                "projectile_initial_position_world": position_world.copy(),
                "projectile_initial_position_shield": position_shield.copy(),
                "initial_arm_q": self.arm_q,
                "initial_arm_qd": self.arm_qd,
                "initial_spacecraft_velocity_world": (
                    self.spacecraft_velocity_world
                ),
                "initial_spacecraft_angular_velocity_world": (
                    self.spacecraft_angular_velocity
                ),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        gym.Env.reset(self, seed=seed, options=options)
        self._randomize_episode_dynamics()
        self._reset_physics()
        self._randomize_robot_initial_state()
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
        self._initial_spacecraft_angular_momentum = (
            self.spacecraft_angular_momentum_world
        )
        self._desired_velocity_world = self._sample_desired_velocity()
        self._minimum_projectile_shield_distance = float("inf")
        self._last_outgoing_velocity.fill(0.0)
        self._last_reward_components = {}
        self._update_sensor()
        self._last_corridor = self.projectile_intercept_corridor()
        self._last_decoded = None
        return self._observation(), {
            "episode_parameters": self._episode_parameters.copy(),
            "true_state": self._true_diagnostics(),
        }

    def _true_diagnostics(self) -> dict[str, np.ndarray]:
        return {
            "projectile_position_world": self.projectile_position_world,
            "projectile_velocity_world": self.projectile_velocity_world,
            "spacecraft_position_world": self.spacecraft_position_world,
            "desired_projectile_velocity_world": self._desired_velocity_world.copy(),
        }

    def _advance_physics(self, torque: np.ndarray) -> None:
        self._step_physics(torque)
        self._filter.predict(self.config.sim_dt)
        self._update_sensor()

    @staticmethod
    def _huber(value: float, delta: float) -> float:
        magnitude = abs(float(value))
        if magnitude <= delta:
            return 0.5 * magnitude**2
        return delta * (magnitude - 0.5 * delta)

    def _post_contact_reference(
        self,
        initial: JointKinematicState,
        elapsed: float,
    ) -> JointKinematicState:
        """Follow the impact velocity briefly, then brake smoothly to rest."""
        reward_config = self.task_config.terminal_reward
        follow = reward_config.follow_through_duration
        stopping = reward_config.stopping_duration
        q0 = np.asarray(initial.q)
        qd0 = np.asarray(initial.qd)
        if elapsed <= follow:
            return JointKinematicState(q0 + elapsed * qd0, qd0, np.zeros_like(qd0))
        q1 = q0 + follow * qd0
        braking_time = min(elapsed - follow, stopping)
        phase = braking_time / stopping
        velocity_scale = 1.0 - 3.0 * phase**2 + 2.0 * phase**3
        position_integral = stopping * (
            phase - phase**3 + 0.5 * phase**4
        )
        acceleration_scale = (-6.0 * phase + 6.0 * phase**2) / stopping
        return JointKinematicState(
            q1 + position_integral * qd0,
            velocity_scale * qd0,
            acceleration_scale * qd0,
        )

    def _evaluate_after_contact(
        self, impact_reference: JointKinematicState
    ) -> tuple[np.ndarray, bool]:
        """Execute deterministic follow-through/braking and estimate exit velocity."""
        reward_config = self.task_config.terminal_reward
        duration = (
            reward_config.follow_through_duration
            + reward_config.stopping_duration
            + reward_config.outgoing_velocity_average_duration
        )
        steps = round(duration / self.config.sim_dt)
        averaging_start = duration - reward_config.outgoing_velocity_average_duration
        velocities: list[np.ndarray] = []
        contact_during_average = False
        torque = np.zeros(self.model.nu)
        for step_index in range(steps):
            elapsed = step_index * self.config.sim_dt
            if step_index % self.config.physics_steps_per_control == 0:
                reference = self._post_contact_reference(impact_reference, elapsed)
                torque = self.controller.compute(
                    self.arm_q, self.arm_qd, reference.q, reference.qd
                )
            self._advance_physics(torque)
            if elapsed >= averaging_start:
                velocities.append(self.projectile_velocity_world)
                contact_during_average = (
                    contact_during_average or self.shield_projectile_contact()
                )
        outgoing = np.mean(velocities, axis=0)
        return outgoing, not contact_during_average

    def _terminal_reward(
        self, *, contact: bool, outgoing_velocity: np.ndarray | None
    ) -> tuple[float, dict[str, float]]:
        config = self.task_config.terminal_reward
        angular_momentum_delta = float(
            np.linalg.norm(
                self.spacecraft_angular_momentum_world
                - self._initial_spacecraft_angular_momentum
            )
        )
        momentum_penalty = config.angular_momentum_weight * self._huber(
            angular_momentum_delta / config.angular_momentum_scale,
            config.huber_delta,
        )
        if contact:
            if outgoing_velocity is None:
                raise ValueError("contact reward requires an outgoing velocity")
            velocity_error = float(
                np.linalg.norm(outgoing_velocity - self._desired_velocity_world)
            )
            velocity_score = float(
                np.exp(-((velocity_error / config.velocity_tolerance) ** 2))
            )
            reward = velocity_score - momentum_penalty
            components = {
                "velocity_error": velocity_error,
                "velocity_error_valid": 1.0,
                "velocity_score": velocity_score,
                "miss_distance_penalty": 0.0,
            }
        else:
            miss_distance_penalty = self._huber(
                self._minimum_projectile_shield_distance
                / config.miss_distance_scale,
                config.huber_delta,
            )
            reward = -config.miss_penalty - miss_distance_penalty - momentum_penalty
            components = {
                "velocity_error": 0.0,
                "velocity_error_valid": 0.0,
                "velocity_score": 0.0,
                "miss_distance_penalty": miss_distance_penalty,
            }
        components.update(
            {
                "spacecraft_angular_momentum_delta": angular_momentum_delta,
                "angular_momentum_penalty": momentum_penalty,
            }
        )
        return float(reward), components

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
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (8,) or not np.all(np.isfinite(action_array)):
            raise ValueError("action must be finite with shape (8,)")
        if np.any(np.abs(action_array) > 1.0 + 1e-6):
            raise ValueError("action must lie within Box(8) bounds")
        action_array = np.clip(action_array, -1.0, 1.0)
        decoded = self._decode(action_array)
        self._last_decoded = decoded if isinstance(decoded, DecodedAction) else None
        ik_feasible = bool(
            self._last_decoded is not None
            and self._last_decoded.ik_solution.converged
            and self._last_decoded.ik_solution.metadata.get(
                "constraints_satisfied", False
            )
        )
        contact = False
        impact_reference: JointKinematicState | None = None
        torque = np.zeros(self.model.nu)
        for step_index in range(self.config.physics_steps_per_action):
            if step_index % self.config.physics_steps_per_control == 0:
                if (
                    ik_feasible
                    and isinstance(decoded, DecodedAction)
                    and decoded.trajectory_result.trajectory
                ):
                    reference = decoded.trajectory_result.trajectory.sample(
                        step_index * self.config.sim_dt
                    )
                else:
                    reference = JointKinematicState(self.arm_q, np.zeros(6))
                torque = self.controller.compute(
                    self.arm_q, self.arm_qd, reference.q, reference.qd
                )
            self._advance_physics(torque)
            distance = float(
                np.linalg.norm(
                    self.projectile_position_world
                    - self.data.site("shield_center").xpos
                )
            )
            self._minimum_projectile_shield_distance = min(
                self._minimum_projectile_shield_distance, distance
            )
            if self.shield_projectile_contact():
                contact = True
                impact_reference = JointKinematicState(
                    self.arm_q,
                    reference.qd,
                    reference.qdd,
                )
                break
            if self.data.time >= self.task_config.episode_duration:
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
        deadline = self.data.time >= self.task_config.episode_duration
        terminated = contact or deadline
        truncated = False
        post_contact_separated = False
        if contact:
            if impact_reference is None:  # pragma: no cover - defensive invariant
                raise RuntimeError("contact detected without a controller reference")
            outgoing_velocity, post_contact_separated = self._evaluate_after_contact(
                impact_reference
            )
            self._last_outgoing_velocity = outgoing_velocity
        else:
            outgoing_velocity = None
        if terminated:
            reward, reward_components = self._terminal_reward(
                contact=contact, outgoing_velocity=outgoing_velocity
            )
            self._last_reward_components = reward_components
        else:
            reward = 0.0
            reward_components = {}
        if self._last_decoded is None:
            trajectory_status = "no_reachable_workspace_intersection"
        elif not ik_feasible:
            trajectory_status = "ik_infeasible"
        else:
            trajectory_status = self._last_decoded.trajectory_result.status
        info: dict[str, Any] = {
            "contact_success": float(contact),
            "terminal_reason": "contact" if contact else ("miss" if deadline else ""),
            "projectile_shield_distance": distance,
            "minimum_projectile_shield_distance": (
                self._minimum_projectile_shield_distance
            ),
            "base_displacement": float(
                np.linalg.norm(
                    self.spacecraft_position_world - self._initial_base_position
                )
            ),
            "ik_position_residual": position_residual,
            "ik_converged": float(ik_feasible),
            "twist_residual_norm": (
                float(np.linalg.norm(self._last_decoded.twist_residual))
                if self._last_decoded is not None
                else 0.0
            ),
            "twist_residual_valid": float(self._last_decoded is not None),
            "trajectory_feasible": (
                float(self._last_decoded.trajectory_result.success)
                if self._last_decoded is not None
                else 0.0
            ),
            "trajectory_status": trajectory_status,
            "desired_projectile_velocity_world": self._desired_velocity_world.copy(),
            "outgoing_projectile_velocity_world": self._last_outgoing_velocity.copy(),
            "outgoing_velocity_valid": float(contact),
            "post_contact_separated": float(post_contact_separated),
            "episode_parameters": self._episode_parameters.copy(),
            **reward_components,
            "true_state": self._true_diagnostics(),
        }
        return self._observation(), float(reward), terminated, truncated, info

"""Modern MuJoCo/Gymnasium port of SpaceRobotEnv's free-floating foundation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SpaceRobotConfig:
    sim_dt: float = 0.001
    control_dt: float = 0.005
    policy_dt: float = 0.250
    initial_arm_q: tuple[float, ...] = (
        0.0,
        -np.pi / 2,
        np.pi / 2,
        -np.pi / 2,
        np.pi / 2,
        0.0,
    )

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_arm_q, dtype=float)
        if initial.shape != (6,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_arm_q must be a finite six-vector")
        if self.sim_dt <= 0 or self.control_dt <= 0 or self.policy_dt <= 0:
            raise ValueError("simulation, control, and policy periods must be positive")
        self._integer_ratio(self.control_dt, self.sim_dt, "control_dt/sim_dt")
        self._integer_ratio(self.policy_dt, self.control_dt, "policy_dt/control_dt")

    @staticmethod
    def _integer_ratio(numerator: float, denominator: float, name: str) -> int:
        ratio = numerator / denominator
        rounded = round(ratio)
        if not np.isclose(ratio, rounded):
            raise ValueError(f"{name} must be an integer")
        return rounded

    @property
    def physics_steps_per_control(self) -> int:
        return self._integer_ratio(self.control_dt, self.sim_dt, "control_dt/sim_dt")

    @property
    def physics_steps_per_action(self) -> int:
        return self._integer_ratio(self.policy_dt, self.sim_dt, "policy_dt/sim_dt")


def default_scene_path() -> Path:
    return Path(__file__).resolve().parents[3] / "assets" / "mjcf" / "scene.xml"


class SpaceRobotEnv(gym.Env[Any, np.ndarray]):
    """Base environment preserving SpaceRobotEnv's simple MuJoCo ownership model."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    ARM_JOINT_NAMES = (
        "arm:shoulder_pan_joint",
        "arm:shoulder_lift_joint",
        "arm:elbow_joint",
        "arm:wrist_1_joint",
        "arm:wrist_2_joint",
        "arm:wrist_3_joint",
    )

    def __init__(
        self,
        config: SpaceRobotConfig | None = None,
        *,
        scene_path: str | Path | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.config = config if config is not None else SpaceRobotConfig()
        # The frozen clock configuration has already validated these ratios.
        # Cache them once rather than repeating NumPy-based validation in every
        # 1 ms physics iteration.
        self._physics_steps_per_control = self.config.physics_steps_per_control
        self._physics_steps_per_action = self.config.physics_steps_per_action
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(
            str(Path(scene_path) if scene_path is not None else default_scene_path())
        )
        self.model.opt.timestep = self.config.sim_dt
        self.data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None
        self._spacecraft_body_id = self.model.body("spacecraft_bus").id
        self._spacecraft_joint_id = self.model.joint("spacecraft_free").id
        self._projectile_body_id = self.model.body("projectile").id
        self._projectile_joint_id = self.model.joint("projectile_free").id
        self._shield_geom_id = self.model.geom("shield_geom").id
        self._projectile_geom_id = self.model.geom("projectile_geom").id
        self._arm_joint_ids = np.array(
            [self.model.joint(name).id for name in self.ARM_JOINT_NAMES]
        )
        self._arm_qpos_indices = self.model.jnt_qposadr[self._arm_joint_ids]
        self._arm_dof_indices = self.model.jnt_dofadr[self._arm_joint_ids]

    @property
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self._arm_qpos_indices].copy()

    @property
    def arm_qd(self) -> np.ndarray:
        return self.data.qvel[self._arm_dof_indices].copy()

    @property
    def projectile_position_world(self) -> np.ndarray:
        return self.data.xpos[self._projectile_body_id].copy()

    @property
    def projectile_velocity_world(self) -> np.ndarray:
        dof = self.model.jnt_dofadr[self._projectile_joint_id]
        return self.data.qvel[dof : dof + 3].copy()

    @property
    def spacecraft_position_world(self) -> np.ndarray:
        return self.data.xpos[self._spacecraft_body_id].copy()

    @property
    def spacecraft_rotation_world(self) -> np.ndarray:
        return self.data.xmat[self._spacecraft_body_id].reshape(3, 3).copy()

    @property
    def spacecraft_velocity_world(self) -> np.ndarray:
        dof = self.model.jnt_dofadr[self._spacecraft_joint_id]
        return self.data.qvel[dof : dof + 3].copy()

    @property
    def spacecraft_angular_velocity(self) -> np.ndarray:
        dof = self.model.jnt_dofadr[self._spacecraft_joint_id]
        return self.data.qvel[dof + 3 : dof + 6].copy()

    @property
    def spacecraft_angular_momentum_world(self) -> np.ndarray:
        """Angular momentum of the bus-and-arm subtree about its center of mass."""
        mujoco.mj_subtreeVel(self.model, self.data)
        return self.data.subtree_angmom[self._spacecraft_body_id].copy()

    @property
    def spacecraft_quaternion_world(self) -> np.ndarray:
        qpos = self.model.jnt_qposadr[self._spacecraft_joint_id]
        return self.data.qpos[qpos + 3 : qpos + 7].copy()

    def _reset_physics(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._arm_qpos_indices] = self.config.initial_arm_q
        projectile_qpos = self.model.jnt_qposadr[self._projectile_joint_id]
        self.data.qpos[projectile_qpos : projectile_qpos + 3] = [0.0, -2.0, 1.7]
        self.data.qpos[projectile_qpos + 3 : projectile_qpos + 7] = [1, 0, 0, 0]
        projectile_dof = self.model.jnt_dofadr[self._projectile_joint_id]
        self.data.qvel[projectile_dof : projectile_dof + 3] = [0.0, 0.6, 0.0]
        mujoco.mj_forward(self.model, self.data)

    def _step_physics(self, torque: npt.ArrayLike) -> None:
        self.data.ctrl[:] = np.asarray(torque, dtype=float)
        mujoco.mj_step(self.model, self.data)

    def shield_projectile_contact(self) -> bool:
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {contact.geom1, contact.geom2}
            if pair == {self._shield_geom_id, self._projectile_geom_id}:
                return True
        return False

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="overview")
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError

    def step(
        self, action: np.ndarray
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        raise NotImplementedError
